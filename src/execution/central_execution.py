from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import BarData
from ibapi.order_state import OrderState
from ibapi.execution import Execution
import math
import threading
from ledger.position_ledger import PositionLedger
from risk.risk_manager import RiskManager
from logger.event_logger import EventLogger

from typing import Dict, Optional, Literal
import pandas as pd
import time

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Literal

from config import CONFIG, GLOBAL
from config import ATR_EXECUTION

import pandas_market_calendars as mcal
from datetime import datetime, time as dtime
import pytz

from functools import lru_cache


@lru_cache(maxsize=4)
def _market_calendar(exchange: str):
    return mcal.get_calendar(exchange)


_schedule_cache: dict = {}   # (exchange, date) -> (open_dt, close_dt) or None


def is_market_open(exchange: str = "NYSE") -> bool:
    now_et = datetime.now(pytz.timezone("America/New_York"))
    key = (exchange, now_et.date())
    if key not in _schedule_cache:
        sched = _market_calendar(exchange).schedule(start_date=now_et.date(), end_date=now_et.date())
        if sched.empty:
            _schedule_cache[key] = None
        else:
            o = sched.iloc[0]["market_open"].tz_convert("America/New_York")
            c = sched.iloc[0]["market_close"].tz_convert("America/New_York")
            _schedule_cache[key] = (o, c)
    window = _schedule_cache[key]
    return window is not None and window[0] <= now_et <= window[1]

import logging
logger = logging.getLogger("executor")

# import these from a seperate file later
class Instrument(BaseModel):
    symbol: str
    asset_class: str
    exchange: str = "SMART"
    sec_type: str = "STK"                    # "STK" (default) or "FUT"
    multiplier: Optional[float] = None       # futures contract multiplier
    last_trade_date: Optional[str] = None    # futures expiry "YYYYMM" or "YYYYMMDD"

class OrderIntent(BaseModel):
    expected_price: Optional[float] = None
    strategy_id: str
    client_order_id: str
    timestamp: str
    schema_version: str
    instrument: Instrument
    intent_type: Literal["delta", "target_position"]

    # used only when intent_type == "delta"
    side: Optional[Literal["buy", "sell"]] = None
    quantity: Optional[float] = None

    # used only when intent_type == "target_position" — signed, no "side" needed
    target_quantity: Optional[float] = None

    order_type: Literal["market", "limit"]
    limit_price: Optional[float] = None
    time_in_force: str = "day"
    metadata: dict = Field(default_factory=dict)

    @field_validator("limit_price")
    @classmethod
    def limit_price_required_for_limit_orders(cls, v, info):
        if info.data.get("order_type") == "limit" and v is None:
            raise ValueError("limit_price is required when order_type is 'limit'")
        return v

    @model_validator(mode="after")
    def validate_intent_fields(self):
        if self.intent_type == "delta":
            if self.side is None or self.quantity is None:
                raise ValueError("'side' and 'quantity' are required when intent_type is 'delta'")
            if self.quantity <= 0:
                raise ValueError("quantity must be positive for delta intents")
        elif self.intent_type == "target_position":
            if self.target_quantity is None:
                raise ValueError("'target_quantity' is required when intent_type is 'target_position'")
            if self.side is not None or self.quantity is not None:
                raise ValueError("'side'/'quantity' should not be set for target_position intents — use 'target_quantity'")

        if self.order_type == "market" and self.expected_price is None:
            raise ValueError("expected_price is required for market orders "
                         "(a systematic strategy always has a reference price at signal time)")
        return self

class CentralExecutor(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)

        # --- order-flow state (owned by the executor) ---
        self._next_order_id: Optional[int] = None
        self._order_id_ready = threading.Event()
        self._order_id_lock = threading.Lock()
        self.order_status: Dict[int, dict] = {}
        self._seen_client_order_ids: Dict[str, Optional[int]] = {}
        self._dedup_lock = threading.Lock()
        self._killed = False  # kill-switch flag (Phase 4)
        self._enforce_market_hours = True  # reject orders while market closed (per-intent override: metadata.allow_when_closed)
        self._multipliers = {}          # symbol -> contract multiplier (1 for equities)
        self._ref_value = {}            # symbol -> price*multiplier, for multiplier-aware risk notional
        self._contract_details = {}     # reqId -> [Contract] (futures front-month resolution)
        self._contract_details_end = {} # reqId -> Event
        self._instruments = {}          # symbol -> instrument dict (to rebuild a contract when flattening)

        # --- position/risk state (owned by their components, NOT duplicated here) ---
        self.ledger = PositionLedger(self, CONFIG)   # CONFIG seeds each strategy's cash position
        self.risk_manager = RiskManager(self.ledger, CONFIG, GLOBAL)
        self._pending_price_reqs: Dict[int, threading.Event] = {}   # reqId -> event fired when price arrives
        self._price_results: Dict[int, float] = {}                   # reqId -> price received
        self._price_req_lock = threading.Lock()
        self._mkt_data_req_id = 9000       
        # Paper fill helper: streaming mkt data subs that keep the paper fill engine alive
        self._paper_mkt_subs: Dict[str, int] = {}   # symbol -> reqId of active streaming sub
        self._paper_mkt_refcount: Dict[str, int] = {}  # symbol -> count of pending orders

        self._open_orders_ready = threading.Event()                           # base, kept away from order IDs
        self._reconcile_mode = False                                             # open-order reconcile flag
        self._reconcile_ib_orders: Dict[int, dict] = {}                          # temp: orderId -> {contract, order, state}
        self._reconcile_orders_done = threading.Event()                          # signalled by openOrderEnd in reconcile mode

        # EventLogger
        self.logger_db = EventLogger()

        self._mark_cache: Dict[str, float] = {}   # symbol -> last good mark
        self._mark_ts: Dict[str, float] = {}       # symbol -> time.time() of last good mark (staleness guard)
        self._mark_lock = threading.Lock()
        self._conn = {"host": "127.0.0.1", "port": 4002, "client_id": 5}  # remembered for auto-reconnect
        self._reconnecting = False
        self._whatif: Dict[int, dict] = {}         # orderId -> margin impact (whatIf openOrder)
        self._whatif_events: Dict[int, threading.Event] = {}
        self._startup_degraded = False   # True if startup reconciliation never succeeded
        self._flatten_retry_ts: Dict[str, float] = {}  # strat -> last flatten retry (ensure_flat)
        self._readonly_retries: Dict[str, int] = {}    # symbol -> read-only retries this episode
        self._daily_baseline: Optional[float] = None  # portfolio equity baseline for the circuit breaker
        self._circuit_broken = False
        self.coordinator = None   # NettingCoordinator (net-pooling); set by server lifespan
        self._api_ready = threading.Event()  # set once Gateway exits read-only mode
        self._alerter = None                 # set by server lifespan (Alerter instance for Telegram DMs)

        # ATR limit-at-pullback execution layer
        from execution.atr_execution import AtrPullbackLayer

        # make meta class to replace this like self.execution_strat 
        self.atr_layer = AtrPullbackLayer(ATR_EXECUTION, executor=self)

        # Historical data infrastructure (used by ATR layer)
        self._hist_data: Dict[int, list] = {}          # reqId -> [BarData, ...]
        self._hist_events: Dict[int, threading.Event] = {}  # reqId -> Event
        self._hist_req_id = 20000
    # ------------------------------------------------------------------
    # Contract builders
    # ------------------------------------------------------------------
    @staticmethod
    def get_contract(symbol: str, sec_type: str, exchange: str, currency: str, **kwargs) -> Contract:
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        for key, value in kwargs.items():
            setattr(contract, key, value)
        return contract

    @staticmethod
    def get_stock_contract(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Contract:
        return CentralExecutor.get_contract(symbol, sec_type="STK", exchange=exchange, currency=currency)

    @staticmethod
    def get_future_contract(symbol: str, exchange: str = "NYMEX", last_trade_date=None,
                            multiplier=None, currency: str = "USD") -> Contract:
        kwargs = {}
        if last_trade_date:
            kwargs["lastTradeDateOrContractMonth"] = str(last_trade_date)
        if multiplier is not None:
            kwargs["multiplier"] = str(int(multiplier))
        return CentralExecutor.get_contract(symbol, sec_type="FUT", exchange=exchange, currency=currency, **kwargs)

    @staticmethod
    def get_forex_contract(pair: str, exchange: str = "IDEALPRO", currency: str = "USD") -> Contract:
        base, quote = pair.split(".")
        return CentralExecutor.get_contract(base, sec_type="CASH", exchange=exchange, currency=quote)

    # ------------------------------------------------------------------
    # Order ID management
    # ------------------------------------------------------------------
    def nextValidId(self, orderId: int) -> None:
        self._next_order_id = orderId
        self._order_id_ready.set()
        logger.info("Next valid order ID: %s", orderId)

    def error(self, reqId: int, errorCode: int, errorString: str,
              advancedOrderRejectJson: str = "") -> None:
        """IB API error / warning callback. Logs everything; order-specific
        errors (reqId matches an open order) are tagged so they stand out.
        Code 321 (read-only) triggers a background retry of the order."""
        if reqId in self.order_status:
            sym = self.order_status[reqId].get("symbol", "?")
            self.order_status[reqId]["last_error"] = {
                "code": errorCode, "message": errorString[:300]}
            if errorCode in self.CANCEL_CONFIRMED_CODES:
                # IB answering OUR cancel, not refusing the order. 202 used to be labelled a
                # rejection (and logged at ERROR), so every successful cancel read as refused.
                logger.info("IB confirmed cancel  orderId=%s  sym=%s  code=%s  %s",
                            reqId, sym, errorCode, errorString)
                self._release_pending(reqId, f"IB code {errorCode}")
            elif errorCode in self.CANCEL_REFUSED_CODES:
                # The cancel failed, usually because the order had already filled. Its
                # pending was released when the cancel was sent, and any fill still to
                # arrive is compensated in _note_fill, so there is nothing to undo here.
                logger.warning("cancel refused  orderId=%s  sym=%s  code=%s  %s",
                               reqId, sym, errorCode, errorString)
            else:
                logger.error("IB ORDER ERROR  orderId=%s  sym=%s  code=%s  %s  %s",
                             reqId, sym, errorCode, errorString,
                             advancedOrderRejectJson or "")
                # Record WHY, so the dashboard and the submitting strategy can both see it
                # instead of watching an order sit at "Submitted" forever.
                self.order_status[reqId]["ack"] = "rejected"
                # --- retry on read-only (code 321) ---
                if errorCode == 321:
                    self._retry_readonly_order(reqId)
        elif errorCode in (2104, 2106, 2158):
            # data-farm connection messages — informational
            logger.debug("IB info  code=%s  %s", errorCode, errorString)
        else:
            logger.warning("IB error  reqId=%s  code=%s  %s  %s",
                           reqId, errorCode, errorString,
                           advancedOrderRejectJson or "")

    #: read-only resubmissions allowed per symbol, per episode. Counted across the whole
    #: chain rather than per order — see _retry_readonly_order.
    READONLY_MAX_RETRIES = 2

    def _retry_readonly_order(self, failed_order_id: int, delay: float = 15.0) -> None:
        """Resubmit an order IB rejected with code 321 (gateway in read-only mode).

        The cap is per SYMBOL and spans the whole chain, which is the only thing that
        actually bounds this. Counting per order does not: the resubmission gets a new
        order id, IB rejects that one too, and ITS error callback starts a fresh chain with
        a fresh budget. What looks like "at most N retries" is then an unbounded loop that
        only a process restart clears — orders accumulate every `delay` seconds for as long
        as the gateway stays read-only.

        Giving up is CRITICAL rather than a warning: the order was never placed, so silence
        here means a position the strategy believes it has and does not.
        """
        info = self.order_status.get(failed_order_id)
        if not info:
            return
        sym = info.get("symbol", "?")
        pending = info.get("pending_qty", 0)
        if abs(pending) < 1e-9:
            return

        attempts = self._readonly_retries.get(sym, 0)
        if attempts >= self.READONLY_MAX_RETRIES:
            # This order will never fill, and IB sends no Inactive for a 321, so without a
            # release its pending sat in the ledger until the process restarted.
            self._release_pending(failed_order_id, "read-only retries exhausted")
            logger.critical(
                "READ-ONLY GIVE UP %s: %d/%d retries used and the gateway is still "
                "refusing orders. %s %g %s was NOT placed and will not be retried — fix the "
                "gateway's read-only setting, then resubmit.",
                sym, attempts, self.READONLY_MAX_RETRIES,
                "sell" if pending < 0 else "buy", abs(pending), sym)
            return
        self._readonly_retries[sym] = attempts + 1

        def _retry():
            logger.warning("READ-ONLY RETRY %s: attempt %d/%d in %.0fs",
                           sym, attempts + 1, self.READONLY_MAX_RETRIES, delay)
            time.sleep(delay)
            try:
                instrument = self._instruments.get(sym, {"symbol": sym})
                ref_price = info.get("expected_price")
                # Release ONLY the failed order's remainder before resubmitting. The pooled
                # path used to pop ALL pending for the symbol — wiping other orders still
                # working in it — and the direct path released nothing at all, so every
                # retried direct order counted its shares twice.
                self._release_pending(failed_order_id, "read-only retry")
                if info.get("net", False):
                    oid = self.place_net_order(sym, pending, instrument, ref_price)
                else:
                    intent = {
                        "client_order_id": f"retry-{sym}-{int(time.time() * 1000)}",
                        "strategy_id": info.get("strategy_id", "unknown"),
                        "instrument": instrument,
                        "side": "buy" if pending > 0 else "sell",
                        "quantity": abs(pending),
                        "order_type": "market",
                        "time_in_force": "day",
                        "expected_price": ref_price,
                    }
                    oid = self.place_order(self.atr_layer.transform(intent))
                if oid:
                    logger.info("READ-ONLY RETRY %s: resubmitted as orderId=%s", sym, oid)
            except Exception as e:
                logger.warning("READ-ONLY RETRY %s attempt %d error: %s",
                               sym, attempts + 1, e)

        threading.Thread(target=_retry, daemon=True, name=f"retry-321-{sym}").start()

    def clear_readonly_retries(self, symbol: str = None) -> None:
        """Forget the retry budget — the gateway is accepting orders again.

        Called on every fill, so a read-only episode months from now starts from a full
        budget instead of being suppressed by a counter left over from this one."""
        if symbol is None:
            self._readonly_retries.clear()
        else:
            self._readonly_retries.pop(symbol, None)

    def get_next_order_id(self, timeout: float = 5.0) -> int:
        if not self._order_id_ready.wait(timeout=timeout):
            raise TimeoutError("Timed out waiting for nextValidId — did connect() actually succeed?")
        with self._order_id_lock:
            order_id = self._next_order_id
            self._next_order_id += 1
        return order_id

    # ------------------------------------------------------------------
    # Order construction & placement
    # ------------------------------------------------------------------
    @staticmethod
    def build_order(intent: dict) -> Order:
        order = Order()
        order.action = intent["side"].upper()
        order.totalQuantity = intent["quantity"]

        if intent["order_type"] == "market":
            order.orderType = "MKT"
        elif intent["order_type"] == "limit":
            order.orderType = "LMT"
            order.lmtPrice = intent["limit_price"]
        elif intent["order_type"] == "stop":
            order.orderType = "STP"
            order.auxPrice = intent["stop_price"]
        elif intent["order_type"] == "stop_limit":
            order.orderType = "STP LMT"
            order.auxPrice = intent["stop_price"]
            order.lmtPrice = intent["limit_price"]
        else:
            raise ValueError(f"Unsupported order_type: {intent['order_type']}")

        tif_map = {"day": "DAY", "gtc": "GTC"}
        order.tif = tif_map.get(intent.get("time_in_force", "day"), "DAY")
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        return order

    # ------------------------------------------------------------------
    # Paper fill helper: streaming market data to wake the fill simulator
    # ------------------------------------------------------------------
    def _paper_subscribe(self, symbol: str, contract) -> None:
        """Start a streaming market data subscription for a symbol if one isn't
        already active. The paper trading engine only simulates fills when
        market data is flowing for that symbol."""
        with self._price_req_lock:
            if symbol in self._paper_mkt_subs:
                self._paper_mkt_refcount[symbol] = self._paper_mkt_refcount.get(symbol, 0) + 1
                return
            self._mkt_data_req_id += 1
            req_id = self._mkt_data_req_id
            self._paper_mkt_subs[symbol] = req_id
            self._paper_mkt_refcount[symbol] = 1
        # Accept delayed/frozen data — paper account may not have live subscriptions,
        # but the fill simulator still works with delayed data flowing
        self.reqMarketDataType(4)
        # snapshot=False -> streaming; keeps the fill simulator active for this symbol
        self.reqMktData(req_id, contract, "", False, False, [])
        logger.debug("paper fill sub: started streaming mkt data for %s (reqId=%d)", symbol, req_id)
        # Restore to type 3 for other callers (snapshot price fetches)
        self.reqMarketDataType(3)

    def _paper_unsubscribe(self, symbol: str) -> None:
        """Decrement the refcount and cancel the streaming sub when no pending orders remain."""
        with self._price_req_lock:
            count = self._paper_mkt_refcount.get(symbol, 0) - 1
            if count > 0:
                self._paper_mkt_refcount[symbol] = count
                return
            self._paper_mkt_refcount.pop(symbol, None)
            req_id = self._paper_mkt_subs.pop(symbol, None)
        if req_id is not None:
            self.cancelMktData(req_id)
            logger.debug("paper fill sub: cancelled streaming mkt data for %s (reqId=%d)", symbol, req_id)

    def place_order(self, intent: dict) -> int:
        instrument = intent["instrument"]
        if instrument.get("sec_type", "STK") == "FUT":
            contract = self.get_future_contract(
                instrument["symbol"], exchange=instrument.get("exchange", "NYMEX"),
                last_trade_date=instrument.get("last_trade_date"),
                multiplier=instrument.get("multiplier"),
            )
        else:
            contract = self.get_stock_contract(instrument["symbol"], exchange=instrument.get("exchange", "SMART"))
        order = self.build_order(intent)

        order_id = self.get_next_order_id()
        self.placeOrder(order_id, contract, order)
        self.logger_db.log_order(order_id, intent)
        self._paper_subscribe(instrument["symbol"], contract)
        signed_qty = intent["quantity"] if intent["side"] == "buy" else -intent["quantity"]
        symbol = instrument["symbol"]

        # multiplier-aware risk notional bookkeeping
        _mult = float(instrument.get("multiplier") or 1.0)
        self._multipliers[symbol] = _mult
        self.ledger.multipliers[symbol] = _mult          # dollar-denominate P&L / drawdown
        self._instruments[symbol] = dict(instrument)     # remembered so we can flatten later
        _px = intent.get("expected_price") or intent.get("limit_price")
        if _px:
            self._ref_value[symbol] = float(_px) * _mult

        # route pending exposure through the ledger, not a local dict
        self.ledger.record_pending(symbol, signed_qty, intent["strategy_id"])

        self.order_status[order_id] = {
            "client_order_id": intent["client_order_id"],
            "strategy_id": intent["strategy_id"],
            "symbol": symbol,
            # "Submitted" here is OUR word, written when the order goes to the socket —
            # IB has said nothing yet, and its own callback later writes the identical
            # string. `ack` is the part that distinguishes them: nothing downstream could
            # tell "we sent it" from "IB accepted it", which is how 27 orders IB never had
            # showed on the dashboard as working.
            "status": "Submitted",
            "ack": "pending",          # pending -> live (IB knows) | rejected (IB refused)
            "sent_at": time.time(),
            "filled": 0,
            "remaining": intent["quantity"],
            "pending_qty": signed_qty,
            "expected_price": intent.get("expected_price"),
            # How the order was actually WORKED, which the dashboard could not show before:
            # order_status recorded neither the type nor the limit. An ATR-transformed order
            # looked identical to a market order that had simply not filled yet, so a limit
            # resting away from the market was indistinguishable from a stuck order.
            "order_type": intent.get("order_type"),
            "limit_price": intent.get("limit_price"),
            "execution_layer": (intent.get("metadata") or {}).get("execution_layer"),
            "exec_filled": 0.0,          # signed, from execDetails — what really traded
            "pending_released": False,   # set once the unfilled remainder leaves pending
        }
        return order_id

    def place_net_order(self, symbol: str, delta: float, instrument: dict, ref_price, urgent: bool = False):
        """Pooled net order (coordinator path): trade the whole net delta for a symbol
        under the synthetic '__net__' strategy. Pending is tracked at the NET level via
        record_net_pending (NOT per-strategy); the fill is decomposed into per-strategy
        sub-fills by the coordinator in execDetails. `delta` is signed (buy>0 / sell<0)."""
        if abs(delta) < 1e-9:
            return None
        instrument = dict(instrument or {"symbol": symbol})
        sym = instrument.get("symbol", symbol)
        side = "buy" if delta > 0 else "sell"
        intent = {
            "client_order_id": f"net-{sym}-{int(time.time() * 1000)}",
            "strategy_id": "__net__",
            "instrument": instrument,
            "side": side,
            "quantity": abs(delta),
            "order_type": "market",
            "time_in_force": "day",
            "expected_price": ref_price,
        }
        # --- ATR execution layer: transform market -> limit-at-pullback ---
        # Skip ATR for urgent orders (flatten / kill_switch) — must close at market.
        # For pooled orders (strategy_id == "__net__"), the ATR layer's strategy filter
        # would never match, so we check if any CONTRIBUTING strategy is ATR-eligible
        # and temporarily swap the strategy_id so the filter passes.

        # change this when I add more execution strats
        if not urgent:
            atr_strats = set(self.atr_layer.strategies)
            if atr_strats and self.coordinator:
                # Find which strategies are driving this symbol's delta
                contributors = {sid for sid, book in self.coordinator.desired.items()
                                if sym in book and abs(book.get(sym, 0)) > 1e-9}
                eligible = contributors & atr_strats
                if eligible:
                    intent["strategy_id"] = next(iter(eligible))
                    intent = self.atr_layer.transform(intent)
                    intent["strategy_id"] = "__net__"
                # else: no ATR-eligible strategy contributes → stays market
            else:
                # No strategy filter (empty list = all) → apply to everything
                intent = self.atr_layer.transform(intent)

        # need to adapt this for more asset classes as well
        if instrument.get("sec_type", "STK") == "FUT":
            contract = self.get_future_contract(
                sym, exchange=instrument.get("exchange", "NYMEX"),
                last_trade_date=instrument.get("last_trade_date"),
                multiplier=instrument.get("multiplier"),
            )
        else:
            contract = self.get_stock_contract(sym, exchange=instrument.get("exchange", "SMART"))
        order = self.build_order(intent)

        order_id = self.get_next_order_id()
        self.placeOrder(order_id, contract, order)
        self.logger_db.log_order(order_id, intent)
        self._paper_subscribe(sym, contract)
        # track ATR-placed limit orders for EOD cancel sweep
        if intent.get("metadata", {}).get("atr_execution"):
            self.atr_layer.record_order(order_id)

        _mult = float(instrument.get("multiplier") or 1.0)
        self._multipliers[sym] = _mult
        self.ledger.multipliers[sym] = _mult
        self._instruments[sym] = dict(instrument)
        if ref_price:
            self._ref_value[sym] = float(ref_price) * _mult

        # NET pending only — attribution to strategies happens on the fill
        self.ledger.record_net_pending(sym, delta)

        self.order_status[order_id] = {
            "client_order_id": intent["client_order_id"],
            "strategy_id": "__net__",
            "symbol": sym,
            # "Submitted" here is OUR word, written when the order goes to the socket —
            # IB has said nothing yet, and its own callback later writes the identical
            # string. `ack` is the part that distinguishes them: nothing downstream could
            # tell "we sent it" from "IB accepted it", which is how 27 orders IB never had
            # showed on the dashboard as working.
            "status": "Submitted",
            "ack": "pending",          # pending -> live (IB knows) | rejected (IB refused)
            "sent_at": time.time(),
            "filled": 0,
            "remaining": abs(delta),
            "pending_qty": delta,
            "expected_price": ref_price,
            "order_type": intent.get("order_type"),
            "limit_price": intent.get("limit_price"),
            "execution_layer": (intent.get("metadata") or {}).get("execution_layer"),
            "exec_filled": 0.0,
            "pending_released": False,
            "net": True,
        }
        return order_id

    def orderStatus(self, orderId: int, status: str, filled: float, remaining: float,
                    avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float,
                    clientId: int, whyHeld: str, mktCapPrice: float) -> None:
        if orderId in self.order_status:
            self.order_status[orderId].update({
                "status": status, "filled": filled,
                "remaining": remaining, "avg_fill_price": avgFillPrice,
                # IB has spoken about this order, so it is real at the broker. "Inactive"
                # is IB's way of saying it refused it.
                "ack": "rejected" if status == "Inactive" else "live",
            })
            self.logger_db.update_order_status(orderId, status)
            # IB ended the order without filling the rest: a day order expiring, a rejection,
            # a cancel from TWS or from us. Nothing released pending here before, so every
            # order IB ended on its own left shares the ledger believed were still on the way
            # — and the next rebalance treated that symbol as already on target.
            if status in self.TERMINAL_UNFILLED:
                self._release_pending(orderId, f"IB {status}")
            # Cancel the paper-fill streaming sub once the order is done
            if status in ("Filled", "Cancelled", "ApiCancelled", "Inactive"):
                sym = self.order_status[orderId].get("symbol")
                if sym:
                    self._paper_unsubscribe(sym)
        logger.debug("OrderStatus - id:%s status:%s filled:%s remaining:%s", orderId, status, filled, remaining)

    # ------------------------------------------------------------------
    # Intent processing (Phase 2 + Phase 4 risk check)
    # ------------------------------------------------------------------
    def process_intent(self, raw_intent: dict) -> dict:
        if self._killed:
            return {"accepted": False, "reason": "executor is in kill-switch state"}

        # --- ATR execution layer: transform market -> limit-at-pullback ---
        raw_intent = self.atr_layer.transform(raw_intent)

        try:
            intent = OrderIntent(**raw_intent)
        except Exception as e:
            return {"accepted": False, "reason": f"schema validation failed: {e}"}

        _is_future = getattr(intent.instrument, "sec_type", "STK") == "FUT"
        if (self._enforce_market_hours and not _is_future
                and not is_market_open() and not intent.metadata.get("allow_when_closed")):
            return {"accepted": False,
                    "reason": "market closed — order not submitted "
                              "(set metadata.allow_when_closed=true to queue for open)"}
        if self._should_pool(intent):
            return self._submit_pooled(intent)
        with self._dedup_lock:
            if intent.client_order_id in self._seen_client_order_ids:
                existing_order_id = self._seen_client_order_ids[intent.client_order_id]
                return {"accepted": True, "order_id": existing_order_id,
                        "note": "duplicate client_order_id — returning existing order, not resubmitting"}
            self._seen_client_order_ids[intent.client_order_id] = None

            try:
                resolved_intent = self._resolve_intent_type(intent)
                resolved_delta = resolved_intent["quantity"] * (1 if resolved_intent["side"] == "buy" else -1)

                # --- Phase 4: risk check, after resolution, before placing ---
                reference_price = self._reference_price(resolved_intent)
                _mult = float(resolved_intent["instrument"].get("multiplier") or 1.0)
                risk_result = self.risk_manager.check_order(
                    resolved_intent, resolved_delta, reference_price,
                    multiplier=_mult, ref_values=self._ref_value,
                )
                if not risk_result["approved"]:
                    del self._seen_client_order_ids[intent.client_order_id]
                    return {"accepted": False, "reason": risk_result["reason"]}

                if GLOBAL.get("pretrade_margin_check"):
                    cap = GLOBAL.get("max_order_init_margin")
                    wi = self.margin_whatif(resolved_intent) or {}
                    im = wi.get("init_margin")
                    if cap is not None and im is not None and im > cap:
                        del self._seen_client_order_ids[intent.client_order_id]
                        return {"accepted": False,
                                "reason": f"pre-trade init margin {im:.0f} > cap {cap:.0f}"}

                order_id = self.place_order(resolved_intent)
                # track ATR-placed limit orders for EOD cancel sweep
                if resolved_intent.get("metadata", {}).get("atr_execution"):
                    self.atr_layer.record_order(order_id)
            except Exception as e:
                del self._seen_client_order_ids[intent.client_order_id]
                return {"accepted": False, "reason": str(e)}

            self._seen_client_order_ids[intent.client_order_id] = order_id

        return {"accepted": True, "order_id": order_id}
    
    def tickPrice(self, reqId: int, tickType: int, price: float, attrib) -> None:
        # 4 = last, 68 = delayed-last, 9 = close (fallbacks in preference order)
        RELEVANT_TICKS = {4, 68, 9}
        if tickType not in RELEVANT_TICKS or price is None or price <= 0:
            return  # IB sends -1 when no data available; ignore
        with self._price_req_lock:
            if reqId in self._pending_price_reqs and reqId not in self._price_results:
                self._price_results[reqId] = price
                self._pending_price_reqs[reqId].set()   # unblock the waiting pull

    def fetch_price(self, symbol: str, exchange: str = "SMART", timeout: float = 3.0) -> Optional[float]:
        contract = self.get_stock_contract(symbol, exchange=exchange)

        with self._order_id_lock:  # reuse a lock to hand out unique market-data reqIds
            self._mkt_data_req_id += 1
            req_id = self._mkt_data_req_id

        event = threading.Event()
        with self._price_req_lock:
            self._pending_price_reqs[req_id] = event

        try:
            # snapshot=True returns a one-off snapshot then auto-cancels — cleaner than streaming
            self.reqMktData(req_id, contract, "", True, False, [])
            if not event.wait(timeout=timeout):
                logger.warning("price fetch for %s timed out", symbol)
                return None
            with self._price_req_lock:
                return self._price_results.get(req_id)
        finally:
            # clean up state; snapshot auto-cancels but cancel anyway to be safe
            self.cancelMktData(req_id)
            with self._price_req_lock:
                self._pending_price_reqs.pop(req_id, None)
                self._price_results.pop(req_id, None)

    # ------------------------------------------------------------------
    # Historical data (IBKR reqHistoricalData) — used by ATR layer
    # ------------------------------------------------------------------
    def historicalData(self, reqId: int, bar: BarData) -> None:
        """EWrapper callback: one bar at a time."""
        self._hist_data.setdefault(reqId, []).append(bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        """EWrapper callback: all bars delivered."""
        ev = self._hist_events.get(reqId)
        if ev:
            ev.set()

    def fetch_atr(self, symbol: str, period: int = 14,
                  bar_size: str = "5 mins", duration: str = "2 D",
                  timeout: float = 10.0) -> Optional[float]:
        """Fetch intraday bars from IBKR and compute ATR(period).
        Returns the ATR value or None on failure."""
        contract = self.get_stock_contract(symbol)

        with self._order_id_lock:
            self._hist_req_id += 1
            req_id = self._hist_req_id

        event = threading.Event()
        self._hist_data[req_id] = []
        self._hist_events[req_id] = event

        try:
            self.reqHistoricalData(
                req_id, contract, "",       # endDateTime="" = now
                duration,                   # e.g. "2 D"
                bar_size,                   # e.g. "5 mins"
                "TRADES",                   # whatToShow
                1,                          # useRTH (regular trading hours only)
                1,                          # formatDate (1 = yyyyMMdd HH:mm:ss)
                False,                      # keepUpToDate
                [],                         # chartOptions
            )
            if not event.wait(timeout=timeout):
                logger.warning("historical data fetch for %s timed out", symbol)
                return None

            bars = self._hist_data.get(req_id, [])
            if len(bars) < period + 1:
                logger.warning("ATR: only %d bars for %s (need %d+1)", len(bars), symbol, period)
                return None

            # Compute ATR from the bars
            highs = [b.high for b in bars]
            lows = [b.low for b in bars]
            closes = [b.close for b in bars]

            true_ranges = []
            for i in range(1, len(bars)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
                true_ranges.append(tr)

            if len(true_ranges) < period:
                return None

            atr = sum(true_ranges[-period:]) / period
            return atr

        except Exception as e:
            logger.warning("ATR fetch error for %s: %s", symbol, e)
            return None
        finally:
            self._hist_data.pop(req_id, None)
            self._hist_events.pop(req_id, None)

    def get_marks(self, symbols, timeout: float = 3.0) -> Dict[str, Optional[float]]:
        """{symbol: mark_price}, snapshot-backed with carry-forward.
        A failed fetch reuses the last good mark; never-marked -> None."""
        symbols = list(dict.fromkeys(symbols))
        results: Dict[str, Optional[float]] = {}

        def _one(sym):
            px = None
            try:
                px = self.fetch_price(sym, timeout=timeout)   # concurrent, unique reqIds
            except Exception as e:
                logger.warning("mark fetch failed for %s: %s", sym, e)
            with self._mark_lock:
                if px is not None and px > 0:
                    self._mark_cache[sym] = px
                    self._mark_ts[sym] = time.time()
                results[sym] = self._mark_cache.get(sym)      # carry-forward

        threads = [threading.Thread(target=_one, args=(s,), daemon=True) for s in symbols]
        for t in threads: t.start()
        for t in threads: t.join(timeout=timeout + 1.0)
        return results

    def _reference_price(self, resolved_intent: dict) -> float:
        # limit orders: the limit price is the reference
        if resolved_intent.get("limit_price") is not None:
            return resolved_intent["limit_price"]
        # market orders: expected_price is guaranteed present by schema validation
        return resolved_intent["expected_price"]
        
    def _resolve_intent_type(self, intent: OrderIntent) -> dict:
        symbol = intent.instrument.symbol

        if intent.intent_type == "delta":
            delta = intent.quantity if intent.side == "buy" else -intent.quantity

        elif intent.intent_type == "target_position":
            # measure against current position + orders already working
            effective_incl_pending = self.ledger.effective_position(symbol)
            gap = intent.target_quantity - effective_incl_pending
            if gap == 0:
                # existing working orders already drive us to target — leave them alone
                raise ValueError("no-op: target already covered by position + working orders")
            # target moved — NOW cancel the stale working orders, then size from the settled position
            self._cancel_open_orders_for_symbol(symbol)
            effective_after_cancel = self.ledger.effective_position(symbol)
            delta = intent.target_quantity - effective_after_cancel

        else:
            raise ValueError(f"Unsupported intent_type: {intent.intent_type}")

        if delta == 0:
            raise ValueError("no-op: resolved delta is zero, nothing to submit")

        resolved = intent.model_dump()
        resolved["side"] = "buy" if delta > 0 else "sell"
        resolved["quantity"] = abs(delta)
        return resolved

    #: IB statuses that end an order without filling the rest of it.
    TERMINAL_UNFILLED = frozenset({"Cancelled", "ApiCancelled", "Inactive"})
    #: IB's answers confirming a cancel: 202 cancelled, 10147 no such order at IB.
    CANCEL_CONFIRMED_CODES = frozenset({202, 10147})
    #: IB refusing a cancel: 161 / 10148, usually because the order already filled.
    CANCEL_REFUSED_CODES = frozenset({161, 10148})

    def _release_pending(self, order_id: int, reason: str) -> float:
        """Release an order's UNFILLED REMAINDER from pending — exactly once.

        The single place pending leaves the ledger for an order that ended without filling.
        It used to happen in several places with different rules, and most paths never did
        it: the kill switch, /flatten, the ATR sweep and every cancel IB made on its own
        released nothing, while the netting cancel released the FULL original quantity even
        after a partial fill, and reconcile could release it a second time.

        Remainder = original quantity minus what execDetails has actually filled; pooled
        orders release at the net level, direct orders per strategy. Returns the quantity
        released (0 if already released, fully filled, or unknown)."""
        st = self.order_status.get(order_id)
        if not st or st.get("pending_released"):
            return 0.0
        st["pending_released"] = True
        sym = st.get("symbol")
        original = float(st.get("pending_qty") or 0.0)
        remainder = original - float(st.get("exec_filled") or 0.0)
        if original == 0.0 or remainder * original <= 0.0:
            return 0.0                              # fully filled, or overfilled
        if sym:
            if st.get("net"):
                self.ledger.record_net_pending(sym, -remainder)
            else:
                self.ledger.record_pending(sym, -remainder, st.get("strategy_id", "?"))
            logger.info("released pending %+g %s from order %s (%s)",
                        -remainder, sym, order_id, reason)
        return remainder

    def _note_fill(self, order_id: int, signed_qty: float) -> None:
        """Count a fill against its order, and keep pending right if the order was cancelled.

        A cancel releases pending straight away, because whoever cancelled usually sizes a
        replacement immediately. If the order fills anyway — the cancel lost the race — the
        fill is about to deduct its quantity from pending a second time. Adding it back first
        nets the two out: the position still moves, because the shares really traded, but
        pending does not go negative."""
        st = self.order_status.get(order_id)
        if not st:
            return
        st["exec_filled"] = float(st.get("exec_filled") or 0.0) + signed_qty
        if st.get("pending_released"):
            sym = st.get("symbol")
            if st.get("net"):
                self.ledger.record_net_pending(sym, signed_qty)
            else:
                self.ledger.record_pending(sym, signed_qty, st.get("strategy_id", "?"))
            logger.warning("late fill %+g %s on order %s after its pending was released — "
                           "the cancel lost the race", signed_qty, sym, order_id)

    def cancel_order(self, order_id: int, reason: str = "") -> bool:
        """Cancel an order at IB, mark it PendingCancel, and release its unfilled remainder.

        EVERY cancel path goes through here. Releasing at send time rather than on IB's
        confirmation is deliberate: the netting rebalance and /flatten size their replacement
        orders the moment they have cancelled, so pending must already exclude the cancelled
        shares. If IB refuses the cancel and the order fills, _note_fill compensates.
        Marking PendingCancel also stops a second path cancelling the same order again.
        Returns False if the cancel could not be sent (pending is then left untouched)."""
        st = self.order_status.get(order_id)
        try:
            self.cancelOrder(order_id)
        except Exception as e:
            logger.error("failed to send cancel for order %s (%s): %s", order_id, reason, e)
            return False
        if st is not None:
            st["status"] = "PendingCancel"
            self._release_pending(order_id, f"cancel: {reason}" if reason else "cancel")
        return True

    def _cancel_open_orders_for_symbol(self, symbol: str) -> None:
        for order_id, status in list(self.order_status.items()):
            if status["symbol"] == symbol and status["status"] in ("PreSubmitted", "Submitted"):
                logger.info("Cancelling stale open order %s for %s before resolving new target",
                            order_id, symbol)
                self.cancel_order(order_id, "rebalance")

    # ------------------------------------------------------------------
    # Fill / position callbacks — all delegate to the ledger
    # ------------------------------------------------------------------
    def execDetails(self, reqId: int, contract: Contract, execution: Execution) -> None:
        order_info = self.order_status.get(execution.orderId, {})
        strategy_id = order_info.get("strategy_id", "unknown")
        # A fill proves the gateway is accepting orders, so this symbol's read-only budget
        # goes back to full. Without a reset the cap is one-shot for the life of the
        # process: the next read-only episode would be refused on a stale counter.
        self.clear_readonly_retries(contract.symbol)
        signed_qty = execution.shares if execution.side == "BOT" else -execution.shares
        self._note_fill(execution.orderId, signed_qty)
        if getattr(contract, "multiplier", None):
            try:
                self.ledger.multipliers[contract.symbol] = float(contract.multiplier)
            except (TypeError, ValueError):
                pass

        # Pooled net order: let the coordinator decompose this fill into per-strategy
        # sub-fills (correct P&L even with opposing legs), then check drawdown per book.
        if order_info.get("net") and self.coordinator is not None:
            # order_id lets the coordinator book this fill to the strategies that OWNED the
            # order, rather than whoever has an open gap in the symbol when it arrives.
            attributed = self.coordinator.attribute_fill(contract.symbol, signed_qty, execution.price,
                                                         order_id=execution.orderId)
            # Log the raw net fill (the actual IB execution)
            self.logger_db.log_fill(
                execution.orderId, execution.execId, contract.symbol,
                execution.side, execution.price, execution.shares,
                "__net__", expected_price=order_info.get("expected_price"),
            )
            # Log per-strategy attributed fills so P&L survives restarts
            for i, (sid, sub_qty) in enumerate(attributed):
                sub_side = "BOT" if sub_qty > 0 else "SLD"
                attr_exec_id = f"{execution.execId}-attr-{sid}-{i}"
                self.logger_db.log_fill(
                    execution.orderId, attr_exec_id, contract.symbol,
                    sub_side, execution.price, abs(sub_qty),
                    sid, expected_price=order_info.get("expected_price"),
                )
            self._check_fill_sanity("__net__", contract.symbol, execution.price, order_info.get("expected_price"))
            # Include the strategies this fill was attributed to, not just the ones with a
            # desired book: after a lost/reset netting.json `desired` is empty, which
            # silently disabled the per-fill drawdown check for every strategy.
            for sid in set(self.coordinator.desired) | {sid for sid, _ in attributed}:
                self.enforce_drawdown(sid, self.ledger.strategy_realized_pnl.get(sid, 0.0))
            self.ledger.save_state(self.logger_db)  # persist after every fill
            logger.info("ExecDetails(net) - %s %s %s @ %s (attributed to %s)",
                        contract.symbol, execution.side, execution.shares, execution.price,
                        ", ".join(f"{sid}:{qty:+.1f}" for sid, qty in attributed))
            return

        self.ledger.record_fill(contract.symbol, signed_qty, execution.price, strategy_id)
        self.logger_db.log_fill(
            execution.orderId, execution.execId, contract.symbol,
            execution.side, execution.price, execution.shares,
            order_info.get("strategy_id", "unknown"),
            expected_price=order_info.get("expected_price"),
        )
        self._check_fill_sanity(strategy_id, contract.symbol, execution.price, order_info.get("expected_price"))
        self.enforce_drawdown(strategy_id, self.ledger.strategy_realized_pnl.get(strategy_id, 0.0))  # halt+flatten on breach
        self.ledger.save_state(self.logger_db)  # persist after every fill
        logger.info("ExecDetails - %s %s %s @ %s", contract.symbol, execution.side, execution.shares, execution.price)

    def position(self, account: str, contract: Contract, position: float, avgCost: float) -> None:
        # write to the LEDGER's broker_positions, not a local copy
        self.ledger.broker_positions[contract.symbol] = position
        _mult = None
        if getattr(contract, "multiplier", None):
            try:
                _mult = float(contract.multiplier)
                self.ledger.multipliers[contract.symbol] = _mult
            except (TypeError, ValueError):
                _mult = None
        # remember how to rebuild this contract, so a drawdown flatten works after a restart
        self._instruments.setdefault(contract.symbol, {
            "symbol": contract.symbol,
            "asset_class": "future" if contract.secType == "FUT" else "equity",
            "sec_type": contract.secType or "STK",
            "exchange": contract.exchange or getattr(contract, "primaryExchange", "")
                        or ("NYMEX" if contract.secType == "FUT" else "SMART"),
            "multiplier": _mult,
            "last_trade_date": getattr(contract, "lastTradeDateOrContractMonth", None) or None,
        })
        logger.info("Position - %s: %s @ avg cost %s", contract.symbol, position, avgCost)

    def positionEnd(self) -> None:
        self.ledger._positions_ready.set()
        logger.info("Position snapshot complete")

    # ------------------------------------------------------------------
    # Futures front-month resolution
    # ------------------------------------------------------------------
    def contractDetails(self, reqId, contractDetails):
        self._contract_details.setdefault(reqId, []).append(contractDetails.contract)

    def contractDetailsEnd(self, reqId):
        ev = self._contract_details_end.get(reqId)
        if ev:
            ev.set()

    def resolve_front_month(self, symbol, exchange="NYMEX", currency="USD",
                            roll_buffer_days=5, timeout=8.0):
        """Front-month futures contract, skipping any expiring within roll_buffer_days
        (avoids the physical-delivery-at-expiry rejection). Returns a dict or None."""
        from datetime import datetime, timedelta
        with self._order_id_lock:
            self._mkt_data_req_id += 1
            rid = self._mkt_data_req_id
        ev = threading.Event()
        self._contract_details_end[rid] = ev
        self._contract_details[rid] = []
        c = self.get_contract(symbol, sec_type="FUT", exchange=exchange, currency=currency)
        self.reqContractDetails(rid, c)
        ev.wait(timeout=timeout)
        cands = self._contract_details.pop(rid, [])
        self._contract_details_end.pop(rid, None)
        cutoff = (datetime.now() + timedelta(days=roll_buffer_days)).strftime("%Y%m%d")
        dated = []
        for k in cands:
            exp = k.lastTradeDateOrContractMonth
            expf = exp if len(exp) == 8 else exp + "01"
            if expf >= cutoff:
                dated.append((expf, k))
        if not dated:
            return None
        dated.sort()
        front = dated[0][1]
        return {"last_trade_date": front.lastTradeDateOrContractMonth,
                "multiplier": float(front.multiplier) if front.multiplier else None,
                "local_symbol": front.localSymbol}

    # ------------------------------------------------------------------
    # Kill switch (Phase 4)
    # ------------------------------------------------------------------
    def connectionClosed(self) -> None:
        """IB socket closed. Expected during shutdown; otherwise a hard disconnect (Gateway
        restart, network drop) — log CRITICAL (-> alert) and, if auto_reconnect is on, kick
        off a background reconnect. Nothing trades until the connection is back."""
        if getattr(self, "_shutting_down", False):
            logger.info("IB connection closed (during shutdown)")
            return
        logger.critical("IB connection closed UNEXPECTEDLY \u2014 trading halted until reconnected")
        if GLOBAL.get("auto_reconnect", True) and not self._reconnecting:
            self._reconnecting = True
            threading.Thread(target=self._reconnect_loop, daemon=True).start()

    def _reconnect_loop(self) -> None:
        """Retry connect() with backoff; on success re-run reconcile + recover open orders."""
        attempts = int(GLOBAL.get("reconnect_max_attempts", 30))
        backoff = float(GLOBAL.get("reconnect_backoff_sec", 10.0))
        c = self._conn
        for i in range(1, attempts + 1):
            if getattr(self, "_shutting_down", False):
                break
            time.sleep(backoff)
            try:
                logger.warning("IB reconnect attempt %d/%d ...", i, attempts)
                self._order_id_ready.clear()
                self.connect(c["host"], c["port"], c["client_id"])
                self._api_thread = threading.Thread(target=self.run, daemon=True)
                self._api_thread.start()
                if not self._order_id_ready.wait(timeout=8.0):
                    logger.warning("reconnect %d: no nextValidId yet", i)
                    continue
                self.reqMarketDataType(3)
                self.reconcile_and_log()
                self.recover_open_orders()
                logger.critical("IB RECONNECTED after %d attempt(s) — reconciled + recovered", i)
                self._reconnecting = False
                return
            except Exception as e:
                logger.warning("reconnect attempt %d failed: %s", i, e)
        self._reconnecting = False
        logger.critical("IB reconnect gave up after %d attempts — manual intervention needed", attempts)

    def _check_fill_sanity(self, strategy_id: str, symbol: str, price, expected) -> None:
        """Post-fill guard: a fill far from the expected price alerts (CRITICAL -> Telegram)
        and, beyond the harder halt threshold, halts + flattens the strategy. Market orders
        can't be pre-rejected, so this is detection after the fact."""
        if not expected or expected <= 0 or not price:
            return
        dev = abs(price - expected) / expected
        alert = GLOBAL.get("fill_slippage_alert_pct")
        haltp = GLOBAL.get("fill_slippage_halt_pct")
        if alert is not None and dev > alert:
            logger.critical("FILL SANITY: %s %s filled @ %.4f vs expected %.4f (%.1f%% off)",
                            strategy_id, symbol, price, expected, dev * 100)
            if (haltp is not None and dev > haltp
                    and strategy_id != "__net__" and self.risk_manager.is_active(strategy_id)):
                self.halt_and_flatten(strategy_id, f"fill deviation {dev * 100:.1f}% on {symbol}")

    def margin_whatif(self, intent: dict, timeout: float = 5.0) -> Optional[dict]:
        """Send a whatIf order (no real order placed) and return its margin impact dict."""
        instrument = intent["instrument"]
        if instrument.get("sec_type", "STK") == "FUT":
            contract = self.get_future_contract(
                instrument["symbol"], exchange=instrument.get("exchange", "NYMEX"),
                last_trade_date=instrument.get("last_trade_date"),
                multiplier=instrument.get("multiplier"))
        else:
            contract = self.get_stock_contract(instrument["symbol"], exchange=instrument.get("exchange", "SMART"))
        order = self.build_order(intent)
        order.whatIf = True
        oid = self.get_next_order_id()
        ev = threading.Event()
        self._whatif_events[oid] = ev
        try:
            self.placeOrder(oid, contract, order)
            ev.wait(timeout=timeout)
            return self._whatif.get(oid)
        finally:
            self._whatif_events.pop(oid, None)
            self._whatif.pop(oid, None)

    def mark_is_fresh(self, symbol: str) -> bool:
        """True if we have a mark for `symbol` no older than GLOBAL['mark_staleness_sec']."""
        max_age = GLOBAL.get("mark_staleness_sec")
        if max_age is None:
            return symbol in self._mark_cache
        with self._mark_lock:
            ts = self._mark_ts.get(symbol)
        return ts is not None and (time.time() - ts) <= float(max_age)

    def trip_circuit_breaker(self, reason: str) -> None:
        """Portfolio circuit breaker: HALT + FLATTEN every strategy and kill new orders.
        Idempotent — fires once until _circuit_broken is cleared."""
        if self._circuit_broken:
            return
        self._circuit_broken = True
        logger.critical("CIRCUIT BREAKER TRIPPED: %s \u2014 flattening ALL strategies, killing new orders", reason)
        for sid in set(CONFIG) | set(self.ledger.strategy_positions):
            try:
                if self.risk_manager.is_active(sid):
                    self.risk_manager.halt_strategy(sid, f"circuit breaker: {reason}")
                if self.coordinator is not None and sid in getattr(self.coordinator, "desired", {}):
                    self.coordinator.halt(sid)
                else:
                    self._flatten_direct(sid)
            except Exception as e:
                logger.error("circuit-breaker flatten failed for %s: %s", sid, e)
        self._killed = True

    def enforce_daily_loss(self, total_equity: float) -> None:
        """Portfolio daily-loss circuit breaker (called by the sampler). Baseline is captured
        on the first call (or reset via reset_daily_baseline); trips when the loss since the
        baseline reaches GLOBAL['max_daily_loss']."""
        if self._daily_baseline is None:
            self._daily_baseline = total_equity
        max_loss = GLOBAL.get("max_daily_loss")
        if max_loss is None or self._circuit_broken:
            return
        loss = self._daily_baseline - total_equity
        if loss >= max_loss:
            self.trip_circuit_breaker(f"daily loss {loss:,.0f} >= {max_loss:,.0f}")

    def reset_daily_baseline(self, total_equity: float = None) -> None:
        """Reset the circuit-breaker baseline (call at the open) and clear a tripped breaker."""
        self._daily_baseline = total_equity
        self._circuit_broken = False

    def enforce_drawdown(self, strat_id: str, pnl: float, source: str = "realized") -> None:
        """If `pnl` (dollar-denominated) breaches the strategy's max_drawdown, HALT it and
        FLATTEN its holdings. `source` is 'realized' (fast path, on each fill) or 'total'
        (periodic, realized + unrealized mark-to-market)."""
        if not self.risk_manager.is_active(strat_id):
            return
        st = self.risk_manager.drawdown_status(strat_id, pnl)
        if st["breached"]:
            reason = (f"DRAWDOWN BREACH ({source}): {strat_id} at {st['drawdown_pct'] * 100:.1f}% "
                      f">= limit {st['max_dd'] * 100:.1f}% (P&L {pnl:,.0f})")
            self.halt_and_flatten(strat_id, reason)

    def halt_and_flatten(self, strat_id: str, reason: str) -> None:
        """Stop a strategy AND close its positions. Pooled strategies unwind via the
        coordinator (so the desired book is zeroed and won't re-open); direct strategies get
        closing market orders. Idempotent — a no-op if already halted."""
        if not self.risk_manager.is_active(strat_id):
            return
        logger.critical(reason)                       # -> AlertingHandler pages Telegram
        self.risk_manager.halt_strategy(strat_id, reason)
        # persist halt state so it survives a restart
        self.logger_db.save_halted_strategies(
            set(), self.risk_manager._active_strategies, set(CONFIG.keys()), reason)
        self.logger_db.log_decision(strat_id, "halt", f"HALTED: {reason}")
        # Kill the strategy's in-flight orders BEFORE closing: an unfilled buy that lands
        # after the flatten would re-open the position the halt just closed.
        try:
            self._cancel_strategy_orders(strat_id)
        except Exception as e:
            logger.error("halt: cancelling working orders failed for %s: %s", strat_id, e)
        try:
            if self.coordinator is not None and strat_id in getattr(self.coordinator, "desired", {}):
                self.coordinator.halt(strat_id)       # zero desired book + unwind (attributes to strat)
            else:
                self._flatten_direct(strat_id)
        except Exception as e:
            logger.error("flatten failed for %s: %s", strat_id, e)

    def _cancel_strategy_orders(self, strat_id: str) -> list:
        """Cancel a strategy's own working orders and reverse their pending contribution.

        A halt that only places closing orders leaves the strategy's in-flight BUYS alive:
        they fill after the flatten and re-open the position the halt just closed. Pooled
        (__net__) orders are skipped — the coordinator's unwind cancels those per symbol,
        and they belong to more strategies than this one."""
        cancelled = []
        for oid, st in list(self.order_status.items()):
            if st.get("status") not in ("PreSubmitted", "Submitted"):
                continue
            if st.get("net") or st.get("strategy_id") != strat_id:
                continue
            if not self.cancel_order(oid, f"halt {strat_id}"):
                continue
            cancelled.append(oid)
        if cancelled:
            logger.warning("halt %s: cancelled working orders %s", strat_id, cancelled)
        return cancelled

    def _has_live_order(self, strat_id: str, symbol: str) -> bool:
        """Is a closing order already working for this symbol? Guards the flatten retry
        against stacking duplicate orders while one sits unfilled (e.g. market closed)."""
        for st in self.order_status.values():
            if st.get("status") not in ("PreSubmitted", "Submitted"):
                continue
            if st.get("symbol") != symbol:
                continue
            if st.get("net") or st.get("strategy_id") == strat_id:
                return True
        return False

    def ensure_flat(self, strat_id: str) -> dict:
        """Re-attempt the unwind of a halted strategy that is still holding.

        halt_and_flatten fires ONCE and returns early forever after (`is_active` is already
        False), so a flatten that failed — broker rejection, exception, market closed, a
        partial fill — left the strategy halted AND still exposed, with nothing retrying.
        Called each sampler cycle for halted strategies. Skips any symbol that already has
        a working order, and re-arms at most every GLOBAL['flatten_retry_sec']."""
        if self.risk_manager.is_active(strat_id):
            return {"retried": False, "reason": "not halted"}
        book = {s: q for s, q in self.ledger.strategy_positions.get(strat_id, {}).items()
                if abs(q) > 1e-9}
        if not book:
            self._flatten_retry_ts.pop(strat_id, None)
            return {"retried": False, "reason": "flat"}

        outstanding = {s: q for s, q in book.items() if not self._has_live_order(strat_id, s)}
        if not outstanding:
            return {"retried": False, "reason": "closing orders already working",
                    "still_holding": book}

        cooldown = float(GLOBAL.get("flatten_retry_sec", 60.0))
        last = self._flatten_retry_ts.get(strat_id, 0.0)
        if time.time() - last < cooldown:
            return {"retried": False, "reason": "cooldown", "still_holding": book}
        self._flatten_retry_ts[strat_id] = time.time()

        logger.critical("HALTED STRATEGY STILL HOLDING — retrying flatten for %s: %s",
                        strat_id, outstanding)
        try:
            if self.coordinator is not None and strat_id in getattr(self.coordinator, "desired", {}):
                self.coordinator.desired[strat_id] = {}
                self.coordinator._save()
                self.coordinator._rebalance(set(outstanding), urgent=True)
            else:
                self._flatten_direct(strat_id, symbols=set(outstanding))
        except Exception as e:
            logger.error("flatten retry failed for %s: %s", strat_id, e)
            return {"retried": True, "ok": False, "error": str(e), "still_holding": book}
        return {"retried": True, "ok": True, "symbols": sorted(outstanding)}

    #: ledger buckets that are bookkeeping, not strategies — a position whose only
    #: attribution is one of these is owned by nobody
    INTERNAL_SIDS = frozenset({"__net__", "flatten_all", "kill_switch"})

    def orphaned_positions(self) -> Dict[str, float]:
        """Broker positions that no real strategy claims — symbol -> signed quantity.

        A position ends up here when its only ledger attribution is an internal bucket
        (``__net__``, ``flatten_all``, ``kill_switch``) or nothing at all: a flatten that
        left a residue, a fill attributed to the net pool, a reconcile that adopted a
        position from the broker.

        These used to be invisible AND unreachable. Every flatten path walks
        ``strategy_positions`` and skips the internal buckets, so ``/flatten`` and the kill
        switch both stepped straight over them — the emergency brake could not close a
        position nobody owned. They are also missing from the equity sampler's NAV, so the
        account can hold real risk that the dashboard values at zero.
        """
        claimed = set()
        for sid, positions in self.ledger.strategy_positions.items():
            if sid in self.INTERNAL_SIDS:
                continue
            claimed |= {s for s, q in positions.items() if abs(q) > 1e-9}
        # A strategy that WANTS a symbol owns it even when the fill was attributed to the
        # net pool — which is the normal outcome for a pooled order. Without this, every
        # coordinator-traded position reads as orphaned the moment its fill lands on
        # __net__, and the report cries wolf about positions a strategy is actively running.
        # Flatten and kill are unaffected: they already sweep every desired symbol.
        coordinator = getattr(self, "coordinator", None)
        if coordinator is not None:
            for sid, book in getattr(coordinator, "desired", {}).items():
                if sid in self.INTERNAL_SIDS:
                    continue
                claimed |= {s for s, q in book.items() if abs(q) > 1e-9}
        return {s: q for s, q in self.ledger.current_positions.items()
                if abs(q) > 1e-9 and s not in claimed}

    def _flatten_orphans(self, symbols: set = None) -> list:
        """Close orphaned positions with market orders (non-pooled path).

        The fill is attributed to ``flatten_all``, which is where most orphans already sit,
        so closing one nets its bucket back to zero rather than inventing a new holding.
        """
        closed = []
        for sym, qty in self.orphaned_positions().items():
            if symbols is not None and sym not in symbols:
                continue
            inst = self._instruments.get(sym)
            if inst is None:
                # We never saw an instrument for this symbol — it came from the broker.
                # A non-unit multiplier means it is a derivative, and guessing STK/SMART
                # would send an order for a contract that does not exist. Say so instead.
                mult = self.ledger.multipliers.get(sym, 1.0)
                if mult and mult != 1.0:
                    logger.error("ORPHAN %s: multiplier %s means this is not an equity and "
                                 "no contract spec is known — close it manually", sym, mult)
                    continue
                inst = {"symbol": sym, "asset_class": "equity",
                        "sec_type": "STK", "exchange": "SMART"}
            intent = {
                "client_order_id": f"orphan-{sym}-{int(time.time() * 1000)}",
                "strategy_id": "flatten_all",
                "instrument": inst,
                "side": "sell" if qty > 0 else "buy",
                "quantity": abs(qty),
                "order_type": "market",
                "time_in_force": "day",
                "expected_price": self._ref_value.get(sym),
            }
            logger.warning("FLATTEN ORPHAN: %s %g %s (claimed by no strategy)",
                           intent["side"], abs(qty), sym)
            self.place_order(intent)
            closed.append({"symbol": sym, "quantity": qty})
        return closed

    def _flatten_direct(self, strat_id: str, symbols: set = None) -> None:
        """Close every non-flat position of a (non-pooled) strategy with market orders.
        Bypasses the active-strategy risk check by calling place_order directly — the
        strategy is halted, but this system-initiated unwind must still go through."""
        book = dict(self.ledger.strategy_positions.get(strat_id, {}))
        for sym, qty in book.items():
            if abs(qty) < 1e-9:
                continue
            if symbols is not None and sym not in symbols:
                continue
            inst = self._instruments.get(sym) or {
                "symbol": sym, "asset_class": "equity", "sec_type": "STK", "exchange": "SMART"}
            intent = {
                "client_order_id": f"flat-{strat_id}-{sym}-{int(time.time() * 1000)}",
                "strategy_id": strat_id,
                "instrument": inst,
                "side": "sell" if qty > 0 else "buy",
                "quantity": abs(qty),
                "order_type": "market",
                "time_in_force": "day",
                "expected_price": self._ref_value.get(sym),
            }
            logger.warning("FLATTEN %s: %s %g %s", strat_id, intent["side"], abs(qty), sym)
            self.place_order(intent)

    # ------------------------------------------------------------------
    # Capital allocation
    # ------------------------------------------------------------------
    @staticmethod
    def _reduction_plan(values: Dict[str, float], shortfall: float, method: str) -> Dict[str, float]:
        """How much CASH each position must raise. `values` is symbol -> SIGNED market value.

        Note the asymmetry: selling a long raises cash, but buying back a short SPENDS it,
        so shorts cannot fund a withdrawal. That is why the two methods differ in what they
        touch:

        pro_rata (default): scale the WHOLE book — longs and shorts — by one fraction, so
            the strategy keeps its shape. The cash raised nets out to exactly the shortfall
            (longs bring cash in, covering shorts pays some back out).
        equal: split the shortfall evenly, in dollars, across the LONG positions only,
            water-filling so that a position smaller than its share gives what it has and
            the rest is redistributed. Shorts are left untouched.

        Both are always feasible for any withdrawal within NAV: the shortfall can never
        exceed net position value (pro_rata), which is itself at most the long value (equal).
        Returns positive dollars for a long being sold, negative for a short being covered."""
        net = sum(values.values())
        if shortfall <= 0 or net <= 0:
            return {}
        shortfall = min(shortfall, net)

        if method != "equal":
            fraction = shortfall / net
            return {sym: val * fraction for sym, val in values.items() if val}

        longs = {sym: val for sym, val in values.items() if val > 0}
        plan = {sym: 0.0 for sym in longs}
        remaining, active = min(shortfall, sum(longs.values())), set(longs)
        while remaining > 1e-6 and active:
            share = remaining / len(active)
            progressed = False
            for sym in sorted(active):
                room = longs[sym] - plan[sym]
                take = min(share, room)
                if take > 1e-12:
                    plan[sym] += take
                    remaining -= take
                    progressed = True
                if plan[sym] >= longs[sym] - 1e-9:
                    active.discard(sym)
            if not progressed:
                break
        return {sym: dollars for sym, dollars in plan.items() if dollars > 0}

    def rebalance_allocation(self, strat_id: str, new_allocation: float,
                             method: str = "pro_rata", dry_run: bool = False) -> dict:
        """Change a strategy's capital allocation.

        An INCREASE lands entirely in cash — the strategy can deploy it on its next signal.
        A DECREASE comes out of cash first; whatever cash can't cover is raised by selling
        positions (`method` decides how that is spread), and the withdrawal is booked
        immediately so NAV is correct before the sales settle: cash goes negative by the
        shortfall and the fills bring it back to zero.

        Capital moves never touch P&L — the basis moves with the cash."""
        cfg = CONFIG.get(strat_id)
        if cfg is None:
            raise ValueError(f"unknown strategy {strat_id}")
        new_allocation = float(new_allocation)
        if not (new_allocation > 0) or not math.isfinite(new_allocation):
            raise ValueError(f"capital_allocation must be a positive number, got {new_allocation!r}")

        current = float(cfg["capital_allocation"])
        delta = new_allocation - current
        cash = self.ledger.cash(strat_id)
        book = {s: q for s, q in self.ledger.strategy_positions.get(strat_id, {}).items()
                if abs(q) > 1e-9}

        # Value the book at live marks, falling back to the strategy's own cost basis.
        marks = self.get_marks(set(book)) if book else {}
        costs = self.ledger.strategy_avg_cost.get(strat_id, {})
        units, valued_at_cost = {}, []
        for sym in book:
            mult = self.ledger.multipliers.get(sym, 1.0)
            px = marks.get(sym)
            if px is None or px <= 0:
                px = costs.get(sym, 0.0)
                if px > 0:
                    valued_at_cost.append(sym)
            units[sym] = px * mult
        # SIGNED market value per position: a short is negative, which is what makes the
        # cash arithmetic below come out right.
        values = {s: book[s] * units[s] for s in book if units.get(s, 0) > 0}
        nav = cash + sum(values.values())

        result = {"strategy_id": strat_id, "method": method, "dry_run": dry_run,
                  "allocation_before": current, "allocation_after": new_allocation,
                  "delta": delta, "cash_before": cash, "nav_before": nav,
                  "valued_at_cost": sorted(valued_at_cost), "liquidations": [], "orders": []}

        if abs(delta) < 1e-9:
            result["note"] = "allocation unchanged"
            return result

        shortfall = 0.0
        if delta < 0:
            withdrawal = -delta
            # A position we can't value makes both the NAV guard and the sell-down plan
            # meaningless, so say THAT rather than reporting a nonsense NAV.
            unvaluable = [s for s in book if units.get(s, 0) <= 0]
            if unvaluable and withdrawal > max(cash, 0.0) + 1e-6:
                raise ValueError(
                    f"cannot size the sell-down for {strat_id}: no price for "
                    f"{', '.join(sorted(unvaluable))}")
            if withdrawal > nav + 1e-6:
                raise ValueError(
                    f"cannot withdraw {withdrawal:,.0f} from {strat_id}: its NAV is only "
                    f"{nav:,.0f} (cash {cash:,.0f} + positions {nav - cash:,.0f})")
            shortfall = max(0.0, withdrawal - max(cash, 0.0))
            if shortfall > 1e-6:
                for sym, dollars in sorted(self._reduction_plan(values, shortfall, method).items()):
                    qty = book[sym]
                    # dollars/unit is signed the same way as qty, so this moves a long down
                    # and a short up — both toward zero — and never past it.
                    units_to_trade = dollars / units[sym]
                    units_to_trade = math.copysign(min(abs(qty), round(abs(units_to_trade))),
                                                   units_to_trade)
                    if abs(units_to_trade) < 1e-9:
                        continue
                    result["liquidations"].append({
                        "symbol": sym, "from_quantity": qty,
                        "to_quantity": qty - units_to_trade,
                        "dollars": dollars, "freed": units_to_trade * units[sym]})

        result["cash_shortfall"] = shortfall
        if dry_run:
            result["note"] = "dry run — nothing was changed"
            return result

        # Book the capital move first so NAV is right immediately; the sells settle into cash.
        moved = self.ledger.adjust_capital(strat_id, delta)
        result.update(cash_after=moved["cash_after"], starting_cash=moved["starting_cash"])
        cfg["capital_allocation"] = new_allocation      # CONFIG is shared with risk + coordinator
        try:
            self.logger_db.save_allocation(strat_id, new_allocation)
        except Exception as e:
            logger.critical("allocation for %s changed to %s but NOT persisted (%s) — "
                            "a restart will revert it", strat_id, f"{new_allocation:,.0f}", e)
        self.ledger.save_state(self.logger_db)

        for cut in result["liquidations"]:
            try:
                result["orders"].append(self._resize_position(strat_id, cut["symbol"],
                                                              cut["to_quantity"], units))
            except Exception as e:
                logger.error("allocation sell-down failed for %s %s: %s",
                             strat_id, cut["symbol"], e)
                result["orders"].append({"symbol": cut["symbol"], "error": str(e)})

        logger.warning("ALLOCATION %s: %s -> %s (cash %s -> %s, selling %d position(s))",
                       strat_id, f"{current:,.0f}", f"{new_allocation:,.0f}",
                       f"{cash:,.0f}", f"{moved['cash_after']:,.0f}",
                       len(result["liquidations"]))
        self.logger_db.log_decision(
            strat_id, "allocation",
            f"allocation {current:,.0f} -> {new_allocation:,.0f} ({method})",
            detail=result, symbols=[c["symbol"] for c in result["liquidations"]])
        return result

    def _resize_position(self, strat_id: str, symbol: str, target_qty: float,
                         units: Dict[str, float]) -> dict:
        """Move one position to `target_qty` — through the coordinator when the strategy is
        pooled (so the book stays authoritative), else with a direct order."""
        if self.coordinator is not None and strat_id in getattr(self.coordinator, "desired", {}):
            price = self.coordinator.ref_price.get(symbol)
            r = self.coordinator.set_target(strat_id, symbol, target_qty, price=price)
            return {"symbol": symbol, "target": target_qty, "pooled": True, **r}
        current = self.ledger.strategy_positions.get(strat_id, {}).get(symbol, 0.0)
        delta = target_qty - current
        inst = self._instruments.get(symbol) or {
            "symbol": symbol, "asset_class": "equity", "sec_type": "STK", "exchange": "SMART"}
        mult = self.ledger.multipliers.get(symbol, 1.0)
        intent = {
            "client_order_id": f"alloc-{strat_id}-{symbol}-{int(time.time() * 1000)}",
            "strategy_id": strat_id,
            "instrument": inst,
            "side": "buy" if delta > 0 else "sell",
            "quantity": abs(delta),
            "order_type": "market",
            "time_in_force": "day",
            "expected_price": (units.get(symbol) or 0.0) / mult or None,
        }
        return {"symbol": symbol, "target": target_qty, "pooled": False,
                "order_id": self.place_order(intent)}

    def clear_kill_switch(self) -> None:
        """Re-enable order flow after a kill switch or a tripped circuit breaker.

        Nothing else clears `_killed`: without this the only way back from a kill — your
        own, or the portfolio breaker's — is a process restart. Strategies halted by the
        breaker stay halted on purpose; reactivate them individually once you've looked."""
        was_killed, was_broken = self._killed, self._circuit_broken
        self._killed = False
        self._circuit_broken = False
        logger.critical("KILL SWITCH CLEARED — new orders accepted again "
                        "(was killed=%s, circuit_broken=%s)", was_killed, was_broken)

    def flatten_strategy(self, strat_id: str) -> dict:
        """Close ONE strategy's positions without halting it — it can trade again on its
        next signal. Cancels its working orders first so an in-flight order can't re-open
        what we just closed."""
        self._cancel_strategy_orders(strat_id)
        book = {s: q for s, q in self.ledger.strategy_positions.get(strat_id, {}).items()
                if abs(q) > 1e-9}
        if not book:
            return {"strategy_id": strat_id, "flattened": [], "note": "already flat"}
        logger.warning("FLATTEN %s (no halt): %s", strat_id, book)
        if self.coordinator is not None and strat_id in getattr(self.coordinator, "desired", {}):
            self.coordinator.desired[strat_id] = {}
            self.coordinator._save()
            self.coordinator._rebalance(set(book), urgent=True)
        else:
            self._flatten_direct(strat_id, symbols=set(book))
        return {"strategy_id": strat_id,
                "flattened": [{"symbol": s, "quantity": q} for s, q in book.items()]}

    def kill_switch(self, flatten: bool = True) -> None:
        self._killed = True
        logger.critical("KILL SWITCH ACTIVATED")

        # 1. cancel all open orders. Through cancel_order, which releases their pending: a
        #    bare cancelOrder did not, so the flatten below sized its closing orders against
        #    shares the ledger still believed were on the way.
        for order_id, status in list(self.order_status.items()):
            if status["status"] in ("PreSubmitted", "Submitted"):
                self.cancel_order(order_id, "kill switch")

        # 2. optionally flatten every position — per-strategy so fills attribute correctly
        if flatten:
            _INTERNAL = {"__net__", "flatten_all", "kill_switch"}
            if getattr(self, "coordinator", None) is not None:
                # Coordinator path: zero desired books and rebalance to flat
                all_syms = set()
                for sid in list(self.coordinator.desired):
                    all_syms |= set(self.coordinator.desired[sid])
                    self.coordinator.desired[sid] = {}
                # Also include symbols from strategy_positions (desired may already be empty)
                for sid, positions in self.ledger.strategy_positions.items():
                    if sid in _INTERNAL:
                        continue
                    all_syms |= {s for s, q in positions.items() if abs(q) > 1e-9}
                # ...and anything the BROKER holds that no strategy claims. Without this the
                # kill switch steps over its own orphans: every set above is built from
                # strategy attribution, so a position owned by nobody survives the kill.
                orphans = self.orphaned_positions()
                if orphans:
                    logger.critical("KILL SWITCH: closing %d orphaned position(s) no strategy "
                                    "claims: %s", len(orphans), orphans)
                    all_syms |= set(orphans)
                self.coordinator._save()
                if all_syms:
                    self.coordinator._rebalance(all_syms, urgent=True)
            else:
                # No coordinator: flatten each strategy directly
                for sid, positions in list(self.ledger.strategy_positions.items()):
                    if sid in _INTERNAL:
                        continue
                    if any(abs(q) > 1e-9 for q in positions.values()):
                        self._flatten_direct(sid)
                self._flatten_orphans()
    
    def reconcile_and_log(self) -> dict:
        result = self.ledger.reconcile()
        self.logger_db.log_reconciliation(result["matched"], result["discrepancies"])
        if not result["matched"]:
            logger.warning("reconciliation found discrepancies: %s", result["discrepancies"])

        # also reconcile open orders
        order_result = self.reconcile_open_orders()
        result["order_reconcile"] = order_result
        return result

    def reconcile_open_orders(self, timeout: float = 5.0) -> dict:
        """Compare executor's order_status against IBKR's live open orders.
        Removes stale entries (orders we think are open but IBKR doesn't),
        adds missing ones (orders IBKR reports but we don't track), and
        returns a summary of discrepancies."""

        # 1. snapshot current order_status IDs that are "live" (not final)
        _LIVE = {"PreSubmitted", "Submitted", "PendingSubmit", "PendingCancel"}
        local_live = {
            oid for oid, st in self.order_status.items()
            if st.get("status") in _LIVE
        }

        # 2. ask IBKR for all open orders
        self._reconcile_ib_orders = {}          # temp dict: orderId -> {contract, order, state}
        self._reconcile_orders_done = threading.Event()
        self._reconcile_mode = True             # flag so openOrder routes to reconcile dict
        self._reconcile_orders_done.clear()
        self.reqAllOpenOrders()
        if not self._reconcile_orders_done.wait(timeout=timeout):
            logger.warning("open-order reconcile: reqAllOpenOrders timed out")
            self._reconcile_mode = False
            return {"timed_out": True}
        self._reconcile_mode = False

        ib_live = set(self._reconcile_ib_orders.keys())

        # 3. stale: in our order_status as live, but IBKR doesn't report them
        stale = local_live - ib_live
        stale_details = []
        for oid in stale:
            st = self.order_status.get(oid, {})
            stale_details.append({
                "order_id": oid,
                "symbol": st.get("symbol"),
                "strategy_id": st.get("strategy_id"),
                "local_status": st.get("status"),
            })
            # mark as dead — set status so it's no longer treated as live
            self.order_status[oid]["status"] = "Reconciled_Stale"
            # Release through the shared helper: once only, the unfilled remainder, at the net
            # level for a pooled order. This used to release the FULL original quantity via the
            # per-strategy path — a second time for an order whose cancel had already released
            # it, and into a phantom strategy_pending["__net__"] for a pooled one.
            self._release_pending(oid, "reconcile: not live at IB")
            logger.warning("reconcile: removed stale order %s (%s %s) — not in IBKR",
                           oid, st.get("symbol"), st.get("strategy_id"))

        # 4. missing: IBKR reports open but we don't track them at all
        missing = ib_live - set(self.order_status.keys())
        missing_details = []
        for oid in missing:
            info = self._reconcile_ib_orders[oid]
            contract, order, state = info["contract"], info["order"], info["state"]
            signed_qty = order.totalQuantity if order.action == "BUY" else -order.totalQuantity
            original = self.logger_db.get_order(oid)
            strategy_id = original["strategy_id"] if original else "recovered"
            client_order_id = original["client_order_id"] if original else f"recovered-{oid}"
            self.order_status[oid] = {
                "client_order_id": client_order_id,
                "strategy_id": strategy_id,
                "symbol": contract.symbol,
                "status": state.status,
                "filled": 0,
                "remaining": order.totalQuantity,
                "pending_qty": signed_qty,
                "expected_price": original["expected_price"] if original else None,
            }
            self.ledger.record_pending(contract.symbol, signed_qty, strategy_id)
            missing_details.append({
                "order_id": oid,
                "symbol": contract.symbol,
                "action": order.action,
                "qty": order.totalQuantity,
                "strategy_id": strategy_id,
            })
            logger.warning("reconcile: recovered missing order %s: %s %s %s",
                           oid, order.action, order.totalQuantity, contract.symbol)

        # clean up temp state
        self._reconcile_ib_orders = {}

        matched = len(stale) == 0 and len(missing) == 0
        result = {
            "matched": matched,
            "stale_removed": stale_details,
            "missing_recovered": missing_details,
            "local_live_count": len(local_live),
            "ib_live_count": len(ib_live),
        }
        if not matched:
            logger.warning("open-order reconcile discrepancies: %s stale, %s missing",
                           len(stale_details), len(missing_details))
        else:
            logger.info("open-order reconcile: all %d orders match", len(ib_live))
        return result

    def recover_open_orders(self, timeout: float = 5.0) -> None:
        self._open_orders_ready.clear()
        self.reqAllOpenOrders()
        if not self._open_orders_ready.wait(timeout=timeout):
            logger.warning("open-order recovery timed out")

    def openOrder(self, orderId, contract, order, orderState):
        # whatIf probe (pre-trade margin check) — capture margin impact, don't treat as live
        if orderId in self._whatif_events:
            def _f(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
            self._whatif[orderId] = {
                "init_margin": _f(orderState.initMarginChange),
                "maint_margin": _f(orderState.maintMarginChange),
                "commission": _f(getattr(orderState, "commission", None)),
            }
            self._whatif_events[orderId].set()
            return

        # reconcile mode: collect IBKR's view into temp dict, don't modify order_status
        if getattr(self, "_reconcile_mode", False):
            self._reconcile_ib_orders[orderId] = {
                "contract": contract,
                "order": order,
                "state": orderState,
            }
            return

        # IB volunteering this order means it holds it — an acknowledgement for one we sent.
        if orderId in self.order_status:
            self.order_status[orderId].setdefault("ack", "live")
            if self.order_status[orderId].get("ack") == "pending":
                self.order_status[orderId]["ack"] = "live"

        # rebuild order_status from what IB reports as live
        if orderId not in self.order_status:
            original = self.logger_db.get_order(orderId)  # you'd add this method
            client_order_id = original["client_order_id"] if original else f"recovered-{orderId}"
            strategy_id = original["strategy_id"] if original else "recovered"
            signed_qty = order.totalQuantity if order.action == "BUY" else -order.totalQuantity
            self.order_status[orderId] = {
                "client_order_id": client_order_id,   # we don't know the original
                "strategy_id": strategy_id,
                "symbol": contract.symbol,
                "status": orderState.status,
                "ack": "live",          # it came from IB, so IB has it
                "sent_at": time.time(),
                "filled": 0,
                "remaining": order.totalQuantity,
                "pending_qty": signed_qty,
                "expected_price": original["expected_price"] if original else None,
            }
            if client_order_id != f"recovered-{orderId}":
                self._seen_client_order_ids[client_order_id] = orderId
            # also restore pending exposure to the ledger
            self.ledger.record_pending(contract.symbol, signed_qty, strategy_id)
            logger.warning("recovered open order %s: %s %s %s",
                        orderId, order.action, order.totalQuantity, contract.symbol)

    def openOrderEnd(self):
        # signal reconcile if in reconcile mode
        if getattr(self, "_reconcile_mode", False):
            self._reconcile_orders_done.set()
        self._open_orders_ready.set()

    def _should_pool(self, intent: OrderIntent) -> bool:
        """Single-front-door routing: which intents go through the netting pool.
        Only target_position intents can pool (delta stays direct). An explicit
        metadata.pool flag wins in both directions; otherwise the default is to pool
        equities (they share a universe) and send futures direct (disjoint / already
        netted upstream by VECM)."""
        if self.coordinator is None or intent.intent_type != "target_position":
            return False
        if "pool" in intent.metadata:
            return bool(intent.metadata["pool"])
        return getattr(intent.instrument, "sec_type", "STK") == "STK"
    def _submit_pooled(self, intent: OrderIntent) -> dict:
        r = self.coordinator.set_target(
            intent.strategy_id,
            intent.instrument.symbol,
            intent.target_quantity,                     # already signed
            instrument=intent.instrument.model_dump(),  # carries sec_type/multiplier/exchange
            price=intent.expected_price or intent.limit_price,
        )
        if not r.get("accepted"):
            return {"accepted": False, "reason": r.get("reason", "rejected by coordinator")}
        orders = r.get("orders", [])                     # 0 or 1 for a single-symbol set_target
        return {
            "accepted": True,
            "pooled": True,
            "order_id": orders[0]["order_id"] if orders else None,
            "orders": orders,
            "note": None if orders else "already at net target — no market order needed",
        }
    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def start(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 5,
              timeout: float = 5.0, attempts: int = 5, retry_delay: float = 3.0) -> dict:
        """Connect to IB and recover state. Retries the nextValidId handshake: IB Gateway
        accepts socket connections (and passes its container healthcheck) seconds before
        its API is ready to answer, so a single 5s wait turns a cold `docker compose up`
        into a failed startup — the app exits, and compose reports the executor unhealthy.
        Same pattern as _reconnect_loop. Worst case here is ~37s, inside the 45s
        healthcheck start_period."""
        self._shutting_down = False
        self._conn = {"host": host, "port": port, "client_id": client_id}

        for attempt in range(1, attempts + 1):
            self._order_id_ready.clear()
            self.connect(host, port, client_id)
            self._api_thread = threading.Thread(target=self.run, daemon=True)  # store the handle
            self._api_thread.start()
            if self._order_id_ready.wait(timeout=timeout):
                if attempt > 1:
                    logger.info("IB connected on attempt %d/%d", attempt, attempts)
                break
            logger.warning("no nextValidId within %.0fs (attempt %d/%d) — Gateway API not "
                           "ready yet; retrying in %.0fs", timeout, attempt, attempts, retry_delay)
            try:
                self.disconnect()               # drop the half-open socket before retrying
            except Exception:
                pass
            if self._api_thread.is_alive():
                self._api_thread.join(timeout=2.0)
            if attempt == attempts:
                raise TimeoutError(
                    f"Timed out waiting for nextValidId after {attempts} attempts — "
                    f"connection to {host}:{port} may have failed")
            time.sleep(retry_delay)

        self.reqMarketDataType(3)

        # Startup reconciliation, but NEVER fatal. IB Gateway can hold a socket open while
        # its uplink to IBKR flaps (error 1100), so reqPositions times out and the old code
        # raised straight out of the FastAPI lifespan: uvicorn stopped serving, the process
        # did NOT exit, and `restart: unless-stopped` never fired — a container that looks
        # up while answering nothing. Far better to come up KILLED and loud: the dashboard,
        # /health and the Telegram control all work, and nothing can trade until you
        # /reconcile and /unkill.
        result = {"matched": None, "discrepancies": {}}
        self._startup_degraded = False
        for attempt in range(1, 4):
            try:
                result = self.reconcile_and_log()   # ledger recovers NET positions from broker
                break
            except Exception as e:
                logger.warning("startup reconciliation attempt %d/3 failed: %s", attempt, e)
                if attempt == 3:
                    self._startup_degraded = True
                    self._killed = True
                    logger.critical(
                        "STARTUP RECONCILIATION FAILED (%s) — executor is UP but KILLED: it "
                        "does not know the broker's positions. Fix the IB connection, then "
                        "POST /reconcile and POST /unkill.", e)
                else:
                    time.sleep(2.0)
        try:
            self.recover_open_orders()         # executor recovers open orders from IB
        except Exception as e:
            logger.critical("open-order recovery failed at startup: %s — working orders may "
                            "be untracked until the next /reconcile", e)
        self._restore_persistent_state()       # restore per-strategy positions, P&L, halts
        # Run read-only probe in background so it doesn't block server startup / health checks
        threading.Thread(target=self._wait_for_api_ready, daemon=True,
                         name="readonly-probe").start()
        return result

    def _wait_for_api_ready(self, max_wait: float = 300.0, poll: float = 10.0) -> None:
        """Block startup until the Gateway exits read-only mode. Probes by sending
        a whatIf order (no real execution) for 1 share of AAPL. If the probe gets
        a 321 error, the gateway is still read-only — retry after `poll` seconds.
        Once the probe succeeds (or times out), set _api_ready and send a Telegram
        alert so you know orders will go through."""
        logger.info("Probing IB Gateway read-only status (up to %.0fs)...", max_wait)
        self._api_ready.clear()

        probe_intent = {
            "client_order_id": "readonly_probe",
            "strategy_id": "__probe__",
            "instrument": {"symbol": "AAPL", "sec_type": "STK", "exchange": "SMART"},
            "side": "buy",
            "quantity": 1,
            "order_type": "market",
            "time_in_force": "day",
            "expected_price": 1.0,
        }

        start = time.time()
        attempt = 0
        while time.time() - start < max_wait:
            attempt += 1
            try:
                # whatIf order: no real order placed. If gateway is read-only, the
                # error callback fires code 321 and margin_whatif returns None.
                result = self.margin_whatif(probe_intent, timeout=10.0)
                if result is not None:
                    # Gateway accepted the whatIf probe — it's writable
                    elapsed = time.time() - start
                    logger.info("IB Gateway API ready — exited read-only mode after %.1fs (%d probes)",
                                elapsed, attempt)
                    self._api_ready.set()
                    # Telegram alert
                    if self._alerter:
                        self._alerter.send(
                            f"✅ IB Gateway exited read-only mode (took {elapsed:.0f}s, {attempt} probes) — orders will go through",
                        )
                    return
                else:
                    logger.warning("Read-only probe %d: whatIf returned None (gateway may still be read-only), "
                                   "retrying in %.0fs...", attempt, poll)
            except Exception as e:
                logger.warning("Read-only probe %d error: %s — retrying in %.0fs...", attempt, e, poll)
            time.sleep(poll)

        # Timed out — set ready anyway (orders will get 321 and retry individually)
        logger.warning("Read-only probe timed out after %.0fs (%d probes) — setting ready anyway, "
                       "individual orders will retry on 321", max_wait, attempt)
        self._api_ready.set()
        if self._alerter:
            self._alerter.send(
                f"⚠️ IB Gateway still in read-only after {max_wait:.0f}s — orders may be rejected (321 retry active)",
                topic="errors",
            )

    def _restore_persistent_state(self) -> None:
        """Reload per-strategy positions, realized P&L, halted strategies and multipliers
        from the SQLite database so the dashboard / risk manager have full state immediately."""
        try:
            self.ledger.restore_state(self.logger_db)
        except Exception as e:
            logger.error("failed to restore ledger state: %s", e)

        # Strategies added at runtime come back FIRST: CONFIG is a fail-closed allowlist,
        # and both the ledger and the risk manager hold a reference to that same dict, so
        # putting the entry back here is what makes the strategy exist again. It has to
        # happen before the allocation restore below, which only touches `sid in CONFIG`.
        try:
            for sid, entry in self.logger_db.load_runtime_strategies().items():
                if sid in CONFIG:
                    continue                     # config.py wins if the id was since added there
                CONFIG[sid] = dict(entry)
                self.risk_manager._active_strategies.add(sid)
                logger.warning("restored runtime strategy %s (allocation %s)",
                               sid, f"{entry['capital_allocation']:,.0f}")
        except Exception as e:
            # Loud: every intent from an unrestored strategy is rejected as "not active".
            logger.critical("failed to restore runtime strategies — any strategy added "
                            "since the last restart will have its orders REJECTED: %s", e)

        # restore runtime allocation changes (CONFIG holds the defaults)
        try:
            for sid, alloc in self.logger_db.load_allocations().items():
                if sid in CONFIG and alloc != CONFIG[sid]["capital_allocation"]:
                    logger.warning("restored allocation for %s: %s (config default %s)",
                                   sid, f"{alloc:,.0f}",
                                   f"{CONFIG[sid]['capital_allocation']:,.0f}")
                    CONFIG[sid]["capital_allocation"] = float(alloc)
        except Exception as e:
            logger.error("failed to restore allocations: %s", e)

        # restore halted strategies
        try:
            halted = self.logger_db.load_halted_strategies()
            for sid in halted:
                if sid in self.risk_manager._active_strategies:
                    self.risk_manager._active_strategies.discard(sid)
                    logger.warning("restored halt for %s", sid)
        except Exception as e:
            logger.error("failed to restore halted strategies: %s", e)
        
    def shutdown(self, timeout: float = 5.0) -> None:
        """Cleanly tear down: disconnect from IB and wait for the socket thread to exit.
        Safe to call multiple times (idempotent) — from a signal handler, finally block,
        or a kill-switch path."""
        if getattr(self, "_shutting_down", False):
            return  # already shutting down, don't double-run
        self._shutting_down = True

        logger.info("Shutting down...")
        try:
            self.logger_db.close()
            if self.isConnected():
                self.disconnect()   # closes the socket, which unblocks run()'s read loop
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

        # wait for the API thread to actually finish, if we have a handle to it
        api_thread = getattr(self, "_api_thread", None)
        if api_thread is not None:
            api_thread.join(timeout=timeout)
            if api_thread.is_alive():
                logger.warning("API thread did not exit within timeout")

        logger.info("Shutdown complete")
