from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import BarData
from ibapi.order_state import OrderState
from ibapi.execution import Execution
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
        self.ledger = PositionLedger(self)
        self.risk_manager = RiskManager(self.ledger, CONFIG, GLOBAL)
        self._pending_price_reqs: Dict[int, threading.Event] = {}   # reqId -> event fired when price arrives
        self._price_results: Dict[int, float] = {}                   # reqId -> price received
        self._price_req_lock = threading.Lock()
        self._mkt_data_req_id = 9000       
        # Paper fill helper: streaming mkt data subs that keep the paper fill engine alive
        self._paper_mkt_subs: Dict[str, int] = {}   # symbol -> reqId of active streaming sub
        self._paper_mkt_refcount: Dict[str, int] = {}  # symbol -> count of pending orders

        self._open_orders_ready = threading.Event()                           # base, kept away from order IDs

        # EventLogger
        self.logger_db = EventLogger()

        self._mark_cache: Dict[str, float] = {}   # symbol -> last good mark
        self._mark_ts: Dict[str, float] = {}       # symbol -> time.time() of last good mark (staleness guard)
        self._mark_lock = threading.Lock()
        self._conn = {"host": "127.0.0.1", "port": 4002, "client_id": 5}  # remembered for auto-reconnect
        self._reconnecting = False
        self._whatif: Dict[int, dict] = {}         # orderId -> margin impact (whatIf openOrder)
        self._whatif_events: Dict[int, threading.Event] = {}
        self._daily_baseline: Optional[float] = None  # portfolio equity baseline for the circuit breaker
        self._circuit_broken = False
        self.coordinator = None   # NettingCoordinator (net-pooling); set by server lifespan

        # ATR limit-at-pullback execution layer
        from execution.atr_execution import AtrPullbackLayer
        self.atr_layer = AtrPullbackLayer(ATR_EXECUTION)
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
        # snapshot=False -> streaming; keeps the fill simulator active for this symbol
        self.reqMktData(req_id, contract, "", False, False, [])
        logger.debug("paper fill sub: started streaming mkt data for %s (reqId=%d)", symbol, req_id)

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
            "status": "Submitted",
            "filled": 0,
            "remaining": intent["quantity"],
            "pending_qty": signed_qty,
            "expected_price": intent.get("expected_price"),
        }
        return order_id

    def place_net_order(self, symbol: str, delta: float, instrument: dict, ref_price):
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
            "status": "Submitted",
            "filled": 0,
            "remaining": abs(delta),
            "pending_qty": delta,
            "expected_price": ref_price,
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
            })
            self.logger_db.update_order_status(orderId, status)
            # Cancel the paper-fill streaming sub once the order is done
            if status in ("Filled", "Cancelled", "Inactive"):
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

    def _cancel_open_orders_for_symbol(self, symbol: str) -> None:
        for order_id, status in list(self.order_status.items()):
            if status["symbol"] == symbol and status["status"] in ("PreSubmitted", "Submitted"):
                logger.info("Cancelling stale open order %s for %s before resolving new target", order_id, symbol)
                self.cancelOrder(order_id)  # FIX: second arg required
                pending_contribution = status.get("pending_qty", 0.0)
                # reverse this order's pending contribution in the ledger
                if status.get("net"):
                    # pooled net order: pending lives at net level only (record_net_pending),
                    # so reverse it there — NOT via record_pending, which would write a
                    # phantom strategy_pending["__net__"] entry.
                    self.ledger.record_net_pending(symbol, -pending_contribution)
                else:
                    self.ledger.record_pending(symbol, -pending_contribution, status["strategy_id"])
                # Mark it cancelled locally so a rapid re-target (another rebalance before IB
                # confirms this cancel) won't cancel it AGAIN and reverse its pending twice.
                status["status"] = "PendingCancel"

    # ------------------------------------------------------------------
    # Fill / position callbacks — all delegate to the ledger
    # ------------------------------------------------------------------
    def execDetails(self, reqId: int, contract: Contract, execution: Execution) -> None:
        order_info = self.order_status.get(execution.orderId, {})
        strategy_id = order_info.get("strategy_id", "unknown")
        signed_qty = execution.shares if execution.side == "BOT" else -execution.shares
        if getattr(contract, "multiplier", None):
            try:
                self.ledger.multipliers[contract.symbol] = float(contract.multiplier)
            except (TypeError, ValueError):
                pass

        # Pooled net order: let the coordinator decompose this fill into per-strategy
        # sub-fills (correct P&L even with opposing legs), then check drawdown per book.
        if order_info.get("net") and self.coordinator is not None:
            attributed = self.coordinator.attribute_fill(contract.symbol, signed_qty, execution.price)
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
            for sid in list(self.coordinator.desired.keys()):
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
        try:
            if self.coordinator is not None and strat_id in getattr(self.coordinator, "desired", {}):
                self.coordinator.halt(strat_id)       # zero desired book + unwind (attributes to strat)
            else:
                self._flatten_direct(strat_id)
        except Exception as e:
            logger.error("flatten failed for %s: %s", strat_id, e)

    def _flatten_direct(self, strat_id: str) -> None:
        """Close every non-flat position of a (non-pooled) strategy with market orders.
        Bypasses the active-strategy risk check by calling place_order directly — the
        strategy is halted, but this system-initiated unwind must still go through."""
        book = dict(self.ledger.strategy_positions.get(strat_id, {}))
        for sym, qty in book.items():
            if abs(qty) < 1e-9:
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

    def kill_switch(self, flatten: bool = True) -> None:
        self._killed = True
        logger.critical("KILL SWITCH ACTIVATED")

        # 1. cancel all open orders
        for order_id, status in list(self.order_status.items()):
            if status["status"] in ("PreSubmitted", "Submitted"):
                self.cancelOrder(order_id)

        # 2. optionally flatten every position — privileged path, bypasses risk + dedup
        if flatten:
            for symbol, qty in list(self.ledger.current_positions.items()):
                if qty != 0:
                    flat_intent = {
                        "strategy_id": "kill_switch",
                        "client_order_id": f"flatten-{symbol}-{time.time()}",
                        "instrument": {"symbol": symbol, "asset_class": "equity", "exchange": "SMART"},
                        "intent_type": "delta",
                        "side": "sell" if qty > 0 else "buy",
                        "quantity": abs(qty),
                        "order_type": "market",
                        "time_in_force": "day",
                        "schema_version": "1.0",
                        "timestamp": "",
                    }
                    self.place_order(flat_intent)  # deliberately direct, not process_intent
    
    def reconcile_and_log(self) -> dict:
        result = self.ledger.reconcile()
        self.logger_db.log_reconciliation(result["matched"], result["discrepancies"])
        if not result["matched"]:
            logger.warning("reconciliation found discrepancies: %s", result["discrepancies"])
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
    def start(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 5, timeout: float = 5.0) -> dict:
        self._shutting_down = False
        self._conn = {"host": host, "port": port, "client_id": client_id}
        self.connect(host, port, client_id)
        self._api_thread = threading.Thread(target=self.run, daemon=True)  # store the handle
        self._api_thread.start()

        if not self._order_id_ready.wait(timeout=timeout):
            raise TimeoutError("Timed out waiting for nextValidId — connection may have failed")
        self.reqMarketDataType(3)
        result = self.reconcile_and_log()      # ledger recovers NET positions from broker
        self.recover_open_orders()             # executor recovers open orders from IB
        self._restore_persistent_state()       # restore per-strategy positions, P&L, halts
        return result

    def _restore_persistent_state(self) -> None:
        """Reload per-strategy positions, realized P&L, halted strategies and multipliers
        from the SQLite database so the dashboard / risk manager have full state immediately."""
        try:
            self.ledger.restore_state(self.logger_db)
        except Exception as e:
            logger.error("failed to restore ledger state: %s", e)

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
