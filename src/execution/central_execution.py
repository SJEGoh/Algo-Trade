from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import BarData
from ibapi.order_state import OrderState
from ibapi.execution import Execution
import threading
import signal
import sys

from ledger.position_ledger import PositionLedger
from risk.risk_manager import RiskManager
from logger.event_logger import EventLogger
from monitoring.logging_config import setup_logging

from typing import Dict, Optional, Literal
import pandas as pd
import time

from pydantic import BaseModel, Field, field_validator, model_validator
from monitoring.logging_config import setup_logging
from typing import Optional, Literal

from config import CONFIG

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

        # --- position/risk state (owned by their components, NOT duplicated here) ---
        self.ledger = PositionLedger(self)
        self.risk_manager = RiskManager(self.ledger, CONFIG)
        self._pending_price_reqs: Dict[int, threading.Event] = {}   # reqId -> event fired when price arrives
        self._price_results: Dict[int, float] = {}                   # reqId -> price received
        self._price_req_lock = threading.Lock()
        self._mkt_data_req_id = 9000       

        self._open_orders_ready = threading.Event()                           # base, kept away from order IDs

        # EventLogger
        self.logger_db = EventLogger()

        self._mark_cache: Dict[str, float] = {}   # symbol -> last good mark
        self._mark_lock = threading.Lock()
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

    def place_order(self, intent: dict) -> int:
        instrument = intent["instrument"]
        contract = self.get_stock_contract(instrument["symbol"], exchange=instrument.get("exchange", "SMART"))
        order = self.build_order(intent)

        order_id = self.get_next_order_id()
        self.placeOrder(order_id, contract, order)
        self.logger_db.log_order(order_id, intent)
        signed_qty = intent["quantity"] if intent["side"] == "buy" else -intent["quantity"]
        symbol = instrument["symbol"]

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
        logger.debug("OrderStatus - id:%s status:%s filled:%s remaining:%s", orderId, status, filled, remaining)

    # ------------------------------------------------------------------
    # Intent processing (Phase 2 + Phase 4 risk check)
    # ------------------------------------------------------------------
    def process_intent(self, raw_intent: dict) -> dict:
        if self._killed:
            return {"accepted": False, "reason": "executor is in kill-switch state"}

        try:
            intent = OrderIntent(**raw_intent)
        except Exception as e:
            return {"accepted": False, "reason": f"schema validation failed: {e}"}

        if self._enforce_market_hours and not is_market_open() and not intent.metadata.get("allow_when_closed"):
            return {"accepted": False,
                    "reason": "market closed — order not submitted "
                              "(set metadata.allow_when_closed=true to queue for open)"}

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
                risk_result = self.risk_manager.check_order(resolved_intent, resolved_delta, reference_price)
                if not risk_result["approved"]:
                    del self._seen_client_order_ids[intent.client_order_id]
                    return {"accepted": False, "reason": risk_result["reason"]}

                order_id = self.place_order(resolved_intent)
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
                self.ledger.record_pending(symbol, -pending_contribution, status["strategy_id"])

    # ------------------------------------------------------------------
    # Fill / position callbacks — all delegate to the ledger
    # ------------------------------------------------------------------
    def execDetails(self, reqId: int, contract: Contract, execution: Execution) -> None:
        order_info = self.order_status.get(execution.orderId, {})
        strategy_id = order_info.get("strategy_id", "unknown")
        signed_qty = execution.shares if execution.side == "BOT" else -execution.shares

        self.ledger.record_fill(contract.symbol, signed_qty, execution.price, strategy_id)
        self.logger_db.log_fill(
            execution.orderId, execution.execId, contract.symbol,
            execution.side, execution.price, execution.shares,
            order_info.get("strategy_id", "unknown"),
            expected_price=order_info.get("expected_price"),
        )
        breached_dd = self.risk_manager.check_drawdown(strategy_id)  # Phase 4: halt if this fill breached DD
        if breached_dd:
            self.halt
        logger.info("ExecDetails - %s %s %s @ %s", contract.symbol, execution.side, execution.shares, execution.price)

    def position(self, account: str, contract: Contract, position: float, avgCost: float) -> None:
        # write to the LEDGER's broker_positions, not a local copy
        self.ledger.broker_positions[contract.symbol] = position
        logger.info("Position - %s: %s @ avg cost %s", contract.symbol, position, avgCost)

    def positionEnd(self) -> None:
        self.ledger._positions_ready.set()
        logger.info("Position snapshot complete")

    # ------------------------------------------------------------------
    # Kill switch (Phase 4)
    # ------------------------------------------------------------------
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
            }
            if client_order_id != f"recovered-{orderId}":
                self._seen_client_order_ids[client_order_id] = orderId
            # also restore pending exposure to the ledger
            self.ledger.record_pending(contract.symbol, signed_qty, strategy_id)
            logger.warning("recovered open order %s: %s %s %s",
                        orderId, order.action, order.totalQuantity, contract.symbol)

    def openOrderEnd(self):
        self._open_orders_ready.set()
    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def start(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 5, timeout: float = 5.0) -> dict:
        self._shutting_down = False
        self.connect(host, port, client_id)
        self._api_thread = threading.Thread(target=self.run, daemon=True)  # store the handle
        self._api_thread.start()

        if not self._order_id_ready.wait(timeout=timeout):
            raise TimeoutError("Timed out waiting for nextValidId — connection may have failed")
        self.reqMarketDataType(3)
        result = self.reconcile_and_log()      # ledger recovers positions
        self.recover_open_orders()             # executor recovers open orders
        return result
        
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
