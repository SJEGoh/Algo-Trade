from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import BarData
from ibapi.order_state import OrderState
from ibapi.execution import Execution
import threading

from typing import Dict, Optional, Literal
import pandas as pd
import time

from test_orders import test_orders, get_test_order, list_test_case_ids, get_burst_test_orders
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Literal

class Instrument(BaseModel):
    symbol: str
    asset_class: str
    exchange: str = "SMART"

class OrderIntent(BaseModel):
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
        return self

class CentralExecutor(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self._next_order_id: Optional[int] = None
        self._order_id_ready = threading.Event()
        self._order_id_lock = threading.Lock()
        self.order_status: Dict[int, dict] = {}
        self._seen_client_order_ids: Dict[str, int] = {}  # client_order_id -> order_id
        self._dedup_lock = threading.Lock()
        self.current_positions: Dict[str, float] = {}
        self.pending_deltas: Dict[str, float] = {} 
        self.broker_positions: Dict[str, float] = {}
        self._positions_ready = threading.Event()

    @staticmethod
    def get_contract(symbol: str, sec_type: str, exchange: str, currency: str, **kwargs) -> Contract:
        """Generic contract builder — holds the shared fields, nothing asset-specific."""
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
        # forex contracts use symbol = base currency, currency = quote currency
        # e.g. EUR.USD -> symbol="EUR", currency="USD"
        base, quote = pair.split(".")
        return CentralExecutor.get_contract(base, sec_type="CASH", exchange=exchange, currency=quote)

    def nextValidId(self, orderId: int) -> None:
        """Called automatically by ibapi once, right after connect()."""
        self._next_order_id = orderId
        self._order_id_ready.set()
        print(f"Next valid order ID: {orderId}")

    def get_next_order_id(self, timeout: float = 5.0) -> int:
        if not self._order_id_ready.wait(timeout=timeout):
            raise TimeoutError("Timed out waiting for nextValidId — did connect() actually succeed?")
        with self._order_id_lock:
            order_id = self._next_order_id
            self._next_order_id += 1
        return order_id
    
    @staticmethod
    def build_order(intent: dict) -> Order:
        order = Order()
        order.action = intent["side"].upper()          # "buy" -> "BUY", "sell" -> "SELL"
        order.totalQuantity = intent["quantity"]

        if intent["order_type"] == "market":
            order.orderType = "MKT"
        elif intent["order_type"] == "limit":
            order.orderType = "LMT"
            order.lmtPrice = intent["limit_price"]
        else:
            raise ValueError(f"Unsupported order_type: {intent['order_type']}")

        tif_map = {"day": "DAY", "gtc": "GTC"}
        order.tif = tif_map.get(intent.get("time_in_force", "day"), "DAY")

        # required on newer ibapi versions or the order gets rejected
        order.eTradeOnly = False
        order.firmQuoteOnly = False

        return order
    
    def place_order(self, intent: dict) -> int:
        instrument = intent["instrument"]
        contract = self.get_stock_contract(
            instrument["symbol"],
            exchange=instrument.get("exchange", "SMART")
        )
        order = self.build_order(intent)

        order_id = self.get_next_order_id()
        self.placeOrder(order_id, contract, order)

        signed_qty = intent["quantity"] if intent["side"] == "buy" else -intent["quantity"]
        symbol = intent["instrument"]["symbol"]
        self.pending_deltas[symbol] = self.pending_deltas.get(symbol, 0.0) + signed_qty
        # track what you submitted before any callback fires
        self.order_status[order_id] = {
            "client_order_id": intent["client_order_id"],
            "strategy_id": intent["strategy_id"],
            "symbol": instrument["symbol"],
            "status": "Submitted",   # local placeholder until IB's own status arrives
            "filled": 0,
            "remaining": intent["quantity"],
            "pending_qty": signed_qty,   # add this
        }
        return order_id

    def orderStatus(self, orderId: int, status: str, filled: float, remaining: float,
                 avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float,
                 clientId: int, whyHeld: str, mktCapPrice: float) -> None:
        if orderId in self.order_status:
            self.order_status[orderId].update({
                "status": status,
                "filled": filled,
                "remaining": remaining,
                "avg_fill_price": avgFillPrice,
            })
        print(f"OrderStatus - id:{orderId} status:{status} filled:{filled} remaining:{remaining}")

    # Phase 2
    def process_intent(self, raw_intent: dict) -> dict:
        # 1. schema validation
        try:
            intent = OrderIntent(**raw_intent)
        except Exception as e:
            return {"accepted": False, "reason": f"schema validation failed: {e}"}

        with self._dedup_lock:
            # 2. dedup check
            if intent.client_order_id in self._seen_client_order_ids:
                existing_order_id = self._seen_client_order_ids[intent.client_order_id]
                return {
                    "accepted": True,
                    "order_id": existing_order_id,
                    "note": "duplicate client_order_id — returning existing order, not resubmitting"
                }
            # reserve the client_order_id immediately, before placing the order,
            # so a second identical request arriving mid-flight still sees it
            self._seen_client_order_ids[intent.client_order_id] = None

            # 3. target_position translation + 4. place the order — both inside the same
            # lock and the same try/except, so any failure in either step releases
            # the reservation cleanly rather than leaving it stuck on None forever
            try:
                resolved_intent = self._resolve_intent_type(intent)
                order_id = self.place_order(resolved_intent)
            except Exception as e:
                del self._seen_client_order_ids[intent.client_order_id]
                return {"accepted": False, "reason": str(e)}

            self._seen_client_order_ids[intent.client_order_id] = order_id

        return {"accepted": True, "order_id": order_id}

    def _resolve_intent_type(self, intent: OrderIntent) -> dict:
        symbol = intent.instrument.symbol

        if intent.intent_type == "target_position":
            # cancel any still-open orders for this symbol before computing delta —
            # otherwise pending exposure from a stale order double-counts or,
            # worse, both orders execute independently
            self._cancel_open_orders_for_symbol(symbol)

        confirmed = self.current_positions.get(symbol, 0.0)
        pending = self.pending_deltas.get(symbol, 0.0)
        effective_current = confirmed + pending

        if intent.intent_type == "delta":
            signed_qty = intent.quantity if intent.side == "buy" else -intent.quantity
            delta = signed_qty
        elif intent.intent_type == "target_position":
            delta = intent.target_quantity - effective_current
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
                print(f"Cancelling stale open order {order_id} for {symbol} before resolving new target")
                self.cancelOrder(order_id)
                # zero out its contribution to pending_deltas — it's being cancelled, not filled
                pending_contribution = status.get("pending_qty", 0.0)
                self.pending_deltas[symbol] = self.pending_deltas.get(symbol, 0.0) - pending_contribution
            
    def execDetails(self, reqId: int, contract: Contract, execution: Execution) -> None:
        signed_qty = execution.shares if execution.side == "BOT" else -execution.shares
        symbol = contract.symbol
        self.current_positions[symbol] = self.current_positions.get(symbol, 0.0) + signed_qty
        self.pending_deltas[symbol] = self.pending_deltas.get(symbol, 0.0) - signed_qty  # this much is no longer "pending", it's confirmed
        print(f"ExecDetails - {symbol} {execution.side} {execution.shares} @ {execution.price}")
    def position(self, account: str, contract: Contract, position: float, avgCost: float) -> None:
        """Fired once per held position when reqPositions() is called."""
        self.broker_positions[contract.symbol] = position
        print(f"Position - {contract.symbol}: {position} @ avg cost {avgCost}")

    def positionEnd(self) -> None:
        """Fired once, after all position() callbacks for this request have been sent."""
        self._positions_ready.set()
        print("Position snapshot complete")

    def fetch_broker_positions(self, timeout: float = 5.0) -> Dict[str, float]:
        self.broker_positions = {}
        self._positions_ready.clear()
        self.reqPositions()
        if not self._positions_ready.wait(timeout=timeout):
            raise TimeoutError("Timed out waiting for reqPositions() to complete")
        return dict(self.broker_positions)
    
    def reconcile_positions(self, auto_correct: bool = True) -> dict:
        broker_positions = self.fetch_broker_positions()

        all_symbols = set(self.current_positions.keys()) | set(broker_positions.keys())
        discrepancies = {}

        for symbol in all_symbols:
            internal = self.current_positions.get(symbol, 0.0)
            broker = broker_positions.get(symbol, 0.0)
            if internal != broker:
                discrepancies[symbol] = {"internal": internal, "broker": broker, "diff": broker - internal}

        if discrepancies and auto_correct:
            print("Auto-correcting internal ledger to match broker...")
            self.current_positions = dict(broker_positions)

        return {
            "reconciled_at": time.time(),
            "matched": len(discrepancies) == 0,
            "discrepancies": discrepancies,
            "broker_positions": broker_positions,
            "internal_positions": dict(self.current_positions),
        }
    def start(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 5, timeout: float = 5.0) -> dict:
        """Connect, wait for the connection to be ready, and reconcile positions before accepting any intents."""
        self.connect(host, port, client_id)
        threading.Thread(target=self.run, daemon=True).start()

        if not self._order_id_ready.wait(timeout=timeout):
            raise TimeoutError("Timed out waiting for nextValidId — connection may have failed")

        reconciliation = self.reconcile_positions()
        if not reconciliation["matched"]:
            print(f"WARNING: startup reconciliation found discrepancies: {reconciliation['discrepancies']}")
        return reconciliation
    
if __name__ == "__main__":
    '''
    app = CentralExecutor()
    app.connect("127.0.0.1", 7497, clientId=5)
    threading.Thread(target=app.run, daemon=True).start()
    app._order_id_ready.wait(timeout=5.0)
    time.sleep(5)

    burst_orders = get_burst_test_orders()
    results = [None] * len(burst_orders)

    def submit(i, intent):
        results[i] = app.process_intent(intent)

    threads = [threading.Thread(target=submit, args=(i, intent)) for i, intent in enumerate(burst_orders)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for r in results:
        print(r)
    '''
