"""tests/test_net_collapse.py — plays out the "+4, +4, -10 nets to -2" question against the
REAL executor + NettingCoordinator, and shows exactly what the broker sees.

The key insight it demonstrates:
  * The RESTING position is -2 either way (invariant always holds).
  * But whether -2 is ALL the broker trades depends on timing:
      - if the three targets arrive with FILLS between them, the broker trades every leg
        (+4, +4, -10 = 18 contracts of turnover) and rests at -2;
      - if they arrive back-to-back while still PENDING (unfilled), the coordinator cancels
        the stale in-flight orders and replaces them, so only the final net (-2, i.e. 2
        contracts) actually crosses the market. THIS is the case where "-2 is all the
        broker sees" is literally true.
  * In BOTH cases each strategy still books its own side (s1=+4, s2=+4, s3=-10) via fill
    attribution, so per-strategy P&L/risk stay correct.

Run it for the narrated play-by-play:   python3 tests/test_net_collapse.py
Run it as a test:                        pytest tests/test_net_collapse.py
"""
import os
import sys

# Allow direct execution (python3 tests/test_net_collapse.py) as well as pytest (conftest).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "src"), os.path.join(_HERE, "..")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest

pytest.importorskip("ibapi")

from ibapi.contract import Contract
from ibapi.execution import Execution

from execution.central_execution import CentralExecutor
from execution.netting import NettingCoordinator

SYMBOL = "CL"
PRICE = 68.5
MULT = 1000.0
INST = {"symbol": SYMBOL, "asset_class": "future", "sec_type": "FUT",
        "exchange": "NYMEX", "multiplier": MULT, "last_trade_date": "20261120"}


class _NullLog:
    def __getattr__(self, name):
        return lambda *a, **k: None


class Harness:
    """Real CentralExecutor with IB calls stubbed. Tracks what the broker RECEIVED, what got
    CANCELLED (withdrawn before it could trade), and what actually FILLED (crossed)."""

    def __init__(self):
        os.environ.setdefault("EXECUTOR_API_KEY", "x")
        ex = CentralExecutor.__new__(CentralExecutor)
        CentralExecutor.__init__(ex)
        self.cfg = {s: {"capital_allocation": 1e9, "max_drawdown": 0.5}
                    for s in ("s1", "s2", "s3")}
        ex.risk_manager._config = self.cfg
        ex.risk_manager._active_strategies = set(self.cfg)
        ex.logger_db = _NullLog()

        self.received = []      # every placeOrder: {oid, qty}
        self.cancelled = set()
        self.filled = set()
        self._oid = 0

        def _next():
            self._oid += 1
            return self._oid
        ex.get_next_order_id = _next

        def _place(oid, contract, order):
            q = order.totalQuantity * (1 if order.action == "BUY" else -1)
            self.received.append({"oid": oid, "qty": q})
        ex.placeOrder = _place
        ex.cancelOrder = lambda oid, *a, **k: self.cancelled.add(oid)

        self.ex = ex
        self.co = NettingCoordinator(ex, self.cfg)
        ex.coordinator = self.co

    def set_target(self, sid, qty):
        return self.co.set_target(sid, SYMBOL, qty, instrument=INST, price=PRICE)

    def live_orders(self):
        """Orders the broker still has working: received, not cancelled, not yet filled."""
        return [o for o in self.received if o["oid"] not in self.cancelled
                and o["oid"] not in self.filled]

    def fill_live(self):
        """Simulate the broker filling every currently-working order (and telling us via
        execDetails + an orderStatus-style status flip, so a filled order isn't re-cancelled)."""
        for o in self.live_orders():
            oid, q = o["oid"], o["qty"]
            c = Contract(); c.symbol = SYMBOL
            e = Execution()
            e.orderId = oid; e.execId = f"e{oid}"
            e.shares = abs(q); e.side = "BOT" if q > 0 else "SLD"; e.price = PRICE
            self.ex.execDetails(1, c, e)                 # routes to coordinator.attribute_fill
            self.ex.order_status[oid]["status"] = "Filled"   # mimic the orderStatus callback
            self.filled.add(oid)

    # ---- reporting ----
    def broker_position(self):
        return self.ex.ledger.current_positions.get(SYMBOL, 0.0)

    def books(self):
        return {s: self.ex.ledger.strategy_positions.get(s, {}).get(SYMBOL, 0.0)
                for s in ("s1", "s2", "s3")}

    def gross_traded(self):
        return sum(abs(o["qty"]) for o in self.received if o["oid"] in self.filled)

    def report(self, title):
        recv = ", ".join(f"{o['qty']:+g}"
                         + ("[cancelled]" if o["oid"] in self.cancelled else
                            "[filled]" if o["oid"] in self.filled else "[working]")
                         for o in self.received)
        print(f"\n  {title}")
        print(f"    broker received : {recv}")
        print(f"    contracts traded: {self.gross_traded():g}  (net that crossed the tape)")
        print(f"    broker position : {self.broker_position():+g}")
        print(f"    strategy books  : {self.books()}")


def _assert_invariant(h):
    assert abs(sum(h.books().values()) - h.broker_position()) < 1e-9
    assert h.books() == {"s1": 4, "s2": 4, "s3": -10}
    assert h.broker_position() == -2


def test_sequential_fills_broker_trades_every_leg():
    """Targets arrive and fill one at a time (typical for market orders). The broker can't
    net what has already filled, so it trades all three legs and rests at -2."""
    h = Harness()
    h.set_target("s1", 4);  h.fill_live()
    h.set_target("s2", 4);  h.fill_live()
    h.set_target("s3", -10); h.fill_live()
    _assert_invariant(h)
    assert h.gross_traded() == 18            # +4, +4, -10 all crossed
    assert len(h.cancelled) == 0


def test_batched_targets_collapse_to_net_only():
    """All three targets arrive before anything fills. Each new target cancels the stale
    in-flight order and re-targets the cumulative net, so ONLY the final -2 crosses the
    market — '-2 is all the broker sees'. Each strategy still books its own side."""
    h = Harness()
    h.set_target("s1", 4)
    h.set_target("s2", 4)
    h.set_target("s3", -10)
    # before any fill: the interim orders were cancelled; one working order remains
    working = h.live_orders()
    assert len(working) == 1 and working[0]["qty"] == -2
    assert len(h.cancelled) == 2             # the two interim orders were withdrawn
    h.fill_live()
    _assert_invariant(h)
    assert h.gross_traded() == 2             # only the net -2 actually traded


if __name__ == "__main__":
    print("=" * 74)
    print("SCENARIO A — targets fill one at a time (e.g. market orders)")
    a = Harness()
    a.set_target("s1", 4);  a.fill_live()
    a.set_target("s2", 4);  a.fill_live()
    a.set_target("s3", -10); a.fill_live()
    a.report("result")
    print("    -> broker traded 18 contracts to rest at -2 (can't un-trade filled legs)")

    print("\n" + "=" * 74)
    print("SCENARIO B — all three targets arrive before any fill (still pending)")
    b = Harness()
    b.set_target("s1", 4)
    b.set_target("s2", 4)
    b.set_target("s3", -10)
    b.report("after the three targets, before filling")
    b.fill_live()
    b.report("after filling the one working order")
    print("    -> broker traded only 2 contracts (net -2). The +4 and +8 interim orders")
    print("       were cancelled before they could fill. THIS is '-2 is all the broker sees'.")

    print("\n" + "=" * 74)
    print("Both rest at -2, and both attribute s1=+4, s2=+4, s3=-10. The difference is")
    print("TURNOVER: netting collapses offsetting intent only while it is still unfilled.")
