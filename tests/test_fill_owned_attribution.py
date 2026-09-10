"""A strategy's position moves when ITS order fills — not when it is placed, and not when
someone else's order fills.

Two defects this pins, both seen in the live database:

  * Attribution re-derived a fill's owners from the live desired book at the moment the
    fill arrived. Any strategy with an open gap in the symbol could claim it: one IB sale
    of 14 MSFT was booked 10 to kalman_vecm — a futures strategy — and 4 to orb_breakout.
    From the submitting strategy's side, its position moved right after it submitted and
    before its own order filled.
  * Internal crossing booked offsetting legs against each other at the reference price the
    moment an order was made, with no fill at all.

Owners are now frozen when the order is placed, and internal crossing is off by default.
Runs the REAL CentralExecutor with only the IB socket stubbed.
"""
import logging
import os

import pytest

pytest.importorskip("ibapi")

from ibapi.contract import Contract
from ibapi.execution import Execution

from execution.central_execution import CentralExecutor
from execution.netting import NettingCoordinator

INST = {"symbol": "MSFT", "asset_class": "equity", "exchange": "SMART"}
CFG = {s: {"capital_allocation": 1e9, "max_drawdown": 0.5} for s in ("s1", "s2", "s3")}


class _Log:
    def __init__(self):
        self.fills = []

    def log_fill(self, *a, **kw):
        self.fills.append(a)

    def __getattr__(self, name):
        return lambda *a, **k: None


def _make(internal_crossing=False):
    os.environ.setdefault("EXECUTOR_API_KEY", "x")
    ex = CentralExecutor.__new__(CentralExecutor)
    CentralExecutor.__init__(ex)
    ex.risk_manager._config = CFG
    ex.risk_manager._active_strategies = set(CFG)
    ex._oid = 0

    def _next(*a, **k):
        ex._oid += 1
        return ex._oid

    ex.get_next_order_id = _next
    ex._cancelled = set()
    ex.placeOrder = lambda oid, c, o: None
    ex.cancelOrder = lambda oid, *a, **k: ex._cancelled.add(oid)
    ex.logger_db = _Log()
    co = NettingCoordinator(ex, CFG, internal_crossing=internal_crossing)
    ex.coordinator = co
    return ex, co


def pos(ex, sid, sym="MSFT"):
    return ex.ledger.strategy_positions.get(sid, {}).get(sym, 0.0)


def live(ex, sym="MSFT"):
    return [oid for oid, st in ex.order_status.items()
            if st.get("net") and st.get("symbol") == sym
            and st.get("status") in ("Submitted", "PreSubmitted")]


def fill(ex, oid, shares=None, price=500.0):
    """Deliver an execDetails fill, then mimic IB's orderStatus so a completed order is not
    later treated as working (and cancelled, reversing pending it no longer holds)."""
    st = ex.order_status[oid]
    q = st["pending_qty"]
    n = abs(q) if shares is None else shares
    c = Contract(); c.symbol = st["symbol"]
    e = Execution()
    e.orderId = oid; e.execId = f"e{oid}-{st.get('filled') or 0}"
    e.shares = n; e.side = "BOT" if q > 0 else "SLD"; e.price = price
    ex.execDetails(1, c, e)
    st["filled"] = (st.get("filled") or 0) + n
    if abs(q) - st["filled"] <= 1e-9:
        st["status"] = "Filled"


def invariant(ex, sym="MSFT"):
    total = sum(book.get(sym, 0.0) for book in ex.ledger.strategy_positions.values())
    assert abs(total - ex.ledger.current_positions.get(sym, 0.0)) < 1e-9


# ------------------------------------------------------------------ submission moves nothing
def test_submitting_does_not_move_a_position():
    ex, co = _make()
    co.set_target("s1", "MSFT", 100, instrument=INST, price=500)
    assert pos(ex, "s1") == 0, "position moved before any fill"
    (oid,) = live(ex)
    fill(ex, oid)
    assert pos(ex, "s1") == 100
    invariant(ex)


def test_an_offsetting_target_books_neither_side_before_the_fill():
    """With crossing on, s2's -60 was booked at submit at the reference price."""
    ex, co = _make()
    co.set_target("s1", "MSFT", 100, instrument=INST, price=500)
    fill(ex, live(ex)[0])
    co.set_target("s2", "MSFT", -60, instrument=INST, price=500)

    assert pos(ex, "s1") == 100 and pos(ex, "s2") == 0
    (oid,) = live(ex)
    fill(ex, oid, price=499.0)
    assert pos(ex, "s2") == -60
    assert ex.ledger.current_positions["MSFT"] == 40
    assert ex.ledger.strategy_avg_cost["s2"]["MSFT"] == pytest.approx(499.0), \
        "booked at the reference price, not the fill price"
    invariant(ex)


# ------------------------------------------------------------------ the mis-attribution
def test_a_late_fill_goes_to_the_order_that_owned_it():
    """s1's order is cancelled by s2's rebalance but fills anyway — the cancel lost the race.
    The old attribution split those 100 shares by the live desired book, 67 to s1 and 33 to
    s2. They were s1's order."""
    ex, co = _make()
    co.set_target("s1", "MSFT", 100, instrument=INST, price=500)
    (first,) = live(ex)
    co.set_target("s2", "MSFT", 50, instrument=INST, price=500)
    assert first in ex._cancelled

    fill(ex, first, shares=100)
    assert pos(ex, "s1") == 100
    assert pos(ex, "s2") == 0, "another strategy's fill landed on s2's open target"
    invariant(ex)


def test_owners_are_recorded_on_the_order():
    ex, co = _make()
    co.set_target("s1", "MSFT", 80, instrument=INST, price=500)
    co.set_target("s2", "MSFT", 40, instrument=INST, price=500)
    (oid,) = live(ex)
    assert ex.order_status[oid]["owners"] == {"s1": 80.0, "s2": 40.0}
    assert co.order_owners[oid]["delta"] == 120.0


def test_partial_fills_split_pro_rata_across_the_orders_owners():
    ex, co = _make()
    co.set_target("s1", "MSFT", 80, instrument=INST, price=500)
    co.set_target("s2", "MSFT", 40, instrument=INST, price=500)
    (oid,) = live(ex)

    fill(ex, oid, shares=60)
    assert pos(ex, "s1") == pytest.approx(40) and pos(ex, "s2") == pytest.approx(20)
    assert oid in co.order_owners, "owners dropped before the order finished filling"
    invariant(ex)

    fill(ex, oid, shares=60)
    assert pos(ex, "s1") == pytest.approx(80) and pos(ex, "s2") == pytest.approx(40)
    assert oid not in co.order_owners
    invariant(ex)


# ------------------------------------------------------------------ exact offsets
def test_an_exact_offset_goes_to_ib_as_two_orders():
    """One net order for zero shares places nothing, so neither position could ever move.
    Each side goes to the broker on its own and moves only when its own order fills."""
    ex, co = _make()
    co.set_target("s1", "MSFT", 60, instrument=INST, price=500)
    co.set_target("s2", "MSFT", -60, instrument=INST, price=500)

    orders = sorted(live(ex), key=lambda o: ex.order_status[o]["pending_qty"])
    assert [ex.order_status[o]["pending_qty"] for o in orders] == [-60, 60]
    assert pos(ex, "s1") == 0 and pos(ex, "s2") == 0

    sell, buy = orders
    assert ex.order_status[buy]["owners"] == {"s1": 60.0}
    assert ex.order_status[sell]["owners"] == {"s2": -60.0}

    fill(ex, buy)
    assert pos(ex, "s1") == 60 and pos(ex, "s2") == 0
    invariant(ex)
    fill(ex, sell)
    assert pos(ex, "s2") == -60 and ex.ledger.current_positions["MSFT"] == 0
    invariant(ex)


def test_rerunning_an_offset_while_its_legs_work_places_nothing_new():
    ex, co = _make()
    co.set_target("s1", "MSFT", 60, instrument=INST, price=500)
    co.set_target("s2", "MSFT", -60, instrument=INST, price=500)
    known, cancelled = set(ex.order_status), set(ex._cancelled)

    r = co.set_target("s2", "MSFT", -60, instrument=INST, price=500)
    assert r["orders"] == []
    assert set(ex.order_status) == known and ex._cancelled == cancelled


# ------------------------------------------------------------------ fallbacks and switches
def test_a_fill_with_no_recorded_owners_falls_back_and_says_so(caplog):
    """An order placed before a restart has no owner record. It is still attributed —
    by the desired book, as before — and the log says that is what happened."""
    ex, co = _make()
    co.desired["s1"] = {"MSFT": 10.0}
    ex.ledger.record_net_pending("MSFT", 10)
    with caplog.at_level(logging.WARNING):
        co.attribute_fill("MSFT", 10, 500.0, order_id=999)
    assert pos(ex, "s1") == 10
    assert any("999" in r.getMessage() and "owners" in r.getMessage() for r in caplog.records)


def test_internal_crossing_is_off_by_default():
    ex, co = _make()
    assert co.internal_crossing is False
    co.set_target("s1", "MSFT", 100, instrument=INST, price=500)
    r = co.set_target("s2", "MSFT", -60, instrument=INST, price=500)
    assert r["internal_crosses"] == []
    assert not any(str(a[1]).startswith("xnet-") for a in ex.logger_db.fills)
    assert pos(ex, "s1") == 0 and pos(ex, "s2") == 0


def test_crossing_can_be_switched_back_on():
    ex, co = _make(internal_crossing=True)
    co.set_target("s1", "MSFT", 100, instrument=INST, price=500)
    r = co.set_target("s2", "MSFT", -60, instrument=INST, price=500)
    assert r["internal_crosses"], "crossing enabled but nothing crossed"


def test_bookkeeping_buckets_are_never_owners():
    ex, co = _make()
    co.desired["__net__"] = {"MSFT": 5.0}
    co.desired["flatten_all"] = {"MSFT": -3.0}
    co.desired["s1"] = {"MSFT": 10.0}
    assert co._owner_gaps("MSFT") == {"s1": 10.0}
