"""Pending is released exactly once, for exactly the unfilled remainder, however an order ends.

Pending is the quantity the executor believes is still on its way. It feeds
effective_position, which the netting rebalance trades against — so pending held for an
order that no longer exists makes a symbol look already on target, and the strategy silently
stops getting fills in it.

Before this, release happened in several places with different rules, and most never did it:

  * the kill switch, /flatten and the daily ATR sweep sent a bare cancelOrder and released
    nothing; neither did any cancel IB made itself (a day order expiring, a rejection);
  * the netting cancel released the FULL original quantity, over-releasing after a fill;
  * reconcile could release an already-cancelled order a second time, and released pooled
    orders into a phantom strategy_pending["__net__"];
  * the read-only retry wiped all pending for the symbol (pooled) or double-counted (direct);
  * IB's code 202 "Order Canceled" was labelled a rejection.
"""
import threading

import pytest
from fastapi.testclient import TestClient
from ibapi.contract import Contract
from ibapi.execution import Execution

import api.server as server
import execution.central_execution as ce
from execution.netting import NettingCoordinator
from risk.risk_manager import RiskManager

CFG = {s: {"capital_allocation": 1e9, "max_drawdown": 0.5} for s in ("s1", "s2")}
API_KEY = "test-key-123"
AUTH = {"X-API-Key": API_KEY}


class FakeDB:
    def __init__(self, *a, **kw): pass
    def __getattr__(self, _name):
        return lambda *a, **kw: None
    def load_strategy_positions(self): return {}, {}
    def load_realized_pnl(self): return {}
    def load_multipliers(self): return {}
    def load_strategy_cash(self): return {}, {}
    def load_halted_strategies(self): return set()


@pytest.fixture
def ex(monkeypatch):
    monkeypatch.setattr(ce, "EventLogger", FakeDB)
    x = ce.CentralExecutor()
    x.logger_db = FakeDB()
    x.risk_manager = RiskManager(x.ledger, CFG, {})
    x._alerter = None
    x.cancelled = []
    counter = [100]

    def next_id(*a, **k):
        counter[0] += 1
        return counter[0]

    monkeypatch.setattr(x, "get_next_order_id", next_id)
    monkeypatch.setattr(x, "placeOrder", lambda oid, c, o: None)
    monkeypatch.setattr(x, "_paper_subscribe", lambda *a, **kw: None)
    monkeypatch.setattr(x, "cancelOrder", lambda oid, *a, **kw: x.cancelled.append(oid))
    return x


def direct(ex, qty=10, sym="AAPL", sid="s1"):
    return ex.place_order({
        "client_order_id": f"c-{sym}-{qty}-{len(ex.order_status)}", "strategy_id": sid,
        "instrument": {"symbol": sym, "asset_class": "equity", "sec_type": "STK",
                       "exchange": "SMART"},
        "side": "buy" if qty > 0 else "sell", "quantity": abs(qty),
        "order_type": "market", "time_in_force": "day", "expected_price": 100.0})


def pooled(ex, delta=10, sym="AAPL"):
    return ex.place_net_order(sym, delta, {"symbol": sym, "sec_type": "STK",
                                           "exchange": "SMART"}, 100.0, urgent=True)


def fill(ex, oid, shares):
    st = ex.order_status[oid]
    c = Contract(); c.symbol = st["symbol"]
    e = Execution()
    e.orderId = oid; e.execId = f"e{oid}-{st.get('exec_filled')}-{shares}"
    e.shares = shares; e.side = "BOT" if st["pending_qty"] > 0 else "SLD"; e.price = 100.0
    ex.execDetails(1, c, e)


def ib_status(ex, oid, status):
    ex.orderStatus(oid, status, 0, 0, 0, 0, 0, 0, 0, "", 0)


def pending(ex, sym="AAPL"):
    return ex.ledger.pending_deltas.get(sym, 0.0)


def drain():
    for t in threading.enumerate():
        if t.name.startswith("retry-321-"):
            t.join(timeout=2)


# ------------------------------------------------------------------ IB ends the order
@pytest.mark.parametrize("status", ["Cancelled", "ApiCancelled", "Inactive"])
def test_ib_ending_an_order_releases_its_pending(ex, status):
    """A day order expiring at the close, a rejection, a cancel from TWS — none of these
    released pending before, so the next rebalance thought those shares were on the way."""
    oid = direct(ex, 10)
    assert pending(ex) == 10
    ib_status(ex, oid, status)
    assert pending(ex) == 0


def test_a_filled_order_releases_nothing_extra(ex):
    oid = direct(ex, 10)
    fill(ex, oid, 10)
    ib_status(ex, oid, "Filled")
    assert pending(ex) == 0
    assert ex.ledger.current_positions["AAPL"] == 10


def test_release_happens_exactly_once(ex):
    """IB can report Cancelled more than once, and a 202 arrives alongside it."""
    oid = direct(ex, 10)
    ex.cancel_order(oid, "test")
    ib_status(ex, oid, "Cancelled")
    ib_status(ex, oid, "Cancelled")
    ex.error(oid, 202, "Order Canceled - reason:")
    assert pending(ex) == 0


# ------------------------------------------------------------------ remainder, not original
def test_cancel_after_a_partial_fill_releases_only_the_unfilled_remainder(ex):
    """The old netting cancel released the full 10 after 4 had filled, driving pending to -4
    and making the next rebalance over-trade by the filled amount."""
    oid = direct(ex, 10)
    fill(ex, oid, 4)
    assert pending(ex) == 6
    ex.cancel_order(oid, "test")
    assert pending(ex) == 0
    assert ex.ledger.current_positions["AAPL"] == 4


def test_a_late_fill_after_a_cancel_moves_the_position_not_pending(ex):
    """The cancel lost the race and the order filled anyway. The shares really traded, so the
    position must move — but pending was already released, and must not go negative."""
    oid = direct(ex, 10)
    ex.cancel_order(oid, "test")
    assert pending(ex) == 0
    fill(ex, oid, 10)
    assert ex.ledger.current_positions["AAPL"] == 10
    assert pending(ex) == 0


def test_a_short_order_releases_in_the_right_direction(ex):
    oid = direct(ex, -10)
    fill(ex, oid, 3)
    assert pending(ex) == -7
    ib_status(ex, oid, "Cancelled")
    assert pending(ex) == 0


# ------------------------------------------------------------------ every cancel path
def test_the_kill_switch_releases_pending(ex):
    oid = direct(ex, 10)
    ex.kill_switch(flatten=False)
    assert oid in ex.cancelled
    assert pending(ex) == 0
    assert ex.order_status[oid]["status"] == "PendingCancel"


def test_a_halt_releases_pending(ex):
    oid = direct(ex, 10)
    ex._cancel_strategy_orders("s1")
    assert pending(ex) == 0


def test_an_order_is_cancelled_at_ib_only_once(ex):
    """/flatten cancelled every order and left it at Submitted, then its own rebalance
    cancelled the same order again — a second cancel request and a second release."""
    oid = pooled(ex, 10)
    ex.cancel_order(oid, "flatten")
    ex._cancel_open_orders_for_symbol("AAPL")
    assert ex.cancelled.count(oid) == 1
    assert pending(ex) == 0


def test_a_pooled_release_creates_no_phantom_net_strategy_pending(ex):
    oid = pooled(ex, 10)
    ib_status(ex, oid, "Cancelled")
    assert pending(ex) == 0
    assert ex.ledger.strategy_pending.get("__net__", {}).get("AAPL", 0.0) == 0.0


def test_the_netting_rebalance_still_sizes_the_replacement_correctly(ex):
    """Release has to happen at send time, not on IB's confirmation: the rebalance sizes its
    replacement the instant it has cancelled."""
    co = NettingCoordinator(ex, CFG)
    ex.coordinator = co
    inst = {"symbol": "AAPL", "asset_class": "equity", "exchange": "SMART"}
    co.set_target("s1", "AAPL", 100, instrument=inst, price=100)
    co.set_target("s2", "AAPL", -60, instrument=inst, price=100)
    live = [o for o, st in ex.order_status.items() if st.get("status") == "Submitted"]
    assert len(live) == 1
    assert ex.order_status[live[0]]["pending_qty"] == 40
    assert pending(ex) == 40


# ------------------------------------------------------------------ reconcile
def _no_ib_orders(ex):
    ex.reqAllOpenOrders = lambda: ex._reconcile_orders_done.set()


def test_reconcile_does_not_release_a_cancelled_order_a_second_time(ex):
    oid = direct(ex, 10)
    ex.cancel_order(oid, "rebalance")            # released, PendingCancel
    _no_ib_orders(ex)
    r = ex.reconcile_open_orders(timeout=1)
    assert [d["order_id"] for d in r["stale_removed"]] == [oid]
    assert pending(ex) == 0, "released twice — pending went negative"


def test_reconcile_releases_an_order_ib_no_longer_has(ex):
    oid = direct(ex, 10)
    _no_ib_orders(ex)
    ex.reconcile_open_orders(timeout=1)
    assert pending(ex) == 0
    assert ex.order_status[oid]["status"] == "Reconciled_Stale"


def test_reconcile_releases_a_pooled_order_at_the_net_level(ex):
    pooled(ex, 10)
    _no_ib_orders(ex)
    ex.reconcile_open_orders(timeout=1)
    assert pending(ex) == 0
    assert ex.ledger.strategy_pending.get("__net__", {}).get("AAPL", 0.0) == 0.0


# ------------------------------------------------------------------ IB's cancel responses
def test_code_202_confirms_a_cancel_and_is_not_a_rejection(ex):
    oid = direct(ex, 10)
    ex.error(oid, 202, "Order Canceled - reason:")
    assert ex.order_status[oid].get("ack") != "rejected"
    assert pending(ex) == 0


def test_code_10147_order_not_found_releases(ex):
    oid = direct(ex, 10)
    ex.error(oid, 10147, "OrderId that needs to be cancelled is not found.")
    assert pending(ex) == 0
    assert ex.order_status[oid].get("ack") != "rejected"


@pytest.mark.parametrize("code", [161, 10148])
def test_a_refused_cancel_is_not_a_rejection_and_the_fill_still_books(ex, code):
    """The order had already filled, so IB could not cancel it. Nothing to undo: pending was
    released on send, and the fill that follows is compensated."""
    oid = direct(ex, 10)
    ex.cancel_order(oid, "test")
    ex.error(oid, code, "Cancel attempted when order is not in a cancellable state")
    assert ex.order_status[oid].get("ack") != "rejected"
    fill(ex, oid, 10)
    assert ex.ledger.current_positions["AAPL"] == 10
    assert pending(ex) == 0


# ------------------------------------------------------------------ read-only retry
def test_a_pooled_retry_releases_only_the_failed_orders_share(ex, monkeypatch):
    """It used to pop ALL pending for the symbol, wiping the direct order still working."""
    monkeypatch.setattr(ce.time, "sleep", lambda s: None)
    direct(ex, 5, sid="s2")
    failed = pooled(ex, 10)
    assert pending(ex) == 15
    ex.error(failed, 321, "The API interface is currently in Read-Only mode.")
    drain()
    assert pending(ex) == 15          # 5 still working + the resubmitted 10


def test_a_direct_retry_does_not_count_its_shares_twice(ex, monkeypatch):
    monkeypatch.setattr(ce.time, "sleep", lambda s: None)
    failed = direct(ex, 10)
    ex.error(failed, 321, "The API interface is currently in Read-Only mode.")
    drain()
    assert pending(ex) == 10, "the failed order's 10 and the retry's 10 both counted"


def test_giving_up_releases_the_orders_pending(ex):
    """IB sends no Inactive for a 321, so a given-up order's pending sat there until restart."""
    ex._readonly_retries["AAPL"] = ex.READONLY_MAX_RETRIES
    failed = direct(ex, 10)
    ex.error(failed, 321, "The API interface is currently in Read-Only mode.")
    assert pending(ex) == 0


# ------------------------------------------------------------------ the API surface
@pytest.fixture
def client(ex, monkeypatch):
    monkeypatch.setattr(server, "executor", ex)
    monkeypatch.setattr(server, "EXECUTOR_API_KEY", API_KEY)
    monkeypatch.setattr(server, "_alert", lambda *a, **kw: None)
    return TestClient(server.app)


def test_flatten_releases_pending(ex, client, monkeypatch):
    monkeypatch.setattr(server, "is_market_open", lambda: False)   # cancel-only mode
    oid = direct(ex, 10)
    assert client.post("/flatten", headers=AUTH).status_code == 200
    assert oid in ex.cancelled
    assert pending(ex) == 0


def test_the_atr_sweep_releases_pending(ex, client):
    oid = direct(ex, 10)
    ex.atr_layer.record_order(oid)
    body = client.post("/atr/cancel", headers=AUTH).json()
    assert body["cancelled"] == [oid]
    assert pending(ex) == 0


def test_pending_endpoint_is_consistent_when_orders_explain_it(ex, client):
    direct(ex, 10)
    body = client.get("/pending").json()
    assert body["pending"] == {"AAPL": 10.0}
    assert body["explained_by_orders"] == {"AAPL": 10.0}
    assert body["consistent"] is True


def test_pending_endpoint_flags_pending_no_order_explains(ex, client):
    """The leak this whole change is about, made visible."""
    ex.ledger.record_pending("MSFT", 7, "s1")
    body = client.get("/pending").json()
    assert body["unexplained"] == {"MSFT": 7.0}
    assert body["consistent"] is False


def test_pending_endpoint_ignores_released_orders(ex, client):
    oid = direct(ex, 10)
    ex.cancel_order(oid, "test")
    body = client.get("/pending").json()
    assert body["orders"] == [] and body["consistent"] is True
