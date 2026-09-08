"""The read-only retry loop, and why `max_retries=5` bounded nothing.

IB rejects an order with code 321 when the gateway is in read-only mode, and the executor
resubmits. The bug was in where the counter lived: it counted retries of ONE order, but each
retry is a NEW order with a NEW id, so IB's rejection of the replacement arrived as a fresh
failure and started a fresh chain with a fresh budget. "At most 5 retries" was really an
unbounded loop that produced an order every 15 seconds until the process was restarted —
observed in production as 27 phantom MCL orders and a still-climbing ARM chain.

The cap is therefore per SYMBOL and spans the chain. These tests drive the error callback
the way IB does — rejecting the replacement too — because that is the case the old shape
got wrong and a per-order test would still pass.
"""
import threading

import pytest

import execution.central_execution as ce
from risk.risk_manager import RiskManager

CFG = {"s1": {"capital_allocation": 100_000.0, "max_drawdown": 0.10}}


class FakeDB:
    def __init__(self, *a, **kw): pass
    def log_fill(self, *a, **kw): pass
    def log_equity(self, *a, **kw): pass
    def log_decision(self, *a, **kw): pass
    def log_risk_event(self, *a, **kw): pass
    def log_reconciliation(self, *a, **kw): pass
    def log_order(self, *a, **kw): pass
    def update_order_status(self, *a, **kw): pass
    def save_strategy_positions(self, *a, **kw): pass
    def save_realized_pnl(self, *a, **kw): pass
    def save_multipliers(self, *a, **kw): pass
    def save_strategy_cash(self, *a, **kw): pass
    def save_halted_strategies(self, *a, **kw): pass
    def load_strategy_positions(self): return {}, {}
    def load_realized_pnl(self): return {}
    def load_multipliers(self): return {}
    def load_strategy_cash(self): return {}, {}
    def load_halted_strategies(self): return set()
    def close(self): pass


@pytest.fixture
def ex(monkeypatch):
    monkeypatch.setattr(ce, "EventLogger", FakeDB)
    monkeypatch.setattr(ce.time, "sleep", lambda s: None)     # no 15s wait in tests
    x = ce.CentralExecutor()
    x.logger_db = FakeDB()
    x.risk_manager = RiskManager(x.ledger, CFG, {})
    x._alerter = None
    x.placed = []

    # Every placement is recorded and gets an order_status entry, exactly as the real
    # place_* methods do — that entry is what the retry path reads `pending_qty` from.
    def fake_place(sym, pending):
        oid = 100 + len(x.placed)
        x.placed.append((oid, sym, pending))
        x.order_status[oid] = {"symbol": sym, "status": "Submitted",
                               "pending_qty": pending, "strategy_id": "s1", "net": True}
        return oid

    x.place_net_order = lambda sym, delta, inst, ref, urgent=False: fake_place(sym, delta)
    x.place_order = lambda intent: fake_place(intent["instrument"]["symbol"],
                                              intent["quantity"])
    return x


def drain():
    """The resubmission happens on a background thread; wait for it so assertions are not
    racing the retry they are about to check."""
    for t in threading.enumerate():
        if t.name.startswith("retry-321-"):
            t.join(timeout=2)


def reject(ex, order_id):
    """IB says 321 for this order, the way the EWrapper callback delivers it."""
    ex.error(order_id, 321, "Error validating request: read-only API")
    drain()


def seed(ex, sym="ARM", pending=8.0, oid=1):
    ex.order_status[oid] = {"symbol": sym, "status": "Submitted",
                            "pending_qty": pending, "strategy_id": "s1", "net": True}
    return oid


# ------------------------------------------------------------------ the cascade
def test_the_chain_stops_after_two_retries(ex):
    """Reject every replacement, as a read-only gateway does. Before the fix this never
    terminated; now the whole chain is worth exactly two resubmissions."""
    oid = seed(ex)
    reject(ex, oid)
    for _ in range(10):                      # keep rejecting whatever comes back
        if not ex.placed:
            break
        last_oid = ex.placed[-1][0]
        before = len(ex.placed)
        reject(ex, last_oid)
        if len(ex.placed) == before:
            break

    assert len(ex.placed) == 2, f"placed {len(ex.placed)} orders, expected the cap of 2"
    assert ex._readonly_retries["ARM"] == 2


def test_a_single_rejection_retries_once(ex):
    reject(ex, seed(ex))
    assert len(ex.placed) == 1


def test_giving_up_is_loud(ex, caplog):
    """The order was never placed. Silence here is a position the strategy thinks it has."""
    import logging
    oid = seed(ex)
    reject(ex, oid)
    reject(ex, ex.placed[-1][0])
    with caplog.at_level(logging.CRITICAL):
        reject(ex, ex.placed[-1][0])

    assert len(ex.placed) == 2
    assert any(r.levelno >= logging.CRITICAL and "ARM" in r.getMessage()
               for r in caplog.records), "gave up without a CRITICAL alert"


# ------------------------------------------------------------------ scope of the budget
def test_the_budget_is_per_symbol(ex):
    """One symbol exhausting its retries must not stop another symbol from getting any."""
    reject(ex, seed(ex, "ARM", 8.0, oid=1))
    reject(ex, ex.placed[-1][0])
    assert ex._readonly_retries["ARM"] == 2

    reject(ex, seed(ex, "ANET", -10.0, oid=50))
    assert ("ANET" in ex._readonly_retries) and ex._readonly_retries["ANET"] == 1
    assert any(sym == "ANET" for _, sym, _ in ex.placed)


def test_a_fill_restores_the_budget(ex):
    """Otherwise the cap is one-shot for the life of the process and a read-only episode
    next month is refused on a counter left over from this one."""
    reject(ex, seed(ex))
    reject(ex, ex.placed[-1][0])
    assert ex._readonly_retries["ARM"] == 2

    ex.clear_readonly_retries("ARM")
    assert "ARM" not in ex._readonly_retries

    reject(ex, seed(ex, "ARM", 8.0, oid=90))
    assert ex._readonly_retries["ARM"] == 1


# ------------------------------------------------------------------ guards
def test_nothing_pending_is_not_retried(ex):
    ex.order_status[1] = {"symbol": "ARM", "status": "Submitted",
                          "pending_qty": 0.0, "strategy_id": "s1", "net": True}
    reject(ex, 1)
    assert ex.placed == []


def test_an_unknown_order_is_ignored(ex):
    ex.error(999, 321, "Error validating request")
    assert ex.placed == []


def test_other_error_codes_do_not_retry(ex):
    """Only 321 means read-only. A rejected order for any other reason is a decision."""
    oid = seed(ex)
    ex.error(oid, 201, "Order rejected - reason: insufficient margin")
    assert ex.placed == []
