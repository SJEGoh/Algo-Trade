"""Ways a drawdown halt left a strategy exposed — each one is now closed.

A halt is only as good as what follows it. `halt_and_flatten` fires ONCE (every later call
returns early because `is_active` is already False), so anything that went wrong during
that single attempt used to be permanent: the strategy showed as HALTED on the dashboard
while still carrying the position that breached its limit.
"""
import time

import pytest

import execution.central_execution as ce
from config import validate_config
from risk.risk_manager import RiskManager

ALLOC = 10_000.0
CFG = {"s1": {"capital_allocation": ALLOC, "max_drawdown": 0.10}}   # halts at -$1,000
BREACH = -1_500.0


class FakeDB:
    """Keeps the real db/executor.db out of the tests."""
    def __init__(self, *a, **kw): self.halts = []
    def log_fill(self, *a, **kw): pass
    def log_equity(self, *a, **kw): pass
    def log_decision(self, *a, **kw): pass
    def log_risk_event(self, *a, **kw): pass
    def log_reconciliation(self, *a, **kw): pass
    def save_strategy_positions(self, *a, **kw): pass
    def save_realized_pnl(self, *a, **kw): pass
    def save_multipliers(self, *a, **kw): pass
    def save_strategy_cash(self, *a, **kw): pass
    def save_halted_strategies(self, halted, active, keys, reason=""):
        self.halts.append(set(keys) - set(active))
    def load_strategy_positions(self): return {}, {}
    def load_realized_pnl(self): return {}
    def load_multipliers(self): return {}
    def load_strategy_cash(self): return {}, {}
    def load_halted_strategies(self): return set()
    def close(self): pass


@pytest.fixture
def ex(monkeypatch):
    """A CentralExecutor with no IB socket and no real database."""
    monkeypatch.setattr(ce, "EventLogger", FakeDB)
    x = ce.CentralExecutor()
    x.logger_db = FakeDB()
    x.risk_manager = RiskManager(x.ledger, CFG, {})
    x.placed, x.cancelled = [], []
    x.place_order = lambda intent: (x.placed.append(intent), len(x.placed))[1]
    x.cancelOrder = lambda oid, *a, **kw: x.cancelled.append(oid)
    x._alerter = None
    return x


def hold(ex, symbol="AAPL", qty=100.0, cost=50.0):
    ex.ledger.strategy_positions["s1"] = {symbol: qty}
    ex.ledger.strategy_avg_cost["s1"] = {symbol: cost}
    ex.ledger.current_positions[symbol] = qty


# ------------------------------------------------------- flatten never retried
def test_a_failed_flatten_is_retried(ex):
    """The broker rejected the closing order. The strategy was halted and left holding,
    and nothing ever tried again — `enforce_drawdown` returns early once halted."""
    hold(ex)

    def reject(intent):
        raise RuntimeError("IB rejected: order not allowed")
    ex.place_order = reject
    ex.enforce_drawdown("s1", BREACH)

    assert ex.risk_manager.is_active("s1") is False
    assert ex.ledger.strategy_positions["s1"]["AAPL"] == 100.0      # still exposed

    ex.place_order = lambda intent: (ex.placed.append(intent), 1)[1]   # broker recovers
    r = ex.ensure_flat("s1")                                            # sampler's next cycle
    assert r["retried"] is True and r["ok"] is True
    assert [(i["side"], i["quantity"]) for i in ex.placed] == [("sell", 100.0)]


def test_retry_does_not_stack_orders_while_one_is_working(ex):
    """Market closed: the closing order sits unfilled. Retrying every cycle would pile up
    duplicate closing orders and oversell on the open."""
    hold(ex)
    ex.enforce_drawdown("s1", BREACH)
    assert len(ex.placed) == 1
    ex.order_status[1] = {"symbol": "AAPL", "status": "Submitted", "strategy_id": "s1",
                          "pending_qty": -100.0}

    for _ in range(5):
        ex._flatten_retry_ts.clear()                 # ignore the cooldown
        assert ex.ensure_flat("s1")["retried"] is False
    assert len(ex.placed) == 1                       # still exactly one closing order


def test_retry_respects_a_cooldown(ex):
    hold(ex)
    ex.enforce_drawdown("s1", BREACH)
    ex.placed.clear()
    assert ex.ensure_flat("s1")["retried"] is True   # first retry goes
    assert ex.ensure_flat("s1")["reason"] == "cooldown"
    ex._flatten_retry_ts["s1"] = time.time() - 10_000
    assert ex.ensure_flat("s1")["retried"] is True


def test_retry_stops_once_flat(ex):
    hold(ex)
    ex.enforce_drawdown("s1", BREACH)
    ex.ledger.strategy_positions["s1"] = {"AAPL": 0.0}      # the flatten filled
    ex.placed.clear()
    assert ex.ensure_flat("s1") == {"retried": False, "reason": "flat"}
    assert ex.placed == []


def test_active_strategies_are_never_flattened_by_the_retry(ex):
    hold(ex)
    assert ex.ensure_flat("s1") == {"retried": False, "reason": "not halted"}
    assert ex.placed == []


def test_retry_only_closes_what_is_still_open(ex):
    """A partial unwind: one leg closed, one didn't. Only the open leg is re-sent."""
    ex.ledger.strategy_positions["s1"] = {"AAPL": 0.0, "MSFT": 40.0}
    ex.ledger.strategy_avg_cost["s1"] = {"AAPL": 50.0, "MSFT": 400.0}
    ex.risk_manager.halt_strategy("s1", "test")
    r = ex.ensure_flat("s1")
    assert r["symbols"] == ["MSFT"]
    assert [i["instrument"]["symbol"] for i in ex.placed] == ["MSFT"]


# ------------------------------------------------------- in-flight orders survive a halt
def test_halt_cancels_the_strategys_working_orders(ex):
    """An unfilled BUY that lands after the flatten re-opens the position the halt closed."""
    hold(ex)
    ex.order_status[42] = {"symbol": "AAPL", "status": "Submitted", "strategy_id": "s1",
                           "pending_qty": 400.0}
    ex.ledger.record_pending("AAPL", 400.0, "s1")

    ex.enforce_drawdown("s1", BREACH)

    assert 42 in ex.cancelled
    assert ex.order_status[42]["status"] == "PendingCancel"
    # the cancelled order's pending contribution is reversed, not left on the books —
    # otherwise the halted strategy still looks like it is buying 400 more
    assert ex.ledger.strategy_pending["s1"]["AAPL"] == pytest.approx(0.0)
    assert ex.ledger.strategy_effective_positions("s1")["AAPL"] == pytest.approx(100.0)


def test_halt_leaves_other_strategies_orders_alone(ex):
    hold(ex)
    ex.order_status[7] = {"symbol": "AAPL", "status": "Submitted", "strategy_id": "other",
                          "pending_qty": 10.0}
    ex.enforce_drawdown("s1", BREACH)
    assert 7 not in ex.cancelled


def test_halt_leaves_pooled_net_orders_to_the_coordinator(ex):
    """__net__ orders belong to more strategies than this one; the coordinator's unwind
    cancels them per symbol."""
    hold(ex)
    ex.order_status[9] = {"symbol": "AAPL", "status": "Submitted", "strategy_id": "__net__",
                          "net": True, "pending_qty": 25.0}
    ex.enforce_drawdown("s1", BREACH)
    assert 9 not in ex.cancelled


# ------------------------------------------------------- config disables the halt silently
def test_the_shipped_config_is_valid():
    assert validate_config() == []


@pytest.mark.parametrize("entry, expect", [
    ({"capital_allocation": 10_000.0}, "max_drawdown"),          # halt silently disabled
    ({"max_drawdown": 0.1}, "capital_allocation"),               # cap silently disabled
    ({"capital_allocation": 0, "max_drawdown": 0.1}, "capital_allocation"),
    ({"capital_allocation": 10_000.0, "max_drawdown": 15}, "max_drawdown"),   # 15 vs 0.15
    ({"capital_allocation": 10_000.0, "max_drawdown": 0}, "max_drawdown"),
])
def test_a_broken_strategy_config_is_caught_at_startup(entry, expect):
    """drawdown_status returns "not breached" when either key is missing, so a typo
    disables the protection instead of announcing it. Startup must refuse to run."""
    problems = validate_config({"x": entry})
    assert problems and any(expect in p for p in problems)


def test_missing_max_drawdown_really_does_disable_the_halt():
    """Why the startup check exists: the predicate itself can't tell you."""
    rm = RiskManager(ce.PositionLedger(None), {"typo": {"capital_allocation": 10_000.0}}, {})
    assert rm.drawdown_status("typo", -1_000_000.0)["breached"] is False


# ------------------------------------------------------- pooled fast path
def test_net_fill_checks_the_strategies_it_was_attributed_to(ex):
    """The per-fill check iterated coordinator.desired only. After a lost/reset
    netting.json that dict is empty — which silently disabled the fast path entirely."""
    checked = []

    class Coord:
        desired = {}                                  # book state lost
        def attribute_fill(self, *a, order_id=None):
            return [("s1", 100.0)]

    ex.coordinator = Coord()
    ex.ledger.strategy_realized_pnl["s1"] = -5_000.0
    ex.enforce_drawdown = lambda sid, pnl, source="realized": checked.append(sid)

    class Execution:
        orderId, execId, side, shares, price = 7, "e1", "BOT", 100, 50.0

    class Contract:
        symbol, secType, exchange, multiplier = "AAPL", "STK", "SMART", None

    ex.order_status[7] = {"symbol": "AAPL", "status": "Filled",
                          "strategy_id": "__net__", "net": True}
    ex.execDetails(1, Contract(), Execution())

    assert "s1" in checked


# ------------------------------------------------------- sampler wiring
class _SamplerExec:
    """Minimal executor for one sampler cycle, with s1 halted and still holding."""

    def __init__(self):
        self.ledger = ce.PositionLedger(None, CFG)
        self.ledger.strategy_positions["s1"] = {"AAPL": 100.0}
        self.ledger.strategy_avg_cost["s1"] = {"AAPL": 50.0}
        self.logger_db = FakeDB()
        self.ensure_flat_calls = []
        self.fresh = True

        class RM:
            def is_active(self_inner, sid): return False      # halted
        self.risk_manager = RM()

    def get_marks(self, symbols): return {s: 40.0 for s in symbols}
    def mark_is_fresh(self, symbol): return self.fresh
    def enforce_daily_loss(self, total): pass
    def enforce_drawdown(self, sid, pnl, source="realized"): pass
    def ensure_flat(self, sid):
        self.ensure_flat_calls.append(sid)
        return {"retried": True, "ok": True}


def _run_one_cycle(server, monkeypatch, executor, cycles=1, config=None):
    import threading
    monkeypatch.setattr(server, "executor", executor)
    monkeypatch.setattr(server, "CONFIG", config or CFG)
    server._sampler_stop.clear()
    server._stale_skips.clear()
    t = threading.Thread(target=server._equity_sampler, args=(0.01,), daemon=True)
    t.start()
    for _ in range(500):
        if len(getattr(executor, "cycles_seen", executor.ensure_flat_calls)) >= cycles:
            break
        server._sampler_stop.wait(0.01)
    server._sampler_stop.set()
    t.join(timeout=5)
    server._sampler_stop.clear()


def test_sampler_retries_the_unwind_of_a_halted_strategy(monkeypatch):
    """The retry is only useful if something calls it every cycle."""
    import api.server as server
    ex = _SamplerExec()
    _run_one_cycle(server, monkeypatch, ex)
    assert "s1" in ex.ensure_flat_calls


def test_unguarded_stale_mark_escalates_to_critical(monkeypatch, caplog):
    """Skipping the unrealized-drawdown check on a stale mark is deliberate — a bad quote
    must not halt a good book — but an unbounded silent skip is an unguarded strategy."""
    import logging
    import api.server as server

    class Active(_SamplerExec):
        def __init__(self):
            super().__init__()
            self.cycles_seen = []
            self.fresh = False                       # no usable mark

            class RM:
                def is_active(self_inner, sid): return True
            self.risk_manager = RM()

        def get_marks(self, symbols):
            self.cycles_seen.append(1)
            return {}

    ex = Active()
    with caplog.at_level(logging.CRITICAL, logger="executor"):
        _run_one_cycle(server, monkeypatch, ex, cycles=server._STALE_SKIP_ALERT_AFTER + 1)
    assert any("UNGUARDED" in r.message for r in caplog.records), \
        "a strategy skipped every cycle must page, not just log a warning"


def test_a_failed_snapshot_write_does_not_disable_the_drawdown_check(monkeypatch):
    """Found while writing these tests: a throw ANYWHERE in the cycle — here a database
    write — aborted the whole cycle before enforcement, so the drawdown check silently
    never ran, every cycle, behind one ERROR log line."""
    import api.server as server

    class BrokenDB(FakeDB):
        def log_equity(self, *a, **kw):
            raise RuntimeError("database is locked")

    class Exec(_SamplerExec):
        def __init__(self):
            super().__init__()
            self.logger_db = BrokenDB()
            self.checked = []
            self.cycles_seen = []

            class RM:
                def is_active(self_inner, sid): return True
            self.risk_manager = RM()

        def get_marks(self, symbols):
            self.cycles_seen.append(1)
            return {s: 40.0 for s in symbols}

        def enforce_drawdown(self, sid, pnl, source="realized"):
            self.checked.append((sid, source))

    ex = Exec()
    _run_one_cycle(server, monkeypatch, ex)
    assert ("s1", "total") in ex.checked, \
        "history logging must never cost us a risk check"


def test_one_bad_strategy_does_not_starve_the_others(monkeypatch):
    """Enforcement is per-strategy: a throw on one must not skip the rest of the book."""
    import api.server as server

    cfg = {"s1": CFG["s1"], "s2": dict(CFG["s1"])}

    class Exec(_SamplerExec):
        def __init__(self):
            super().__init__()
            self.checked = []
            self.cycles_seen = []

            class RM:
                def is_active(self_inner, sid): return True
            self.risk_manager = RM()

        def get_marks(self, symbols):
            self.cycles_seen.append(1)
            return {s: 40.0 for s in symbols}

        def mark_is_fresh(self, symbol):
            return True

        def enforce_drawdown(self, sid, pnl, source="realized"):
            if sid == "s1":
                raise RuntimeError("boom")
            self.checked.append(sid)

    ex = Exec()
    _run_one_cycle(server, monkeypatch, ex, config=cfg)
    assert "s2" in ex.checked
