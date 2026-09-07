"""Equity sampler: the portfolio curve is the sum of every position of every strategy.

The sampler is what feeds both the dashboard's equity curve (via equity_snapshots) and
the portfolio circuit breaker, so these tests pin down that it writes a NAV per strategy
(cash + positions), totals them across strategies, and still hands the drawdown check
plain P&L.
"""
import threading

import pytest

import api.server as server
from ledger.position_ledger import PositionLedger
from logger.event_logger import EventLogger

CFG = {
    "alpha": {"capital_allocation": 100_000.0, "max_drawdown": 0.15},
    "beta": {"capital_allocation": 50_000.0, "max_drawdown": 0.15},
}


class FakeRiskManager:
    def is_active(self, strategy_id):
        return True


class FakeExecutor:
    """Just enough executor for one sampler cycle: a real ledger + DB, canned marks."""

    def __init__(self, db, mark=60.0):
        self.ledger = PositionLedger(None, CFG)
        self.logger_db = db
        self.risk_manager = FakeRiskManager()
        self._mark = mark
        self.breaker_saw = None
        self.drawdown_calls = []

    def get_marks(self, symbols):
        return {s: self._mark for s in symbols}

    def mark_is_fresh(self, symbol):
        return True

    def enforce_daily_loss(self, total_equity):
        self.breaker_saw = total_equity

    def enforce_drawdown(self, sid, pnl, source="realized"):
        self.drawdown_calls.append((sid, pnl, source))


@pytest.fixture
def sampled(tmp_path, monkeypatch):
    """Run exactly one sampler cycle and hand back the executor it ran against."""
    db = EventLogger(db_path=tmp_path / "executor.db")
    ex = FakeExecutor(db)
    ex.ledger.record_fill("AAPL", +100, 50.0, "alpha")   # cash 95k, marked 60 -> 6k of stock
    monkeypatch.setattr(server, "executor", ex)
    monkeypatch.setattr(server, "CONFIG", CFG)
    server._sampler_stop.clear()
    server._last_equity.update({"ts": None, "strategies": {}, "totals": {}, "marks": {}})

    t = threading.Thread(target=server._equity_sampler, args=(60.0,), daemon=True)
    t.start()
    for _ in range(200):                                 # wait for the first cycle to land
        if server._last_equity["ts"]:
            break
        server._sampler_stop.wait(0.01)
    server._sampler_stop.set()
    t.join(timeout=5)
    yield ex, db
    server._sampler_stop.clear()
    db.close()


def test_portfolio_nav_is_the_sum_over_strategies(sampled):
    ex, _ = sampled
    totals = server._last_equity["totals"]
    assert totals["nav"] == pytest.approx(151_000.0)     # alpha 101k + beta's untouched 50k
    assert totals["cash"] == pytest.approx(145_000.0)
    assert totals["position_value"] == pytest.approx(6_000.0)
    assert totals["equity"] == pytest.approx(1_000.0)    # P&L only
    assert totals["nav"] == pytest.approx(totals["cash"] + totals["position_value"])
    assert totals["nav"] == pytest.approx(totals["starting_cash"] + totals["equity"])


def test_flat_strategy_still_contributes_its_cash(sampled):
    ex, _ = sampled
    beta = server._last_equity["strategies"]["beta"]
    assert beta["cash"] == 50_000.0 and beta["nav"] == 50_000.0


def test_snapshots_persist_nav_per_strategy(sampled):
    ex, db = sampled
    rows = {r["strategy_id"]: r for r in db.get_equity_history()}
    assert rows["alpha"]["nav"] == pytest.approx(101_000.0)
    assert rows["alpha"]["cash"] == pytest.approx(95_000.0)
    assert rows["alpha"]["equity"] == pytest.approx(1_000.0)   # P&L column unchanged
    assert sum(r["nav"] for r in rows.values()) == pytest.approx(151_000.0)


def test_circuit_breaker_sees_portfolio_nav(sampled):
    ex, _ = sampled
    assert ex.breaker_saw == pytest.approx(151_000.0)


def test_drawdown_check_still_gets_pnl_not_nav(sampled):
    """A 15% drawdown limit is a fraction of allocation — feeding it NAV would never trip."""
    ex, _ = sampled
    by_sid = {sid: pnl for sid, pnl, src in ex.drawdown_calls if src == "total"}
    assert by_sid["alpha"] == pytest.approx(1_000.0)
    assert by_sid["beta"] == pytest.approx(0.0)


def test_equity_endpoint_serves_the_cached_snapshot(sampled):
    body = server.get_equity()
    assert body["totals"]["nav"] == pytest.approx(151_000.0)
    assert body["marks"] == {"AAPL": 60.0}
    assert body["strategies"]["alpha"]["position_value"] == pytest.approx(6_000.0)
