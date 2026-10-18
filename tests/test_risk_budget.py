"""Mark fetching, whole-book risk, and the risk-budget scale.

Pinned down:
  * get_marks never has more than mark_max_concurrent snapshots in flight — past IB's line
    limit the excess came back empty, a different handful each cycle;
  * a symbol with no live mark falls back to its last cached daily close, and that fallback
    is never FRESH, so drawdown checks and exits do not act on it;
  * book risk is today's positions replayed over real returns, unmeasured names listed;
  * the scale recommendation is clamped and stepped, and GET /risk/book changes nothing.
"""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from execution.central_execution import CentralExecutor
from portfolio import risk_budget
from portfolio.pca_hedge import DailyHistoryCache


# ---------------------------------------------------------------------------
# get_marks
# ---------------------------------------------------------------------------
def marker(prices, cap=5, history=None, delay=0.02):
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def fetch_price(sym, timeout=3.0, contract=None):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(delay)
        with lock:
            state["now"] -= 1
        return prices.get(sym)

    ex = SimpleNamespace(
        _mark_contract=lambda s: None, fetch_price=fetch_price,
        _mark_lock=threading.Lock(), _mark_cache={}, _mark_ts={},
        _mark_gate=threading.BoundedSemaphore(cap), price_history=history)
    ex.get_marks = lambda syms, timeout=3.0: CentralExecutor.get_marks(ex, syms, timeout)
    ex.mark_is_fresh = lambda s, max_age=120.0: CentralExecutor.mark_is_fresh(ex, s, max_age)
    return ex, state


def test_snapshots_in_flight_never_exceed_the_cap():
    syms = [f"S{i}" for i in range(60)]
    ex, state = marker({s: 10.0 for s in syms}, cap=5)
    out = ex.get_marks(syms)
    assert all(out[s] == 10.0 for s in syms)          # all priced, despite the queueing
    assert state["peak"] <= 5


def test_no_live_mark_falls_back_to_last_close_and_is_never_fresh(tmp_path):
    fetched = []
    history = DailyHistoryCache(lambda s: fetched.append(s) or {"20261015": 42.0,
                                                                "20261016": 43.5},
                                path=tmp_path / "h.json")
    history.get(["OLD"])                                  # cached already
    ex, _ = marker({"LIVE": 10.0}, history=history)

    out = ex.get_marks(["LIVE", "OLD", "NEW"])
    assert out["LIVE"] == 10.0 and ex.mark_is_fresh("LIVE")
    assert out["OLD"] == 43.5 and not ex.mark_is_fresh("OLD")   # valued, but stale
    assert "OLD" not in ex._mark_cache                    # never becomes a carry-forward
    assert ex.marks_from_close == {"OLD"}
    assert out["NEW"] is None                             # no close yet...
    for _ in range(100):                                  # ...but one is being fetched
        if "NEW" in fetched:
            break
        time.sleep(0.01)
    assert "NEW" in fetched


def test_a_live_mark_still_wins_over_the_close(tmp_path):
    history = DailyHistoryCache(lambda s: {"20261016": 1.0}, path=tmp_path / "h.json")
    history.get(["A"])
    ex, _ = marker({"A": 99.0}, history=history)
    assert ex.get_marks(["A"])["A"] == 99.0 and ex.marks_from_close == set()


# ---------------------------------------------------------------------------
# book risk + scale
# ---------------------------------------------------------------------------
def returns(n=250, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-10-01", periods=n)
    return pd.DataFrame({"A": rng.normal(0, 0.02, n), "B": rng.normal(0, 0.02, n)}, index=idx)


def test_book_risk_is_the_replayed_pnl():
    r = returns()
    risk = risk_budget.book_risk({"A": 100_000.0, "B": -50_000.0, "NEW": 5_000.0}, r, 1e6)
    pnl = r["A"] * 100_000 - r["B"] * 50_000
    assert risk["daily_vol"] == pytest.approx(pnl.std(ddof=1) / 1e6)
    assert risk["annual_vol"] == pytest.approx(risk["daily_vol"] * np.sqrt(252))
    assert risk["var99"] == pytest.approx(-np.percentile(pnl, 1) / 1e6)
    assert risk["unmeasured_symbols"] == ["NEW"] and risk["days"] == 250


def test_offsetting_positions_show_less_risk_than_either_alone():
    r = returns()
    r["B"] = r["A"] + np.random.default_rng(1).normal(0, 0.002, len(r))   # B ~ A
    hedged = risk_budget.book_risk({"A": 100_000.0, "B": -100_000.0}, r, 1e6)
    naked = risk_budget.book_risk({"A": 100_000.0}, r, 1e6)
    assert hedged["daily_vol"] < naked["daily_vol"] / 5


@pytest.mark.parametrize("vol, step, capped", [
    (0.04, 1.25, 2.0),      # 2.5x wanted: capped at 2x, stepped to 1.25x
    (0.09, 1.11, 1.11),     # small move: taken in one step
    (0.10, 1.00, 1.00),     # on target: hold
    (0.40, 0.75, 0.5),      # far over: capped at 0.5x, stepped to 0.75x
])
def test_scale_is_capped_then_stepped(vol, step, capped):
    sc = risk_budget.recommend_scale(vol, 0.10)
    assert sc["step"] == pytest.approx(step, abs=0.01)
    assert sc["capped"] == pytest.approx(capped, abs=0.01)


def test_unmeasurable_book_holds():
    assert risk_budget.recommend_scale(None, 0.10)["step"] == 1.0


# ---------------------------------------------------------------------------
# GET /risk/book
# ---------------------------------------------------------------------------
def test_risk_book_endpoint_is_read_only(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import api.server as server

    r = returns()
    closes = {s: 100 * (1 + r[s]).cumprod() for s in r}
    positions = {"orb": {"A": 1_000.0}, "hedge_overlay": {"B": -200.0}}
    ex = SimpleNamespace(
        ledger=SimpleNamespace(strategy_positions=positions, multipliers={}),
        fetch_daily_closes=lambda s: {d.strftime("%Y%m%d"): float(v)
                                      for d, v in closes[s].items()},
        price_history=None, marks_from_close={"B"})
    monkeypatch.setattr(server, "executor", ex)
    monkeypatch.setattr(server, "DB_DIR", tmp_path)
    monkeypatch.setitem(server._last_equity, "marks", {"A": 100.0, "B": 100.0})
    monkeypatch.setitem(server._last_equity, "totals", {"nav": 1_000_000.0})

    out = TestClient(server.app).get("/risk/book", params={"target_vol": 0.10}).json()
    assert out["measured_notional"] == pytest.approx(120_000.0)   # overlay counts: real risk
    assert out["priced_from_close"] == ["B"]
    assert out["scale"]["step"] > 1.0                             # tiny book: raise
    assert positions == {"orb": {"A": 1_000.0}, "hedge_overlay": {"B": -200.0}}


def test_rebalance_survives_the_risk_budget_failing(monkeypatch, caplog):
    import logging
    import run_rebalance

    class Down:
        def __init__(self, *a, **kw): pass
        def _request(self, *a, **kw): raise RuntimeError("executor busy")

    monkeypatch.setattr(run_rebalance, "ExecutorClient", Down)
    args = SimpleNamespace(target_vol=0.1, scale_min=0.5, scale_max=2.0, max_step=0.25)
    with caplog.at_level(logging.WARNING, logger="run-rebalance"):
        assert run_rebalance.report_risk_budget(args, []) is None
    assert "risk budget unavailable" in caplog.text
