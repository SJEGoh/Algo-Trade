"""Min-variance hedge sized by the overlay's capital.

Pinned down:
  * the hedge never spends more gross than the budget, and spends exactly the budget when
    the budget binds;
  * it never spends past the minimum-variance hedge — more capital would add variance;
  * more budget never removes less variance (diminishing returns, not reversals);
  * a tight budget goes to the ETF that covers the most variance (sparsity);
  * churn control keeps a held hedge unless re-solving is materially better, and an
    allocation cut below the held gross always resizes;
  * GET /exposure/minvar and `run_hedge.py --model minvar` respect the allocation.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from portfolio import minvar_hedge

PRICES = {"SPY": 500.0, "IWM": 200.0, "XLE": 90.0, "TLT": 95.0}


def market(n=300, seed=5):
    """A book that is mostly small-cap beta plus a little energy, and idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-06-02", periods=n)
    mkt, size, oil = (rng.normal(0, s, n) for s in (0.010, 0.006, 0.012))
    etf = pd.DataFrame({"SPY": mkt + rng.normal(0, 0.001, n),
                        "IWM": 1.1 * mkt + size + rng.normal(0, 0.001, n),
                        "XLE": 0.7 * mkt + oil + rng.normal(0, 0.001, n),
                        "TLT": rng.normal(0, 0.006, n)}, index=idx)
    pnl = pd.Series(300_000 * (1.2 * mkt + size) + 60_000 * oil
                    + rng.normal(0, 1_500, n), index=idx)
    return pnl, etf


def test_a_binding_budget_is_spent_exactly_and_never_exceeded():
    pnl, etf = market()
    for budget in (20_000, 80_000, 150_000):
        p = minvar_hedge.plan(pnl, etf, budget, 1e6, PRICES)
        assert p.binding
        assert p.gross <= budget
        assert p.gross > budget - max(PRICES.values()) * len(PRICES)   # only share rounding


def test_it_never_spends_past_the_minimum_variance_hedge():
    pnl, etf = market()
    full = minvar_hedge.plan(pnl, etf, 10_000_000, 1e6, PRICES)
    assert not full.binding and full.gross < 1_000_000
    assert full.variance_removed > 0.8                    # the rest is idiosyncratic noise


def test_more_budget_never_removes_less_variance():
    pnl, etf = market()
    removed = [minvar_hedge.plan(pnl, etf, b, 1e6, PRICES).variance_removed
               for b in (10_000, 50_000, 100_000, 200_000, 400_000)]
    assert removed == sorted(removed)
    assert removed[0] > 0


def test_a_tight_budget_goes_to_the_best_single_hedge():
    pnl, etf = market()
    p = minvar_hedge.plan(pnl, etf, 15_000, 1e6, PRICES)
    assert list(p.legs) == ["IWM"] and p.legs["IWM"] < 0   # small-cap beta: short IWM


def test_a_held_hedge_close_to_optimal_is_kept():
    pnl, etf = market()
    first = minvar_hedge.plan(pnl, etf, 80_000, 1e6, PRICES)
    nudged = {s: q - (2 if q > 0 else -2) for s, q in first.legs.items()}   # still fits
    again = minvar_hedge.plan(pnl, etf, 80_000, 1e6, PRICES, current_hedge=nudged)
    assert again.held_kept and again.legs == nudged


def test_an_allocation_cut_below_the_held_hedge_always_resizes():
    pnl, etf = market()
    held = minvar_hedge.plan(pnl, etf, 150_000, 1e6, PRICES).legs
    cut = minvar_hedge.plan(pnl, etf, 50_000, 1e6, PRICES, current_hedge=held)
    assert not cut.held_kept and cut.gross <= 50_000


def test_a_held_name_outside_the_universe_is_unwound():
    pnl, etf = market()
    p = minvar_hedge.plan(pnl, etf, 50_000, 1e6, {**PRICES, "SMH": 250.0},
                          current_hedge={"SMH": -40.0})
    targets = {i["instrument"]["symbol"]: i["target_quantity"] for i in p.book}
    assert targets["SMH"] == 0.0


def test_too_little_history_refuses():
    pnl, etf = market(n=60)
    with pytest.raises(ValueError):
        minvar_hedge.plan(pnl, etf, 50_000, 1e6, PRICES)


# ---------------------------------------------------------------------------
# endpoint + runner
# ---------------------------------------------------------------------------
@pytest.fixture
def server_with_book(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import api.server as server

    pnl, etf = market()
    hold = pd.Series(pnl.values / 300_000, index=pnl.index)        # one name, ~$300k
    series = {**{s: etf[s] for s in etf}, "BOOK": hold}

    def closes(sym):
        px = 100 * (1 + series[sym]).cumprod()
        return {d.strftime("%Y%m%d"): float(v) for d, v in px.items()}

    ex = SimpleNamespace(
        ledger=SimpleNamespace(strategy_positions={"orb": {"BOOK": 3_000.0}}, multipliers={}),
        fetch_daily_closes=closes, price_history=None, marks_from_close=set(),
        get_marks=lambda syms: {s: PRICES.get(s) for s in syms})
    monkeypatch.setattr(server, "executor", ex)
    monkeypatch.setattr(server, "DB_DIR", tmp_path)
    monkeypatch.setitem(server.CONFIG, "hedge_overlay",
                        {"capital_allocation": 100_000.0, "max_drawdown": 0.95})
    monkeypatch.setitem(server._last_equity, "marks", {"BOOK": 100.0, **PRICES})
    monkeypatch.setitem(server._last_equity, "totals", {"nav": 1_000_000.0})
    return TestClient(server.app)


def test_endpoint_spends_the_overlays_allocation_with_headroom(server_with_book):
    r = server_with_book.get("/exposure/minvar").json()
    assert r["budget"] == pytest.approx(95_000.0)
    assert r["gross"] <= 95_000.0 and r["variance_removed"] > 0
    assert r["book"]


def test_endpoint_refuses_a_budget_over_the_allocation(server_with_book):
    assert server_with_book.get("/exposure/minvar", params={"budget": 200_000}).status_code == 422


def test_run_hedge_minvar_submits_through_the_overlay(server_with_book, monkeypatch):
    import run_hedge
    submitted = []

    class FakeClient:
        def __init__(self, *a, **kw): pass
        def preflight(self): return {"market_open": True}
        def _request(self, method, path, params=None, **kw):
            return server_with_book.get(path, params=params).json()
        def submit_book(self, book, strategy_id=None):
            submitted.append(strategy_id); return {"order_ids": [1]}
        @staticmethod
        def order_ids(result): return [1]
        def wait_for_acks(self, ids, timeout=30.0):
            return {"live": [1], "rejected": [], "pending": []}
        def journal(self, *a, **kw): return {"logged": True}

    monkeypatch.setattr(run_hedge, "ExecutorClient", FakeClient)
    assert run_hedge.main(["--model", "minvar"]) == run_hedge.EXIT_OK
    assert submitted == ["hedge_overlay"]


def test_minvar_has_no_thresholds():
    import run_hedge
    with pytest.raises(SystemExit):
        run_hedge.main(["--model", "minvar", "--dry-run", "--trigger", "0.01"])
