"""PCA factor hedge — strategy-agnostic, report-only.

Pinned down:
  * factors are tradable ETF portfolios (unit gross), PC1 is net long, and signs stay put
    from one fit to the next;
  * shorting $H of factor k's portfolio moves only factor k's exposure, by exactly H — the
    property that lets the hedge shave one factor without disturbing the others;
  * the gate is the factor's contribution to the book's daily vol: a book under the trigger
    is left alone, one over it is shaved to the target, never to zero;
  * the hedge never measures itself, and a holding with no usable history is reported;
  * GET /exposure/pca and `run_hedge.py --model pca` are read-only.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from portfolio import pca_hedge
from portfolio.hedger import BucketPolicy

ETFS = ["SPY", "IWM", "QQQ", "XLE", "TLT"]


def synthetic(n=300, seed=7):
    """ETF returns driven by a market factor, a size factor and a rates factor, plus noise."""
    rng = np.random.default_rng(seed)
    mkt, size, rates = (rng.normal(0, s, n) for s in (0.010, 0.005, 0.006))
    load = {"SPY": (1.0, 0.0, 0.0), "IWM": (1.2, 1.0, 0.0), "QQQ": (1.2, -0.4, 0.0),
            "XLE": (0.8, 0.3, 0.0), "TLT": (-0.2, 0.0, 1.0)}
    idx = pd.bdate_range("2025-06-02", periods=n)
    etf = pd.DataFrame({s: a * mkt + b * size + c * rates + rng.normal(0, 0.002, n)
                        for s, (a, b, c) in load.items()}, index=idx)
    return etf, idx, rng, mkt


def test_factors_are_unit_gross_with_pc1_long_and_stable_signs():
    etf, *_ = synthetic()
    m = pca_hedge.fit_factors(etf, n_components=3, min_obs=100)
    assert np.allclose(np.abs(m.weights).sum(axis=1), 1.0)
    assert m.weights[0].sum() > 0                          # PC1 is the market, net long
    assert m.explained[0] > m.explained[1] > m.explained[2]

    flipped = pca_hedge.fit_factors(etf.iloc[5:], n_components=3, min_obs=100,
                                    prev_weights=-m.weights)
    assert all(np.dot(a, b) < 0 for a, b in zip(flipped.weights, m.weights))   # follows prev


def test_holding_a_factor_portfolio_loads_only_on_that_factor():
    etf, *_ = synthetic()
    m = pca_hedge.fit_factors(etf, n_components=3, min_obs=100)
    for k in range(3):
        port = pd.DataFrame({"P": etf[m.universe].values @ m.weights[k]}, index=etf.index)
        loads, _ = pca_hedge.loadings(port, m.returns)
        expected = np.eye(3)[k]
        assert np.allclose(loads["P"][0], expected, atol=1e-8)


def _book(beta_scale, n_names=10):
    """Fit factors, then build holdings with market beta ~`beta_scale`."""
    etf, idx, rng, mkt = synthetic()
    m = pca_hedge.fit_factors(etf, n_components=3, min_obs=100)
    names = [f"N{i}" for i in range(n_names)]
    hold = pd.DataFrame({s: beta_scale * mkt + rng.normal(0, 0.01, len(idx)) for s in names},
                        index=idx)
    loads, short = pca_hedge.loadings(hold, m.returns)
    return m, names, loads, short


PRICES = {"SPY": 500.0, "IWM": 200.0, "QQQ": 450.0, "XLE": 90.0, "TLT": 95.0}


def test_a_book_under_the_trigger_is_left_alone():
    m, names, loads, _ = _book(1.0)
    values = {s: 20_000.0 for s in names}                   # 200k long, NAV 1M
    p = pca_hedge.plan(m, values, loads, 1_000_000.0, PRICES)
    assert p.legs == {} and p.book == []
    assert p.decisions[0].vol_contribution < pca_hedge.DEFAULT_POLICY.trigger


def test_a_book_over_the_trigger_is_shaved_to_target_not_flat():
    m, names, loads, _ = _book(1.5)
    nav = 1_000_000.0
    values = {s: 90_000.0 for s in names}                   # 900k of beta-1.5 names
    p = pca_hedge.plan(m, values, loads, nav, PRICES)
    pc1 = p.decisions[0]
    assert pc1.vol_contribution > pca_hedge.DEFAULT_POLICY.trigger
    assert pc1.hedge_notional < 0                           # short the market factor
    after = abs(pc1.exposure + pc1.hedge_notional) * m.sigma[0] / nav
    assert after == pytest.approx(pca_hedge.DEFAULT_POLICY.target, rel=1e-9)
    assert p.legs and all(q < 0 for s, q in p.legs.items() if m.weights[0][m.universe.index(s)] > 0)
    assert p.decisions[2].hedge_notional == 0.0             # PC3 is report-only


def test_only_the_first_hedge_components_may_act():
    m, names, loads, _ = _book(1.5)
    values = {s: 90_000.0 for s in names}
    p = pca_hedge.plan(m, values, loads, 1_000_000.0, PRICES, hedge_components=0)
    assert p.legs == {} and all(d.hedge_notional == 0 for d in p.decisions)


def test_hysteresis_holds_a_hedge_until_release():
    m, names, loads, _ = _book(1.0)
    values = {s: 45_000.0 for s in names}
    nav = 1_000_000.0
    vol = pca_hedge.plan(m, values, loads, nav, PRICES).decisions[0].vol_contribution
    pol = BucketPolicy(trigger=vol * 1.1, target=vol * 0.8, release=vol * 0.7)
    cold = pca_hedge.plan(m, values, loads, nav, PRICES, policy=pol)
    warm = pca_hedge.plan(m, values, loads, nav, PRICES, policy=pol, current_hedge={"SPY": -10})
    assert cold.legs == {}                                  # under trigger, not hedged
    assert warm.legs                                        # over release, keep hedging


def test_the_overlay_is_excluded_and_gaps_are_reported():
    positions = {"orb": {"AAA": 10.0, "BBB": 5.0}, "hedge_overlay": {"SPY": -100.0}}
    values, unpriced = pca_hedge.position_values(positions, {"AAA": 50.0, "SPY": 500.0})
    assert values == {"AAA": 500.0} and unpriced == {"BBB": 5.0}

    m, names, loads, _ = _book(1.0)
    p = pca_hedge.plan(m, {"N0": 1_000.0, "NEW": 5_000.0}, loads, 1e6, PRICES)
    assert p.unmeasured == {"NEW": 5_000.0}
    assert p.coverage["unmeasured_symbols"] == ["NEW"]


def test_history_cache_fetches_once_a_day_and_falls_back(tmp_path):
    calls = []

    def fetch(sym):
        calls.append(sym)
        return None if sym == "BAD" else {"20260101": 10.0, "20260102": 11.0}

    c = pca_hedge.DailyHistoryCache(fetch, path=tmp_path / "h.json")
    out = c.get(["SPY", "BAD"])
    assert list(out) == ["SPY"] and out["SPY"].iloc[-1] == 11.0
    c.get(["SPY"])
    assert calls.count("SPY") == 1                          # cached for the day
    again = pca_hedge.DailyHistoryCache(fetch, path=tmp_path / "h.json")
    again.get(["SPY"])
    assert calls.count("SPY") == 1                          # and across a restart


def test_exposure_pca_endpoint_is_read_only(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import api.server as server

    etf, idx, rng, mkt = synthetic()
    hold = {"N0": 1.5 * mkt + rng.normal(0, 0.01, len(idx))}

    def closes_for(sym):
        r = etf[sym] if sym in etf else pd.Series(hold[sym], index=idx)
        px = 100 * (1 + r).cumprod()
        return {d.strftime("%Y%m%d"): float(v) for d, v in px.items()}

    placed = []
    ex = SimpleNamespace(
        ledger=SimpleNamespace(strategy_positions={"orb": {"N0": 9_000.0}}, multipliers={}),
        fetch_daily_closes=closes_for,
        get_marks=lambda syms: {s: PRICES.get(s, 100.0) for s in syms},
        place_order=lambda *a, **k: placed.append(a))
    monkeypatch.setattr(server, "executor", ex)
    monkeypatch.setattr(server, "DB_DIR", tmp_path)
    monkeypatch.setitem(server._pca_cache, "history", None)
    monkeypatch.setitem(server._pca_cache, "prev_weights", None)
    monkeypatch.setitem(server._last_equity, "marks", {"N0": 100.0})
    monkeypatch.setitem(server._last_equity, "totals", {"nav": 1_000_000.0})

    r = TestClient(server.app).get("/exposure/pca", params={"components": 3}).json()
    assert [f["factor"] for f in r["factors"]] == ["PC1", "PC2", "PC3"]
    assert r["exposures"][0]["hedge_notional"] < 0          # 900k of beta-1.5: over trigger
    assert r["book"] and placed == []


@pytest.mark.parametrize("argv", [["--model", "pca"], ["--model", "pca", "--any-time"]])
def test_run_hedge_pca_refuses_to_trade(argv):
    import run_hedge
    with pytest.raises(SystemExit):
        run_hedge.main(argv)


def test_run_hedge_pca_report_prints_and_submits_nothing(monkeypatch, tmp_path, caplog):
    """End to end through the real endpoint: the report renders and nothing is submitted."""
    import logging
    from fastapi.testclient import TestClient
    import api.server as server
    import run_hedge

    test_exposure_pca_endpoint_is_read_only(monkeypatch, tmp_path)   # wires the fake executor
    api = TestClient(server.app)

    class FakeClient:
        def __init__(self, *a, **kw): pass
        def preflight(self): return {"market_open": False}
        def _request(self, method, path, params=None, **kw):
            return api.get(path, params={**(params or {}), "components": 3}).json()
        def submit_book(self, *a, **kw): raise AssertionError("must not submit")

    monkeypatch.setattr(run_hedge, "ExecutorClient", FakeClient)
    with caplog.at_level(logging.INFO, logger="run-hedge"):
        assert run_hedge.main(["--model", "pca", "--dry-run", "--any-time"]) == run_hedge.EXIT_OK
    text = caplog.text
    assert "PC1" in text and "leg" in text and "NOT submitted" in text
