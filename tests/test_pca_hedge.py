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
    monkeypatch.setitem(server._pca_cache, "prev_weights", None)
    monkeypatch.setitem(server._last_equity, "marks", {"N0": 100.0})
    monkeypatch.setitem(server._last_equity, "totals", {"nav": 1_000_000.0})

    r = TestClient(server.app).get("/exposure/pca", params={"components": 3}).json()
    assert [f["factor"] for f in r["factors"]] == ["PC1", "PC2", "PC3"]
    assert r["exposures"][0]["hedge_notional"] < 0          # 900k of beta-1.5: over trigger
    assert r["book"] and placed == []


@pytest.mark.parametrize("argv", [["--model", "pca", "--any-time"],
                                  ["--model", "pca", "--exclude", "IWM"]])
def test_run_hedge_pca_refuses_to_trade_a_what_if(argv):
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


def test_exclude_is_a_what_if_on_the_measured_book(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import api.server as server

    test_exposure_pca_endpoint_is_read_only(monkeypatch, tmp_path)   # fake executor: N0 only
    server.executor.ledger.strategy_positions["pairs"] = {"IWM": -2_000.0}
    server._last_equity["marks"]["IWM"] = 200.0            # held, so the sampler marks it
    api = TestClient(server.app)

    full = api.get("/exposure/pca").json()
    without = api.get("/exposure/pca", params={"exclude": ["iwm"]}).json()
    no_pairs = api.get("/exposure/pca", params={"exclude_strategy": ["pairs"]}).json()

    assert "IWM" in full["exposures"][0]["top_contributors"]
    assert "IWM" not in without["exposures"][0]["top_contributors"]
    assert without["exposures"][0]["exposure"] > full["exposures"][0]["exposure"]   # short gone
    assert no_pairs["exposures"][0]["exposure"] == pytest.approx(without["exposures"][0]["exposure"])
    assert without["excluded"] == {"symbols": ["IWM"], "strategies": []}
    assert server.executor.ledger.strategy_positions["pairs"] == {"IWM": -2_000.0}  # untouched


def test_exclude_needs_the_pca_model():
    import run_hedge
    with pytest.raises(SystemExit):
        run_hedge.main(["--dry-run", "--exclude", "IWM"])


# ---------------------------------------------------------------------------
# No-trade band
# ---------------------------------------------------------------------------
def test_a_held_factor_hedge_survives_weight_drift_untraded():
    """Refit on a shifted window (weights drift a little), feed back the legs from the first
    plan as the held hedge: nothing should trade."""
    etf, idx, rng, mkt = synthetic()
    m1 = pca_hedge.fit_factors(etf.iloc[:280], n_components=3, min_obs=100)
    m2 = pca_hedge.fit_factors(etf.iloc[20:], n_components=3, min_obs=100,
                               prev_weights=m1.weights)
    assert not np.allclose(m1.weights, m2.weights)            # it really did drift
    names = [f"N{i}" for i in range(10)]
    hold = pd.DataFrame({s: 1.5 * mkt + rng.normal(0, 0.01, len(idx)) for s in names}, index=idx)
    values = {s: 90_000.0 for s in names}
    first = pca_hedge.plan(m1, values, pca_hedge.loadings(hold, m1.returns)[0], 1e6, PRICES)
    assert first.legs
    again = pca_hedge.plan(m2, values, pca_hedge.loadings(hold, m2.returns)[0], 1e6, PRICES,
                           current_hedge=first.legs)
    assert again.legs == first.legs
    assert "band" in again.decisions[0].reason


def test_a_factor_hedge_below_target_unwinds_never_grows():
    m, names, loads, _ = _book(1.0)
    nav = 1e6
    big = {s: 90_000.0 for s in names}
    hedged = pca_hedge.plan(m, big, loads, nav, PRICES,
                            policy=BucketPolicy(trigger=0.004, target=0.003, release=0.001))
    assert hedged.legs
    # the book shrinks to between release and target: the hedge must go, not grow
    small = {s: v / 3.5 for s, v in big.items()}
    after = pca_hedge.plan(m, small, loads, nav, PRICES, current_hedge=hedged.legs,
                           policy=BucketPolicy(trigger=0.004, target=0.003, release=0.001))
    pc1 = after.decisions[0]
    assert 0.001 < pc1.vol_contribution < 0.003
    assert pc1.hedge_notional == 0.0
    assert all(abs(after.legs.get(s, 0)) <= abs(q) for s, q in hedged.legs.items())


# ---------------------------------------------------------------------------
# Live PCA path
# ---------------------------------------------------------------------------
def _pca_view(unmeasured=0.0):
    leg = {"instrument": {"symbol": "SPY", "asset_class": "equity", "sec_type": "STK",
                          "exchange": "SMART"}, "target_quantity": -40.0, "expected_price": 500.0}
    return {"nav": 1e6, "history_days": 250, "pc1_explains_60pct": False,
            "policy": {"trigger": 0.006, "target": 0.005, "release": 0.004},
            "factors": [{"factor": "PC1", "explained": 0.4, "daily_vol": 0.009,
                         "weights": {"SPY": 0.5}}],
            "exposures": [{"factor": "PC1", "exposure": 9e5, "fraction": 0.9,
                           "vol_contribution": 0.008, "hedge_notional": -2e4, "held": 0.0,
                           "reason": "over trigger", "top_contributors": {}}],
            "legs": {"SPY": -40.0}, "book": [leg], "book_risk": {},
            "coverage": {"measured_notional": 9e5, "unmeasured_notional": unmeasured,
                         "unmeasured_symbols": ["X"] if unmeasured else [],
                         "unpriced_symbols": [], "factor_r2": 0.4}}


def _client(view, submitted, acks=None):
    class FakeClient:
        def __init__(self, *a, **kw): pass
        def preflight(self): return {"market_open": True}
        def _request(self, method, path, **kw): return view
        def submit_book(self, book, strategy_id=None):
            submitted.append((strategy_id, book)); return {"order_ids": [7]}
        @staticmethod
        def order_ids(result): return [7]
        def wait_for_acks(self, ids, timeout=30.0):
            return acks or {"live": [7], "rejected": [], "pending": []}
        def journal(self, *a, **kw): return {"logged": True}
    return FakeClient


def test_live_pca_submits_the_book_to_the_overlay(monkeypatch):
    import run_hedge
    submitted = []
    monkeypatch.setattr(run_hedge, "ExecutorClient", _client(_pca_view(), submitted))
    assert run_hedge.main(["--model", "pca"]) == run_hedge.EXIT_OK
    assert submitted == [("hedge_overlay", _pca_view()["book"])]


def test_live_pca_unconfirmed_at_the_broker_exits_non_zero(monkeypatch):
    import run_hedge
    monkeypatch.setattr(run_hedge, "ExecutorClient", _client(
        _pca_view(), [], acks={"live": [], "rejected": [], "pending": [7]}))
    assert run_hedge.main(["--model", "pca"]) == run_hedge.EXIT_NOT_ACKED


def test_live_pca_refuses_when_too_much_of_the_book_is_unmeasured(monkeypatch):
    import run_hedge
    submitted = []
    monkeypatch.setattr(run_hedge, "ExecutorClient",
                        _client(_pca_view(unmeasured=6e5), submitted))   # 40% unmeasured
    assert run_hedge.main(["--model", "pca"]) == run_hedge.EXIT_REFUSED
    assert submitted == []


def test_the_scheduler_runs_one_hedge_model_and_warms_pca(monkeypatch):
    import importlib
    from datetime import datetime
    import tools.day_scheduler as ds

    def hedge_events(model):
        monkeypatch.setenv("HEDGE_MODEL", model)
        importlib.reload(ds)
        o, c = datetime(2026, 10, 19, 9, 30), datetime(2026, 10, 19, 16, 0)
        return [e for e in ds.build_events(o, c) if "run_hedge.py" in " ".join(e[3] or [])]

    buckets = hedge_events("buckets")
    assert len(buckets) == 1 and "pca" not in buckets[0][3]
    pca = hedge_events("pca")
    assert len(pca) == 2
    warm, live = sorted(pca, key=lambda e: e[0])
    assert "--dry-run" in warm[3] and "--dry-run" not in live[3]
    assert (live[0] - warm[0]).total_seconds() == 3600
