"""Annualised Sharpe on the dashboard's strategy cards (GET /strategies).

Daily NAV returns, weekdays only, risk-free 0, cached; under 10 trading days it is None —
a ratio from a week of data is noise, and a card showing one invites trusting it.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.server as server


def history(days, seed=2, minute_noise=True):
    """Equity snapshots several times a day, every calendar day, for one strategy."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-08-03", periods=days, freq="D", tz="UTC")
    rets = pd.Series(rng.normal(0.001, 0.01, days), index=idx)
    rets[rets.index.dayofweek >= 5] = 0.0                 # markets shut: NAV does not move
    nav = 100_000 * (1 + rets).cumprod()
    rows = []
    for ts, v in nav.items():
        for h in (14, 18, 21):                            # intraday samples, last one wins
            jitter = rng.normal(0, 50) if minute_noise and h != 21 else 0.0
            rows.append({"ts": (ts + pd.Timedelta(hours=h)).isoformat(),
                         "strategy_id": "alpha", "nav": float(v + jitter)})
    return rows, nav


@pytest.fixture
def client(monkeypatch):
    def use(rows, active=True):
        db = SimpleNamespace(get_equity_history=lambda **kw: rows)
        ex = SimpleNamespace(logger_db=db,
                             risk_manager=SimpleNamespace(is_active=lambda sid: active))
        monkeypatch.setattr(server, "executor", ex)
        monkeypatch.setattr(server, "CONFIG", {"alpha": {"capital_allocation": 100_000.0,
                                                         "max_drawdown": 0.15}})
        monkeypatch.setitem(server._sharpe_cache, "at", None)
        monkeypatch.setitem(server._sharpe_cache, "by_strategy", {})
        return TestClient(server.app)
    return use


def test_sharpe_is_annualised_daily_weekday_returns(client):
    rows, nav = history(60)
    card = client(rows).get("/strategies").json()["strategies"][0]
    daily = nav.pct_change().dropna()
    daily = daily[daily.index.dayofweek < 5]
    expected = daily.mean() / daily.std(ddof=1) * np.sqrt(252)
    assert card["sharpe"] == pytest.approx(expected, rel=1e-6)
    assert card["sharpe_days"] == len(daily)


def test_too_little_history_shows_no_ratio(client):
    rows, _ = history(9)
    card = client(rows).get("/strategies").json()["strategies"][0]
    assert card["sharpe"] is None and 0 < card["sharpe_days"] < 10


def test_it_is_cached_between_polls(client, monkeypatch):
    rows, _ = history(60)
    api = client(rows)
    first = api.get("/strategies").json()["strategies"][0]["sharpe"]
    calls = []
    server.executor.logger_db.get_equity_history = lambda **kw: calls.append(1) or []
    assert api.get("/strategies").json()["strategies"][0]["sharpe"] == first
    assert calls == []


def test_a_broken_history_never_breaks_the_endpoint(client):
    api = client([])
    def boom(**kw): raise RuntimeError("database is locked")
    server.executor.logger_db.get_equity_history = boom
    card = api.get("/strategies").json()["strategies"][0]
    assert card["sharpe"] is None and card["capital_allocation"] == 100_000.0


def spy_closes(nav, seed=7):
    """Independent random SPY closes on the weekdays of `nav`."""
    rng = np.random.default_rng(seed)
    days = nav.index.tz_localize(None).normalize()
    weekdays = days[days.dayofweek < 5]
    rets = pd.Series(rng.normal(0.0005, 0.012, len(weekdays)), index=weekdays)
    return 400 * (1 + rets).cumprod()


def use_spy(monkeypatch, closes):
    cache = SimpleNamespace(get=lambda syms: {"SPY": closes} if closes is not None else {})
    monkeypatch.setattr(server, "_price_history", lambda: cache)


def test_beta_is_ols_of_weekday_returns_on_spy(client, monkeypatch):
    rows, nav = history(60)
    spy = spy_closes(nav)
    use_spy(monkeypatch, spy)
    card = client(rows).get("/strategies").json()["strategies"][0]
    daily = nav.pct_change().dropna()
    daily = daily[daily.index.dayofweek < 5]
    daily.index = daily.index.tz_localize(None).normalize()
    both = pd.concat([daily, spy.pct_change().dropna()], axis=1, join="inner").dropna()
    expected = both.iloc[:, 0].cov(both.iloc[:, 1]) / both.iloc[:, 1].var()
    assert card["beta"] == pytest.approx(expected, rel=1e-6)
    assert card["beta_days"] == len(both)


def test_beta_recovers_a_known_exposure(client, monkeypatch):
    idx = pd.bdate_range("2026-08-03", periods=80)
    rng = np.random.default_rng(3)
    spy_r = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    strat_r = 0.6 * spy_r + rng.normal(0, 0.0005, len(idx))
    nav = 100_000 * (1 + strat_r).cumprod()
    rows = [{"ts": (ts.tz_localize("UTC") + pd.Timedelta(hours=21)).isoformat(),
             "strategy_id": "alpha", "nav": float(v)} for ts, v in nav.items()]
    use_spy(monkeypatch, 400 * (1 + spy_r).cumprod())
    card = client(rows).get("/strategies").json()["strategies"][0]
    assert card["beta"] == pytest.approx(0.6, abs=0.02)


def test_no_spy_history_leaves_sharpe_alone(client, monkeypatch):
    rows, _ = history(60)
    use_spy(monkeypatch, None)
    card = client(rows).get("/strategies").json()["strategies"][0]
    assert card["beta"] is None and card["beta_days"] == 0
    assert card["sharpe"] is not None
