"""The hedge and rebalance events the day scheduler fires, and the runners behind them.

Two things are pinned here that a reading of the code would not make obvious:

  * WHEN they fire. The hedge is placed at the session MIDPOINT rather than a clock time, so
    a half-day still hedges half way through it. The rebalance is deliberately kept clear of
    the c-10/c-5/c-2 cluster, because applying one SELLS positions to raise cash and must not
    run while ovn_volsurge is entering.
  * That an unconfirmed hedge is a FAILURE. A hedge the broker never took, reported as
    success, is worse than no hedge — the book gets sized as though it were covered.
"""
from datetime import datetime, timedelta

import pandas as pd
import pytest
import pytz

import tools.day_scheduler as ds

ET = pytz.timezone("America/New_York")


def session(open_h=9, open_m=30, close_h=16, close_m=0, day=9):
    o = ET.localize(datetime(2026, 9, day, open_h, open_m))
    c = ET.localize(datetime(2026, 9, day, close_h, close_m))
    return o, c


def events_of(kind_label, o, c):
    return [(w, lbl) for w, lbl, _k, _p in ds.build_events(o, c) if kind_label in lbl]


# ------------------------------------------------------------------ when they fire
def test_the_hedge_runs_at_the_session_midpoint():
    o, c = session()
    (when, _), = events_of("portfolio hedge", o, c)
    assert when == o + (c - o) / 2
    assert when.strftime("%H:%M") == "12:45"


def test_a_half_day_hedges_half_way_through_it():
    """A fixed clock time would put the hedge 20 minutes before an early close, measuring
    exposure the day is about to stop carrying."""
    o, c = session(close_h=13, close_m=0)
    (when, _), = events_of("portfolio hedge", o, c)
    assert when == o + (c - o) / 2
    assert when.strftime("%H:%M") == "11:15"


def test_the_rebalance_runs_late_but_clear_of_the_closing_cluster():
    o, c = session()
    (when, _), = events_of("capital rebalance", o, c)

    assert when == c - timedelta(minutes=ds.REBALANCE_BEFORE_CLOSE_MIN)
    for label in ("ovn_volsurge enter", "ATR cancel", "rrg rotation"):
        for other, _ in events_of(label, o, c):
            assert when < other, f"rebalance collides with {label}"


def test_the_hedge_runs_before_the_rebalance():
    """Hedge the exposure the day is carrying; reallocate once the day is nearly done."""
    o, c = session()
    (hedge, _), = events_of("portfolio hedge", o, c)
    (rebal, _), = events_of("capital rebalance", o, c)
    assert hedge < rebal


def test_the_rebalance_is_scheduled_report_only():
    """--apply SELLS positions. It must be added deliberately, never inherited from a default."""
    o, c = session()
    cmds = [p for _w, lbl, _k, p in ds.build_events(o, c) if "capital rebalance" in lbl]
    assert cmds and all("--apply" not in (c or []) for c in cmds)


@pytest.mark.parametrize("flag,label", [("hedge", "portfolio hedge"),
                                        ("rebalance", "capital rebalance")])
def test_each_can_be_switched_off(monkeypatch, flag, label):
    monkeypatch.setitem(ds.ENABLE, flag, False)
    o, c = session()
    assert events_of(label, o, c) == []


# ------------------------------------------------------------------ the runners
def test_history_is_resampled_to_days_not_left_as_ticks():
    """The sampler writes ~1 row/minute, so a week is ~10,000 rows — a sample that looks
    enormous and is not. Feeding raw ticks would defeat the allocator's calendar-day guards."""
    from run_rebalance import returns_from_history

    rows = []
    for day in range(1, 4):
        for minute in range(0, 600, 5):        # 120 snapshots a day
            ts = datetime(2026, 9, day, 14, 0) + timedelta(minutes=minute)
            rows.append({"ts": ts.isoformat() + "+00:00", "strategy_id": "s1",
                         "nav": 100_000 + day * 100 + minute})
    out = returns_from_history(rows)

    assert len(out) <= 3, f"kept {len(out)} rows from 3 days of ticks"
    assert list(out.columns) == ["s1"]


def test_test_fixtures_are_not_given_real_capital():
    from run_rebalance import TEST_PREFIXES
    for sid in ("test_suite", "test_suite_small_alloc", "halt_test_1", "demo_momentum"):
        assert sid.startswith(TEST_PREFIXES)
    for sid in ("orb_breakout", "ovn_volsurge", "kalman_vecm"):
        assert not sid.startswith(TEST_PREFIXES)


def test_an_unconfirmed_hedge_exits_non_zero(monkeypatch):
    """The whole point of the runner. The executor accepted the legs; IB did not take them.
    Exiting 0 here would report cover that does not exist."""
    import run_hedge

    class FakeClient:
        def __init__(self, *a, **kw): pass
        def preflight(self): return {"market_open": True}
        def _request(self, method, path, **kw):
            return {"nav": 1_000_000.0, "exposures": [], "unhedgeable": {},
                    "coverage": {"unpriced_symbols": []},
                    "hedge": [{"bucket": "semiconductors", "symbol": "SMH",
                               "quantity": -800.0, "reason": "over trigger"}],
                    "book": [{"instrument": {"symbol": "SMH"}, "target_quantity": -800.0,
                              "expected_price": 250.0}]}
        def submit_book(self, book, strategy_id=None):
            return {"orders": [{"symbol": "SMH", "order_id": 7}]}
        @staticmethod
        def order_ids(result): return [7]
        def wait_for_acks(self, ids, timeout=30.0):
            return {"live": [], "rejected": [], "pending": [7], "acks": {}}
        def journal(self, *a, **kw): return {"logged": True}

    monkeypatch.setattr(run_hedge, "ExecutorClient", FakeClient)
    assert run_hedge.main([]) == run_hedge.EXIT_NOT_ACKED


def test_a_confirmed_hedge_exits_zero(monkeypatch):
    import run_hedge

    class FakeClient:
        def __init__(self, *a, **kw): pass
        def preflight(self): return {"market_open": True}
        def _request(self, method, path, **kw):
            return {"nav": 1_000_000.0, "exposures": [], "unhedgeable": {},
                    "coverage": {"unpriced_symbols": []},
                    "hedge": [], "book": [{"instrument": {"symbol": "SMH"},
                                           "target_quantity": -800.0,
                                           "expected_price": 250.0}]}
        def submit_book(self, book, strategy_id=None):
            return {"orders": [{"symbol": "SMH", "order_id": 7}]}
        @staticmethod
        def order_ids(result): return [7]
        def wait_for_acks(self, ids, timeout=30.0):
            return {"live": [7], "rejected": [], "pending": [], "acks": {}}
        def journal(self, *a, **kw): return {"logged": True}

    monkeypatch.setattr(run_hedge, "ExecutorClient", FakeClient)
    assert run_hedge.main([]) == run_hedge.EXIT_OK


def test_a_closed_market_is_a_clean_skip(monkeypatch):
    import run_hedge

    class FakeClient:
        def __init__(self, *a, **kw): pass
        def preflight(self): return {"market_open": False}
        def _request(self, *a, **kw):
            raise AssertionError("asked for exposure with the market closed")

    monkeypatch.setattr(run_hedge, "ExecutorClient", FakeClient)
    assert run_hedge.main([]) == run_hedge.EXIT_OK


def test_a_dry_run_submits_nothing(monkeypatch):
    import run_hedge

    class FakeClient:
        def __init__(self, *a, **kw): pass
        def preflight(self): return {"market_open": True}
        def _request(self, method, path, **kw):
            return {"nav": 1_000_000.0, "exposures": [], "unhedgeable": {},
                    "coverage": {"unpriced_symbols": []}, "hedge": [],
                    "book": [{"instrument": {"symbol": "SMH"}, "target_quantity": -800.0,
                              "expected_price": 250.0}]}
        def submit_book(self, *a, **kw):
            raise AssertionError("submitted during a dry run")

    monkeypatch.setattr(run_hedge, "ExecutorClient", FakeClient)
    assert run_hedge.main(["--dry-run"]) == run_hedge.EXIT_OK


# ------------------------------------------------------------------ shakeout settings
def _cmd_for(label, o, c):
    return [p for _w, lbl, _k, p in ds.build_events(o, c) if label in lbl][0]


def test_the_hedge_thresholds_are_passed_explicitly_not_left_to_defaults():
    """The production default of 30% would never fire against a book carrying ~5% per
    bucket. The low values are a deliberate shakeout, so they belong at the call site where
    they are visible — reverting is deleting arguments."""
    cmd = _cmd_for("portfolio hedge", *session())
    assert "--trigger" in cmd
    trigger = float(cmd[cmd.index("--trigger") + 1])
    target = float(cmd[cmd.index("--target") + 1])
    release = float(cmd[cmd.index("--release") + 1])

    assert trigger < 0.10, "trigger is too high to fire on a young book"
    assert release <= target <= trigger


def test_the_rebalance_resamples_intraday_with_a_matching_annualiser():
    """6 hours cannot make a daily return series — one bar, zero returns. Intraday bars fix
    that, but leaving the annualiser at 252 would understate the covariance ~26x."""
    cmd = _cmd_for("capital rebalance", *session())
    freq = cmd[cmd.index("--freq") + 1]
    ppy = float(cmd[cmd.index("--periods-per-year") + 1])

    assert freq == "15min"
    assert ppy == pytest.approx(26 * 252, rel=0.05), "annualiser does not match the frequency"


def test_the_shakeout_rebalance_never_carries_apply():
    """These settings defeat the guards on purpose. Moving money on them would be the exact
    failure the guards exist to prevent."""
    cmd = _cmd_for("capital rebalance", *session())
    assert "--apply" not in cmd


# ------------------------------------------------------------------ plumbing strategies
def test_the_test_strategies_fire_repeatedly_through_the_session():
    """They run on 30-minute bars and exist to trip a 1% drawdown halt. Once a day would
    re-evaluate a signal that changed 13 times, and would take a week to reach the limit."""
    o, c = session()
    fired = sorted(w for w, lbl in events_of("plumbing test", o, c))

    assert len(fired) > 10, f"only {len(fired)} runs scheduled"
    assert fired[0] < o + timedelta(hours=2), "first run is too late to leave time to halt"
    assert fired[-1] <= c - timedelta(minutes=ds.TEST_STRATS_STOP_BEFORE_CLOSE_MIN)
    assert len(set(fired)) == len(fired), "two runs scheduled at the same instant"


def test_both_test_strategies_fire_on_every_cycle():
    o, c = session()
    per_strategy = {}
    for when, lbl in events_of("plumbing test", o, c):
        per_strategy.setdefault(lbl.split(" ")[0], []).append(when)
    assert set(per_strategy) == {"halt_test_macd", "halt_test_bollinger"}
    a, b = (len(v) for v in per_strategy.values())
    assert abs(a - b) <= 1, f"cadences drifted apart: {a} vs {b}"


def test_they_submit_an_authoritative_book():
    """`resync` closes names the signal has dropped. Without it a flat signal would leave
    yesterday's positions in place and the halt would never be reached."""
    o, c = session()
    cmds = [p for _w, lbl, _k, p in ds.build_events(o, c) if "plumbing test" in lbl]
    assert cmds and all(cmd[-1] == "resync" for cmd in cmds)
    assert {cmd[-2] for cmd in cmds} == {"halt_test_macd", "halt_test_bollinger"}


def test_they_can_be_switched_off(monkeypatch):
    monkeypatch.setitem(ds.ENABLE, "test_strats", False)
    assert events_of("plumbing test", *session()) == []


def test_their_drawdown_limits_are_tight_enough_to_actually_halt():
    from config import CONFIG
    for sid in ("halt_test_macd", "halt_test_bollinger"):
        cfg = CONFIG[sid]
        assert cfg["max_drawdown"] <= 0.01
        # $100 of loss on a $10k allocation — reachable in a day by a $5k book
        assert cfg["capital_allocation"] * cfg["max_drawdown"] <= 100.0


def test_they_are_excluded_from_capital_reallocation():
    """The `halt_test` prefix is load-bearing: the rebalancer must never hand real capital
    to a disposable fixture."""
    from run_rebalance import TEST_PREFIXES
    for sid in ("halt_test_macd", "halt_test_bollinger"):
        assert sid.startswith(TEST_PREFIXES)


def test_the_book_fits_inside_the_allocation_cap():
    """5 names x $1,000 against a $10,000 cap. If the book could exceed the cap the orders
    would be rejected by the allocation check before reaching the halt logic being tested."""
    from models.test_strategies import TEST_UNIVERSE, _IntradayBarStrategy
    from config import CONFIG
    worst_case = len(TEST_UNIVERSE) * _IntradayBarStrategy("x").lot_dollars
    assert worst_case <= CONFIG["halt_test_macd"]["capital_allocation"]
