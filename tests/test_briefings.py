"""The Telegram briefings, and the ways they reported confidently wrong things.

Three real defects are pinned here. Each was silent — the briefing arrived every day looking
plausible, which is worse than not arriving at all:

  * the post-market fill list filtered on `timestamp`, a key the fill record does not have
    (it is `filled_at`), so it reported "(no fills today)" every day regardless of trading;
  * timestamps are stored in UTC and the briefings are headed in ET, so a 19:58 fill was
    printed next to a session that closed at 16:00;
  * the strategy and P&L sections skipped `test_suite*` and `halt_test*` but not `demo_*`
    or the ledger's internal buckets, so `__net__` was listed as a strategy and a $600k
    demo allocation was reported as capital at work.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tools"))

from briefing_common import (ET, HEDGE_SID, et, is_strategy,  # noqa: E402
                             is_today_et, money, pnl_sections)


# ------------------------------------------------------------------ timestamps
def test_utc_fill_times_are_shown_in_et():
    """A 19:58 UTC fill is 15:58 ET — two minutes before the close, not four hours after."""
    assert et("2026-09-09T19:58:07.138231+00:00") == "15:58:07"


def test_a_missing_timestamp_does_not_crash_the_briefing():
    assert et(None) == "--:--:--"
    assert et("") == "--:--:--"


def test_todays_fills_are_selected_on_the_ET_date():
    now = datetime(2026, 9, 9, 16, 10, tzinfo=ET)
    assert is_today_et("2026-09-09T19:58:07+00:00", now)      # 15:58 ET, today
    assert is_today_et("2026-09-09T13:31:00+00:00", now)      # 09:31 ET, today


def test_an_evening_utc_fill_is_still_todays_session():
    """20:30 UTC is 16:30 ET — same session. A naive UTC-date comparison keeps this one,
    but the pair below is where it goes wrong."""
    now = datetime(2026, 9, 9, 16, 40, tzinfo=ET)
    assert is_today_et("2026-09-09T20:30:00+00:00", now)


def test_an_early_utc_morning_belongs_to_the_previous_session():
    """01:00 UTC on the 10th is 21:00 ET on the 9th — yesterday's session, even though the
    UTC date says otherwise."""
    now = datetime(2026, 9, 10, 8, 0, tzinfo=ET)
    assert not is_today_et("2026-09-10T01:00:00+00:00", now)


def test_the_old_field_name_would_have_matched_nothing():
    """The regression itself: fill records carry `filled_at`, never `timestamp`, so the old
    `f.get("timestamp","").startswith(today)` was False for every fill ever recorded."""
    fill = {"symbol": "AAPL", "filled_at": "2026-09-09T19:58:07+00:00", "price": 314.64}
    assert fill.get("timestamp") is None
    assert is_today_et(fill.get("filled_at"), datetime(2026, 9, 9, 16, 10, tzinfo=ET))


# ------------------------------------------------------------------ who counts
@pytest.mark.parametrize("sid", ["cross_sectional_momentum", "orb_breakout",
                                 "ovn_volsurge", "kalman_vecm", "kalman_rrg_combined"])
def test_real_strategies_are_reported(sid):
    assert is_strategy(sid)


@pytest.mark.parametrize("sid", ["__net__", "flatten_all", "kill_switch"])
def test_ledger_buckets_are_not_strategies(sid):
    """They carry real P&L and real positions, which is why they slipped through — but a
    reader shown `__net__: -$153.80` in a strategy list looks for a signal behind it."""
    assert not is_strategy(sid)


@pytest.mark.parametrize("sid", ["demo_momentum", "demo_meanrev", "test_suite",
                                 "test_suite_small_alloc", "halt_test_1", "halt_test_macd"])
def test_fixtures_are_not_strategies(sid):
    assert not is_strategy(sid)


def test_the_hedge_overlay_is_not_a_strategy():
    assert not is_strategy(HEDGE_SID)


# ------------------------------------------------------------------ P&L split
def test_pnl_is_split_three_ways():
    strat, fixture, internal = pnl_sections({
        "cross_sectional_momentum": -8291.50,
        "orb_breakout": -0.70,
        "halt_test_macd": 2.57,
        "demo_momentum": 500.0,
        "__net__": -153.80,
        "flatten_all": 0.0,
        "ovn_volsurge": 0.0,
    })
    assert [s for s, _v in strat] == ["cross_sectional_momentum", "orb_breakout"]
    assert [s for s, _v in fixture] == ["demo_momentum", "halt_test_macd"]
    assert [s for s, _v in internal] == ["__net__"]


def test_zero_pnl_is_dropped_everywhere():
    strat, fixture, internal = pnl_sections({"a": 0.0, "__net__": 0.0, "demo_x": 0.0})
    assert (strat, fixture, internal) == ([], [], [])


def test_the_strategy_total_excludes_buckets_and_fixtures():
    """The old total summed everything that was not a test fixture, so `__net__` was folded
    into a figure labelled as strategy performance."""
    strat, _fx, internal = pnl_sections({"orb_breakout": -100.0, "__net__": -153.80,
                                         "demo_momentum": 999.0})
    assert sum(v for _s, v in strat) == pytest.approx(-100.0)
    assert sum(v for _s, v in internal) == pytest.approx(-153.80)


def test_money_always_carries_a_sign():
    assert money(-8358.6) == "$-8,358.60"
    assert money(13.14) == "$+13.14"


# ------------------------------------------------------------------ schedule is derived
def test_the_schedule_comes_from_the_scheduler_not_a_hardcoded_list():
    """The briefing used to hand-write the day's times, so every scheduler change made it
    quietly wrong — by the time the hedge, the rebalance and the plumbing strategies existed
    it was describing a day that no longer happened."""
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_ROOT / "src"))
    from premarket_briefing import _schedule_lines

    o = datetime(2026, 9, 10, 9, 30, tzinfo=ET)
    c = datetime(2026, 9, 10, 16, 0, tzinfo=ET)
    text = "\n".join(_schedule_lines(o, c))

    for expected in ("portfolio hedge", "capital rebalance", "halt_test_macd",
                     "orb_breakout resync", "ovn_volsurge enter"):
        assert expected in text, f"{expected} missing from the briefing schedule"


def test_repeating_events_are_collapsed_not_listed_one_by_one():
    """22 plumbing runs plus 12 ORB resyncs plus 7 reconciles is a 45-line briefing nobody
    reads. Anything that repeats becomes a range and a cadence."""
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_ROOT / "src"))
    from premarket_briefing import _schedule_lines

    lines = _schedule_lines(datetime(2026, 9, 10, 9, 30, tzinfo=ET),
                            datetime(2026, 9, 10, 16, 0, tzinfo=ET))
    assert len(lines) < 20, f"briefing schedule is {len(lines)} lines"
    assert any("every 30 min" in ln and "runs)" in ln for ln in lines)
