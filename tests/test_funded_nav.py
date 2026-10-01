"""tests/test_funded_nav.py — portfolio NAV counts real capital only.

Every strategy's allocation seeds its cash, so the test fixtures, demo rows and the hedge
overlay in config.py put ~$1.1M of starting cash that no one set aside into portfolio NAV.
What must hold:

  * an unfunded strategy adds its P&L and positions to the totals, never its basis —
    real exposure on a fixture still shows, made-up capital doesn't;
  * the balance-sheet identities hold for the totals: nav == cash + position_value and
    equity == nav - starting_cash;
  * "funded" in a config entry overrides the name-based default either way.
"""
import pytest

import api.server as server
from config import is_funded


def _book(starting_cash, realized=0.0, unrealized=0.0, position_value=0.0):
    nav = starting_cash + realized + unrealized
    return {"starting_cash": starting_cash, "realized": realized, "unrealized": unrealized,
            "equity": realized + unrealized, "position_value": position_value,
            "cash": nav - position_value, "nav": nav}


@pytest.mark.parametrize("sid, funded", [
    ("pair_break_fade", True), ("cross_sectional_momentum", True),
    ("test_suite", False), ("test_suite_small_alloc", False), ("halt_test_macd", False),
    ("demo_momentum", False), ("hedge_overlay", False),
])
def test_default_funding(sid, funded):
    assert is_funded(sid, {}) is funded


def test_an_explicit_flag_wins():
    assert is_funded("test_suite", {"funded": True}) is True
    assert is_funded("pair_break_fade", {"funded": False}) is False


def test_totals_leave_out_unfunded_starting_cash():
    totals = server._portfolio_totals({
        "pair_break_fade": _book(500_000, realized=1_200, unrealized=-300, position_value=80_000),
        "test_suite": _book(1_000_000),                                   # never traded
        "test_suite_small_alloc": _book(1_000, unrealized=-2, position_value=11),  # 1 share F
        "hedge_overlay": _book(10_000, realized=-50),
    })
    assert totals["starting_cash"] == pytest.approx(500_000)
    assert totals["nav"] == pytest.approx(500_000 + 1_200 - 300 - 2 - 50)
    assert totals["position_value"] == pytest.approx(80_011)            # fixture's share counts
    assert totals["equity"] == pytest.approx(1_200 - 300 - 2 - 50)
    assert totals["nav"] == pytest.approx(totals["cash"] + totals["position_value"])
    assert totals["equity"] == pytest.approx(totals["nav"] - totals["starting_cash"])
