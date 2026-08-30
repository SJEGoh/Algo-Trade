# tests/test_position_ledger.py
import pytest
import sys
import os

# make the ledger importable — adjust path to your layout
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "execution"))
from position_ledger import PositionLedger


@pytest.fixture
def ledger():
    """Fresh ledger with no IB connection — executor=None is fine since
    attribution logic never touches self._executor."""
    return PositionLedger(executor=None)


def test_open_long_sets_avg_cost(ledger):
    ledger.record_fill("AAPL", +100, 50.0, "s1")
    assert ledger.strategy_positions["s1"]["AAPL"] == 100.0
    assert ledger.strategy_avg_cost["s1"]["AAPL"] == 50.0
    assert ledger.strategy_realized_pnl["s1"] == 0.0  # no P&L realized on an opening trade


def test_adding_to_long_updates_weighted_avg_cost(ledger):
    ledger.record_fill("AAPL", +100, 50.0, "s1")
    ledger.record_fill("AAPL", +100, 60.0, "s1")
    assert ledger.strategy_positions["s1"]["AAPL"] == 200.0
    assert ledger.strategy_avg_cost["s1"]["AAPL"] == 55.0  # (50*100 + 60*100) / 200


def test_partial_reduce_realizes_pnl_and_keeps_avg_cost(ledger):
    ledger.record_fill("AAPL", +100, 50.0, "s1")
    ledger.record_fill("AAPL", +100, 60.0, "s1")  # avg cost 55, qty 200
    ledger.record_fill("AAPL", -150, 70.0, "s1")  # sell 150 @ 70
    assert ledger.strategy_realized_pnl["s1"] == pytest.approx(2250.0)  # (70-55)*150
    assert ledger.strategy_positions["s1"]["AAPL"] == 50.0
    assert ledger.strategy_avg_cost["s1"]["AAPL"] == 55.0  # unchanged on partial reduce


def test_open_short_sets_avg_cost(ledger):
    ledger.record_fill("TSLA", -100, 300.0, "s2")
    assert ledger.strategy_positions["s2"]["TSLA"] == -100.0
    assert ledger.strategy_avg_cost["s2"]["TSLA"] == 300.0


def test_adding_to_short_updates_weighted_avg_cost(ledger):
    ledger.record_fill("TSLA", -100, 300.0, "s2")
    ledger.record_fill("TSLA", -100, 320.0, "s2")
    assert ledger.strategy_positions["s2"]["TSLA"] == -200.0
    assert ledger.strategy_avg_cost["s2"]["TSLA"] == pytest.approx(310.0)


def test_flip_short_to_long_realizes_and_reopens(ledger):
    ledger.record_fill("TSLA", -100, 300.0, "s2")   # short 100 @ 300
    ledger.record_fill("TSLA", +150, 280.0, "s2")   # buy 150: close 100 @ profit, flip to long 50
    assert ledger.strategy_realized_pnl["s2"] == pytest.approx(2000.0)  # (280-300)*100*(-1)
    assert ledger.strategy_positions["s2"]["TSLA"] == 50.0
    assert ledger.strategy_avg_cost["s2"]["TSLA"] == 280.0  # new long leg at fill price


def test_full_close_to_flat_zeroes_avg_cost(ledger):
    ledger.record_fill("AAPL", +100, 50.0, "s1")
    ledger.record_fill("AAPL", -100, 60.0, "s1")
    assert ledger.strategy_positions["s1"]["AAPL"] == 0.0
    assert ledger.strategy_realized_pnl["s1"] == pytest.approx(1000.0)  # (60-50)*100
    assert ledger.strategy_avg_cost["s1"]["AAPL"] == 0.0


def test_two_strategies_are_isolated(ledger):
    ledger.record_fill("AAPL", +100, 50.0, "s1")
    ledger.record_fill("AAPL", +100, 80.0, "s2")
    # same symbol, different strategies — cost bases must not bleed into each other
    assert ledger.strategy_avg_cost["s1"]["AAPL"] == 50.0
    assert ledger.strategy_avg_cost["s2"]["AAPL"] == 80.0
