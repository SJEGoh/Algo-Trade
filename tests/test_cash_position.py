"""Cash-as-a-position tests.

Each strategy holds CASH alongside its instruments: it starts at the strategy's capital
basis and every fill moves it by -signed_qty * price * multiplier. The invariant these
tests defend is

    nav = cash + position_value == starting_cash + realized + unrealized

for longs, shorts, flips, multiplier-bearing futures, internal crosses and restarts —
because the dashboard's overall equity curve is the sum of that NAV across strategies.
"""
import pytest

from ledger.position_ledger import PositionLedger
from logger.event_logger import EventLogger

CFG = {
    "alpha": {"capital_allocation": 100_000.0, "max_drawdown": 0.15},
    "beta": {"capital_allocation": 50_000.0, "max_drawdown": 0.15},
    "gamma": {"capital_allocation": 1_000_000.0, "starting_cash": 250_000.0,
              "max_drawdown": 0.15},
}


@pytest.fixture
def ledger():
    return PositionLedger(executor=None, config=CFG)


def assert_nav_identity(led, marks):
    """nav must equal cash + positions AND starting cash + P&L, for every strategy."""
    for sid, v in led.equity_snapshot(marks).items():
        assert v["nav"] == pytest.approx(v["cash"] + v["position_value"])
        assert v["nav"] == pytest.approx(v["starting_cash"] + v["realized"] + v["unrealized"])


# ---------------------------------------------------------------- seeding
def test_cash_starts_at_capital_allocation(ledger):
    assert ledger.cash("alpha") == 100_000.0
    assert ledger.cash("beta") == 50_000.0


def test_starting_cash_overrides_allocation(ledger):
    """`starting_cash` funds a strategy with less than its risk allocation."""
    assert ledger.cash("gamma") == 250_000.0
    assert ledger.starting_cash["gamma"] == 250_000.0


def test_unconfigured_strategy_starts_flat(ledger):
    """An unknown strategy has no capital basis, so its NAV is just P&L."""
    ledger.record_fill("AAPL", +10, 100.0, "not_in_config")
    snap = ledger.equity_snapshot({"AAPL": 110.0})["not_in_config"]
    assert snap["starting_cash"] == 0.0
    assert snap["cash"] == -1_000.0
    assert snap["nav"] == pytest.approx(100.0) == pytest.approx(snap["equity"])


# ---------------------------------------------------------------- cash moves
def test_buy_pays_cash_sell_receives_it(ledger):
    ledger.record_fill("AAPL", +100, 50.0, "alpha")     # -$5,000
    assert ledger.cash("alpha") == 95_000.0
    ledger.record_fill("AAPL", -40, 60.0, "alpha")      # +$2,400
    assert ledger.cash("alpha") == 97_400.0


def test_nav_holds_through_a_long_round_trip(ledger):
    ledger.record_fill("AAPL", +100, 50.0, "alpha")
    ledger.record_fill("AAPL", +100, 60.0, "alpha")     # avg cost 55
    assert_nav_identity(ledger, {"AAPL": 70.0})
    snap = ledger.equity_snapshot({"AAPL": 70.0})["alpha"]
    assert snap["position_value"] == pytest.approx(14_000.0)
    assert snap["nav"] == pytest.approx(103_000.0)      # 100k + (70-55)*200

    ledger.record_fill("AAPL", -200, 70.0, "alpha")     # fully closed -> all cash
    snap = ledger.equity_snapshot({"AAPL": 70.0})["alpha"]
    assert snap["position_value"] == 0.0
    assert snap["cash"] == pytest.approx(103_000.0)
    assert snap["nav"] == pytest.approx(103_000.0)


def test_short_position_raises_cash_and_nav_still_ties(ledger):
    ledger.record_fill("TSLA", -100, 300.0, "beta")     # +$30,000 cash
    assert ledger.cash("beta") == 80_000.0
    snap = ledger.equity_snapshot({"TSLA": 280.0})["beta"]
    assert snap["position_value"] == pytest.approx(-28_000.0)   # short leg is negative value
    assert snap["nav"] == pytest.approx(52_000.0)               # 50k + 2k unrealized
    assert_nav_identity(ledger, {"TSLA": 280.0})


def test_nav_holds_through_a_flip(ledger):
    ledger.record_fill("TSLA", -100, 300.0, "beta")
    ledger.record_fill("TSLA", +150, 280.0, "beta")     # close short, flip long 50 @ 280
    assert_nav_identity(ledger, {"TSLA": 290.0})
    snap = ledger.equity_snapshot({"TSLA": 290.0})["beta"]
    assert snap["realized"] == pytest.approx(2_000.0)
    assert snap["unrealized"] == pytest.approx(500.0)
    assert snap["nav"] == pytest.approx(52_500.0)


def test_futures_multiplier_is_dollar_denominated(ledger):
    ledger.multipliers["CL"] = 1000.0
    ledger.record_fill("CL", +2, 68.0, "gamma")         # 2 * 68 * 1000 = $136,000
    assert ledger.cash("gamma") == pytest.approx(114_000.0)
    snap = ledger.equity_snapshot({"CL": 69.0})["gamma"]
    assert snap["position_value"] == pytest.approx(138_000.0)
    assert snap["nav"] == pytest.approx(252_000.0)      # 250k + $2,000
    assert_nav_identity(ledger, {"CL": 69.0})


def test_internal_cross_is_cash_neutral_across_strategies(ledger):
    total_before = ledger.cash("alpha") + ledger.cash("beta")
    ledger.apply_internal_cross("MSFT", +10, 400.0, "alpha")
    ledger.apply_internal_cross("MSFT", -10, 400.0, "beta")
    assert ledger.cash("alpha") == 96_000.0
    assert ledger.cash("beta") == 54_000.0
    assert ledger.cash("alpha") + ledger.cash("beta") == total_before
    assert ledger.current_positions.get("MSFT", 0.0) == 0.0   # unchanged at the broker
    assert_nav_identity(ledger, {"MSFT": 410.0})


def test_attributed_net_fill_moves_cash(ledger):
    ledger.record_net_pending("AAPL", +10)
    ledger.apply_attributed_fill("AAPL", +10, 100.0, "alpha")
    assert ledger.cash("alpha") == 99_000.0
    assert ledger.current_positions["AAPL"] == 10.0


# ---------------------------------------------------------------- snapshot shape
def test_unmarked_position_is_valued_at_cost(ledger):
    """No mark -> 0 unrealized, and NAV falls back to cost basis rather than vanishing."""
    ledger.record_fill("AAPL", +100, 50.0, "alpha")
    snap = ledger.equity_snapshot({})["alpha"]           # no marks at all
    assert snap["unrealized"] == 0.0
    assert snap["position_value"] == pytest.approx(5_000.0)
    assert snap["nav"] == pytest.approx(100_000.0)
    assert snap["unmarked"] == ["AAPL"]


def test_flat_strategies_still_report_their_cash(ledger):
    """Every configured strategy appears in the snapshot, so the portfolio NAV is the
    sum over all strategies even before the first trade."""
    snap = ledger.equity_snapshot({})
    assert set(snap) == set(CFG)
    assert sum(v["nav"] for v in snap.values()) == pytest.approx(400_000.0)


def test_strategy_book_lists_cash_first(ledger):
    ledger.record_fill("AAPL", +100, 50.0, "alpha")
    book = ledger.strategy_book("alpha", {"AAPL": 55.0})
    assert book[0]["symbol"] == "CASH" and book[0]["is_cash"] is True
    assert book[0]["market_value"] == 95_000.0
    assert book[1]["symbol"] == "AAPL"
    assert book[1]["market_value"] == pytest.approx(5_500.0)
    assert sum(r["market_value"] for r in book) == pytest.approx(100_500.0)


def test_set_cash_reconciles_a_balance(ledger):
    ledger.set_cash("alpha", 42_000.0)
    assert ledger.cash("alpha") == 42_000.0
    assert ledger.starting_cash["alpha"] == 100_000.0    # basis untouched by default
    ledger.set_cash("alpha", 42_000.0, reset_basis=True)
    assert ledger.starting_cash["alpha"] == 42_000.0


# ---------------------------------------------------------------- persistence
def test_cash_survives_a_restart(tmp_path):
    db = EventLogger(db_path=tmp_path / "executor.db")
    led = PositionLedger(executor=None, config=CFG)
    led.record_fill("AAPL", +100, 50.0, "alpha")
    led.save_state(db)

    restored = PositionLedger(executor=None, config=CFG)
    restored.restore_state(db)
    assert restored.cash("alpha") == 95_000.0
    assert restored.starting_cash["alpha"] == 100_000.0
    # a strategy that never traded keeps the basis seeded from config
    assert restored.cash("beta") == 50_000.0
    db.close()


def test_equity_history_round_trips_nav(tmp_path):
    db = EventLogger(db_path=tmp_path / "executor.db")
    db.log_equity("2026-01-02T15:00:00Z", "alpha", 100.0, 50.0, 150.0,
                  cash=95_000.0, position_value=5_150.0, nav=100_150.0)
    row = db.get_equity_history(strategy_id="alpha")[0]
    assert row["equity"] == 150.0            # P&L, unchanged meaning
    assert row["nav"] == 100_150.0
    assert row["cash"] == 95_000.0
    db.close()


def test_legacy_snapshot_rows_have_null_nav(tmp_path):
    """Rows written before cash existed still load — the dashboard rebases them."""
    db = EventLogger(db_path=tmp_path / "executor.db")
    db.log_equity("2026-01-02T15:00:00Z", "alpha", 10.0, 5.0, 15.0)
    row = db.get_equity_history(strategy_id="alpha")[0]
    assert row["nav"] is None and row["equity"] == 15.0
    db.close()


def test_write_off_returns_cost_basis_to_cash(ledger):
    """Clearing a phantom position (already flat at the broker) must not silently
    destroy NAV — the cost basis goes back to cash and no P&L is booked."""
    ledger.record_fill("AAPL", +100, 50.0, "alpha")
    nav_before = ledger.equity_snapshot({"AAPL": 60.0})["alpha"]["nav"]
    assert nav_before == pytest.approx(101_000.0)

    assert ledger.write_off_position("alpha", "AAPL") == 100.0
    snap = ledger.equity_snapshot({"AAPL": 60.0})["alpha"]
    assert snap["cash"] == pytest.approx(100_000.0)     # 95k + 100 * 50
    assert snap["position_value"] == 0.0
    assert snap["realized"] == 0.0                      # no invented P&L
    assert snap["nav"] == pytest.approx(100_000.0)
    assert_nav_identity(ledger, {"AAPL": 60.0})
    assert ledger.write_off_position("alpha", "AAPL") == 0.0   # idempotent
