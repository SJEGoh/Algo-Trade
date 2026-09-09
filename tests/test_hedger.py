"""Threshold hedging: shave abnormal exposures, leave the rest to diversification.

The hedger does nothing until a bucket grows past a trigger, then shaves the excess down to
a target rather than to flat. Most of what is pinned here is restraint — the cases where it
must NOT act — plus the three failure modes that would be silent in production:

  * counting its own hedge as exposure, which oscillates forever;
  * hedging on a stale threshold so the book thrashes on the boundary;
  * quietly skipping exposure it could not price or could not hedge, so the report says
    "covered" about something nobody is watching.
"""
import pytest

from portfolio.hedger import (BucketPolicy, HEDGE_STRATEGY_ID, UNCLASSIFIED,
                              bucket_exposures, default_bucket_of, plan)

NAV = 1_000_000.0
PRICES = {"SMH": 250.0, "IGV": 100.0, "QQQ": 500.0, "XLE": 90.0}


def exposures_for(fractions, nav=NAV):
    """Build exposure objects directly — bucketing is tested separately."""
    from portfolio.hedger import Exposure
    return {b: Exposure(b, f * nav, f) for b, f in fractions.items()}


# ------------------------------------------------------------------ bucketing
def test_equities_bucket_by_sector_and_others_by_asset_class():
    assert default_bucket_of("NVDA") == "semiconductors"
    assert default_bucket_of("CRWD") == "software"
    assert default_bucket_of("MCL", {"asset_class": "future"}) == "future"


def test_an_unknown_equity_is_unclassified_not_guessed():
    assert default_bucket_of("WXYZ", {"asset_class": "equity"}) == UNCLASSIFIED


def test_exposure_sums_across_strategies_and_applies_multipliers():
    positions = {"a": {"NVDA": 100.0}, "b": {"NVDA": 50.0, "MCL": 2.0}}
    exp, unpriced = bucket_exposures(
        positions, marks={"NVDA": 200.0, "MCL": 90.0}, nav=NAV,
        multipliers={"MCL": 100.0}, instruments={"MCL": {"asset_class": "future"}})

    assert exp["semiconductors"].notional == pytest.approx(150 * 200)
    assert exp["future"].notional == pytest.approx(2 * 90 * 100)   # multiplier applied
    assert unpriced == {}


def test_the_hedges_own_positions_are_excluded_from_exposure():
    """The defining bug of the design. If the hedge counts toward the exposure it is sizing,
    it sees its own work, unwinds, sees the exposure return, and oscillates forever."""
    positions = {"momentum": {"NVDA": 100.0},
                 HEDGE_STRATEGY_ID: {"SMH": -80.0}}
    exp, _ = bucket_exposures(positions, marks={"NVDA": 200.0, "SMH": 250.0}, nav=NAV)

    assert exp["semiconductors"].notional == pytest.approx(20_000)
    assert all("SMH" not in e.symbols for e in exp.values())


def test_a_symbol_with_no_mark_is_reported_not_dropped():
    """Exposure that could not be measured is exactly the exposure worth knowing about."""
    exp, unpriced = bucket_exposures({"a": {"NVDA": 100.0, "ARM": 50.0}},
                                     marks={"NVDA": 200.0}, nav=NAV)
    assert unpriced == {"ARM": 50.0}
    assert exp["semiconductors"].notional == pytest.approx(20_000)


def test_zero_nav_is_refused():
    with pytest.raises(ValueError, match="NAV must be positive"):
        bucket_exposures({"a": {"NVDA": 1.0}}, marks={"NVDA": 1.0}, nav=0.0)


# ------------------------------------------------------------------ restraint
def test_an_exposure_below_the_trigger_is_left_alone():
    """Nothing held, nothing wanted — the submission should be empty rather than a no-op
    leg for every hedge instrument in the map."""
    p = plan(exposures_for({"semiconductors": 0.25}), NAV, PRICES)
    assert [d.quantity for d in p.decisions] == [0.0]
    assert p.book == []


def test_an_exposure_at_the_trigger_is_left_alone():
    p = plan(exposures_for({"semiconductors": 0.30}), NAV, PRICES,
             default_policy=BucketPolicy(0.30, 0.25, 0.20))
    assert p.decisions[0].quantity == 0.0


def test_a_tiny_excess_is_not_worth_an_order():
    p = plan(exposures_for({"semiconductors": 0.2505}), NAV, PRICES,
             default_policy=BucketPolicy(0.25, 0.25, 0.20), min_hedge_notional=1_000.0)
    assert p.decisions[0].quantity == 0.0
    assert "floor" in p.decisions[0].reason


# ------------------------------------------------------------------ shaving
def test_the_excess_is_shaved_to_the_target_not_to_flat():
    """Hedging to zero would discard a position the strategies took on purpose."""
    p = plan(exposures_for({"semiconductors": 0.45}), NAV, PRICES,
             default_policy=BucketPolicy(0.30, 0.25, 0.20))
    d = p.decisions[0]

    assert d.hedge_notional == pytest.approx(-0.20 * NAV, rel=0.01)   # 45% - 25%
    assert d.quantity == pytest.approx(-800, abs=1)                   # 200k / 250
    assert d.hedge_symbol == "SMH"


def test_a_short_exposure_is_hedged_long():
    p = plan(exposures_for({"semiconductors": -0.45}), NAV, PRICES,
             default_policy=BucketPolicy(0.30, 0.25, 0.20))
    assert p.decisions[0].quantity > 0, "a large short was not hedged with a long"


def test_each_bucket_uses_its_own_instrument():
    p = plan(exposures_for({"semiconductors": 0.45, "software": 0.40}), NAV, PRICES,
             default_policy=BucketPolicy(0.30, 0.25, 0.20))
    got = {d.bucket: d.hedge_symbol for d in p.decisions if d.quantity}
    assert got == {"semiconductors": "SMH", "software": "IGV"}


# ------------------------------------------------------------------ hysteresis
def test_an_unhedged_bucket_is_judged_against_the_trigger():
    p = plan(exposures_for({"semiconductors": 0.27}), NAV, PRICES,
             default_policy=BucketPolicy(0.30, 0.25, 0.20), current_hedge={})
    assert p.decisions[0].quantity == 0.0, "hedged below the trigger"


def test_an_already_hedged_bucket_is_judged_against_the_release():
    """Between release and trigger the hedge is kept, so a book hovering on the boundary
    does not hedge and unwind on every cycle."""
    p = plan(exposures_for({"semiconductors": 0.27}), NAV, PRICES,
             default_policy=BucketPolicy(0.30, 0.25, 0.20), current_hedge={"SMH": -800.0})
    assert p.decisions[0].quantity < 0, "unwound while still above the release level"


def test_falling_below_the_release_unwinds_completely():
    p = plan(exposures_for({"semiconductors": 0.18}), NAV, PRICES,
             default_policy=BucketPolicy(0.30, 0.25, 0.20), current_hedge={"SMH": -800.0})
    assert p.decisions[0].quantity == 0.0


def test_hysteresis_state_comes_from_the_positions_themselves():
    """No separate store to go stale across a restart — the hedge book IS the state."""
    exposures = exposures_for({"semiconductors": 0.27})
    policy = BucketPolicy(0.30, 0.25, 0.20)
    cold = plan(exposures, NAV, PRICES, default_policy=policy, current_hedge={})
    warm = plan(exposures, NAV, PRICES, default_policy=policy, current_hedge={"SMH": -800.0})
    assert cold.decisions[0].quantity == 0.0 and warm.decisions[0].quantity < 0


# ------------------------------------------------------------------ honest reporting
def test_an_unclassified_exposure_is_reported_and_never_hedged():
    """A guessed hedge against an unknown exposure is a second unknown position."""
    p = plan(exposures_for({UNCLASSIFIED: 0.50}), NAV, PRICES)
    assert p.unhedgeable[UNCLASSIFIED] == pytest.approx(0.50 * NAV)
    assert p.book == []


def test_a_bucket_with_no_hedge_instrument_is_reported():
    p = plan(exposures_for({"biotech": 0.60}), NAV, PRICES)
    assert "biotech" in p.unhedgeable


def test_an_unpriceable_hedge_instrument_fails_closed():
    """Sizing off a missing price would put on a position of unknown size."""
    p = plan(exposures_for({"semiconductors": 0.45}), NAV, prices={},
             default_policy=BucketPolicy(0.30, 0.25, 0.20))
    assert "semiconductors" in p.unhedgeable
    assert p.book == []


def test_coverage_reports_what_was_not_measured():
    p = plan(exposures_for({"semiconductors": 0.45}), NAV, PRICES,
             unpriced={"WXYZ": 100.0}, default_policy=BucketPolicy(0.30, 0.25, 0.20))
    assert p.coverage["unpriced_symbols"] == ["WXYZ"]
    assert p.coverage["nav"] == NAV


# ------------------------------------------------------------------ staleness
def test_a_hedge_whose_exposure_vanished_is_unwound():
    """Otherwise a hedge outlives the position it was taken against and becomes a naked
    short in its own right."""
    p = plan(exposures_for({"software": 0.10}), NAV, PRICES, current_hedge={"SMH": -800.0})
    stale = [d for d in p.decisions if d.hedge_symbol == "SMH"]
    assert stale and stale[0].quantity == 0.0
    assert "unwinding" in stale[0].reason


def test_the_book_is_absolute_and_includes_the_unwinds():
    """/targets is authoritative, so listing a zero target says the unwind out loud in the
    journal instead of leaving it to be inferred from an omission."""
    p = plan(exposures_for({"semiconductors": 0.45, "software": 0.10}), NAV, PRICES,
             default_policy=BucketPolicy(0.30, 0.25, 0.20), current_hedge={"IGV": -50.0})
    targets = {i["instrument"]["symbol"]: i["target_quantity"] for i in p.book}
    assert targets["SMH"] < 0
    assert targets["IGV"] == 0.0


# ------------------------------------------------------------------ policy sanity
@pytest.mark.parametrize("bad", [(0.2, 0.25, 0.3), (0.3, 0.35, 0.2), (0.3, 0.25, 0.0)])
def test_an_incoherent_policy_is_refused(bad):
    with pytest.raises(ValueError):
        BucketPolicy(*bad)


def test_per_bucket_policies_override_the_default():
    p = plan(exposures_for({"semiconductors": 0.35, "software": 0.35}), NAV, PRICES,
             default_policy=BucketPolicy(0.30, 0.25, 0.20),
             policies={"software": BucketPolicy(0.50, 0.45, 0.40)})
    acted = {d.bucket for d in p.decisions if d.quantity}
    assert acted == {"semiconductors"}
