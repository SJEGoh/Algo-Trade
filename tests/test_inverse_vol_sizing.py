"""Position sizing from the live allocation, weighted inversely to volatility.

Sizing used to be a fixed dollar lot baked into each strategy, so `/allocate` could move a
strategy's capital and the strategy carried on at its old size — too small to use an
increase, and after a cut its orders bounced off the allocation check, which looks like the
strategy quietly failing rather than being resized.

Equal dollar lots were also the wrong risk unit: $2,000 of a 60%-vol name is several times
the risk of $2,000 of a 15%-vol one. Inverse-vol weighting equalises the risk contribution
instead of the notional.
"""
import numpy as np
import pandas as pd
import pytest

from models.equity_strategies import _EquityBase, OrbBreakoutStrategy, inverse_vol_sizing


class AlwaysHold(_EquityBase):
    """Holds everything it is given. Lets the sizing pipeline be tested without depending
    on a signal firing on synthetic data — a skipped test proves nothing about sizing."""
    def _selected(self, data):
        return set(data)


def bars(vol, n=120, price=100.0, seed=0):
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0, vol, n)))
    idx = pd.date_range("2026-01-01", periods=n, freq="30min", tz="UTC")
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": np.full(n, 1e6)}, index=idx)


# ------------------------------------------------------------------ the weighting
def test_the_calmer_name_gets_more_capital():
    data = {"CALM": bars(0.002, seed=1), "WILD": bars(0.02, seed=2)}
    d = inverse_vol_sizing(data, {"CALM", "WILD"}, 100_000, max_weight=1.0)
    assert d["CALM"] > d["WILD"] * 3, f"risk was not equalised: {d}"


def test_a_single_name_is_still_capped():
    """The concentration guard has to bind hardest exactly when the book is thinnest —
    a one-name day is when 'deploy the whole allocation' becomes 'bet it all on one name'."""
    d = inverse_vol_sizing({"A": bars(0.01)}, {"A"}, 200_000, max_weight=0.25)
    assert d["A"] == pytest.approx(50_000)


def test_capping_does_not_renormalise():
    """Portfolio weights must sum to 1, so capping there redistributes the excess. Here the
    cap exists to limit concentration — redistributing would defeat it, handing two names
    50% each under a 25% cap. A thin book deploys less instead."""
    data = {"A": bars(0.01, seed=3), "B": bars(0.01, seed=4)}
    d = inverse_vol_sizing(data, {"A", "B"}, 100_000, max_weight=0.25)
    assert sum(d.values()) == pytest.approx(50_000)
    assert all(v <= 25_000 + 1e-6 for v in d.values())


def test_nothing_selected_deploys_nothing():
    assert inverse_vol_sizing({"A": bars(0.01)}, set(), 100_000) == {}


def test_a_name_with_no_usable_volatility_falls_back_to_equal_weight():
    """A flat or too-short series has zero volatility, and 1/0 is not a weight. Sizing it at
    zero would silently drop a name the signal chose."""
    flat = bars(0.01)
    flat["close"] = 100.0                       # zero variance
    d = inverse_vol_sizing({"FLAT": flat}, {"FLAT"}, 100_000, max_weight=0.5)
    assert d["FLAT"] == pytest.approx(50_000)


# ------------------------------------------------------------------ allocation awareness
def test_doubling_the_allocation_doubles_the_position():
    data = {"AAA": bars(0.01, seed=5)}
    a = AlwaysHold("s", universe=["AAA"], capital_allocation=100_000,
                   max_weight=1.0)._targets(data)["AAA"]
    b = AlwaysHold("s", universe=["AAA"], capital_allocation=200_000,
                   max_weight=1.0)._targets(data)["AAA"]
    assert a > 0
    assert b == pytest.approx(2 * a, rel=0.02)


def test_deploy_fraction_scales_the_whole_book():
    data = {"AAA": bars(0.01, seed=5)}
    full = AlwaysHold("s", universe=["AAA"], capital_allocation=200_000,
                      max_weight=1.0)._targets(data)["AAA"]
    half = AlwaysHold("s", universe=["AAA"], capital_allocation=200_000,
                      deploy_fraction=0.5, max_weight=1.0)._targets(data)["AAA"]
    assert full > 0
    assert half == pytest.approx(full / 2, rel=0.02)


def test_without_an_allocation_the_old_fixed_lot_still_applies():
    """A caller that has not been taught to fetch the allocation keeps working unchanged,
    rather than silently sizing to zero."""
    data = {"AAA": bars(0.01, seed=5)}
    strat = AlwaysHold("s", universe=["AAA"], lot_dollars=10_000)
    assert strat.capital_allocation is None
    price = float(data["AAA"]["close"].iloc[-1])
    assert strat._targets(data)["AAA"] == int(10_000 / price)


def test_selection_is_independent_of_sizing():
    """The signal decides WHAT to hold; the allocation decides HOW MUCH. Changing the
    allocation must not change which names are chosen."""
    data = {"AAA": bars(0.01, seed=5), "BBB": bars(0.02, seed=6)}
    a = OrbBreakoutStrategy(universe=["AAA", "BBB"], capital_allocation=10_000,
                            ohlc_fn=lambda: data)
    b = OrbBreakoutStrategy(universe=["AAA", "BBB"], capital_allocation=900_000,
                            ohlc_fn=lambda: data)
    assert a._selected(data) == b._selected(data)
