"""Capital allocation across strategies, and the ways it goes wrong.

Mean-variance is not dangerous because the maths is hard; it is dangerous because it is
confident. Fed a sample mean estimated from a fortnight, a max-Sharpe objective does not
average the error out — it seeks it, because "the strategy with the highest estimated
return" and "the strategy whose return is most over-estimated" are largely the same set.

So most of what is pinned here is refusal: refusing to use means without enough history,
refusing to judge a strategy that has barely started, and refusing to let any one name take
the book however good it looks.
"""
import numpy as np
import pandas as pd
import pytest

from portfolio.allocator import (Allocation, equal_weights, history_days,
                                 inverse_vol_weights, max_sharpe_weights, recommend,
                                 shrunk_covariance, strategy_returns)


def series(n_days, strategies, seed=0, vols=None, means=None, start="2026-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_days, freq="1D", tz="UTC")
    vols = vols or {s: 0.01 for s in strategies}
    means = means or {s: 0.0 for s in strategies}
    return pd.DataFrame(
        {s: rng.normal(means[s], vols[s], n_days) for s in strategies}, index=idx)


# ------------------------------------------------------------------ Ledoit-Wolf
def test_shrinkage_is_reported_and_bounded():
    cov, shrinkage = shrunk_covariance(series(300, ["a", "b", "c"]))
    assert 0.0 <= shrinkage <= 1.0
    assert cov.shape == (3, 3)
    assert list(cov.index) == ["a", "b", "c"]


def test_shrinkage_conditions_a_near_singular_covariance():
    """The reason this estimator is here. With few observations relative to strategies the
    sample covariance is nearly singular, and every mean-variance optimiser inverts it —
    amplifying exactly the directions estimated worst."""
    r = series(8, [f"s{i}" for i in range(6)], seed=3)
    sample = np.cov(r.values, rowvar=False)
    shrunk, coeff = shrunk_covariance(r, annualise=False)

    assert np.linalg.cond(shrunk.values) < np.linalg.cond(sample)
    assert coeff > 0.0, "no shrinkage applied where it was most needed"


def test_shrinkage_is_larger_when_there_is_less_data():
    _, few = shrunk_covariance(series(10, ["a", "b", "c", "d"], seed=1))
    _, many = shrunk_covariance(series(400, ["a", "b", "c", "d"], seed=1))
    assert few > many


def test_covariance_is_annualised_by_default():
    daily, _ = shrunk_covariance(series(300, ["a", "b"]), annualise=False)
    annual, _ = shrunk_covariance(series(300, ["a", "b"]), annualise=True)
    assert np.allclose(annual.values, daily.values * 252)


def test_shrinkage_preserves_real_correlation():
    """Shrinkage must not flatten structure that is actually there — otherwise it would be
    buying stability by discarding the diversification information it is meant to supply."""
    rng = np.random.default_rng(12)
    idx = pd.date_range("2026-01-01", periods=500, freq="1D", tz="UTC")
    common = rng.normal(0, 0.01, 500)
    r = pd.DataFrame({"a": common + rng.normal(0, 0.002, 500),
                      "b": common + rng.normal(0, 0.002, 500),
                      "c": rng.normal(0, 0.01, 500)}, index=idx)
    cov, shrinkage = shrunk_covariance(r, annualise=False)

    corr_ab = cov.loc["a", "b"] / np.sqrt(cov.loc["a", "a"] * cov.loc["b", "b"])
    corr_ac = cov.loc["a", "c"] / np.sqrt(cov.loc["a", "a"] * cov.loc["c", "c"])
    assert corr_ab > 0.8, f"shrank away a real correlation (got {corr_ab:.2f})"
    assert abs(corr_ac) < 0.3
    assert shrinkage < 1.0


# ------------------------------------------------------------------ weight schemes
def test_weights_are_a_long_only_simplex():
    r = series(300, ["a", "b", "c"], seed=2)
    cov, _ = shrunk_covariance(r)
    w = max_sharpe_weights(r.mean() * 252, cov, max_weight=0.6)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()


def test_the_cap_binds_however_good_a_strategy_looks():
    """The protection that matters. A strategy with a spectacular estimated Sharpe must
    still not be handed the book — the estimate is what is in doubt, not the arithmetic."""
    strategies = ["star", "b", "c", "d"]
    r = series(300, strategies, seed=4,
               means={"star": 0.02, "b": 0.0001, "c": 0.0001, "d": 0.0001},
               vols={s: 0.005 for s in strategies})
    cov, _ = shrunk_covariance(r)
    w = max_sharpe_weights(r.mean() * 252, cov, max_weight=0.30, l2_lambda=0.0)

    assert w["star"] <= 0.30 + 1e-6
    assert w["star"] == pytest.approx(0.30, abs=1e-3), "cap should bind on the best name"


def test_l2_pulls_toward_equal_weight():
    strategies = ["star", "b", "c", "d"]
    r = series(300, strategies, seed=5,
               means={"star": 0.01, "b": 0.0001, "c": 0.0001, "d": 0.0001},
               vols={s: 0.005 for s in strategies})
    cov, mu = shrunk_covariance(r)[0], r.mean() * 252

    loose = max_sharpe_weights(mu, cov, max_weight=1.0, l2_lambda=0.0)
    tight = max_sharpe_weights(mu, cov, max_weight=1.0, l2_lambda=50.0)

    spread = lambda w: float(((w - 0.25) ** 2).sum())
    assert spread(tight) < spread(loose), "L2 did not spread the book"


def test_inverse_vol_favours_the_calmer_strategy():
    r = series(300, ["calm", "wild"], seed=6, vols={"calm": 0.002, "wild": 0.02})
    w = inverse_vol_weights(shrunk_covariance(r)[0])
    assert w["calm"] > w["wild"]


def test_a_cap_that_cannot_fill_the_book_is_rejected():
    r = series(300, ["a", "b", "c"], seed=7)
    with pytest.raises(ValueError, match="cannot fill"):
        max_sharpe_weights(r.mean() * 252, shrunk_covariance(r)[0], max_weight=0.2)


# ------------------------------------------------------------------ the method ladder
def test_a_young_book_gets_equal_weight_not_an_optimisation():
    r = series(10, ["a", "b", "c"], seed=8)
    alloc = recommend(r, min_history_days=0.0)
    assert alloc.method == "equal_weight"
    assert alloc.weights.nunique() == 1


def test_a_medium_book_sizes_by_risk_and_ignores_returns():
    """Volatility estimates well from a short window; means do not. This rung of the ladder
    is the difference between degrading gracefully and failing catastrophically."""
    # three names and a loose cap, so the cap cannot force the answer and the test is
    # actually measuring what inverse-vol does with a tempting mean
    r = series(40, ["calm", "wild", "mid"], seed=9,
               vols={"calm": 0.002, "wild": 0.02, "mid": 0.008},
               means={"calm": 0.0, "wild": 0.05, "mid": 0.0})
    alloc = recommend(r, min_history_days=0.0, max_weight=0.9)

    assert alloc.method == "inverse_volatility"
    assert alloc.weights["calm"] > alloc.weights["mid"] > alloc.weights["wild"], \
        "sized on the lucky mean instead of on risk"


def test_a_long_book_is_allowed_to_optimise():
    r = series(400, ["a", "b", "c"], seed=10)
    alloc = recommend(r, min_history_days=0.0)
    assert alloc.method == "max_sharpe"
    assert alloc.shrinkage is not None


def test_a_lucky_fortnight_cannot_take_the_book():
    """The whole point, end to end: one strategy has a spectacular short run. The ladder
    must refuse to reward it, because two weeks cannot tell luck from skill."""
    strategies = ["lucky", "b", "c"]
    r = series(14, strategies, seed=11,
               means={"lucky": 0.05, "b": 0.0, "c": 0.0},
               vols={s: 0.001 for s in strategies})
    alloc = recommend(r, min_history_days=0.0)

    assert alloc.method != "max_sharpe"
    assert alloc.weights["lucky"] <= 0.4, f"gave {alloc.weights['lucky']:.0%} to a lucky fortnight"


# ------------------------------------------------------------------ youth exemption
def young_and_old(old_days=200, young_days=5, seed=20):
    """One seasoned pair, plus a newcomer whose short life has been spectacular."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=old_days, freq="1D", tz="UTC")
    df = pd.DataFrame({"veteran_a": rng.normal(0.0003, 0.01, old_days),
                       "veteran_b": rng.normal(0.0003, 0.01, old_days)}, index=idx)
    df["newcomer"] = np.nan
    df.iloc[-young_days:, df.columns.get_loc("newcomer")] = rng.normal(0.05, 0.001, young_days)
    return df


def test_history_is_measured_per_strategy_not_per_frame():
    ages = history_days(young_and_old())
    assert ages["veteran_a"] > 150
    assert ages["newcomer"] < 14


def test_a_newcomer_is_exempt_and_holds_its_current_weight():
    alloc = recommend(young_and_old(), current_weights={"newcomer": 0.10},
                      min_history_days=14.0, min_coverage=0.0)

    assert "newcomer" in alloc.exempt
    assert alloc.exempt["newcomer"] == pytest.approx(0.10)
    assert "newcomer" not in alloc.weights, "a 5-day-old strategy was sized by the optimiser"


def test_the_rest_of_the_book_is_sized_around_the_exemption():
    alloc = recommend(young_and_old(), current_weights={"newcomer": 0.10},
                      min_history_days=14.0, min_coverage=0.0)

    assert alloc.weights.sum() == pytest.approx(0.90)      # the remaining budget
    assert alloc.all_weights.sum() == pytest.approx(1.0)   # and the book still totals 1


def test_a_spectacular_newcomer_cannot_pull_capital():
    """The failure this exists to prevent: five days at a 5% daily mean would dominate any
    max-Sharpe fit. Exemption means the optimiser never sees it."""
    alloc = recommend(young_and_old(), current_weights={"newcomer": 0.05},
                      min_history_days=14.0, min_coverage=0.0)
    assert alloc.all_weights["newcomer"] == pytest.approx(0.05)


def test_a_newcomer_does_not_distort_the_others_covariance():
    """It is held out of the estimate too, not merely out of the answer — a short noisy
    column would mis-size the strategies that HAVE earned their history."""
    alloc = recommend(young_and_old(), current_weights={"newcomer": 0.10},
                      min_history_days=14.0, min_coverage=0.0)
    assert "newcomer" not in (alloc.diagnostics.get("annualised_mean") or {})
    assert alloc.diagnostics["exempt_for_youth"] == ["newcomer"]


def test_a_strategy_that_matures_rejoins_the_optimisation():
    alloc = recommend(young_and_old(young_days=60), current_weights={"newcomer": 0.10},
                      min_history_days=14.0, min_coverage=0.0)
    assert alloc.exempt == {}
    assert "newcomer" in alloc.weights


def test_an_all_new_book_changes_nothing():
    """Every strategy is too young to judge. Sizing on that would be sizing on noise, so
    the honest answer is to leave the allocations alone and say why."""
    r = series(5, ["a", "b"], seed=21)
    alloc = recommend(r, current_weights={"a": 0.5, "b": 0.5}, min_history_days=14.0)

    assert alloc.method == "hold"
    assert alloc.weights.empty
    assert alloc.all_weights.to_dict() == {"a": 0.5, "b": 0.5}


def test_exempt_strategies_holding_the_whole_book_is_refused():
    r = series(5, ["a", "b"], seed=22)
    with pytest.raises(ValueError, match="reallocate them manually"):
        recommend(r, current_weights={"a": 0.8, "b": 0.7}, min_history_days=14.0)


def test_an_unknown_current_weight_defaults_to_zero():
    """A newcomer the caller did not price gets no reserved budget — it is reported as
    exempt so the caller decides, rather than silently receiving capital."""
    alloc = recommend(young_and_old(), min_history_days=14.0, min_coverage=0.0)
    assert alloc.exempt == {"newcomer": 0.0}
    assert alloc.weights.sum() == pytest.approx(1.0)


# ------------------------------------------------------------------ non-allocatable
from portfolio.allocator import NON_ALLOCATABLE          # noqa: E402
from portfolio.hedger import HEDGE_STRATEGY_ID           # noqa: E402


def book_with_hedge(n=300, seed=30):
    """Two real strategies plus a hedge overlay behaving as a hedge does: small negative
    drift, and negatively correlated with the book it protects."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="1D", tz="UTC")
    common = rng.normal(0.0005, 0.01, n)
    return pd.DataFrame({
        "alpha_a": common + rng.normal(0, 0.003, n),
        "alpha_b": common + rng.normal(0, 0.003, n),
        HEDGE_STRATEGY_ID: -0.8 * common + rng.normal(-0.0002, 0.001, n),
    }, index=idx)


def test_the_hedge_overlay_is_in_the_default_exclusion_set():
    assert HEDGE_STRATEGY_ID in NON_ALLOCATABLE
    assert {"__net__", "flatten_all", "kill_switch"} <= NON_ALLOCATABLE


def test_the_hedge_is_never_sized_by_the_optimiser():
    """Its expected return is negative by design and its correlation negative by
    construction — a Sharpe term would short it and a variance term would buy more of it,
    both fighting the hedger that actually sizes it."""
    alloc = recommend(book_with_hedge(), current_weights={HEDGE_STRATEGY_ID: 0.05},
                      min_history_days=0.0)

    assert HEDGE_STRATEGY_ID not in alloc.weights
    assert alloc.reserved[HEDGE_STRATEGY_ID] == pytest.approx(0.05)
    assert set(alloc.weights.index) == {"alpha_a", "alpha_b"}


def test_the_hedge_is_kept_out_of_the_covariance_too():
    """Dropped before estimation, not merely from the answer: a series negatively
    correlated with everything would distort the whole matrix, not just its own weight."""
    alloc = recommend(book_with_hedge(), current_weights={HEDGE_STRATEGY_ID: 0.05},
                      min_history_days=0.0)
    assert HEDGE_STRATEGY_ID not in (alloc.diagnostics.get("annualised_mean") or {})
    assert alloc.diagnostics["reserved_non_allocatable"] == [HEDGE_STRATEGY_ID]
    assert alloc.diagnostics["strategies"] == 2


def test_the_reserved_weight_comes_off_the_budget():
    alloc = recommend(book_with_hedge(), current_weights={HEDGE_STRATEGY_ID: 0.05},
                      min_history_days=0.0)
    assert alloc.weights.sum() == pytest.approx(0.95)
    assert alloc.all_weights.sum() == pytest.approx(1.0)


def test_reserved_and_exempt_both_reduce_the_budget():
    df = book_with_hedge()
    df["newcomer"] = np.nan
    df.iloc[-4:, df.columns.get_loc("newcomer")] = 0.01

    alloc = recommend(df, current_weights={HEDGE_STRATEGY_ID: 0.05, "newcomer": 0.10},
                      min_history_days=14.0, min_coverage=0.0)

    assert alloc.reserved[HEDGE_STRATEGY_ID] == pytest.approx(0.05)
    assert alloc.exempt["newcomer"] == pytest.approx(0.10)
    assert alloc.weights.sum() == pytest.approx(0.85)
    assert alloc.all_weights.sum() == pytest.approx(1.0)


def test_a_reserved_strategy_never_rejoins_however_long_it_runs():
    """Unlike youth exemption, this is permanent — the hedge does not graduate."""
    alloc = recommend(book_with_hedge(n=900), current_weights={HEDGE_STRATEGY_ID: 0.05},
                      min_history_days=14.0)
    assert HEDGE_STRATEGY_ID not in alloc.weights


def test_a_book_of_nothing_but_non_allocatables_is_refused():
    idx = pd.date_range("2026-01-01", periods=50, freq="1D", tz="UTC")
    df = pd.DataFrame({HEDGE_STRATEGY_ID: np.zeros(50), "__net__": np.zeros(50)}, index=idx)
    with pytest.raises(ValueError, match="every strategy is non-allocatable"):
        recommend(df)
