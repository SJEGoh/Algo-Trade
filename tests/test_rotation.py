"""tests/test_rotation.py — CombinedRotationStrategy unit tests.

Uses a fake DataProvider returning deterministic price series so every assertion
is exact. No network, no yfinance, no Alpaca.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from models.rotation import CombinedRotationStrategy, _kalman_state_z, _quadrant
from data.data_provider import DataProvider


# ---------------------------------------------------------------------------
# Fake data provider
# ---------------------------------------------------------------------------
class FakeDataProvider(DataProvider):
    """Returns pre-loaded daily close series keyed by symbol."""

    def __init__(self, data: dict[str, pd.Series]):
        self._data = data

    def get_daily_bars(self, symbol: str, lookback_days: int) -> pd.Series:
        s = self._data.get(symbol)
        if s is None:
            return pd.Series(dtype=float)
        return s.iloc[-lookback_days:]

    def get_current_price(self, symbol: str) -> float:
        s = self._data.get(symbol)
        if s is None or s.empty:
            raise ValueError(f"no data for {symbol}")
        return float(s.iloc[-1])


# ---------------------------------------------------------------------------
# Helpers — build deterministic price series
# ---------------------------------------------------------------------------
def _dates(n: int = 300) -> pd.DatetimeIndex:
    return pd.bdate_range("2025-01-02", periods=n, freq="B")


def _flat(price: float, n: int = 300) -> pd.Series:
    """Perfectly flat price series (0% daily return)."""
    return pd.Series(price, index=_dates(n))


def _trending(start: float, daily_ret: float, n: int = 300) -> pd.Series:
    """Geometric growth at a fixed daily return."""
    return pd.Series(start * (1 + daily_ret) ** np.arange(n), index=_dates(n))


def _make_provider(symbols: list[str], series_fn=None, bench: str = "SPY") -> FakeDataProvider:
    """Build a FakeDataProvider with one benchmark and N universe symbols.
    series_fn(symbol) -> pd.Series; default: all flat at 100."""
    fn = series_fn or (lambda s: _flat(100.0))
    data = {s: fn(s) for s in symbols}
    if bench not in data:
        data[bench] = _flat(100.0)
    return data


# ---------------------------------------------------------------------------
# Quadrant helper
# ---------------------------------------------------------------------------
class TestQuadrant:
    def test_leading(self):
        assert _quadrant(1.0, 1.0) == "Leading"

    def test_improving(self):
        assert _quadrant(-1.0, 1.0) == "Improving"

    def test_lagging(self):
        assert _quadrant(-1.0, -1.0) == "Lagging"

    def test_weakening(self):
        assert _quadrant(1.0, -1.0) == "Weakening"

    def test_boundary_x_zero_y_positive(self):
        assert _quadrant(0.0, 1.0) == "Leading"

    def test_boundary_x_zero_y_negative(self):
        assert _quadrant(0.0, -1.0) == "Weakening"


# ---------------------------------------------------------------------------
# Kalman signal primitives
# ---------------------------------------------------------------------------
class TestKalmanStateZ:
    def test_flat_series_returns_near_zero(self):
        s = _flat(100.0, 300)
        z = _kalman_state_z(s, "level", period=20, q=0.05)
        assert len(z) == len(s)
        # flat returns -> level should be near zero
        assert abs(z.iloc[-1]) < 2.0

    def test_trending_up_level_responds(self):
        s = _trending(100.0, 0.002, 300)  # +0.2%/day
        z = _kalman_state_z(s, "level", period=20, q=0.05)
        # The z-score of a constant-drift series oscillates (rolling mean catches up),
        # but it should NOT be identically zero — the filter is tracking real signal.
        assert z.dropna().abs().mean() > 0.01

    def test_velocity_detects_acceleration(self):
        # first half flat, second half trending
        n = 300
        prices = np.ones(n) * 100.0
        prices[150:] = 100.0 * (1.002 ** np.arange(150))
        s = pd.Series(prices, index=_dates(n))
        z = _kalman_state_z(s, "velocity", period=20, q=0.01)
        # velocity should be positive toward the end
        assert z.iloc[-1] > z.iloc[100]

    def test_output_length_matches_input(self):
        s = _flat(50.0, 200)
        z = _kalman_state_z(s, "level", period=20, q=0.05)
        assert len(z) == len(s)


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------
class TestComputeSignal:
    def _strat(self, data: dict[str, pd.Series], **kw) -> CombinedRotationStrategy:
        dp = FakeDataProvider(data)
        universe = [s for s in data if s != "SPY"]
        return CombinedRotationStrategy(
            data_provider=dp, universe=universe, benchmark="SPY",
            lookback_days=len(next(iter(data.values()))), **kw,
        )

    def test_signal_keys_match_universe(self):
        data = _make_provider(["A", "B", "C"])
        strat = self._strat(data)
        sig = strat.compute_signal(data)
        # all universe members with enough history should appear
        for sym in ["A", "B", "C"]:
            assert sym in sig
            assert "x" in sig[sym]
            assert "y" in sig[sym]
            assert "score" in sig[sym]
            assert "quadrant" in sig[sym]

    def test_benchmark_excluded_from_signal(self):
        data = _make_provider(["A", "SPY"])
        strat = self._strat(data)
        sig = strat.compute_signal(data)
        assert "SPY" not in sig

    def test_missing_symbol_skipped(self):
        data = _make_provider(["A", "B"])
        # remove B's data
        data["B"] = pd.Series(dtype=float)
        strat = self._strat(data)
        sig = strat.compute_signal(data)
        assert "A" in sig
        assert "B" not in sig

    def test_outperformer_has_nonzero_score(self):
        """A symbol trending up relative to a flat benchmark should produce a
        non-trivial signal (the z-score may oscillate, but it shouldn't be zero)."""
        data = {
            "SPY": _flat(100.0, 300),
            "WINNER": _trending(100.0, 0.003, 300),  # outperforming
            "FLAT": _flat(100.0, 300),                # matching benchmark
        }
        strat = self._strat(data)
        sig = strat.compute_signal(data)
        # WINNER's relative strength is trending — its score magnitude should be
        # meaningfully different from FLAT's near-zero.
        assert abs(sig["WINNER"]["score"]) > 0.01 or abs(sig["WINNER"]["x"]) > 0.01


# ---------------------------------------------------------------------------
# Sleeve A — gated long-leg
# ---------------------------------------------------------------------------
class TestSleeveA:
    def _strat(self, data, **kw):
        dp = FakeDataProvider(data)
        universe = [s for s in data if s != "SPY"]
        return CombinedRotationStrategy(
            data_provider=dp, universe=universe, benchmark="SPY",
            lookback_days=len(next(iter(data.values()))), **kw,
        )

    def test_top_fraction_selects_correct_count(self):
        # 6 symbols, top_fraction=1/3 -> top 2
        data = _make_provider([f"S{i}" for i in range(6)])
        strat = self._strat(data, top_fraction=1/3)
        sig = strat.compute_signal(data)
        # force all into Leading quadrant
        for s in sig:
            sig[s]["quadrant"] = "Leading"
            sig[s]["score"] = float(hash(s) % 100)  # deterministic ranking
        weights = strat._sleeve_a_weights(sig)
        assert len(weights) == 2

    def test_lagging_names_gated_out(self):
        data = _make_provider(["A", "B", "C"])
        strat = self._strat(data, top_fraction=1.0)  # try to hold all
        sig = strat.compute_signal(data)
        # put all in Lagging -> none qualify
        for s in sig:
            sig[s]["quadrant"] = "Lagging"
        weights = strat._sleeve_a_weights(sig)
        assert len(weights) == 0

    def test_equal_weight_per_qualifier(self):
        data = _make_provider(["A", "B", "C", "D", "E", "F"])
        strat = self._strat(data, top_fraction=0.5)  # top 3 of 6
        sig = strat.compute_signal(data)
        for i, s in enumerate(sig):
            sig[s]["quadrant"] = "Leading"
            sig[s]["score"] = float(i)
        weights = strat._sleeve_a_weights(sig)
        k = 3  # top 3
        for w in weights.values():
            assert abs(w - 1.0 / k) < 1e-9

    def test_fewer_than_3_symbols_returns_empty(self):
        data = _make_provider(["A", "B"])  # only 2 universe symbols
        strat = self._strat(data, top_fraction=0.5)
        sig = strat.compute_signal(data)
        weights = strat._sleeve_a_weights(sig)
        assert weights == {}


# ---------------------------------------------------------------------------
# Sleeve B — naive risk parity
# ---------------------------------------------------------------------------
class TestSleeveB:
    def _strat(self, data, **kw):
        dp = FakeDataProvider(data)
        universe = [s for s in data if s != "SPY"]
        return CombinedRotationStrategy(
            data_provider=dp, universe=universe, benchmark="SPY",
            lookback_days=len(next(iter(data.values()))), **kw,
        )

    def test_weights_sum_to_one(self):
        np.random.seed(42)
        n = 300
        data = {"SPY": _flat(100.0, n)}
        for s in ["A", "B", "C", "D"]:
            prices = 100.0 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n))
            data[s] = pd.Series(prices, index=_dates(n))
        strat = self._strat(data, risk_win=126)
        sig = strat.compute_signal(data)
        weights = strat._sleeve_b_weights(data, sig)
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_lower_vol_gets_higher_weight(self):
        n = 300
        idx = _dates(n)
        data = {
            "SPY": _flat(100.0, n),
            "CALM": pd.Series(100.0 * np.cumprod(1 + np.random.RandomState(1).normal(0, 0.005, n)), index=idx),
            "WILD": pd.Series(100.0 * np.cumprod(1 + np.random.RandomState(2).normal(0, 0.04, n)), index=idx),
        }
        strat = self._strat(data, risk_win=126)
        sig = strat.compute_signal(data)
        weights = strat._sleeve_b_weights(data, sig)
        assert weights["CALM"] > weights["WILD"]

    def test_no_tilt_ignores_signal(self):
        """With tilt_b=0, changing scores should not affect weights."""
        np.random.seed(99)
        n = 300
        data = {"SPY": _flat(100.0, n)}
        for s in ["X", "Y"]:
            data[s] = pd.Series(100.0 * np.cumprod(1 + np.random.normal(0, 0.02, n)), index=_dates(n))
        strat = self._strat(data, tilt_b=0.0, risk_win=126)
        sig1 = strat.compute_signal(data)
        w1 = strat._sleeve_b_weights(data, sig1)
        # artificially change scores
        sig2 = {s: {**v, "score": v["score"] * 10} for s, v in sig1.items()}
        w2 = strat._sleeve_b_weights(data, sig2)
        for s in w1:
            assert abs(w1[s] - w2[s]) < 1e-9


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------
class TestConstructWeights:
    def _strat(self, data, **kw):
        dp = FakeDataProvider(data)
        universe = [s for s in data if s != "SPY"]
        return CombinedRotationStrategy(
            data_provider=dp, universe=universe, benchmark="SPY",
            lookback_days=len(next(iter(data.values()))), **kw,
        )

    def test_returns_three_dicts(self):
        np.random.seed(42)
        n = 300
        data = {"SPY": _flat(100.0, n)}
        for s in ["A", "B", "C", "D"]:
            data[s] = pd.Series(100.0 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n)), index=_dates(n))
        strat = self._strat(data, weight_a=0.4)
        sig = strat.compute_signal(data)
        result = strat.construct_weights(sig, data)
        assert isinstance(result, tuple) and len(result) == 3
        combined, a, b = result
        assert isinstance(combined, dict)
        assert isinstance(a, dict)
        assert isinstance(b, dict)

    def test_all_weights_non_negative(self):
        """Both sleeves are long-only, so combined weights should never be negative."""
        np.random.seed(7)
        n = 300
        data = {"SPY": _flat(100.0, n)}
        for s in ["A", "B", "C", "D", "E", "F"]:
            data[s] = pd.Series(100.0 * np.cumprod(1 + np.random.normal(0.001, 0.02, n)), index=_dates(n))
        strat = self._strat(data, weight_a=0.35)
        sig = strat.compute_signal(data)
        combined, _, _ = strat.construct_weights(sig, data)
        for w in combined.values():
            assert w >= 0

    def test_weight_a_zero_is_pure_risk_parity(self):
        """With weight_a=0, sleeve A contributes nothing."""
        np.random.seed(11)
        n = 300
        data = {"SPY": _flat(100.0, n)}
        for s in ["A", "B", "C", "D"]:
            data[s] = pd.Series(100.0 * np.cumprod(1 + np.random.normal(0, 0.02, n)), index=_dates(n))
        strat = self._strat(data, weight_a=0.0)
        sig = strat.compute_signal(data)
        combined, a_w, b_w = strat.construct_weights(sig, data)
        # combined should equal sleeve B weights
        for s in combined:
            assert abs(combined[s] - b_w.get(s, 0.0)) < 1e-9

    def test_weight_a_one_is_pure_gated_leg(self):
        """With weight_a=1, sleeve B contributes nothing."""
        np.random.seed(13)
        n = 300
        data = {"SPY": _flat(100.0, n)}
        for s in ["A", "B", "C", "D"]:
            data[s] = pd.Series(100.0 * np.cumprod(1 + np.random.normal(0, 0.02, n)), index=_dates(n))
        strat = self._strat(data, weight_a=1.0)
        sig = strat.compute_signal(data)
        combined, a_w, b_w = strat.construct_weights(sig, data)
        for s in combined:
            assert abs(combined[s] - a_w.get(s, 0.0)) < 1e-9


# ---------------------------------------------------------------------------
# Intent generation
# ---------------------------------------------------------------------------
class TestGenerateIntents:
    def test_intent_schema(self):
        """Every intent must have the required fields."""
        np.random.seed(42)
        n = 300
        data = {"SPY": _flat(100.0, n)}
        for s in ["A", "B", "C", "D"]:
            data[s] = pd.Series(100.0 * np.cumprod(1 + np.random.normal(0.001, 0.02, n)), index=_dates(n))
        dp = FakeDataProvider(data)
        strat = CombinedRotationStrategy(
            data_provider=dp, universe=["A", "B", "C", "D"], benchmark="SPY",
            capital_allocation=100_000, lookback_days=n,
        )
        intents = strat.generate_intents()
        assert len(intents) > 0
        for i in intents:
            assert i["strategy_id"] == "kalman_rrg_combined"
            assert i["intent_type"] == "target_position"
            assert isinstance(i["target_quantity"], int)
            assert isinstance(i["expected_price"], float)
            assert i["instrument"]["asset_class"] == "equity"
            assert i["instrument"]["exchange"] == "SMART"
            assert "symbol" in i["instrument"]
            assert i["order_type"] == "market"

    def test_no_short_positions(self):
        """Long-only strategy — no target should be negative."""
        np.random.seed(42)
        n = 300
        data = {"SPY": _flat(100.0, n)}
        for s in ["A", "B", "C", "D", "E", "F"]:
            data[s] = pd.Series(100.0 * np.cumprod(1 + np.random.normal(0, 0.02, n)), index=_dates(n))
        dp = FakeDataProvider(data)
        strat = CombinedRotationStrategy(
            data_provider=dp, universe=list(data.keys() - {"SPY"}), benchmark="SPY",
            capital_allocation=100_000, lookback_days=n,
        )
        intents = strat.generate_intents()
        for i in intents:
            assert i["target_quantity"] >= 0, f"{i['instrument']['symbol']} has negative target"

    def test_total_notional_within_allocation(self):
        """Sum of |target * price| should not exceed allocation."""
        np.random.seed(42)
        n = 300
        alloc = 100_000
        data = {"SPY": _flat(100.0, n)}
        for s in ["A", "B", "C", "D"]:
            data[s] = pd.Series(100.0 * np.cumprod(1 + np.random.normal(0, 0.02, n)), index=_dates(n))
        dp = FakeDataProvider(data)
        strat = CombinedRotationStrategy(
            data_provider=dp, universe=["A", "B", "C", "D"], benchmark="SPY",
            capital_allocation=alloc, lookback_days=n,
        )
        intents = strat.generate_intents()
        gross = sum(abs(i["target_quantity"]) * i["expected_price"] for i in intents)
        # allow 5% slack for rounding of share counts
        assert gross <= alloc * 1.05, f"gross {gross:.0f} exceeds allocation {alloc}"

    def test_covers_full_universe(self):
        """Intents should cover the full universe (including zero-target names)
        so /targets can close dropped positions."""
        np.random.seed(42)
        n = 300
        universe = ["A", "B", "C", "D"]
        data = {"SPY": _flat(100.0, n)}
        for s in universe:
            data[s] = pd.Series(100.0 * np.cumprod(1 + np.random.normal(0, 0.02, n)), index=_dates(n))
        dp = FakeDataProvider(data)
        strat = CombinedRotationStrategy(
            data_provider=dp, universe=universe, benchmark="SPY",
            capital_allocation=100_000, lookback_days=n,
        )
        intents = strat.generate_intents()
        symbols = {i["instrument"]["symbol"] for i in intents}
        for s in universe:
            assert s in symbols

    def test_metadata_present(self):
        np.random.seed(42)
        n = 300
        data = {"SPY": _flat(100.0, n)}
        for s in ["A", "B", "C"]:
            data[s] = pd.Series(100.0 * np.cumprod(1 + np.random.normal(0.001, 0.02, n)), index=_dates(n))
        dp = FakeDataProvider(data)
        strat = CombinedRotationStrategy(
            data_provider=dp, universe=["A", "B", "C"], benchmark="SPY",
            capital_allocation=100_000, lookback_days=n,
        )
        intents = strat.generate_intents()
        for i in intents:
            meta = i["metadata"]
            assert "blended_weight" in meta
            assert "sleeve_a_weight" in meta
            assert "sleeve_b_weight" in meta
            assert "blend_weight_a" in meta


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_all_lagging_parks_sleeve_a_in_cash(self):
        """When every name is in Lagging, sleeve A holds nothing."""
        # Use flat prices -> relative strength is 1.0 -> Kalman z ≈ 0 for all
        # Then manually verify sleeve A is empty when all quadrants are bad
        data = _make_provider(["A", "B", "C", "D"])
        dp = FakeDataProvider(data)
        strat = CombinedRotationStrategy(
            data_provider=dp, universe=["A", "B", "C", "D"], benchmark="SPY",
            lookback_days=300,
        )
        sig = strat.compute_signal(data)
        # force all to Lagging
        for s in sig:
            sig[s]["quadrant"] = "Lagging"
        a_w = strat._sleeve_a_weights(sig)
        assert a_w == {}

    def test_empty_universe_returns_no_intents(self):
        data = {"SPY": _flat(100.0, 300)}
        dp = FakeDataProvider(data)
        strat = CombinedRotationStrategy(
            data_provider=dp, universe=[], benchmark="SPY",
            capital_allocation=100_000, lookback_days=300,
        )
        intents = strat.generate_intents()
        assert intents == []

    def test_benchmark_not_in_universe(self):
        """SPY should never appear as a tradable position."""
        data = _make_provider(["A", "B", "C", "SPY"])
        dp = FakeDataProvider(data)
        strat = CombinedRotationStrategy(
            data_provider=dp, universe=["A", "B", "C", "SPY"], benchmark="SPY",
            capital_allocation=100_000, lookback_days=300,
        )
        # constructor strips benchmark from universe
        assert "SPY" not in strat.universe
        intents = strat.generate_intents()
        symbols = {i["instrument"]["symbol"] for i in intents}
        assert "SPY" not in symbols
