# strategies/combined_rotation.py
"""
Combined two-factor rotation strategy (production intent generator).
 
Mirrors the research in the Kalman/RRG notebook as a single, broker-agnostic strategy that
emits `target_position` intents — same shape as MomentumStrategy.
 
The book is a capital blend of two long-only sleeves:
 
    combined_weight = weight_a * sleeve_A + (1 - weight_a) * sleeve_B
 
  * Sleeve A — "gated long-leg": rank the universe by a Kalman rotation score (built on each
    asset's relative strength vs a benchmark), hold the top `top_fraction` **only if** each name
    is also in a bullish RRG quadrant (Leading / Improving). Unqualified slots stay in cash, so a
    broadly weak universe pulls the sleeve toward cash. Equal weight (1/k) per qualifier.
 
  * Sleeve B — "NRP" (naive risk parity): weight by inverse trailing volatility. Signal-free by
    default (`tilt_b = 0`); set `tilt_b > 0` to lean the risk-parity weights toward the rotation
    leaders (then `period_b` becomes live).
 
Because both sleeves are long-only the combined book never shorts; its gross exposure is
`weight_a * sum(sleeve_A) + (1 - weight_a)` — i.e. <= 1, dropping below 1 exactly when the gate
parks Sleeve A in cash.
 
The signal is causal (forward-only Kalman + trailing z-score); live, the target is computed from
the most recent close and traded thereafter, so there is no look-ahead to correct for.
"""
 
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data.data_provider import DataProvider  # the interface, not a concrete impl
 
 
# --------------------------------------------------------------------------------------
# Signal primitives — faithful port of proj_1/src/features.py :: kalman_first / kalman_second
# --------------------------------------------------------------------------------------
def _kalman_state_z(series: pd.Series, which: str, period: int, q: float, r: float = 1.0) -> pd.Series:
    """Causal constant-velocity Kalman filter on the RETURNS of `series`.
 
    Returns the chosen hidden state ('level' -> x1, 'velocity' -> x2), standardised into a
    trailing `period`-window z-score. Forward pass only => no look-ahead.
    """
    s = pd.Series(series).astype(float)
    orig_index = s.index
    ret = s.pct_change().dropna()
    if ret.empty:
        return pd.Series(0.0, index=orig_index)
 
    x = np.array([[ret.iloc[0]], [0.0]])
    P = np.eye(2)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * q
    R = np.array([[r]])
    I = np.eye(2)
 
    out = []
    for z in ret.values:
        x = F @ x
        P = F @ P @ F.T + Q
        y = z - (H @ x)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (I - K @ H) @ P
        out.append(x[0, 0] if which == "level" else x[1, 0])
 
    state = pd.Series(out, index=ret.index)
    z = (state - state.rolling(period).mean()) / state.rolling(period).std()
    return z.reindex(orig_index).fillna(0.0)
 
 
def _quadrant(x: float, y: float) -> str:
    if x >= 0 and y >= 0:
        return "Leading"
    if x < 0 and y >= 0:
        return "Improving"
    if x < 0 and y < 0:
        return "Lagging"
    return "Weakening"
 
 
class CombinedRotationStrategy:
    def __init__(
        self,
        data_provider: DataProvider,
        universe: list[str],
        benchmark: str = "SPY",
        *,
        # signal
        period_a: int = 20,          # Kalman look-back for Sleeve A (gated long-leg)
        period_b: int = 40,          # Kalman look-back for Sleeve B (only used if tilt_b > 0)
        w_level: float = 0.5,        # composite score = w_level * leadership + w_velocity * momentum
        w_velocity: float = 0.5,
        # sleeve A
        top_fraction: float = 1 / 3,
        long_quadrants: tuple[str, ...] = ("Leading", "Improving"),
        # sleeve B
        tilt_b: float = 0.0,         # 0 = pure inverse-vol (signal-free); >0 = rotation-tilted
        risk_win: int = 126,         # trailing window (days) for inverse-vol
        # blend + sizing
        weight_a: float = 0.35,       # capital share of Sleeve A (rest to Sleeve B)
        capital_allocation: float = 100_000,
        lookback_days: int | None = None,
        state_path: Optional[str] = None,
    ):
        self._data = data_provider   # injected — strategy doesn't know or care about the broker/cache
        self.universe = [s for s in universe if s != benchmark]   # benchmark is not a tradable sleeve
        self.benchmark = benchmark
 
        self.period_a = period_a
        self.period_b = period_b
        self.w_level = w_level
        self.w_velocity = w_velocity
 
        self.top_fraction = top_fraction
        self.long_quadrants = tuple(long_quadrants)
 
        self.tilt_b = tilt_b
        self.risk_win = risk_win
 
        self.weight_a = weight_a
        self.capital_allocation = capital_allocation
        self.strategy_id = "kalman_rrg_combined"
 
        self.state_path = Path(state_path) if state_path else None

        # enough history for the slowest of: Kalman z-score warm-up, and the risk-parity window
        min_needed = max(risk_win, 3 * max(period_a, period_b)) + 10
        self.lookback_days = max(lookback_days or 0, min_needed, 252)
 
    # ---------------------------------------------------------------- data
    def _fetch_all_bars(self) -> dict[str, pd.Series]:
        """Pull daily bars for every symbol AND the benchmark via the data provider."""
        symbols = list(dict.fromkeys([*self.universe, self.benchmark]))
        return {sym: self._data.get_daily_bars(sym, self.lookback_days) for sym in symbols}
 
    def _price_frame(self, bars: dict[str, pd.Series]) -> pd.DataFrame:
        """Align every series onto a common calendar (inner-ish, forward-filled)."""
        df = pd.DataFrame(bars).sort_index().ffill()
        return df.dropna(how="all")
 
    # ---------------------------------------------------------------- signal
    def compute_signal(self, price_history: dict[str, pd.Series]) -> dict[str, dict]:
        """Latest leadership (x), momentum (y), composite score and RRG quadrant per asset.
 
        Built on relative strength vs the benchmark, so the whole signal is expressed relative to
        the market — the essence of rotation.
        """
        df = self._price_frame(price_history)
        if self.benchmark not in df.columns:
            raise ValueError(f"benchmark {self.benchmark!r} missing from data")
        bench = df[self.benchmark]
 
        signal: dict[str, dict] = {}
        for sym in self.universe:
            if sym not in df.columns:
                continue
            rs = (df[sym] / bench).dropna()
            if len(rs) < 3 * self.period_a:          # not enough history for a trustworthy z-score
                continue
            x = float(_kalman_state_z(rs, "level", self.period_a, q=0.05).iloc[-1])
            y = float(_kalman_state_z(rs, "velocity", self.period_a, q=1.0 / self.period_a ** 2).iloc[-1])
            signal[sym] = {
                "x": x,
                "y": y,
                "score": self.w_level * x + self.w_velocity * y,
                "quadrant": _quadrant(x, y),
            }
        return signal
 
    # ---------------------------------------------------------------- sleeves
    def _sleeve_a_weights(self, signal: dict[str, dict]) -> dict[str, float]:
        """Gated long-leg: top-`top_fraction` by score AND in a bullish quadrant; 1/k each, cash the rest."""
        scored = {s: v["score"] for s, v in signal.items()}
        if len(scored) < 3:
            return {}
        k = max(1, round(len(scored) * self.top_fraction))
        ranked = sorted(scored, key=scored.get, reverse=True)
        weights = {}
        for sym in ranked[:k]:                       # de-risk: an unqualified slot simply stays in cash
            if signal[sym]["quadrant"] in self.long_quadrants:
                weights[sym] = 1.0 / k
        return weights
 
    def _sleeve_b_weights(self, price_history: dict[str, pd.Series], signal: dict[str, dict]) -> dict[str, float]:
        """Naive risk parity (inverse trailing vol), optionally tilted toward the rotation leaders."""
        df = self._price_frame(price_history)
        rets = df[[s for s in self.universe if s in df.columns]].pct_change().iloc[-self.risk_win:]
        rets = rets.dropna(axis=1, how="any")        # only assets with a full risk window
        if rets.shape[1] == 0:
            return {}
        vol = rets.std()
        inv_vol = (1.0 / vol.replace(0.0, np.nan)).dropna()
        base = inv_vol / inv_vol.sum()
 
        if self.tilt_b and signal:                   # optional rotation tilt of the risk-parity base
            valid = [s for s in base.index if s in signal]
            if len(valid) >= 2:
                sc = pd.Series({s: signal[s]["score"] for s in valid})
                z = (sc - sc.mean()) / (sc.std() if sc.std() > 0 else 1.0)
                tilted = base.loc[valid] * np.exp(self.tilt_b * z)
                base = tilted / tilted.sum()
 
        return base.to_dict()
 
    def construct_weights(self, signal: dict[str, dict],
                          price_history: dict[str, pd.Series]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Blend the two long-only sleeves by capital weight."""
        a = self._sleeve_a_weights(signal)
        b = self._sleeve_b_weights(price_history, signal)
        combined = {}
        for sym in self.universe:
            w = self.weight_a * a.get(sym, 0.0) + (1.0 - self.weight_a) * b.get(sym, 0.0)
            if w != 0.0:
                combined[sym] = w
        return combined, a, b
 
    # ---------------------------------------------------------------- state persistence
    def _save_state(self, signal: dict, weights: dict, a_w: dict, b_w: dict) -> None:
        if not self.state_path:
            return
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": signal,
            "combined_weights": {s: round(w, 6) for s, w in weights.items()},
            "sleeve_a_weights": {s: round(w, 6) for s, w in a_w.items()},
            "sleeve_b_weights": {s: round(w, 6) for s, w in b_w.items()},
            "params": {
                "weight_a": self.weight_a,
                "period_a": self.period_a,
                "top_fraction": self.top_fraction,
                "capital_allocation": self.capital_allocation,
            },
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(state, indent=2, default=str))
        except Exception:
            pass  # best-effort — never crash the strategy over state I/O

    def load_state(self) -> Optional[dict]:
        if self.state_path and self.state_path.exists():
            try:
                text = self.state_path.read_text().strip()
                if text:
                    return json.loads(text)
            except Exception:
                pass
        return None

    # ---------------------------------------------------------------- intents
    def generate_intents(self) -> list[dict]:
        """Full rebalance cycle: fetch -> signal -> weights -> target_position intents."""
        bars = self._fetch_all_bars()
        signal = self.compute_signal(bars)
        weights, a_w, b_w = self.construct_weights(signal, bars)
        self._save_state(signal, weights, a_w, b_w)

        ts = datetime.now(timezone.utc).isoformat()
        stamp = int(time.time())
        intents: list[dict] = []
        for symbol in self.universe:                 # emit for the full universe so dropped names get flattened
            if symbol not in bars or bars[symbol].dropna().empty:
                continue
            weight = weights.get(symbol, 0.0)
            current_price = float(bars[symbol].dropna().iloc[-1])
            target_dollars = weight * self.capital_allocation
            target_shares = round(target_dollars / current_price)
            sig = signal.get(symbol, {})
            intents.append({
                "strategy_id": self.strategy_id,
                "client_order_id": f"{self.strategy_id}-{symbol}-{stamp}",
                "timestamp": ts,
                "schema_version": "1.0",
                "instrument": {"symbol": symbol, "asset_class": "equity", "exchange": "SMART"},
                "intent_type": "target_position",
                "target_quantity": target_shares,
                "order_type": "market",
                "expected_price": current_price,
                "time_in_force": "day",
                "metadata": {
                    "blended_weight": round(weight, 6),
                    "sleeve_a_weight": round(a_w.get(symbol, 0.0), 6),
                    "sleeve_b_weight": round(b_w.get(symbol, 0.0), 6),
                    "signal_score": round(sig.get("score", float("nan")), 6) if sig else None,
                    "quadrant": sig.get("quadrant"),
                    "blend_weight_a": self.weight_a,
                },
            })
        return intents
