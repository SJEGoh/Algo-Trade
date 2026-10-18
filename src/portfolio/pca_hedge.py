"""
portfolio/pca_hedge.py — statistical factor hedge, knowing nothing about the strategies.

The book is treated as a black box: whatever is held, by whichever strategy, for whatever
reason. Its risk is measured against a handful of statistical factors, and only a factor
whose contribution to the book's risk has grown abnormally large is shaved back.

Why PCA on a FIXED ETF universe, not on the holdings
----------------------------------------------------
PCA on the holdings gives factors made of the holdings themselves: untradable, re-estimated
from a book that turns over daily (so they rotate and flip sign), and noisy (~100 names, a
year of data). PCA on ~15 liquid ETFs gives factors that are themselves ETF portfolios — so
a hedge is directly tradable — and a basis that only drifts slowly. Each holding is then
regressed on the factor returns, so a name no one has seen before is measured automatically.
As the strategy mix changes (small-cap today, mega-cap next quarter) the loadings change and
the hedge mix with them; nothing is hard-coded to a style.

The algebra that makes the hedge exact
--------------------------------------
Factor k's return is f_k = w_k . r, where w_k is the k-th covariance eigenvector scaled to
unit gross. Holding $H of the w_k portfolio has loading exactly H on factor k and zero on
every other factor (eigenvectors are orthogonal). So "short $H of factor k" is "short
H * w_k[j] dollars of ETF j", and it moves only the exposure it is aimed at.

The gate is RISK, not exposure, and not variance-explained
----------------------------------------------------------
A factor is judged by its contribution to the book's daily volatility, |E_k| * sigma_k as a
share of NAV. PC2 is a quieter factor than PC1, so a PC2 bet only trips the gate when it is
genuinely large. Gating on "PC1 explains < 60% of the universe" was considered and rejected:
that number describes the market regime (it rises in a selloff), not this book — it would
skip PC2 in exactly the stressed markets where a large PC2 bet hurts. It is reported, never
used to decide.

Nothing here places an order or touches IB. `plan()` returns the book it would submit.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from portfolio.hedger import BucketPolicy, HEDGE_STRATEGY_ID

logger = logging.getLogger("pca_hedge")

#: The factor universe. Broad enough to span equity style (size, growth), the sectors, and
#: the non-equity sleeves; small and liquid enough that every factor is cheap to trade.
FACTOR_UNIVERSE = ["SPY", "IWM", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
                   "XLP", "XLU", "XLRE", "XLB", "TLT", "GLD", "DBC"]

#: Hedge thresholds as daily-volatility contribution, share of NAV. 0.6% of NAV a day is
#: roughly a 60%-of-NAV market position at ~1% daily market vol.
DEFAULT_POLICY = BucketPolicy(trigger=0.006, target=0.005, release=0.004)


# ---------------------------------------------------------------------------
# Daily price history, cached a day at a time
# ---------------------------------------------------------------------------
class DailyHistoryCache:
    """{symbol: pd.Series of daily closes}, refreshed at most once per UTC day per symbol and
    persisted to JSON, so a restart does not re-pull a year of bars for every holding.

    `fetch(symbol) -> {'YYYYMMDD': close} | None` is the IB call. Fetches run a few at a time:
    daily bars are not under IB's small-bar pacing limits, but a burst of ~130 requests still
    gets soft-throttled."""

    def __init__(self, fetch: Callable[[str], Optional[Dict[str, float]]],
                 path: Optional[Path] = None, workers: int = 4):
        self._fetch, self._path, self._workers = fetch, Path(path) if path else None, workers
        self._lock = threading.Lock()
        self._data: Dict[str, dict] = {}       # symbol -> {"day": "YYYY-MM-DD", "closes": {...}}
        if self._path and self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
            except Exception as e:
                logger.warning("price history cache unreadable (%s) — starting empty", e)

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def get(self, symbols) -> Dict[str, pd.Series]:
        today = self._today()
        symbols = sorted(set(symbols))
        with self._lock:
            stale = [s for s in symbols if (self._data.get(s) or {}).get("day") != today]
        if stale:
            with ThreadPoolExecutor(max_workers=self._workers) as pool:
                fetched = dict(zip(stale, pool.map(self._fetch_one, stale)))
            with self._lock:
                for s, closes in fetched.items():
                    if closes:
                        self._data[s] = {"day": today, "closes": closes}
                self._save()
        out = {}
        with self._lock:
            for s in symbols:
                closes = (self._data.get(s) or {}).get("closes")
                if closes:   # a failed refresh falls back to yesterday's history
                    ser = pd.Series(closes, dtype=float)
                    ser.index = pd.to_datetime(ser.index, format="%Y%m%d")
                    out[s] = ser.sort_index()
        return out

    def _fetch_one(self, symbol):
        try:
            return self._fetch(symbol)
        except Exception as e:
            logger.warning("history fetch failed for %s: %s", symbol, e)
            return None

    def _save(self):
        if not self._path:
            return
        try:
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data))
            tmp.replace(self._path)
        except Exception as e:
            logger.warning("price history cache not saved: %s", e)


def daily_returns(closes: Dict[str, pd.Series], lookback: int = 250) -> pd.DataFrame:
    """Simple daily returns, dates x symbols, last `lookback` days. Gaps stay NaN."""
    if not closes:
        return pd.DataFrame()
    px = pd.DataFrame(closes).sort_index()
    return px.pct_change(fill_method=None).iloc[1:].tail(lookback)


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------
@dataclass
class FactorModel:
    universe: List[str]
    weights: np.ndarray            # (k, m) — each row unit gross, a tradable ETF portfolio
    sigma: np.ndarray              # (k,) daily vol of each factor's return
    explained: np.ndarray          # (k,) share of universe variance
    returns: pd.DataFrame          # dates x ["PC1", ...]

    @property
    def names(self):
        return [f"PC{i + 1}" for i in range(len(self.sigma))]


def fit_factors(etf_returns: pd.DataFrame, n_components: int = 3,
                prev_weights: Optional[np.ndarray] = None, min_obs: int = 120) -> FactorModel:
    """PCA of the ETF covariance (not correlation: the weights must be dollar weights).

    Signs are pinned so the hedge does not flip when nothing has changed: aligned to
    `prev_weights` when given, else PC1 net long (it is the market) and every other PC with
    its largest weight positive."""
    r = etf_returns.dropna(axis=1, thresh=min_obs).dropna()
    if len(r) < min_obs or r.shape[1] < n_components:
        raise ValueError(f"need {min_obs}+ complete days across {n_components}+ ETFs, "
                         f"got {len(r)} days x {r.shape[1]} ETFs")
    cov = np.cov(r.values, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    k = n_components

    W = []
    for i in range(k):
        v = vecs[:, i]
        if prev_weights is not None and i < len(prev_weights) and len(prev_weights[i]) == len(v):
            v = v if np.dot(v, prev_weights[i]) >= 0 else -v
        elif i == 0:
            v = v if v.sum() >= 0 else -v
        else:
            v = v if v[np.argmax(np.abs(v))] >= 0 else -v
        W.append(v / np.abs(v).sum())
    W = np.array(W)
    fr = pd.DataFrame(r.values @ W.T, index=r.index, columns=[f"PC{i + 1}" for i in range(k)])
    return FactorModel(universe=list(r.columns), weights=W, sigma=fr.std(ddof=1).values,
                       explained=vals[:k] / vals.sum(), returns=fr)


def loadings(holding_returns: pd.DataFrame, factor_returns: pd.DataFrame,
             min_obs: int = 60) -> tuple:
    """OLS of each holding's daily return on the factor returns (with intercept).
    Returns ({symbol: (betas array, r2)}, [symbols with too little overlapping history])."""
    out, short = {}, []
    for sym in holding_returns.columns:
        df = pd.concat([holding_returns[sym], factor_returns], axis=1, join="inner").dropna()
        if len(df) < min_obs:
            short.append(sym)
            continue
        y = df.iloc[:, 0].values
        X = np.column_stack([np.ones(len(df)), df.iloc[:, 1:].values])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        tss = ((y - y.mean()) ** 2).sum()
        out[sym] = (coef[1:], 1.0 - (resid ** 2).sum() / tss if tss > 0 else 0.0)
    return out, short


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
@dataclass
class FactorDecision:
    factor: str
    exposure: float               # signed $ of the unit-gross factor portfolio
    fraction: float               # exposure / NAV
    vol_contribution: float       # |exposure| * sigma / NAV — what the gate compares
    hedge_notional: float         # signed $ of the factor portfolio to add (opposite sign)
    reason: str
    top_contributors: Dict[str, float] = field(default_factory=dict)


@dataclass
class PCAPlan:
    model: FactorModel
    decisions: List[FactorDecision]
    legs: Dict[str, float]                  # ETF -> signed whole shares
    prices: Dict[str, float]
    nav: float
    unmeasured: Dict[str, float]            # symbol -> $ value held with no loading
    r2: Dict[str, float]
    current_hedge: Dict[str, float] = field(default_factory=dict)
    r2_weights: Dict[str, float] = field(default_factory=dict)   # measured symbol -> $ value

    @property
    def book(self) -> list:
        """The desired overlay book in the /targets shape. Absolute, like the bucket hedger:
        a held ETF no longer wanted is stated as an explicit zero."""
        syms = {s for s, q in self.legs.items() if abs(q) >= 1} | set(self.current_hedge)
        return [{"instrument": {"symbol": s, "asset_class": "equity",
                                "sec_type": "STK", "exchange": "SMART"},
                 "target_quantity": float(self.legs.get(s, 0.0)),
                 "expected_price": self.prices.get(s)}
                for s in sorted(syms) if self.prices.get(s)]

    @property
    def coverage(self) -> dict:
        measured = sum(abs(v) for v in self.r2_weights.values())
        return {"measured_notional": measured,
                "unmeasured_notional": sum(abs(v) for v in self.unmeasured.values()),
                "unmeasured_symbols": sorted(self.unmeasured),
                # value-weighted R^2: how much of the book's daily moves the factors explain
                "factor_r2": (sum(abs(v) * self.r2[s] for s, v in self.r2_weights.items())
                              / measured) if measured else None}


def plan(model: FactorModel, position_values: Dict[str, float],
         holding_loadings: Dict[str, tuple], nav: float, prices: Dict[str, float],
         policy: BucketPolicy = DEFAULT_POLICY, hedge_components: int = 2,
         current_hedge: Dict[str, float] = None, min_hedge_notional: float = 1_000.0,
         unmeasured: Dict[str, float] = None) -> PCAPlan:
    """Decide the overlay for the book. `position_values` is signed $ per symbol over every
    strategy EXCEPT the overlay (the hedge must never measure itself). Only the first
    `hedge_components` factors may act; the rest are reported."""
    if nav <= 0:
        raise ValueError(f"NAV must be positive (got {nav})")
    current_hedge = {s: q for s, q in (current_hedge or {}).items() if abs(q) > 1e-9}
    unmeasured = dict(unmeasured or {})
    k = len(model.sigma)

    exposure = np.zeros(k)
    contrib: Dict[str, np.ndarray] = {}
    weights_used = {}
    for sym, value in position_values.items():
        if abs(value) < 1e-9:
            continue
        if sym not in holding_loadings:
            unmeasured[sym] = value
            continue
        betas, _ = holding_loadings[sym]
        contrib[sym] = value * betas
        exposure += contrib[sym]
        weights_used[sym] = value

    # Hysteresis state is read off the overlay's own holdings, as in the bucket hedger: a
    # factor counts as hedged if the overlay holds anything at all in the factor universe.
    hedged = any(s in model.universe for s in current_hedge)

    decisions, dollars = [], np.zeros(len(model.universe))
    for i in range(k):
        name = model.names[i]
        e, sig = exposure[i], model.sigma[i]
        vol = abs(e) * sig / nav
        top = dict(sorted(((s, float(c[i])) for s, c in contrib.items()),
                          key=lambda kv: -abs(kv[1]))[:5])
        base = dict(factor=name, exposure=float(e), fraction=float(e / nav),
                    vol_contribution=float(vol), top_contributors=top)
        if i >= hedge_components:
            decisions.append(FactorDecision(**base, hedge_notional=0.0,
                                            reason="reported only — beyond the hedged factors"))
            continue
        threshold = policy.release if hedged else policy.trigger
        label = "release" if hedged else "trigger"
        if vol <= threshold:
            decisions.append(FactorDecision(**base, hedge_notional=0.0,
                                            reason=f"{vol:.2%}/day at or below {label} {threshold:.2%}"))
            continue
        # shave to target: keep |E| * sigma = target * NAV
        keep = policy.target * nav / sig
        h = -math.copysign(abs(e) - keep, e)
        if abs(h) < min_hedge_notional:
            decisions.append(FactorDecision(**base, hedge_notional=0.0,
                                            reason=f"excess {abs(h):,.0f} below the {min_hedge_notional:,.0f} floor"))
            continue
        dollars += h * model.weights[i]
        decisions.append(FactorDecision(**base, hedge_notional=float(h),
                                        reason=f"{vol:.2%}/day over {label} {threshold:.2%} — "
                                               f"shaving to {policy.target:.2%}"))

    legs = {}
    for j, sym in enumerate(model.universe):
        px = prices.get(sym)
        if abs(dollars[j]) < 1e-9:
            continue
        if not px or not math.isfinite(px) or px <= 0:
            logger.error("no price for factor ETF %s — its %+,.0f leg is dropped", sym, dollars[j])
            continue
        q = math.copysign(math.floor(abs(dollars[j]) / px), dollars[j])
        if abs(q) >= 1:
            legs[sym] = q

    return PCAPlan(model=model, decisions=decisions, legs=legs, prices=dict(prices), nav=nav,
                   unmeasured=unmeasured,
                   r2={s: holding_loadings[s][1] for s in weights_used},
                   current_hedge=current_hedge, r2_weights=weights_used)


def position_values(strategy_positions: Dict[str, Dict[str, float]], marks: Dict[str, float],
                    multipliers: Dict[str, float] = None, exclude=frozenset({HEDGE_STRATEGY_ID})
                    ) -> tuple:
    """Signed $ per symbol netted across strategies (overlay excluded), and the unpriced."""
    multipliers = multipliers or {}
    qty: Dict[str, float] = {}
    for sid, pos in strategy_positions.items():
        if sid in exclude:
            continue
        for s, q in pos.items():
            if abs(q) > 1e-9:
                qty[s] = qty.get(s, 0.0) + q
    values, unpriced = {}, {}
    for s, q in qty.items():
        px = marks.get(s)
        if px is None or not math.isfinite(px) or px <= 0:
            unpriced[s] = q
            continue
        values[s] = q * px * float(multipliers.get(s, 1.0) or 1.0)
    return values, unpriced
