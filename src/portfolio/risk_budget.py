"""
portfolio/risk_budget.py — how much total risk the book runs, and how far to scale it.

The allocator divides a FIXED total between strategies (its weights sum to one); nothing
asked whether that total is the right size. This answers that question, and only that one.

Book risk by historical replay
------------------------------
Today's positions are replayed over the last year of each holding's actual daily returns:
P&L(t) = sum_i value_i * r_i(t). The spread of that series is the book's daily volatility,
its 1st percentile the 1-day 99% loss. No factor model is involved, so idiosyncratic risk
is counted in full and diversification is whatever the data says it was — unlike the PCA
report, which only sees the ~third of the book's moves the factors explain.

It is the risk of TODAY's book, not of the strategies' track records: a book that turns over
daily is measured as it stands, which is the thing a scale factor would multiply.

The scale is a recommendation, bounded twice
--------------------------------------------
target / current is clamped to [min_scale, max_scale], then to one step of +/- max_step.
Realised volatility jumps in a selloff; an unbounded vol-target cuts risk after the drop and
re-adds it after the rebound, selling pro-rata each time. The step limit makes it move
slowly enough to watch. Nothing here changes an allocation.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def book_risk(values: Dict[str, float], returns: pd.DataFrame, nav: float,
              min_obs: int = 60) -> dict:
    """Daily vol, annualised vol and 1-day 99% historical VaR of the current book.

    `values` is signed $ per symbol (hedge overlay already excluded or not, caller's choice);
    `returns` is dates x symbols of daily returns. A symbol with fewer than `min_obs` days
    is left out and listed, never silently dropped. A missing day for a measured symbol
    (a holiday on one exchange, a halt) counts as a zero return."""
    if nav <= 0:
        raise ValueError(f"NAV must be positive (got {nav})")
    measured, unmeasured = {}, {}
    for s, v in values.items():
        if abs(v) < 1e-9:
            continue
        if s in returns.columns and returns[s].notna().sum() >= min_obs:
            measured[s] = v
        else:
            unmeasured[s] = v
    base = {"nav": nav, "measured_notional": sum(abs(v) for v in measured.values()),
            "unmeasured_notional": sum(abs(v) for v in unmeasured.values()),
            "unmeasured_symbols": sorted(unmeasured)}
    if not measured:
        return {**base, "days": 0, "daily_vol": None, "annual_vol": None,
                "var99": None, "worst_day": None}

    r = returns[list(measured)]
    r = r[r.notna().any(axis=1)].fillna(0.0)
    pnl = r.values @ np.array(list(measured.values()))
    sd = float(np.std(pnl, ddof=1))
    var99 = float(-np.percentile(pnl, 1))
    return {**base, "days": int(len(pnl)),
            "daily_vol": sd / nav,                          # share of NAV per day
            "annual_vol": sd / nav * math.sqrt(TRADING_DAYS),
            "var99": var99 / nav,                           # 1-day 99% loss, share of NAV
            "worst_day": float(-pnl.min()) / nav}


def recommend_scale(annual_vol: Optional[float], target_vol: float, min_scale: float = 0.5,
                    max_scale: float = 2.0, max_step: float = 0.25) -> dict:
    """How far to scale every allocation to bring the book to `target_vol` (annualised).

    Assumes risk scales with allocation — true for a strategy that sizes off its
    allocation, false for one that trades a fixed share count. Check before relying on it."""
    if not (0 < min_scale <= 1 <= max_scale) or not (0 < max_step < 1) or target_vol <= 0:
        raise ValueError("need 0 < min_scale <= 1 <= max_scale, 0 < max_step < 1, target_vol > 0")
    if not annual_vol or annual_vol <= 0:
        return {"raw": None, "capped": None, "step": 1.0,
                "reason": "book risk not measurable — hold"}
    raw = target_vol / annual_vol
    capped = min(max(raw, min_scale), max_scale)
    step = min(max(capped, 1 - max_step), 1 + max_step)
    if abs(step - 1.0) < 0.02:
        reason = f"book at {annual_vol:.1%}/yr vs target {target_vol:.1%} — hold"
    else:
        reason = (f"book at {annual_vol:.1%}/yr vs target {target_vol:.1%} — "
                  f"{'raise' if step > 1 else 'cut'} allocations x{step:.2f} this step"
                  + (f" (full move x{capped:.2f}" + (", capped" if capped != raw else "") + ")"
                     if step != capped else ""))
    return {"raw": raw, "capped": capped, "step": step, "reason": reason}
