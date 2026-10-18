"""
portfolio/minvar_hedge.py — spend the overlay's capital where it removes the most variance.

The hedge overlay's capital_allocation is a gross-notional budget B. This finds the ETF
positions h (dollars, long or short) that minimise the variance of the book's P&L:

    min_h  Var(p + R h)     subject to   sum_j |h_j| <= B

where p is TODAY's book replayed over the last year of each holding's returns (the same
series portfolio/risk_budget.py reports) and R the daily returns of the hedge universe.
That is a lasso regression of -p on R with the L1 norm pinned to the budget — solved
exactly on the LARS path, so the constraint is met precisely rather than approximately.

Why this shape
--------------
  * The budget IS the knob. More capital buys more variance reduction, with diminishing
    returns; the report says how much each dollar bought.
  * The L1 constraint is sparse: with a tight budget it picks the one or two ETFs that cover
    the most variance, rather than smearing small legs across all of them.
  * It never spends past the minimum-variance hedge. If the unconstrained least-squares
    hedge fits inside the budget, that is the answer — more capital would ADD variance.
  * It is measured against everything, idiosyncratic included. What it cannot remove (the
    stock-specific part no ETF tracks) is reported, not hidden.

Unlike the bucket and PCA models this hedges continuously: whatever the book holds, the
budget is spent cutting its variance, including exposure the strategies took on purpose.
That is the trade-off of sizing from capital rather than from a threshold — the allocation
is how you choose how much of it to give back.

Churn control: a held hedge is kept unless re-solving would cut variance by at least
`min_improvement` (share of the book's variance) more than the hedge already held, or the
held hedge no longer fits the budget; and a leg moving by less than `min_trade_notional`
keeps its quantity.

Nothing here places an order. `plan()` returns the book it would submit.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import lars_path

logger = logging.getLogger("minvar_hedge")


def solve(p: np.ndarray, R: np.ndarray, budget: float) -> tuple:
    """Dollar hedge h minimising Var(p + R h) with ||h||_1 <= budget.
    Inputs are demeaned here. Returns (h, binding) — `binding` False when the unconstrained
    minimum-variance hedge already fits inside the budget."""
    p = p - p.mean()
    R = R - R.mean(axis=0)
    if budget <= 0 or not np.any(p):
        return np.zeros(R.shape[1]), False
    h_ols, *_ = np.linalg.lstsq(R, -p, rcond=None)
    if np.abs(h_ols).sum() <= budget:
        return h_ols, False
    # LARS-lasso path: coefficients are piecewise linear in the knots, and so is their L1
    # norm, so the point where ||h||_1 == budget is found exactly by interpolation.
    _, _, coefs = lars_path(R, -p, method="lasso")
    norms = np.abs(coefs).sum(axis=0)
    i = int(np.searchsorted(norms, budget))
    if i >= coefs.shape[1]:
        return coefs[:, -1], True
    if i == 0:
        return np.zeros(R.shape[1]), True
    a, b = norms[i - 1], norms[i]
    t = 0.0 if b <= a else (budget - a) / (b - a)
    return coefs[:, i - 1] + t * (coefs[:, i] - coefs[:, i - 1]), True


@dataclass
class MinVarPlan:
    universe: list
    dollars: Dict[str, float]               # target $ per ETF, after rounding to shares
    legs: Dict[str, float]                  # ETF -> signed whole shares
    prices: Dict[str, float]
    nav: float
    budget: float
    binding: bool                           # the budget, not the optimum, set the size
    held_kept: bool                         # churn control kept the existing hedge
    vol_before: float                       # daily, share of NAV
    vol_after: float
    vol_with_held: float
    days: int
    reason: str
    current_hedge: Dict[str, float] = field(default_factory=dict)

    @property
    def variance_removed(self) -> float:
        if self.vol_before <= 0:
            return 0.0
        return 1.0 - (self.vol_after / self.vol_before) ** 2

    @property
    def gross(self) -> float:
        return sum(abs(q) * self.prices.get(s, 0.0) for s, q in self.legs.items())

    @property
    def book(self) -> list:
        """Absolute overlay book, /targets shape. A held name no longer wanted is stated as
        an explicit zero, so a hedge from another model is unwound, never orphaned."""
        syms = {s for s, q in self.legs.items() if abs(q) >= 1} | set(self.current_hedge)
        return [{"instrument": {"symbol": s, "asset_class": "equity",
                                "sec_type": "STK", "exchange": "SMART"},
                 "target_quantity": float(self.legs.get(s, 0.0)),
                 "expected_price": self.prices.get(s)}
                for s in sorted(syms) if self.prices.get(s)]


def plan(book_pnl: pd.Series, etf_returns: pd.DataFrame, budget: float, nav: float,
         prices: Dict[str, float], current_hedge: Dict[str, float] = None,
         min_improvement: float = 0.05, min_trade_notional: float = 2_500.0,
         min_obs: int = 120) -> MinVarPlan:
    """Decide the overlay. `book_pnl` is the daily $ P&L of today's book EXCLUDING the
    overlay (the hedge must never measure itself); `etf_returns` the hedge universe."""
    if nav <= 0:
        raise ValueError(f"NAV must be positive (got {nav})")
    current_hedge = {s: q for s, q in (current_hedge or {}).items() if abs(q) > 1e-9}
    df = pd.concat([book_pnl.rename("__book__"), etf_returns], axis=1, join="inner")
    df = df.dropna(axis=1, thresh=min_obs).dropna()
    universe = [c for c in df.columns if c != "__book__"]
    if "__book__" not in df.columns or len(df) < min_obs or not universe:
        raise ValueError(f"need {min_obs}+ overlapping days of book and ETF history, "
                         f"got {len(df)}")
    p, R = df["__book__"].values, df[universe].values
    priced = [s for s in universe if (prices.get(s) or 0) > 0]
    if len(priced) < len(universe):
        logger.error("no price for %s — excluded from the hedge universe",
                     sorted(set(universe) - set(priced)))
        R = R[:, [universe.index(s) for s in priced]]
        universe = priced

    def vol(h):
        return float(np.std(p + R @ h, ddof=1)) / nav

    held_h = np.array([current_hedge.get(s, 0.0) * prices[s] for s in universe])
    held_gross = sum(abs(q) * (prices.get(s) or 0.0) for s, q in current_hedge.items())
    h, binding = solve(p, R, budget)
    v0, v_new, v_held = vol(np.zeros(len(universe))), vol(h), vol(held_h)

    kept = False
    if current_hedge and held_gross <= budget * 1.0001 and v0 > 0:
        gain = (v_held ** 2 - v_new ** 2) / v0 ** 2
        if gain < min_improvement:
            h, kept = held_h, True
            reason = (f"held hedge is within {gain:.1%} of the best variance cut "
                      f"(threshold {min_improvement:.0%}) — holding")
    if not kept:
        reason = ("budget-bound: every dollar of the allocation is spent" if binding else
                  "full minimum-variance hedge fits inside the budget")

    legs, dollars = {}, {}
    for j, sym in enumerate(universe):
        px, now = prices[sym], float(current_hedge.get(sym, 0.0))
        if kept:
            q = now
        else:
            q = math.copysign(math.floor(abs(h[j]) / px), h[j]) if abs(h[j]) >= px else 0.0
            if now and q and abs(q - now) * px < min_trade_notional:
                q = now
        if abs(q) >= 1:
            legs[sym] = q
            dollars[sym] = q * px
    # rounding can only shrink a leg, but the min-trade hold can keep an old, larger one
    while not kept and sum(abs(d) for d in dollars.values()) > budget and dollars:
        worst = max(dollars, key=lambda s: abs(dollars[s]))
        q = legs[worst] - math.copysign(1, legs[worst])
        legs[worst], dollars[worst] = q, q * prices[worst]
        if abs(q) < 1:
            legs.pop(worst), dollars.pop(worst)

    final = np.array([dollars.get(s, 0.0) for s in universe])
    return MinVarPlan(universe=universe, dollars=dollars, legs=legs,
                      prices={s: prices.get(s) for s in set(universe) | set(current_hedge)},
                      nav=nav, budget=budget, binding=binding, held_kept=kept,
                      vol_before=v0, vol_after=vol(final), vol_with_held=v_held,
                      days=int(len(df)), reason=reason, current_hedge=current_hedge)
