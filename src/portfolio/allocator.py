"""
portfolio/allocator.py — how much capital each strategy should hold.

Mean-variance allocation across strategy return streams, with the two things that decide
whether it helps or hurts: a shrunk covariance, and a refusal to optimise on data that
cannot support it.

Why the guard rails are the important part
------------------------------------------
The sample covariance is badly conditioned when the number of observations is not much
larger than the number of strategies, and the *sample mean* is the worst-estimated input in
finance. A max-Sharpe optimiser fed noisy means does not average those errors out — it
systematically loads on whichever strategy's return was most OVER-estimated, because that is
what "highest Sharpe" selects for. Michaud called it error maximisation, and it is why
naive mean-variance so often loses to equal weight out of sample (DeMiguel, Garlappi &
Uppal, 2009).

Two defences here, in order of importance:

  1. `min_days_for_means` — REFUSE to use expected returns until there is enough history,
     and fall back to a method that does not need them. This matters more than any penalty
     term. Note it counts CALENDAR SPAN, not rows: the equity sampler writes a snapshot a
     minute, so a week of trading is ~10,000 rows. That looks like a large sample and is
     not one — resampling more finely tightens a covariance estimate slightly and tells you
     nothing extra about a mean.

  2. `max_weight` and `l2_lambda` — a hard cap per strategy, and a pull toward equal
     weight. These are what stop one lucky week dominating the book.

     An L1 (LASSO) penalty is the wrong tool for that. L1 induces SPARSITY: it drives
     weights to zero and concentrates the book into fewer strategies, the opposite of
     "don't let one strategy dominate". It is also literally inert here — under
     `w >= 0` and `sum(w) == 1`, ||w||_1 == 1 for every feasible w, so the penalty is a
     constant and cannot change the solution. L2 toward equal weight spreads; a box
     constraint caps. Those are the two that do what is wanted.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from portfolio.hedger import HEDGE_STRATEGY_ID

logger = logging.getLogger("allocator")

#: Strategy ids whose capital is not an investment decision, so they must never be sized by
#: the optimiser.
#:
#: The hedge overlay is the important one. Its expected return is DESIGNED to be negative —
#: it is insurance — and it is negatively correlated with the rest of the book by
#: construction. Those confuse a mean-variance objective in opposite and equally wrong
#: directions: the Sharpe term wants to SHORT it (i.e. to un-hedge), while the variance term
#: loves the negative correlation and would pile capital into it. Its size is set by the
#: exposure it must offset, not by anything an optimiser can see.
#:
#: The rest are ledger bookkeeping buckets, not strategies. Duplicated from
#: CentralExecutor.INTERNAL_SIDS on purpose — importing that here would pull ibapi into a
#: module which otherwise needs only numpy and sklearn.
NON_ALLOCATABLE = frozenset({HEDGE_STRATEGY_ID, "__net__", "flatten_all", "kill_switch"})

TRADING_DAYS = 252
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Return series
# ---------------------------------------------------------------------------
def strategy_returns(logger_db, strategy_ids: List[str] = None, since: str = None,
                     freq: str = "1D") -> pd.DataFrame:
    """Per-strategy periodic returns, from the equity sampler's snapshots.

    Returns are computed on NAV (cash + position value), not on P&L, so a strategy holding
    mostly cash correctly shows a small return on its allocated capital rather than a wild
    one on a small base.

    Resampled to `freq` (daily by default) using the LAST snapshot in each bucket. The
    sampler writes roughly once a minute; using those raw ticks would inflate the apparent
    sample size without adding independent information, which is exactly the illusion the
    guard rails downstream exist to prevent.
    """
    rows = logger_db.get_equity_history(since=since)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if strategy_ids:
        df = df[df["strategy_id"].isin(strategy_ids)]
    if df.empty:
        return pd.DataFrame()

    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="mixed")
    # `nav` arrived in a later migration; older rows only carry realized+unrealized.
    basis = "nav" if "nav" in df.columns and df["nav"].notna().any() else "equity"
    if basis == "equity":
        logger.warning("no NAV column in equity_snapshots — falling back to P&L-based "
                       "returns, which are not comparable across strategies of different size")

    wide = (df.pivot_table(index="ts", columns="strategy_id", values=basis, aggfunc="last")
              .sort_index()
              .resample(freq).last()
              .dropna(how="all"))
    return wide.pct_change().dropna(how="all")


def history_days(returns: pd.DataFrame) -> pd.Series:
    """Calendar days between each strategy's FIRST and LAST actual observation.

    Per strategy, not per frame: a book that has run for a year plus a strategy added on
    Tuesday has one year of history in aggregate and two days for the newcomer. Judging the
    newcomer on the frame's span is how a strategy with three good days gets sized as though
    it had proven itself.
    """
    out = {}
    for sid in returns.columns:
        valid = returns[sid].dropna()
        if valid.empty:
            out[sid] = 0.0
            continue
        out[sid] = (valid.index.max() - valid.index.min()).total_seconds() / 86400.0
    return pd.Series(out)


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------
def shrunk_covariance(returns: pd.DataFrame, annualise: bool = True,
                      periods_per_year: float = TRADING_DAYS) -> tuple:
    """Ledoit-Wolf shrunk covariance. Returns (covariance DataFrame, shrinkage coefficient).

    Shrinks the sample covariance toward a scaled identity, by an amount the estimator
    chooses analytically to minimise expected squared error. The sample covariance of a
    handful of strategies over a short window is nearly singular, and inverting it — which
    any mean-variance optimiser does implicitly — amplifies exactly the directions estimated
    worst. Shrinkage is the cheap, standard fix.

    A shrinkage coefficient near 1 is the estimator telling you your sample covariance
    carries almost no usable information. Treat that as a warning, not a detail.
    """
    if returns.empty or returns.shape[1] == 0:
        raise ValueError("no returns to estimate a covariance from")

    clean = returns.dropna(axis=1, how="all").fillna(0.0)
    lw = LedoitWolf().fit(clean.values)
    cov = pd.DataFrame(lw.covariance_, index=clean.columns, columns=clean.columns)
    if annualise:
        # Must match the sampling frequency of `returns`. Scaling a 15-minute series by 252
        # understates the annualised covariance ~26x, which would make every Sharpe number
        # downstream quietly wrong rather than obviously wrong.
        cov *= periods_per_year
    return cov, float(lw.shrinkage_)


# ---------------------------------------------------------------------------
# Weight schemes
# ---------------------------------------------------------------------------
def equal_weights(names) -> pd.Series:
    names = list(names)
    return pd.Series(1.0 / len(names), index=names)


def inverse_vol_weights(cov: pd.DataFrame, max_weight: float = 1.0) -> pd.Series:
    """Weight by 1/volatility. Needs no expected returns, which is the point: volatility
    estimates from a short window are far more trustworthy than mean estimates, so this
    degrades gracefully where max-Sharpe degrades catastrophically."""
    vol = np.sqrt(np.diag(cov.values))
    vol = np.where(vol < _EPS, np.inf, vol)          # a zero-vol strategy gets no weight
    w = 1.0 / vol
    if not np.isfinite(w).any() or w.sum() < _EPS:
        return equal_weights(cov.index)
    return _apply_cap(pd.Series(w / w.sum(), index=cov.index), max_weight)


def max_sharpe_weights(mu: pd.Series, cov: pd.DataFrame, max_weight: float = 0.40,
                       l2_lambda: float = 0.0, risk_free: float = 0.0,
                       restarts: int = 8, seed: int = 0) -> pd.Series:
    """Long-only max-Sharpe, capped per strategy, optionally pulled toward equal weight.

    `max_weight` is the hard cap and `l2_lambda` the soft pull: the objective adds
    `l2_lambda * ||w - 1/N||^2`, so raising it moves the answer continuously from
    unconstrained max-Sharpe toward equal weight. Both exist because the ranking that the
    Sharpe objective maximises is estimated from noise, and neither a cap nor a pull needs
    to be right to be protective — they only need to bound how wrong the optimiser can get.

    Solved with SLSQP from several starting points; the constrained Sharpe surface is not
    guaranteed convex, and restarts are cheaper than a bad local optimum.
    """
    names = list(mu.index)
    n = len(names)
    if n == 0:
        raise ValueError("no strategies to allocate between")
    if n == 1:
        return pd.Series([1.0], index=names)
    if max_weight * n < 1.0 - 1e-9:
        raise ValueError(f"max_weight={max_weight} cannot fill a book of {n} strategies "
                         f"(need at least {1.0 / n:.3f})")

    mu_v, cov_v = mu.values.astype(float), cov.values.astype(float)
    eq = np.full(n, 1.0 / n)

    def negative_sharpe(w):
        excess = float(w @ mu_v) - risk_free
        vol = float(np.sqrt(max(w @ cov_v @ w, _EPS)))
        penalty = l2_lambda * float(((w - eq) ** 2).sum())
        return -(excess / vol) + penalty

    constraints = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, max_weight)] * n

    rng = np.random.default_rng(seed)
    best, best_val = None, np.inf
    starts = [eq] + [rng.dirichlet(np.ones(n)) for _ in range(max(0, restarts - 1))]
    for w0 in starts:
        w0 = np.clip(w0, 0.0, max_weight)
        w0 = w0 / w0.sum() if w0.sum() > _EPS else eq
        res = minimize(negative_sharpe, w0, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 500, "ftol": 1e-10})
        if res.success and res.fun < best_val:
            best, best_val = res.x, res.fun

    if best is None:
        logger.warning("max-Sharpe optimisation did not converge — falling back to "
                       "inverse-volatility, which needs no expected returns")
        return inverse_vol_weights(cov, max_weight)

    w = np.clip(best, 0.0, max_weight)
    return pd.Series(w / w.sum(), index=names)


def _apply_cap(w: pd.Series, max_weight: float) -> pd.Series:
    """Clip to the cap and redistribute the excess over the uncapped names, repeatedly —
    one pass is not enough, because redistributing can push another name over the cap."""
    w = w.clip(lower=0.0)
    if w.sum() < _EPS:
        return equal_weights(w.index)
    w = w / w.sum()
    for _ in range(100):
        over = w > max_weight + 1e-12
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        room = ~over
        if not room.any():
            break
        w[room] += excess * (w[room] / w[room].sum() if w[room].sum() > _EPS
                             else 1.0 / room.sum())
    return w / w.sum()


# ---------------------------------------------------------------------------
# The thing you actually call
# ---------------------------------------------------------------------------
@dataclass
class Allocation:
    #: seasoned strategies the optimiser actually sized
    weights: pd.Series
    method: str                      # what was actually used
    reason: str                      # why that method and not another
    #: strategy -> weight held unchanged because it is too young to judge
    exempt: Dict[str, float] = field(default_factory=dict)
    #: strategy -> weight held unchanged because it is not an investment decision at all
    #: (the hedge overlay, ledger buckets). Unlike `exempt`, never rejoins the optimisation.
    reserved: Dict[str, float] = field(default_factory=dict)
    shrinkage: Optional[float] = None
    days: float = 0.0
    diagnostics: Dict = field(default_factory=dict)

    @property
    def all_weights(self) -> pd.Series:
        """Optimised and exempt together — the whole book, summing to 1."""
        combined = dict(self.weights)
        combined.update(self.exempt)
        combined.update(self.reserved)
        return pd.Series(combined)

    def capital(self, total: float) -> Dict[str, float]:
        return {sid: float(total * w) for sid, w in self.all_weights.items()}

    def __str__(self):
        lines = [f"{self.method} — {self.reason}",
                 f"  history: {self.days:.1f} days"
                 + (f", shrinkage {self.shrinkage:.2f}" if self.shrinkage is not None else "")]
        for sid, w in self.weights.sort_values(ascending=False).items():
            lines.append(f"  {sid:28} {w:7.2%}")
        for sid, w in sorted(self.exempt.items(), key=lambda kv: -kv[1]):
            age = (self.diagnostics.get("history_days") or {}).get(sid)
            lines.append(f"  {sid:28} {w:7.2%}  (exempt"
                         + (f", {age:.1f}d old)" if age is not None else ")"))
        for sid, w in sorted(self.reserved.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {sid:28} {w:7.2%}  (reserved — not an allocation decision)")
        return "\n".join(lines)


def recommend(returns: pd.DataFrame, current_weights: Dict[str, float] = None,
              exclude: set = None, min_history_days: float = 14.0, max_weight: float = 0.40,
              l2_lambda: float = 0.5, min_days_for_means: float = 90.0,
              min_days_for_cov: float = 20.0, min_coverage: float = 0.5,
              periods_per_year: float = TRADING_DAYS) -> Allocation:
    """Pick weights, holding young strategies out of the optimisation entirely.

    Exemption (`min_history_days`, two weeks by default)
    ---------------------------------------------------
    A strategy with less than `min_history_days` of live history is EXEMPT: its weight is
    held at whatever it has now and the optimiser sizes the rest of the book around it.

    This is not politeness toward new strategies, it is protection from them. The first two
    weeks are when a strategy is most likely to look extraordinary by chance — a handful of
    observations, no drawdown yet, an annualised Sharpe computed from noise. Feeding that
    into a max-Sharpe objective is the fastest way to hand the book to whichever strategy
    just got lucky. Holding it out costs nothing: it keeps trading at its current size and
    joins the optimisation once there is something to judge.

    Exempting also protects the OTHER strategies. A short, noisy column distorts the shared
    covariance matrix, so a newcomer can mis-size names that have earned their history.

    Non-allocatable strategies (`exclude`, default `NON_ALLOCATABLE`)
    ----------------------------------------------------------------
    The hedge overlay and the ledger's bookkeeping buckets are dropped before anything is
    estimated. The hedge is not a bet whose size an optimiser should choose: its expected
    return is negative by design and its correlation to the book is negative by
    construction, so a Sharpe objective would try to short it and a variance objective would
    try to buy more of it — both of which fight the thing sizing it for real. Their current
    weight is reserved off the top and the rest of the book is sized around it.

    The method ladder, applied to the seasoned strategies only:

      max-Sharpe        needs means AND a covariance   -> min_days_for_means
      inverse-vol       needs a covariance only        -> min_days_for_cov
      equal weight      needs nothing

    Falling down that ladder is the correct outcome for a young book, not a conservatism to
    tune away. `current_weights` supplies the held weights for exempt strategies; without
    it they are reported but left out of the sizing, for the caller to decide.
    """
    if returns.empty or returns.shape[1] == 0:
        raise ValueError("no return history — cannot allocate")

    current_weights = dict(current_weights or {})
    exclude = NON_ALLOCATABLE if exclude is None else set(exclude)

    # Dropped BEFORE anything is estimated, so they cannot reach the covariance either — a
    # hedge that is negatively correlated with everything would distort the whole matrix,
    # not merely its own weight.
    non_allocatable = [sid for sid in returns.columns if sid in exclude]
    reserved = {sid: float(current_weights.get(sid, 0.0)) for sid in non_allocatable}
    returns = returns[[c for c in returns.columns if c not in exclude]]
    if returns.shape[1] == 0:
        raise ValueError("every strategy is non-allocatable — nothing to allocate between")

    # Drop strategies that barely reported: a mostly-NaN series contributes noise to the
    # covariance and a meaningless mean.
    coverage = returns.notna().mean()
    keep = coverage[coverage >= min_coverage].index
    dropped = [s for s in returns.columns if s not in keep]
    returns = returns[keep]
    if returns.shape[1] == 0:
        raise ValueError("every strategy fell below min_coverage — nothing to allocate")

    ages = history_days(returns)
    young = [sid for sid in returns.columns if ages[sid] < min_history_days]
    seasoned = [sid for sid in returns.columns if sid not in young]

    exempt = {sid: float(current_weights.get(sid, 0.0)) for sid in young}
    held_back = sum(exempt.values()) + sum(reserved.values())
    if held_back > 1.0:
        raise ValueError(f"exempt and reserved strategies already hold {held_back:.0%} of "
                         "the book — reallocate them manually before optimising the rest")
    budget = 1.0 - held_back

    diagnostics = {
        "dropped_for_coverage": dropped,
        "observations": int(returns.shape[0]),
        "strategies": int(returns.shape[1]),
        "history_days": {k: round(float(v), 2) for k, v in ages.items()},
        "exempt_for_youth": young,
        "reserved_non_allocatable": non_allocatable,
        "budget_after_exemptions": round(budget, 4),
    }

    if not seasoned:
        # Nothing has served long enough to be judged. Sizing the book on this would be
        # sizing it on noise, so say so and change nothing.
        return Allocation(
            pd.Series(dtype=float), "hold", exempt=exempt, reserved=reserved,
            reason=(f"no strategy has {min_history_days:.0f} days of history yet "
                    f"(oldest {ages.max():.1f}d) — leaving allocations untouched"),
            days=float(ages.max()), diagnostics=diagnostics)

    sub = returns[seasoned]
    span = sub.dropna(how="all")
    days = float(ages[seasoned].max())

    # `max_weight` caps a share of the WHOLE book, but the optimiser only sizes the budget
    # left after exemptions, so convert it. It can become infeasible — three seasoned
    # strategies cannot absorb 90% of the book under a 25% cap — in which case the cap is
    # relaxed to the tightest feasible value and the change is reported. Failing outright
    # would block the allocator precisely when a young book needs it most, and silently
    # keeping the nominal cap would be a lie.
    cap_in_budget = (max_weight / budget) if budget > _EPS else 1.0
    floor = 1.0 / len(seasoned)
    effective_cap = min(1.0, max(cap_in_budget, floor))
    cap_relaxed = effective_cap > cap_in_budget + 1e-12
    if cap_relaxed:
        logger.warning("max_weight %.0f%% is infeasible for %d seasoned strategies holding "
                       "%.0f%% of the book — relaxed to %.0f%%", max_weight * 100,
                       len(seasoned), budget * 100, effective_cap * budget * 100)
    diagnostics["effective_max_weight"] = round(effective_cap * budget, 4)
    diagnostics["max_weight_relaxed"] = cap_relaxed

    def scaled(w: pd.Series) -> pd.Series:
        return w * budget if budget < 1.0 else w

    note = (f" ({len(young)} exempt: {', '.join(young)})" if young else "")
    if cap_relaxed:
        note += (f"; cap raised {max_weight:.0%} -> {effective_cap * budget:.0%} to fill "
                 f"the book with {len(seasoned)} seasoned strateg"
                 f"{'y' if len(seasoned) == 1 else 'ies'}")

    if days < min_days_for_cov:
        return Allocation(
            scaled(equal_weights(seasoned)), "equal_weight", exempt=exempt, reserved=reserved,
            reason=(f"only {days:.1f} days of seasoned history (need {min_days_for_cov:.0f} "
                    f"to estimate a covariance) — equal weight assumes nothing{note}"),
            days=days, diagnostics=diagnostics)

    cov, shrinkage = shrunk_covariance(span[seasoned],
                                       periods_per_year=periods_per_year)

    if days < min_days_for_means:
        return Allocation(
            scaled(inverse_vol_weights(cov, effective_cap)), "inverse_volatility",
            exempt=exempt, reserved=reserved,
            reason=(f"{days:.1f} days supports a covariance but not expected returns "
                    f"(need {min_days_for_means:.0f}) — sizing by risk only{note}"),
            shrinkage=shrinkage, days=days, diagnostics=diagnostics)

    mu = span[seasoned].mean() * periods_per_year
    weights = max_sharpe_weights(mu, cov, max_weight=effective_cap,
                                 l2_lambda=l2_lambda)
    diagnostics["annualised_mean"] = {k: round(float(v), 4) for k, v in mu.items()}
    return Allocation(
        scaled(weights), "max_sharpe", exempt=exempt, reserved=reserved,
        reason=(f"{days:.1f} days of history, cap {max_weight:.0%}, "
                f"L2 pull {l2_lambda}{note}"),
        shrinkage=shrinkage, days=days, diagnostics=diagnostics)
