"""Hard risk limits applied AFTER sizing and BEFORE any order is sent.

Notebook Section 8: "decide a hard cap on gross notional and per-leg notional before
running this live, independent of what the signal says -- a model bug or data feed glitch
should not be able to size an unbounded position."

These caps are deliberately dumb and non-negotiable. If a target breaches a cap it is
clamped (not silently dropped), and the breach is recorded so the daily log/alert shows
it. If anything looks structurally wrong (NaN, absurd price), `validate_inputs` aborts
the run before it can trade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from vecm.sizing import LegSizing


@dataclass
class RiskLimits:
    max_contracts_per_leg: int = 10          # absolute cap on |contracts| per leg
    max_gross_notional: float = 500_000.0    # sum of |leg notional| across legs (USD)
    max_leg_notional: float = 250_000.0      # |notional| per single leg (USD)
    # abort if any leg's fractional target explodes past this (data/model sanity check)
    max_abs_fractional: float = 50.0


@dataclass
class RiskDecision:
    approved_contracts: Dict[str, int]
    breaches: List[str] = field(default_factory=list)
    aborted: bool = False
    abort_reason: str = ""


def validate_inputs(sizings: Dict[str, LegSizing], limits: RiskLimits) -> RiskDecision:
    """Sanity-check sizings for NaN / absurd values. Returns an aborted decision if bad."""
    for leg, s in sizings.items():
        if not math.isfinite(s.price) or s.price <= 0:
            return RiskDecision({}, aborted=True,
                                abort_reason=f"{leg}: non-finite/non-positive price {s.price}")
        if not math.isfinite(s.eff_pos):
            return RiskDecision({}, aborted=True,
                                abort_reason=f"{leg}: non-finite eff_pos")
        if abs(s.fractional) > limits.max_abs_fractional:
            return RiskDecision({}, aborted=True,
                                abort_reason=(f"{leg}: fractional target {s.fractional:.2f} exceeds "
                                              f"sanity cap {limits.max_abs_fractional} -- likely a "
                                              f"data/model error, aborting before trading."))
    return RiskDecision({leg: s.contracts for leg, s in sizings.items()})


def apply_limits(sizings: Dict[str, LegSizing], limits: RiskLimits) -> RiskDecision:
    """Clamp target contracts to the configured caps. Records every clamp as a breach."""
    decision = validate_inputs(sizings, limits)
    if decision.aborted:
        return decision

    approved: Dict[str, int] = {}
    breaches: List[str] = []

    # 1. per-leg contract + notional caps
    for leg, s in sizings.items():
        n = s.contracts
        # per-leg contract cap
        if abs(n) > limits.max_contracts_per_leg:
            capped = int(math.copysign(limits.max_contracts_per_leg, n))
            breaches.append(f"{leg}: contracts {n} -> {capped} (per-leg contract cap)")
            n = capped
        # per-leg notional cap
        notional = abs(n) * s.price * s.multiplier
        if notional > limits.max_leg_notional and notional > 0:
            capped = int(math.copysign(math.floor(limits.max_leg_notional / (s.price * s.multiplier)), n))
            breaches.append(f"{leg}: notional ${notional:,.0f} > cap "
                            f"${limits.max_leg_notional:,.0f}, contracts {n} -> {capped}")
            n = capped
        approved[leg] = n

    # 2. gross notional cap -- scale everything down proportionally if breached
    gross = sum(abs(approved[leg]) * sizings[leg].price * sizings[leg].multiplier for leg in approved)
    if gross > limits.max_gross_notional and gross > 0:
        factor = limits.max_gross_notional / gross
        breaches.append(f"gross notional ${gross:,.0f} > cap "
                        f"${limits.max_gross_notional:,.0f}, scaling positions by {factor:.3f}")
        for leg in approved:
            approved[leg] = int(approved[leg] * factor)

    return RiskDecision(approved_contracts=approved, breaches=breaches)
