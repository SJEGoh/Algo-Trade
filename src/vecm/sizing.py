"""Translate normalized, vol-scaled exposures into integer futures-contract counts.

From notebook Section 7.2:

    contracts_leg = round( NAV * eff_pos_leg / (price_leg * multiplier_leg) )

where eff_pos_leg = normalized_position_leg * vol_scale_multiplier_leg, and NAV already
includes whatever overall leverage / capital allocation you have chosen for the strategy
(see `capital_allocation` in config).

Because the WTI/Brent/RBOB contracts are large (1,000 bbl, 1,000 bbl, 42,000 gal), a
small account will round most legs to zero. `LegSizing` exposes the *fractional* target
alongside the rounded integer so the caller can log how much rounding is distorting the
intended exposure -- a live-only effect the continuous-notional backtest never saw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class LegSizing:
    leg: str
    eff_pos: float          # normalized * vol_scale (can exceed +/-1)
    price: float
    multiplier: int
    fractional: float       # unrounded target contract count
    contracts: int          # rounded integer target
    notional: float         # signed USD notional of the rounded position


def target_contracts(
    eff_pos: Dict[str, float],
    prices: Dict[str, float],
    multipliers: Dict[str, int],
    nav: float,
) -> Dict[str, LegSizing]:
    """Return a per-leg LegSizing. `eff_pos`, `prices`, `multipliers` share leg keys."""
    out: Dict[str, LegSizing] = {}
    for leg, pos in eff_pos.items():
        px = float(prices[leg])
        mult = int(multipliers[leg])
        if px <= 0 or mult <= 0:
            raise ValueError(f"non-positive price/multiplier for {leg}")
        frac = nav * pos / (px * mult)
        n = int(round(frac))
        out[leg] = LegSizing(
            leg=leg,
            eff_pos=float(pos),
            price=px,
            multiplier=mult,
            fractional=float(frac),
            contracts=n,
            notional=float(n * px * mult),
        )
    return out
