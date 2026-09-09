"""
portfolio/hedger.py — shave abnormal exposures, leave the rest to diversification.

Not a continuous overlay. The book is left alone until one bucket grows past a threshold,
and then only the EXCESS is hedged — the position the strategies deliberately took is kept,
and the anomaly on top of it is shaved off.

Why this shape rather than a beta-neutral overlay
-------------------------------------------------
A continuous hedge cancels the exposure some strategies exist to harvest: a timing strategy
(intraday breakout, overnight vol-surge) earns its return by being in the market at the right
moment, so neutralising the market removes the edge along with the risk. A threshold hedge
does not have that problem, because it does nothing until the exposure is larger than anything
the strategies intended.

It is also far more robust to estimation error. Sizing a continuous hedge needs an accurate
beta, and a wrong one mis-hedges every day. This needs only a reliable answer to "is this
bucket unusually large", which a notional sum answers without fitting anything. When the
threshold is wrong you occasionally hedge something you did not need to — a small cost —
rather than systematically hedging the wrong amount forever.

The three properties that make it behave
----------------------------------------
  * HEDGE TO A TARGET, NOT TO ZERO. Hedging a 45% exposure down to 25% shaves the anomaly;
    hedging it to flat throws away a position that was taken on purpose.
  * HYSTERESIS. Separate trigger and release levels, so a book hovering on the threshold
    does not hedge and unwind repeatedly. Whether a bucket is currently hedged is read off
    the hedge positions themselves, so there is no extra state to persist or get stale.
  * EXPOSURE EXCLUDES THE HEDGE. Measured over every strategy EXCEPT the hedge overlay. If
    the hedge counted toward the exposure it was sizing, it would see its own work, unwind,
    see the exposure return, and oscillate forever. This is the defining bug of the design
    and it is silent, so it is enforced in one place: `bucket_exposures(exclude=...)`.

Nothing here places an order. `plan()` returns the book it would submit.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("hedger")

HEDGE_STRATEGY_ID = "hedge_overlay"
UNCLASSIFIED = "unclassified"


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------
#: Symbol -> bucket. Deliberately coarse and economic rather than GICS: GICS puts NVDA in
#: Information Technology, TSLA in Consumer Discretionary and GOOGL in Communication
#: Services, which would report a book spread across three sectors when it is one trade.
DEFAULT_SECTORS: Dict[str, str] = {
    **{s: "semiconductors" for s in
       ("NVDA AMD AVGO MU ARM MRVL TSM ASML KLAC LRCX AMAT QCOM TXN ON MCHP NXPI SMCI"
        ).split()},
    **{s: "software" for s in
       ("CRWD PANW FTNT SNOW NET ZS OKTA HUBS NOW CRM ADBE INTU CDNS SNPS PLTR APP DDOG"
        ).split()},
    **{s: "megacap_tech" for s in "AMZN META GOOGL MSFT ORCL NFLX AAPL".split()},
    **{s: "consumer_internet" for s in "UBER ABNB SHOP MELI COIN HOOD RDDT".split()},
    **{s: "hardware" for s in "ANET VRT DELL HPQ".split()},
    **{s: "autos" for s in "TSLA RIVN LCID F GM".split()},
    **{s: "energy" for s in "XOM CVX COP SLB OXY".split()},
    **{s: "real_estate" for s in "VNQ AMT PLD SPG".split()},
    **{s: "precious_metals" for s in "GLD IAU GDX".split()},
}

#: Bucket -> the instrument used to hedge it. A bucket with no entry is reported as
#: UNHEDGEABLE rather than quietly skipped — an exposure nobody is watching is worse than
#: one nobody is hedging.
DEFAULT_HEDGE_INSTRUMENTS: Dict[str, str] = {
    "semiconductors": "SMH",
    "software": "IGV",
    "megacap_tech": "QQQ",
    "consumer_internet": "XLY",
    "hardware": "XLK",
    "autos": "XLY",
    "energy": "XLE",
    "real_estate": "XLRE",
    "precious_metals": "GLD",
    "commodity": "DBC",
    "rates": "TLT",
}


def default_bucket_of(symbol: str, instrument: dict = None) -> str:
    """Bucket a symbol. Equities go to a sector; anything else falls back to its asset class.

    `asset_class` is used rather than a sector taxonomy for non-equities on purpose: it is
    coarse, unambiguous and already carried on every instrument, where a sector label for a
    futures contract would be a guess. A symbol that cannot be placed lands in
    `unclassified`, which is reported, never hedged, and never silently dropped.
    """
    sector = DEFAULT_SECTORS.get(symbol.upper())
    if sector:
        return sector
    asset_class = ((instrument or {}).get("asset_class") or "").strip().lower()
    if asset_class and asset_class != "equity":
        return asset_class
    return UNCLASSIFIED


@dataclass
class Exposure:
    bucket: str
    notional: float                 # signed dollars
    fraction: float                 # signed share of NAV
    symbols: Dict[str, float] = field(default_factory=dict)


def bucket_exposures(strategy_positions: Dict[str, Dict[str, float]], marks: Dict[str, float],
                     nav: float, multipliers: Dict[str, float] = None,
                     instruments: Dict[str, dict] = None, bucket_of=default_bucket_of,
                     exclude: set = None) -> tuple:
    """Signed notional per bucket, as a share of NAV.

    Returns (exposures, unpriced) — `unpriced` is every symbol held with no usable mark, and
    it is returned rather than logged because exposure that could not be measured is exactly
    the exposure worth knowing about.

    `exclude` defaults to the hedge overlay itself. Including it would make the hedge count
    toward the exposure it is sizing, which oscillates.
    """
    exclude = {HEDGE_STRATEGY_ID} if exclude is None else set(exclude)
    multipliers = multipliers or {}
    instruments = instruments or {}
    if nav <= 0:
        raise ValueError(f"NAV must be positive to express exposure as a fraction (got {nav})")

    combined: Dict[str, float] = {}
    for sid, positions in strategy_positions.items():
        if sid in exclude:
            continue
        for symbol, qty in positions.items():
            if abs(qty) > 1e-9:
                combined[symbol] = combined.get(symbol, 0.0) + qty

    exposures: Dict[str, Exposure] = {}
    unpriced: Dict[str, float] = {}
    for symbol, qty in combined.items():
        price = marks.get(symbol)
        if price is None or not math.isfinite(price) or price <= 0:
            unpriced[symbol] = qty
            continue
        notional = qty * price * float(multipliers.get(symbol, 1.0) or 1.0)
        bucket = bucket_of(symbol, instruments.get(symbol))
        entry = exposures.setdefault(bucket, Exposure(bucket, 0.0, 0.0))
        entry.notional += notional
        entry.symbols[symbol] = notional

    for entry in exposures.values():
        entry.fraction = entry.notional / nav
    return exposures, unpriced


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
@dataclass
class BucketPolicy:
    """Thresholds for one bucket, as absolute shares of NAV.

    Absolute rather than relative-to-history on purpose. A trailing z-score normalises away
    exactly the concentration worth catching: a book that is always 80% one sector never
    looks abnormal against its own past.

    trigger > target >= release, and all are magnitudes (a -35% short trips a 30% trigger).
    """
    trigger: float = 0.30      # start hedging above this
    target: float = 0.25       # hedge down to this, NOT to zero
    release: float = 0.20      # fully unwind below this

    def __post_init__(self):
        if not (0 < self.release <= self.target <= self.trigger <= 1.0):
            raise ValueError(
                f"need 0 < release <= target <= trigger <= 1, got release={self.release}, "
                f"target={self.target}, trigger={self.trigger}")


@dataclass
class HedgeDecision:
    bucket: str
    exposure_fraction: float
    hedge_symbol: Optional[str]
    hedge_notional: float          # signed dollars; opposite sign to the exposure
    quantity: float                # signed, whole units
    price: Optional[float]
    reason: str


@dataclass
class HedgePlan:
    decisions: List[HedgeDecision]
    exposures: Dict[str, Exposure]
    unpriced: Dict[str, float] = field(default_factory=dict)
    unhedgeable: Dict[str, float] = field(default_factory=dict)
    nav: float = 0.0
    current_hedge: Dict[str, float] = field(default_factory=dict)

    @property
    def book(self) -> list:
        """The desired hedge book, in the shape POST /targets expects.

        Carries a hedge we want, and an explicit zero for a hedge we currently HOLD and no
        longer want — `/targets` is authoritative, so stating the unwind puts it in the
        journal rather than leaving it to be inferred from an omission.

        A bucket that is flat and staying flat is left out entirely. Listing every hedge
        instrument at zero on every cycle would submit a dozen no-op legs a minute and bury
        the entries that mean something.
        """
        return [{"instrument": {"symbol": d.hedge_symbol, "asset_class": "equity",
                                "sec_type": "STK", "exchange": "SMART"},
                 "target_quantity": d.quantity,
                 "expected_price": d.price}
                for d in self.decisions
                if d.hedge_symbol and d.price
                and (abs(d.quantity) > 1e-9 or d.hedge_symbol in self.current_hedge)]

    @property
    def coverage(self) -> dict:
        """How much of the book this plan can actually speak to. The unmeasured and
        unhedgeable totals are the honest part of the report."""
        measured = sum(abs(e.notional) for e in self.exposures.values())
        return {"measured_notional": measured,
                "unhedgeable_notional": sum(abs(v) for v in self.unhedgeable.values()),
                "unpriced_symbols": sorted(self.unpriced),
                "nav": self.nav}

    def __str__(self):
        lines = [f"NAV {self.nav:,.0f}"]
        for b, e in sorted(self.exposures.items(), key=lambda kv: -abs(kv[1].fraction)):
            lines.append(f"  {b:20} {e.fraction:+7.1%}  {e.notional:+14,.0f}")
        acting = [d for d in self.decisions if abs(d.quantity) > 0]
        lines.append("  -- hedge --" if acting else "  -- no bucket above its trigger --")
        for d in acting:
            lines.append(f"  {d.bucket:20} {d.hedge_symbol:>5} {d.quantity:+10,.0f} "
                         f"({d.hedge_notional:+,.0f}) — {d.reason}")
        if self.unpriced:
            lines.append(f"  !! unpriced (exposure NOT measured): {sorted(self.unpriced)}")
        if self.unhedgeable:
            lines.append(f"  !! no hedge instrument: "
                         + ", ".join(f"{k} {v:+,.0f}" for k, v in self.unhedgeable.items()))
        return "\n".join(lines)


def plan(exposures: Dict[str, Exposure], nav: float, prices: Dict[str, float],
         policies: Dict[str, BucketPolicy] = None, default_policy: BucketPolicy = None,
         hedge_instruments: Dict[str, str] = None,
         current_hedge: Dict[str, float] = None,
         min_hedge_notional: float = 1_000.0,
         unpriced: Dict[str, float] = None) -> HedgePlan:
    """Decide the hedge book for the exposures given.

    `current_hedge` is the overlay's existing positions (symbol -> quantity) and supplies the
    hysteresis state: a bucket already carrying a hedge is judged against `release`, an
    unhedged one against `trigger`. Reading the state off the positions means there is no
    separate store to fall out of sync with reality after a restart.
    """
    policies = policies or {}
    default_policy = default_policy or BucketPolicy()
    hedge_instruments = hedge_instruments or DEFAULT_HEDGE_INSTRUMENTS
    current_hedge = current_hedge or {}

    decisions: List[HedgeDecision] = []
    unhedgeable: Dict[str, float] = {}

    for bucket, exposure in sorted(exposures.items()):
        if bucket == UNCLASSIFIED:
            # Never hedge what we could not identify — a guessed hedge on an unknown
            # exposure is a second unknown position, not a smaller one.
            unhedgeable[bucket] = exposure.notional
            continue

        symbol = hedge_instruments.get(bucket)
        if not symbol:
            unhedgeable[bucket] = exposure.notional
            continue

        policy = policies.get(bucket, default_policy)
        held = float(current_hedge.get(symbol, 0.0))
        currently_hedged = abs(held) > 1e-9
        threshold = policy.release if currently_hedged else policy.trigger
        magnitude = abs(exposure.fraction)

        if magnitude <= threshold:
            reason = (f"{magnitude:.1%} at or below "
                      f"{'release' if currently_hedged else 'trigger'} {threshold:.0%}")
            decisions.append(HedgeDecision(bucket, exposure.fraction, symbol, 0.0, 0.0,
                                           prices.get(symbol), reason))
            continue

        # Shave the excess over `target`, keeping the exposure the strategies intended.
        excess_fraction = magnitude - policy.target
        hedge_notional = -math.copysign(excess_fraction * nav, exposure.fraction)

        price = prices.get(symbol)
        if price is None or not math.isfinite(price) or price <= 0:
            # Fail closed and loudly: an unpriceable hedge cannot be sized, and guessing
            # would put on a position of unknown size against a known exposure.
            logger.error("no price for hedge instrument %s — %s left UNHEDGED at %.1f%%",
                         symbol, bucket, magnitude * 100)
            unhedgeable[bucket] = exposure.notional
            continue

        if abs(hedge_notional) < min_hedge_notional:
            decisions.append(HedgeDecision(
                bucket, exposure.fraction, symbol, 0.0, 0.0, price,
                f"excess {abs(hedge_notional):,.0f} below the {min_hedge_notional:,.0f} floor"))
            continue

        quantity = math.copysign(math.floor(abs(hedge_notional) / price), hedge_notional)
        if abs(quantity) < 1:
            decisions.append(HedgeDecision(
                bucket, exposure.fraction, symbol, 0.0, 0.0, price,
                f"excess {abs(hedge_notional):,.0f} is under one share of {symbol}"))
            continue

        decisions.append(HedgeDecision(
            bucket, exposure.fraction, symbol, quantity * price, quantity, price,
            f"{magnitude:.1%} over {'release' if currently_hedged else 'trigger'} "
            f"{threshold:.0%} — shaving to target {policy.target:.0%}"))

    # Anything the overlay holds for a bucket that has since vanished must be unwound, or a
    # stale hedge outlives the exposure it was taken against.
    planned = {d.hedge_symbol for d in decisions}
    for symbol, qty in current_hedge.items():
        if symbol not in planned and abs(qty) > 1e-9:
            decisions.append(HedgeDecision(
                f"stale:{symbol}", 0.0, symbol, 0.0, 0.0, prices.get(symbol),
                "no exposure left in the bucket this hedged — unwinding"))

    return HedgePlan(decisions=decisions, exposures=exposures, unpriced=dict(unpriced or {}),
                     unhedgeable=unhedgeable, nav=nav,
                     current_hedge={k: v for k, v in current_hedge.items()
                                    if abs(v) > 1e-9})
