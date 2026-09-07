from ledger.position_ledger import PositionLedger
import math
import threading
from typing import Dict, Optional, Literal

import logging
logger = logging.getLogger("executor")

def _is_priceable(price) -> bool:
    """A price usable for a notional check: a real, finite, strictly positive number.
    None / NaN / 0 / negative all mean "cannot value" — never "no limit"."""
    try:
        p = float(price)
    except (TypeError, ValueError):
        return False
    return math.isfinite(p) and p > 0


class RiskManager:
    def __init__(self, ledger, config: dict, global_config: dict = None):
        self._ledger = ledger
        self._config = config
        self._global = global_config or {}
        self._active_strategies: set = set(config.keys())
        self._lock = threading.Lock()

    def check_order(self, intent: dict, resolved_delta: float, price, multiplier: float = 1.0,
                    ref_values: dict = None) -> dict:
        strategy_id = intent["strategy_id"]
        if strategy_id not in self._active_strategies:
            return {"approved": False, "reason": f"strategy {strategy_id} is not active"}

        # FAIL CLOSED. An unpriced order can't be valued, so approving it waives the
        # allocation cap entirely — that is how an oversized order gets through.
        if not _is_priceable(price):
            logger.warning("No usable reference price for %s (%r) — order REJECTED", strategy_id, price)
            return {"approved": False,
                    "reason": f"no usable reference price ({price!r}) — cannot value order"}

        alloc = self._config[strategy_id]["capital_allocation"]
        symbol = intent["instrument"]["symbol"]

        # Per-symbol contract value = price * multiplier (multiplier = 1 for equities).
        # ref_values carries prior legs' values so gross is valued per symbol, not all at
        # one price — matters once futures with large multipliers share the book.
        unit = dict(ref_values or {})
        unit[symbol] = price * float(multiplier)

        eff = self._ledger.strategy_effective_positions(strategy_id)
        eff[symbol] = eff.get(symbol, 0.0) + resolved_delta
        fallback = price * float(multiplier)
        projected_gross = sum(abs(q) * self._unit_value(strategy_id, s, unit, fallback)
                              for s, q in eff.items())

        # A NaN gross compares False against every limit, so it would silently pass.
        if not math.isfinite(projected_gross):
            return {"approved": False,
                    "reason": f"projected gross notional is not a number ({projected_gross!r}) "
                              "— a leg has a bad price or multiplier"}

        if projected_gross > alloc:
            return {"approved": False,
                    "reason": f"order would exceed allocation: projected gross {projected_gross:.0f} > {alloc:.0f}"}

        # Global gross-exposure cap across ALL strategies (portfolio-level), if configured.
        max_gross = self._global.get("max_gross_exposure")
        if max_gross is not None:
            total_gross = 0.0
            for other_sid in set(self._ledger.strategy_positions) | {strategy_id}:
                eff_other = self._ledger.strategy_effective_positions(other_sid)
                if other_sid == strategy_id:
                    eff_other[symbol] = eff_other.get(symbol, 0.0) + resolved_delta
                for s, q in eff_other.items():
                    total_gross += abs(q) * self._unit_value(other_sid, s, unit, fallback)
            if not math.isfinite(total_gross):
                return {"approved": False,
                        "reason": "portfolio gross notional is not a number — a leg has a bad price"}
            if total_gross > max_gross:
                return {"approved": False,
                        "reason": f"order would exceed GLOBAL gross exposure: projected {total_gross:.0f} > {max_gross:.0f}"}
        return {"approved": True}

    def _unit_value(self, strat_id: str, symbol: str, unit: dict, fallback: float) -> float:
        """Value of one unit of `symbol` for the gross-notional sum.

        `unit` holds real traded references (price * multiplier) for symbols this session
        has priced. It is EMPTY for a symbol never traded since the last restart, while
        positions in that symbol restore from the database — so falling straight back to
        the incoming order's price valued an existing $338k holding at whatever the new
        order happened to cost. The strategy's own average cost is a far better estimate,
        and it's already in the ledger."""
        if symbol in unit:
            return unit[symbol]
        try:
            cost = (getattr(self._ledger, "strategy_avg_cost", {}) or {}).get(strat_id, {}).get(symbol)
            if cost:
                mult = float((getattr(self._ledger, "multipliers", {}) or {}).get(symbol, 1.0))
                value = abs(float(cost)) * mult
                if math.isfinite(value) and value > 0:
                    return value
        except (TypeError, ValueError, AttributeError):
            pass
        return fallback

    def _strategy_gross_notional(self, strat_id: str, price: float) -> float:
        positions = self._ledger.strategy_positions.get(strat_id, {})
        return sum(abs(qty) * price for qty in positions.values())
    
    def halt_strategy(self, strat_id: str, reason: str) -> None:
        with self._lock:
            self._active_strategies.discard(strat_id)
        logger.warning("HALTED: %s - %s", strat_id, reason)

    def reactivate_strategy(self, strat_id: str) -> None:
        with self._lock:
            self._active_strategies.add(strat_id)
        logger.info("REACTIVATED: %s", strat_id)

    def check_drawdown(self, strat_id: str) -> bool:
        cfg = self._config.get(strat_id, {})
        alloc = cfg.get("capital_allocation")
        max_dd = cfg.get("max_drawdown")

        if alloc is None or max_dd is None:
            return

        pnl = self._ledger.strategy_realized_pnl.get(strat_id, 0.0)
        drawdown_pct = -pnl/alloc if pnl < 0 else 0.0
        if drawdown_pct >= max_dd:
            logger.critical("DRAWDOWN BREACH: %s at %.1f%% >= limit %.1f%%",
                            strat_id, drawdown_pct * 100, max_dd * 100)
            self.halt_strategy(strat_id, f"DRAWDOWN BREACH: {strat_id} at {drawdown_pct * 100:.1f} >= limit {max_dd * 100:.1f}")

    def drawdown_status(self, strat_id: str, pnl: float) -> dict:
        """Pure predicate: is `pnl` (realized, or realized+unrealized) a drawdown breach for
        this strategy? pnl is dollar-denominated (multiplier-aware). Enforcement (halt +
        flatten) is the executor's job — see CentralExecutor.enforce_drawdown."""
        cfg = self._config.get(strat_id, {})
        alloc = cfg.get("capital_allocation")
        max_dd = cfg.get("max_drawdown")
        if alloc is None or max_dd is None or alloc <= 0:
            return {"breached": False, "drawdown_pct": 0.0, "max_dd": max_dd or 0.0}
        dd = (-pnl / alloc) if pnl < 0 else 0.0
        return {"breached": dd >= max_dd, "drawdown_pct": dd, "max_dd": max_dd}

    def max_gross_exposure(self):
        """Portfolio-wide gross-notional cap, or None if unset. Public so the netting
        coordinator can enforce the same limit on pooled books."""
        return self._global.get("max_gross_exposure")

    def is_active(self, strategy_id: str) -> bool:
        with self._lock:
            return strategy_id in self._active_strategies
