from ledger.position_ledger import PositionLedger
import threading
from typing import Dict, Optional, Literal

import logging
logger = logging.getLogger("executor")

class RiskManager:
    def __init__(self, ledger, config: dict):
        self._ledger = ledger
        self._config = config
        self._active_strategies: set = set(config.keys())
        self._lock = threading.Lock()

    def check_order(self, intent: dict, resolved_delta: float, price, multiplier: float = 1.0,
                    ref_values: dict = None) -> dict:
        strategy_id = intent["strategy_id"]
        if strategy_id not in self._active_strategies:
            return {"approved": False, "reason": f"strategy {strategy_id} is not active"}

        if price is None:
            logger.warning("No reference price for %s — notional check skipped", strategy_id)
            return {"approved": True}

        alloc = self._config[strategy_id]["capital_allocation"]
        symbol = intent["instrument"]["symbol"]

        # Per-symbol contract value = price * multiplier (multiplier = 1 for equities).
        # ref_values carries prior legs' values so gross is valued per symbol, not all at
        # one price — matters once futures with large multipliers share the book.
        unit = dict(ref_values or {})
        unit[symbol] = price * float(multiplier)

        eff = self._ledger.strategy_effective_positions(strategy_id)
        eff[symbol] = eff.get(symbol, 0.0) + resolved_delta
        projected_gross = sum(abs(q) * unit.get(s, price * float(multiplier)) for s, q in eff.items())

        if projected_gross > alloc:
            return {"approved": False,
                    "reason": f"order would exceed allocation: projected gross {projected_gross:.0f} > {alloc:.0f}"}
        return {"approved": True}

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

    def is_active(self, strategy_id: str) -> bool:
        with self._lock:
            return strategy_id in self._active_strategies
