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

    def check_order(self, intent: dict, resolved_delta: float, price: Optional[float]) -> dict:
        strategy_id = intent["strategy_id"]

        if strategy_id not in self._active_strategies:
            return {"approved": False, "reason": f"strategy {strategy_id} is not active"}

        alloc = self._config[strategy_id]["capital_allocation"]

        if price is not None:
            order_notional = abs(resolved_delta) * price
            current_notional = self._strategy_gross_notional(strategy_id, price)
            if current_notional + order_notional > alloc:
                return {"approved": False,
                        "reason": f"order would exceed allocation: {current_notional + order_notional:.0f} > {alloc:.0f}"}
        else:
            logger.warning("No reference price for %s — notional check skipped", strategy_id)

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

    def check_drawdown(self, strat_id: str) -> None:
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
