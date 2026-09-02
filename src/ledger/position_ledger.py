from typing import Dict, Optional, Literal
import threading
from typing import TYPE_CHECKING

import logging
logger = logging.getLogger("executor")

if TYPE_CHECKING:
    from execution.central_execution import CentralExecutor  # only imported by type checkers, never at runtime


class PositionLedger:
    def __init__(self, executor: "CentralExecutor"):
        self._executor = executor  # needed to call reqPositions() via the live connection
        self.current_positions: Dict[str, float] = {}
        self.pending_deltas: Dict[str, float] = {}
        self.broker_positions: Dict[str, float] = {}
        self.strategy_positions: Dict[str, Dict[str, float]] = {}
        self.strategy_avg_cost: Dict[str, Dict[str, float]] = {}
        self.strategy_realized_pnl: Dict[str, float] = {}  # Phase 3 addition
        self._positions_ready = threading.Event()
        self._lock = threading.Lock()
        self.strategy_pending: Dict[str, Dict[str, float]] = {}
        self.multipliers: Dict[str, float] = {}  # symbol -> contract multiplier (1 for equities); makes P&L dollar-denominated

    def record_fill(self, symbol: str, signed_qty: float, price: float, strat_id: str) -> None:
        with self._lock:
            self.current_positions[symbol] = self.current_positions.get(symbol, 0.0) + signed_qty
            self.pending_deltas[symbol]    = self.pending_deltas.get(symbol, 0.0) - signed_qty
            sp = self.strategy_pending.setdefault(strat_id, {})
            sp[symbol] = sp.get(symbol, 0.0) - signed_qty        # <-- new: keep lockstep with pending_deltas
            self._attribute_fill(symbol, signed_qty, price, strat_id)
    def strategy_effective_positions(self, strat_id: str) -> Dict[str, float]:
        """Filled + pending, per symbol, for one strategy — a snapshot under lock."""
        with self._lock:
            filled  = self.strategy_positions.get(strat_id, {})
            pending = self.strategy_pending.get(strat_id, {})
            return {s: filled.get(s, 0.0) + pending.get(s, 0.0) for s in set(filled) | set(pending)}
        
    def record_pending(self, symbol: str, signed_qty: float, strat_id: str) -> None:
        with self._lock:
            self.pending_deltas[symbol] = self.pending_deltas.get(symbol, 0.0) + signed_qty
            sp = self.strategy_pending.setdefault(strat_id, {})
            sp[symbol] = sp.get(symbol, 0.0) + signed_qty

    def effective_position(self, symbol: str) -> float:
        with self._lock:
            return self.current_positions.get(symbol, 0.0) + self.pending_deltas.get(symbol, 0.0)

    def record_net_pending(self, symbol: str, signed_qty: float) -> None:
        """Netting path: adjust NET pending only (no per-strategy attribution).
        Used when placing a pooled net order; attribution happens on the fill."""
        with self._lock:
            self.pending_deltas[symbol] = self.pending_deltas.get(symbol, 0.0) + signed_qty

    def apply_attributed_fill(self, symbol: str, signed_qty: float, price: float, strat_id: str) -> None:
        """Netting fill: update net position + this strategy's book & realized P&L, and
        reverse the NET pending. Does NOT touch strategy_pending (net pending is tracked
        separately via record_net_pending)."""
        with self._lock:
            self.current_positions[symbol] = self.current_positions.get(symbol, 0.0) + signed_qty
            self.pending_deltas[symbol] = self.pending_deltas.get(symbol, 0.0) - signed_qty
            self._attribute_fill(symbol, signed_qty, price, strat_id)

    def apply_internal_cross(self, symbol: str, signed_qty: float, price: float, strat_id: str) -> None:
        """Internal crossing fill: update only this strategy's book & realized P&L.
        Does NOT touch current_positions or pending_deltas — internal crosses are
        zero-sum across strategies and don't change the net broker position."""
        with self._lock:
            self._attribute_fill(symbol, signed_qty, price, strat_id)

    def _attribute_fill(self, symbol: str, signed_qty: float, price: float, strat_id: str) -> None:
        strat_pos = self.strategy_positions.setdefault(strat_id, {})
        strat_cost = self.strategy_avg_cost.setdefault(strat_id, {})
        self.strategy_realized_pnl.setdefault(strat_id, 0.0)

        prev_qty = strat_pos.get(symbol, 0.0)
        prev_cost = strat_cost.get(symbol, 0.0)
        new_qty = prev_qty + signed_qty

        # inrease exposure or opening new position
        if prev_qty == 0.0 or ((prev_qty > 0) == (signed_qty > 0)):
            total_cost = prev_cost * abs(prev_qty) + price * abs(signed_qty)
            strat_cost[symbol] = total_cost / abs(new_qty) if new_qty != 0 else 0.0

        else:
            closed_qty = min(abs(signed_qty), abs(prev_qty))
            direction = 1 if prev_qty > 0 else -1
            mult = self.multipliers.get(symbol, 1.0)   # dollar-denominate (futures multiplier)
            self.strategy_realized_pnl[strat_id] += (price - prev_cost) * closed_qty * direction * mult

            if abs(signed_qty) > abs(prev_qty):
                strat_cost[symbol] = price
            elif new_qty == 0:
                strat_cost[symbol] = 0.0

        strat_pos[symbol] = new_qty

    def fetch_broker_positions(self, timeout: float = 5.0) -> Dict[str, float]:
        self.broker_positions = {}
        self._positions_ready.clear()
        self._executor.reqPositions()
        if not self._positions_ready.wait(timeout=timeout):
            raise TimeoutError("Timed out waiting for reqPositions()")
        return dict(self.broker_positions)

    def reconcile(self, auto_correct: bool = True) -> dict:
        broker = self.fetch_broker_positions()
        all_symbols = set(self.current_positions) | set(broker)
        discrepancies = {
            s: {"internal": self.current_positions.get(s, 0.0), "broker": broker.get(s, 0.0)}
            for s in all_symbols
            if self.current_positions.get(s, 0.0) != broker.get(s, 0.0)
        }
        if discrepancies and auto_correct:
            self.current_positions = dict(broker)
        return {"matched": not discrepancies, "discrepancies": discrepancies}

    def equity_snapshot(self, marks: Dict[str, float]) -> Dict[str, dict]:
        """Per-strategy realized + unrealized (mark-to-market) + cumulative total.
        (mark - avg_cost) * qty is sign-correct for long and short. Missing mark
        -> that leg contributes 0. Read under lock for a torn-free snapshot."""
        out = {}
        with self._lock:
            strats = set(self.strategy_positions) | set(self.strategy_realized_pnl)
            for strat in strats:
                positions = self.strategy_positions.get(strat, {})
                costs = self.strategy_avg_cost.get(strat, {})
                realized = self.strategy_realized_pnl.get(strat, 0.0)
                unrealized = 0.0
                for sym, qty in positions.items():
                    mark = marks.get(sym)
                    if qty == 0 or mark is None:
                        continue
                    unrealized += (mark - costs.get(sym, 0.0)) * qty * self.multipliers.get(sym, 1.0)
                out[strat] = {"realized": realized, "unrealized": unrealized,
                              "equity": realized + unrealized}
        return out
    
if __name__ == "__main__":
    led = PositionLedger(executor=None)  # no connection needed to test attribution

    # long side
    led.record_fill("AAPL", +100, 50.0, "s1")   # open long 100 @ 50
    led.record_fill("AAPL", +100, 60.0, "s1")   # add 100 @ 60 -> avg cost 55, qty 200
    led.record_fill("AAPL", -150, 70.0, "s1")   # sell 150 @ 70 -> realize (70-55)*150 = +2250, qty 50 left

    print("realized:", led.strategy_realized_pnl["s1"])   # expect 2250.0
    print("position:", led.strategy_positions["s1"]["AAPL"])  # expect 50.0
    print("avg cost:", led.strategy_avg_cost["s1"]["AAPL"])   # expect 55.0 (unchanged on partial reduce)

    # short side + flip
    led.record_fill("TSLA", -100, 300.0, "s2")  # open short 100 @ 300
    led.record_fill("TSLA", +150, 280.0, "s2")  # buy 150 @ 280: close 100 @ profit, flip to long 50 @ 280
    print("realized:", led.strategy_realized_pnl["s2"])   # expect (280-300)*100*(-1) = +2000
    print("position:", led.strategy_positions["s2"]["TSLA"])  # expect +50.0
    print("avg cost:", led.strategy_avg_cost["s2"]["TSLA"])   # expect 280.0 (new long leg)
