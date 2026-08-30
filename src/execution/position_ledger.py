from typing import Dict, Optional, Literal
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from central_execution import CentralExecutor  # only imported by type checkers, never at runtime


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


    def record_fill(self, symbol: str, signed_qty: float, price: float, strat_id: str) -> None:
        with self._lock:
            self.current_positions[symbol] = self.current_positions.get(symbol, 0.0) + signed_qty
            self.pending_deltas[symbol] = self.pending_deltas.get(symbol, 0.0) - signed_qty
            self._attribute_fill(symbol, signed_qty, price, strat_id)

    def record_pending(self, symbol: str, signed_qty: float) -> None:
        with self._lock:
            self.pending_deltas[symbol] = self.pending_deltas.get(symbol, 0.0) + signed_qty

    def effective_position(self, symbol: str) -> float:
        with self._lock:
            return self.current_positions.get(symbol, 0.0) + self.pending_deltas.get(symbol, 0.0)

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
            self.strategy_realized_pnl[strat_id] += (price - prev_cost) * closed_qty * direction

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
