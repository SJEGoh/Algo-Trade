from typing import Dict, Optional, Literal
import threading
from typing import TYPE_CHECKING

import logging
logger = logging.getLogger("executor")

if TYPE_CHECKING:
    from execution.central_execution import CentralExecutor  # only imported by type checkers, never at runtime


class PositionLedger:
    #: key used for the cash leg wherever positions are presented as a book
    CASH_SYMBOL = "CASH"

    def __init__(self, executor: "CentralExecutor", config: Optional[dict] = None):
        self._executor = executor  # needed to call reqPositions() via the live connection
        self._config = config or {}
        self.current_positions: Dict[str, float] = {}
        self.pending_deltas: Dict[str, float] = {}
        self.broker_positions: Dict[str, float] = {}
        self.strategy_positions: Dict[str, Dict[str, float]] = {}
        self.strategy_avg_cost: Dict[str, Dict[str, float]] = {}
        self.strategy_realized_pnl: Dict[str, float] = {}  # Phase 3 addition
        #: strat -> cumulative commissions. Already deducted from realized P&L and cash by
        #: apply_fee; tallied separately so fees stay visible instead of dissolving into P&L.
        self.strategy_fees: Dict[str, float] = {}
        self._positions_ready = threading.Event()
        self._lock = threading.Lock()
        self.strategy_pending: Dict[str, Dict[str, float]] = {}
        self.multipliers: Dict[str, float] = {}  # symbol -> contract multiplier (1 for equities); makes P&L dollar-denominated
        # --- cash as a position -------------------------------------------------
        # Each strategy holds CASH alongside its instruments: it starts at the strategy's
        # capital basis (config `starting_cash`, else `capital_allocation`) and every fill
        # moves it by -signed_qty * price * multiplier. Strategy NAV is then
        #   cash + market value of positions  ==  starting_cash + realized + unrealized.
        self.starting_cash: Dict[str, float] = {}   # strat -> capital basis (fixed)
        self.strategy_cash: Dict[str, float] = {}   # strat -> live cash balance
        for sid, cfg in self._config.items():
            basis = self._basis_from_config(cfg)
            self.starting_cash[sid] = basis
            self.strategy_cash[sid] = basis

    # ------------------------------------------------------------------
    # Cash book
    # ------------------------------------------------------------------
    @staticmethod
    def _basis_from_config(cfg: dict) -> float:
        """A strategy's capital basis: explicit `starting_cash`, else its allocation."""
        if not isinstance(cfg, dict):
            return 0.0
        basis = cfg.get("starting_cash")
        if basis is None:
            basis = cfg.get("capital_allocation", 0.0)
        try:
            return float(basis)
        except (TypeError, ValueError):
            return 0.0

    def _basis(self, strat_id: str) -> float:
        """Capital basis for a strategy (memoised). Unknown strategies start at 0.0, so
        their NAV is just P&L."""
        if strat_id not in self.starting_cash:
            self.starting_cash[strat_id] = self._basis_from_config(self._config.get(strat_id, {}))
        return self.starting_cash[strat_id]

    def _cash(self, strat_id: str) -> float:
        """Live cash for a strategy, seeded from its basis. Caller holds the lock."""
        if strat_id not in self.strategy_cash:
            self.strategy_cash[strat_id] = self._basis(strat_id)
        return self.strategy_cash[strat_id]

    def cash(self, strat_id: str) -> float:
        """Public, thread-safe read of one strategy's cash position."""
        with self._lock:
            return self._cash(strat_id)

    def cash_book(self) -> Dict[str, float]:
        """Snapshot of every strategy's cash position."""
        with self._lock:
            return dict(self.strategy_cash)

    def adjust_capital(self, strat_id: str, delta: float) -> dict:
        """Move capital INTO (delta > 0) or OUT OF (delta < 0) a strategy.

        Cash and the capital basis move together, so P&L is untouched: funding a strategy
        is not a profit, and withdrawing from it is not a loss. NAV moves by exactly delta.

        Withdrawing more than the cash on hand is allowed and leaves cash NEGATIVE — that
        shortfall is the amount of stock the caller must sell; the sale proceeds land back
        in cash on the fill and bring it up to zero. See CentralExecutor.rebalance_allocation."""
        with self._lock:
            before = self._cash(strat_id)
            self.strategy_cash[strat_id] = before + delta
            self.starting_cash[strat_id] = self._basis(strat_id) + delta
            return {"strategy_id": strat_id, "delta": delta,
                    "cash_before": before, "cash_after": self.strategy_cash[strat_id],
                    "starting_cash": self.starting_cash[strat_id]}

    def set_cash(self, strat_id: str, amount: float, reset_basis: bool = False) -> None:
        """Overwrite a strategy's cash balance (e.g. reconciling against the broker or
        funding a new strategy). `reset_basis` also moves the capital basis, so the
        starting point of the NAV curve follows."""
        with self._lock:
            self.strategy_cash[strat_id] = float(amount)
            if reset_basis:
                self.starting_cash[strat_id] = float(amount)

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

    def apply_fee(self, strat_id: str, amount: float) -> None:
        """Charge a commission to a strategy: out of its cash AND its realized P&L.

        Both, so  nav == starting_cash + realized + unrealized  still holds. From cash alone,
        NAV would disagree with P&L by every fee paid; from P&L alone, NAV would overstate by
        the same amount. And because the drawdown checks read realized P&L, a fee now counts
        toward a halt — which matters most on a tight limit, where fees can be most of it."""
        if not amount:
            return
        with self._lock:
            self.strategy_cash[strat_id] = self._cash(strat_id) - amount
            self.strategy_realized_pnl[strat_id] = (
                self.strategy_realized_pnl.get(strat_id, 0.0) - amount)
            self.strategy_fees[strat_id] = self.strategy_fees.get(strat_id, 0.0) + amount

    def _attribute_fill(self, symbol: str, signed_qty: float, price: float, strat_id: str) -> None:
        strat_pos = self.strategy_positions.setdefault(strat_id, {})
        strat_cost = self.strategy_avg_cost.setdefault(strat_id, {})
        self.strategy_realized_pnl.setdefault(strat_id, 0.0)
        mult = self.multipliers.get(symbol, 1.0)   # dollar-denominate (futures multiplier)

        # Cash leg: a buy pays out, a sell takes in. Zero-sum on internal crosses, and it
        # keeps  cash + market value == starting_cash + realized + unrealized  exact.
        self.strategy_cash[strat_id] = self._cash(strat_id) - signed_qty * price * mult

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
            self.strategy_realized_pnl[strat_id] += (price - prev_cost) * closed_qty * direction * mult

            if abs(signed_qty) > abs(prev_qty):
                strat_cost[symbol] = price
            elif new_qty == 0:
                strat_cost[symbol] = 0.0

        strat_pos[symbol] = new_qty

    def write_off_position(self, strat_id: str, symbol: str) -> float:
        """Repair path: drop a strategy position that is already flat at the broker,
        crediting its cost basis back to cash. Books NO P&L (the exit price is unknown),
        which keeps  nav == starting_cash + realized + unrealized  exact — the strategy
        simply stops carrying an unrealized mark it never really had.
        Returns the quantity written off."""
        with self._lock:
            qty = self.strategy_positions.get(strat_id, {}).get(symbol, 0.0)
            if abs(qty) < 1e-9:
                return 0.0
            cost = self.strategy_avg_cost.get(strat_id, {}).get(symbol, 0.0)
            mult = self.multipliers.get(symbol, 1.0)
            self.strategy_cash[strat_id] = self._cash(strat_id) + cost * qty * mult
            self.strategy_positions[strat_id][symbol] = 0.0
            self.strategy_avg_cost.setdefault(strat_id, {})[symbol] = 0.0
            return qty

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

    # ------------------------------------------------------------------
    # Persistence — save / restore strategy-level state across restarts
    # ------------------------------------------------------------------
    def save_state(self, logger_db) -> None:
        """Persist per-strategy positions, avg costs, realized P&L, and multipliers
        to the EventLogger's SQLite database. Called after every fill and periodically."""
        with self._lock:
            logger_db.save_strategy_positions(
                dict(self.strategy_positions), dict(self.strategy_avg_cost)
            )
            logger_db.save_realized_pnl(dict(self.strategy_realized_pnl))
            logger_db.save_multipliers(dict(self.multipliers))
            logger_db.save_strategy_cash(dict(self.strategy_cash), dict(self.starting_cash))
            # getattr: logger backends that predate fee tracking still save everything else.
            # The fees are already inside the realized P&L and cash persisted above, so all a
            # missing method loses is the separate tally, not the money.
            save_fees = getattr(logger_db, "save_strategy_fees", None)
            if save_fees:
                save_fees(dict(self.strategy_fees))

    def restore_state(self, logger_db) -> None:
        """Reload per-strategy positions, avg costs, realized P&L, and multipliers
        from the database. Called once at startup, AFTER reconcile has set
        current_positions from the broker."""
        positions, avg_cost = logger_db.load_strategy_positions()
        realized = logger_db.load_realized_pnl()
        multipliers = logger_db.load_multipliers()
        cash, basis = logger_db.load_strategy_cash()
        load_fees = getattr(logger_db, "load_strategy_fees", None)
        fees = load_fees() if load_fees else {}
        with self._lock:
            self.strategy_positions = positions
            self.strategy_avg_cost = avg_cost
            self.strategy_realized_pnl = realized
            self.strategy_fees = dict(fees)
            self.multipliers.update(multipliers)
            # Cash survives restarts; a strategy with no saved row keeps the basis seeded
            # from config (a newly added strategy starts fully in cash).
            self.starting_cash.update(basis)
            self.strategy_cash.update(cash)
        logger.info("restored strategy state: %d strategies, %d symbols, %d multipliers, "
                    "%d cash balances",
                    len(positions), sum(len(p) for p in positions.values()), len(multipliers),
                    len(cash))

    def equity_snapshot(self, marks: Dict[str, float]) -> Dict[str, dict]:
        """Per-strategy P&L *and* NAV, treating cash as one more position.

        Keys per strategy:
          realized / unrealized / equity   P&L only (`equity` = realized + unrealized) —
                                           this is what the drawdown checks consume.
          cash / position_value / nav      the balance-sheet view: nav = cash + position_value
                                           = starting_cash + realized + unrealized.
        (mark - avg_cost) * qty is sign-correct for long and short. A symbol with no mark
        contributes 0 unrealized and is valued at cost, so NAV stays consistent. Read under
        lock for a torn-free snapshot."""
        out = {}
        with self._lock:
            strats = (set(self.strategy_positions) | set(self.strategy_realized_pnl)
                      | set(self.strategy_cash) | set(self.starting_cash))
            for strat in strats:
                positions = self.strategy_positions.get(strat, {})
                costs = self.strategy_avg_cost.get(strat, {})
                realized = self.strategy_realized_pnl.get(strat, 0.0)
                cash = self._cash(strat)
                unrealized = 0.0
                position_value = 0.0
                stale = []
                for sym, qty in positions.items():
                    if qty == 0:
                        continue
                    mult = self.multipliers.get(sym, 1.0)
                    cost = costs.get(sym, 0.0)
                    mark = marks.get(sym)
                    if mark is None:
                        stale.append(sym)          # value at cost -> contributes 0 unrealized
                        position_value += cost * qty * mult
                        continue
                    unrealized += (mark - cost) * qty * mult
                    position_value += mark * qty * mult
                out[strat] = {
                    "realized": realized,
                    "unrealized": unrealized,
                    "equity": realized + unrealized,          # P&L (unchanged meaning)
                    "cash": cash,
                    "position_value": position_value,
                    "nav": cash + position_value,
                    "starting_cash": self._basis(strat),
                    "unmarked": stale,
                }
        return out

    def strategy_book(self, strat_id: str, marks: Dict[str, float] = None) -> list:
        """One strategy's holdings as a book of positions with CASH as the first line —
        [{symbol, quantity, avg_cost, mark, multiplier, market_value, unrealized}, ...].
        Cash has quantity == market_value and no cost basis."""
        marks = marks or {}
        with self._lock:
            rows = [{
                "symbol": self.CASH_SYMBOL, "quantity": self._cash(strat_id),
                "avg_cost": None, "mark": None, "multiplier": 1.0,
                "market_value": self._cash(strat_id), "unrealized": 0.0, "is_cash": True,
            }]
            for sym, qty in sorted(self.strategy_positions.get(strat_id, {}).items()):
                if qty == 0:
                    continue
                mult = self.multipliers.get(sym, 1.0)
                cost = self.strategy_avg_cost.get(strat_id, {}).get(sym, 0.0)
                mark = marks.get(sym)
                px = cost if mark is None else mark
                rows.append({
                    "symbol": sym, "quantity": qty, "avg_cost": cost, "mark": mark,
                    "multiplier": mult, "market_value": px * qty * mult,
                    "unrealized": 0.0 if mark is None else (mark - cost) * qty * mult,
                    "is_cash": False,
                })
        return rows

if __name__ == "__main__":
    led = PositionLedger(executor=None, config={"s1": {"capital_allocation": 100_000.0},
                                               "s2": {"capital_allocation": 100_000.0}})

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

    # cash as a position: NAV = cash + market value == starting cash + realized + unrealized
    snap = led.equity_snapshot({"AAPL": 72.0, "TSLA": 285.0})
    for sid, v in sorted(snap.items()):
        print(f"{sid}: cash={v['cash']:,.2f} positions={v['position_value']:,.2f} "
              f"nav={v['nav']:,.2f} (start {v['starting_cash']:,.0f} + pnl {v['equity']:,.2f})")
        assert abs(v["nav"] - (v["starting_cash"] + v["equity"])) < 1e-6
    print("book s1:", led.strategy_book("s1", {"AAPL": 72.0}))
