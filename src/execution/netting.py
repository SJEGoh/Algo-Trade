"""src/execution/netting.py — multi-strategy net-pooling coordinator.

Each strategy owns a DESIRED BOOK (its target position per symbol). The coordinator
holds ONE net position per symbol at the broker (= sum of all books) and attributes
fills back to strategies so per-strategy P&L and risk stay correct.

Two ways a strategy updates its book:
  * set_target(sid, symbol, qty, ...)  — incremental (one symbol); good for event-driven
    strategies. ABSOLUTE targets only (self-correcting). Exits are EXPLICIT: send qty=0.
  * submit_book(sid, intents)          — full-book replace; the authoritative snapshot that
    self-heals drift and closes any name the strategy stopped mentioning. Run periodically.

Invariant maintained:  sum_over_strategies(strategy_positions[*][sym]) == net position[sym].
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

_EPS = 1e-9


class NettingCoordinator:
    NET_SID = "__net__"

    def __init__(self, executor, config, state_path=None):
        self.ex = executor
        self.config = config
        self.desired = {}      # {strategy_id: {symbol: qty}}
        self.instrument = {}   # {symbol: instrument dict}  (how to build the contract)
        self.ref_price = {}    # {symbol: price}            (notional + net order)
        self.state_path = Path(state_path) if state_path else None
        self._lock = threading.RLock()
        self._load()

    # ---------------- persistence ----------------
    def _load(self):
        if self.state_path and self.state_path.exists():
            try:
                d = json.loads(self.state_path.read_text())
                self.desired = {s: {k: float(v) for k, v in b.items()}
                                for s, b in d.get("desired", {}).items()}
                self.instrument = d.get("instrument", {})
                self.ref_price = {k: float(v) for k, v in d.get("ref_price", {}).items()}
            except Exception:
                pass

    def _save(self):
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(
                {"desired": self.desired, "instrument": self.instrument, "ref_price": self.ref_price}))

    # ---------------- helpers ----------------
    def _mult(self, sym):
        return float((self.instrument.get(sym) or {}).get("multiplier") or 1.0)

    def _gross(self, sid):
        return sum(abs(q) * self.ref_price.get(s, 0.0) * self._mult(s)
                   for s, q in self.desired.get(sid, {}).items())

    def net(self) -> dict:
        out = {}
        for book in self.desired.values():
            for s, q in book.items():
                out[s] = out.get(s, 0.0) + q
        return out

    # ---------------- book updates ----------------
    def set_target(self, sid, symbol, qty, instrument=None, price=None):
        """Incremental: set ONE symbol's target for a strategy, then re-net it."""
        with self._lock:
            if not self.ex.risk_manager.is_active(sid):
                return {"accepted": False, "reason": f"{sid} not active"}
            if instrument is not None:
                self.instrument[symbol] = instrument
            if price is not None:
                self.ref_price[symbol] = float(price)
            book = self.desired.setdefault(sid, {})
            prev = book.get(symbol)
            if qty == 0:
                book.pop(symbol, None)
            else:
                book[symbol] = float(qty)
            alloc = self.config[sid]["capital_allocation"]
            if self._gross(sid) > alloc:
                if prev is None:
                    book.pop(symbol, None)
                else:
                    book[symbol] = prev
                return {"accepted": False,
                        "reason": f"{sid} desired gross exceeds allocation {alloc:.0f}"}
            self._save()
            rebal = self._rebalance({symbol})
            return {"accepted": True, **rebal}

    def submit_book(self, sid, intents):
        """Full-book replace: authoritative snapshot. Closes any name dropped from the book."""
        with self._lock:
            if not self.ex.risk_manager.is_active(sid):
                return {"accepted": False, "reason": f"{sid} not active"}
            new_book = {}
            for it in intents:
                sym = it["instrument"]["symbol"]
                self.instrument[sym] = it["instrument"]
                self.ref_price[sym] = float(it.get("expected_price") or it.get("limit_price") or 0.0)
                q = float(it["target_quantity"])
                if q != 0:
                    new_book[sym] = q
            old = self.desired.get(sid, {})
            self.desired[sid] = new_book
            alloc = self.config[sid]["capital_allocation"]
            if self._gross(sid) > alloc:
                self.desired[sid] = old
                return {"accepted": False,
                        "reason": f"{sid} desired gross exceeds allocation {alloc:.0f}"}
            self._save()
            rebal = self._rebalance(set(old) | set(new_book))
            return {"accepted": True, **rebal}

    def halt(self, sid):
        """Flatten a strategy's book (keep the entry so its unwind attributes to it)."""
        with self._lock:
            old = self.desired.get(sid) or {}
            self.desired[sid] = {}
            self._save()
            return self._rebalance(set(old))

    # ---------------- internal crossing ----------------
    def _internal_cross(self, symbols):
        """Cross offsetting strategy deltas internally at the reference price.
        Returns list of dicts: {symbol, strategy_id, side, quantity, price}.
        Updates per-strategy positions via apply_internal_cross (zero-sum,
        no change to current_positions or pending_deltas)."""
        import logging
        logger = logging.getLogger("executor")
        crosses = []
        _counter = [0]

        for sym in symbols:
            price = self.ref_price.get(sym)
            if not price or price <= 0:
                continue

            # Per-strategy deltas: what each strategy still needs
            deltas = {}
            for sid, book in self.desired.items():
                want = book.get(sym, 0.0)
                have = self.ex.ledger.strategy_positions.get(sid, {}).get(sym, 0.0)
                d = want - have
                if abs(d) > _EPS:
                    deltas[sid] = d

            if not deltas:
                continue

            buyers = {sid: d for sid, d in deltas.items() if d > 0}
            sellers = {sid: abs(d) for sid, d in deltas.items() if d < 0}

            if not buyers or not sellers:
                continue  # all same direction — nothing to cross

            total_buy = sum(buyers.values())
            total_sell = sum(sellers.values())
            crossable = min(total_buy, total_sell)

            if crossable < _EPS:
                continue

            # Pro-rata allocation of the cross to each side
            buy_scale = crossable / total_buy
            sell_scale = crossable / total_sell

            for sid, qty in buyers.items():
                fill_qty = qty * buy_scale
                self.ex.ledger.apply_internal_cross(sym, fill_qty, price, sid)
                crosses.append({
                    "symbol": sym, "strategy_id": sid,
                    "side": "BOT", "quantity": abs(fill_qty), "price": price,
                })
                # Log to DB
                _counter[0] += 1
                exec_id = f"xnet-{sym}-{int(time.time()*1000)}-{_counter[0]}"
                self.ex.logger_db.log_fill(
                    0, exec_id, sym, "BOT", price, abs(fill_qty), sid,
                    expected_price=price,
                )

            for sid, qty in sellers.items():
                fill_qty = qty * sell_scale
                self.ex.ledger.apply_internal_cross(sym, -fill_qty, price, sid)
                crosses.append({
                    "symbol": sym, "strategy_id": sid,
                    "side": "SLD", "quantity": abs(fill_qty), "price": price,
                })
                _counter[0] += 1
                exec_id = f"xnet-{sym}-{int(time.time()*1000)}-{_counter[0]}"
                self.ex.logger_db.log_fill(
                    0, exec_id, sym, "SLD", price, abs(fill_qty), sid,
                    expected_price=price,
                )

            logger.info("InternalCross %s: crossed %.1f shares @ %.2f (%d buyers, %d sellers)",
                        sym, crossable, price, len(buyers), len(sellers))

        if crosses:
            self.ex.ledger.save_state(self.ex.logger_db)  # persist after internal crosses
        return crosses

    # ---------------- rebalance to net ----------------
    def _rebalance(self, symbols):
        """Cross internally first, then send residual net delta to IB.
        Returns {"orders": [...], "internal_crosses": [...]}."""
        crosses = self._internal_cross(symbols)
        net = self.net()
        placed = []
        for sym in symbols:
            target = net.get(sym, 0.0)
            if abs(target - self.ex.ledger.effective_position(sym)) < _EPS:
                continue
            self.ex._cancel_open_orders_for_symbol(sym)          # cancel stale in-flight first
            delta = target - self.ex.ledger.effective_position(sym)
            if abs(delta) < _EPS:
                continue
            oid = self.ex.place_net_order(sym, delta, self.instrument.get(sym), self.ref_price.get(sym))
            placed.append({"symbol": sym, "delta": delta, "order_id": oid})
        return {"orders": placed, "internal_crosses": crosses}

    # ---------------- fill attribution ----------------
    def attribute_fill(self, symbol, filled_signed, price):
        """Decompose a net fill into per-strategy sub-fills at the fill price, so each
        strategy books only its own change (correct P&L even with opposing legs).
        Returns a list of (strategy_id, attributed_qty) tuples for DB logging."""
        with self._lock:
            changes = {}
            for sid, book in self.desired.items():
                want = book.get(symbol, 0.0)
                have = self.ex.ledger.strategy_positions.get(sid, {}).get(symbol, 0.0)
                if abs(want - have) > _EPS:
                    changes[sid] = want - have
            total = sum(changes.values())
            if abs(total) < _EPS:
                self.ex.ledger.apply_attributed_fill(symbol, filled_signed, price, self.NET_SID)
                return [(self.NET_SID, filled_signed)]
            scale = filled_signed / total                       # pro-rata; exact on a full fill
            attributed = []
            for sid, ch in changes.items():
                sub_qty = ch * scale
                self.ex.ledger.apply_attributed_fill(symbol, sub_qty, price, sid)
                attributed.append((sid, sub_qty))
            return attributed
