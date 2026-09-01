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
            return {"accepted": True, "orders": self._rebalance({symbol})}

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
            return {"accepted": True, "orders": self._rebalance(set(old) | set(new_book))}

    def halt(self, sid):
        """Flatten a strategy's book (keep the entry so its unwind attributes to it)."""
        with self._lock:
            old = self.desired.get(sid) or {}
            self.desired[sid] = {}
            self._save()
            return self._rebalance(set(old))

    # ---------------- rebalance to net ----------------
    def _rebalance(self, symbols):
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
        return placed

    # ---------------- fill attribution ----------------
    def attribute_fill(self, symbol, filled_signed, price):
        """Decompose a net fill into per-strategy sub-fills at the fill price, so each
        strategy books only its own change (correct P&L even with opposing legs)."""
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
                return
            scale = filled_signed / total                       # pro-rata; exact on a full fill
            for sid, ch in changes.items():
                self.ex.ledger.apply_attributed_fill(symbol, ch * scale, price, sid)
