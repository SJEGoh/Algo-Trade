"""src/execution/netting.py — multi-strategy net-pooling coordinator.

Each strategy owns a DESIRED BOOK (its target position per symbol). The coordinator
holds ONE net position per symbol at the broker (= sum of all books) and attributes
each fill to the strategies that OWNED the order that filled, so per-strategy P&L and risk
stay correct. Strategy positions move on fills, never on submission.

Two ways a strategy updates its book:
  * set_target(sid, symbol, qty, ...)  — incremental (one symbol); good for event-driven
    strategies. ABSOLUTE targets only (self-correcting). Exits are EXPLICIT: send qty=0.
  * submit_book(sid, intents)          — full-book replace; the authoritative snapshot that
    self-heals drift and closes any name the strategy stopped mentioning. Run periodically.

Invariant maintained:  sum_over_strategies(strategy_positions[*][sym]) == net position[sym].
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path

_EPS = 1e-9

logger = logging.getLogger("executor")

#: Ledger buckets. They hold residue, not intent, so they are never an order's owner.
_BOOKKEEPING = frozenset({"__net__", "flatten_all", "kill_switch"})


class NettingCoordinator:
    NET_SID = "__net__"

    def __init__(self, executor, config, state_path=None, internal_crossing: bool = False):
        self.ex = executor
        self.config = config
        self.desired = {}      # {strategy_id: {symbol: qty}}
        self.instrument = {}   # {symbol: instrument dict}  (how to build the contract)
        self.ref_price = {}    # {symbol: price}            (notional + net order)
        #: Book offsetting legs against each other at the reference price, with NO fill.
        #: Off by default: a strategy's position should move only when the broker fills.
        self.internal_crossing = bool(internal_crossing)
        #: order_id -> {symbol, gaps {sid: qty}, delta, filled}. Who each working order is
        #: FOR, frozen when it is placed. In memory only, on purpose: IB reuses order ids
        #: across restarts, so a persisted entry could attach to an unrelated order in the
        #: next session.
        self.order_owners = {}
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
        """Contract multiplier, or None when a futures leg's multiplier is unknown —
        defaulting a future to 1.0 understates its notional by the multiplier (1,000x on
        CL), which is a hole in the allocation cap, not a harmless default."""
        inst = self.instrument.get(sym) or {}
        m = inst.get("multiplier")
        if m is None:
            m = getattr(self.ex.ledger, "multipliers", {}).get(sym)   # learned from IB contracts
        if m is None:
            return None if inst.get("sec_type") == "FUT" else 1.0
        try:
            m = float(m)
        except (TypeError, ValueError):
            return None
        return m if math.isfinite(m) and m > 0 else None

    def _unit(self, sym):
        """Value of ONE unit of `sym` (price * multiplier), or None if it can't be valued.
        None must never be read as zero: an unpriced leg counted as $0 of notional is an
        allocation cap that any order size passes."""
        px = self.ref_price.get(sym)
        try:
            px = float(px)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(px) or px <= 0:
            return None
        mult = self._mult(sym)
        return None if mult is None else px * mult

    def _exposure(self, sid):
        """(gross_notional, unvaluable_symbols) for a strategy.

        Counts every symbol the strategy WANTS or still HOLDS, at the larger of the two
        quantities. Held positions matter because a desired book that was reset — lost
        netting.json, a fresh container volume, a restart before the first resync — looks
        empty while the strategy is still carrying the risk, so checking the book alone
        lets it re-book a full allocation on top of what it already owns."""
        want = self.desired.get(sid, {}) or {}
        held = (getattr(self.ex.ledger, "strategy_positions", {}) or {}).get(sid, {}) or {}
        gross, unvaluable = 0.0, []
        for sym in set(want) | set(held):
            qty = max(abs(float(want.get(sym, 0.0))), abs(float(held.get(sym, 0.0))))
            if qty <= _EPS:
                continue
            unit = self._unit(sym)
            if unit is None:
                unvaluable.append(sym)
                continue
            gross += qty * unit
        return gross, sorted(unvaluable)

    def _gross(self, sid):
        """Gross notional a strategy is exposed to (unvaluable legs excluded — callers
        must check `_exposure` for those rather than trusting this number alone)."""
        return self._exposure(sid)[0]

    def _check_allocation(self, sid, previous_gross=None):
        """The pooled path's allocation gate. Returns a rejection dict, or None to accept.
        Fails closed: a leg that can't be valued is a rejection, never a free pass.

        `previous_gross` is the strategy's exposure BEFORE this book change. A book that
        shrinks is always allowed, whatever the cap says — otherwise lowering a strategy's
        allocation could never be carried out, because `_exposure` counts what it still
        HOLDS: the reduction itself would be rejected for breaching the new, smaller cap."""
        gross, unvaluable = self._exposure(sid)
        if unvaluable:
            return {"accepted": False,
                    "reason": f"{sid}: cannot value {', '.join(unvaluable)} — missing/invalid "
                              f"reference price or contract multiplier; order rejected"}
        if not math.isfinite(gross):
            return {"accepted": False, "reason": f"{sid}: gross notional is not a number"}
        if previous_gross is not None and gross <= previous_gross + _EPS:
            return None                       # de-risking is never blocked
        alloc = float(self.config[sid]["capital_allocation"])
        if gross > alloc:
            return {"accepted": False,
                    "reason": f"{sid} desired gross {gross:,.0f} exceeds allocation {alloc:,.0f}"}

        # Same portfolio-wide cap /orders enforces — pooled books were never checked against it.
        max_gross = None
        rm = getattr(self.ex, "risk_manager", None)
        if rm is not None and hasattr(rm, "max_gross_exposure"):
            max_gross = rm.max_gross_exposure()
        if max_gross is not None:
            total = 0.0
            for other in set(self.desired) | set(getattr(self.ex.ledger, "strategy_positions", {})):
                g, u = self._exposure(other)
                if u:
                    return {"accepted": False,
                            "reason": f"cannot value {', '.join(u)} for {other} — "
                                      "portfolio gross check failed closed"}
                total += g
            if total > float(max_gross):
                return {"accepted": False,
                        "reason": f"portfolio gross {total:,.0f} exceeds GLOBAL "
                                  f"max_gross_exposure {float(max_gross):,.0f}"}
        return None

    def _set_ref_price(self, sym, price) -> None:
        """Record a reference price ONLY if it can actually value the leg. Storing 0.0 (or
        NaN) for a missing price is what let an unpriced book pass the allocation cap."""
        try:
            px = float(price)
        except (TypeError, ValueError):
            return
        if math.isfinite(px) and px > 0:
            self.ref_price[sym] = px

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
            self._set_ref_price(symbol, price)
            before_gross, before_unvaluable = self._exposure(sid)
            book = self.desired.setdefault(sid, {})
            prev = book.get(symbol)
            if qty == 0:
                book.pop(symbol, None)
            else:
                book[symbol] = float(qty)
            # Exits always pass: closing risk can't be blocked by a missing price.
            rejection = None if qty == 0 else self._check_allocation(
                sid, None if before_unvaluable else before_gross)
            if rejection is not None:
                if prev is None:
                    book.pop(symbol, None)
                else:
                    book[symbol] = prev
                return rejection
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
                # NOT `or 0.0` — an absent price must stay absent so the allocation check
                # can reject the book, instead of valuing the leg at zero and passing it.
                self._set_ref_price(sym, it.get("expected_price") if it.get("expected_price")
                                    is not None else it.get("limit_price"))
                q = float(it["target_quantity"])
                if q != 0:
                    new_book[sym] = q
            old = self.desired.get(sid, {})
            before_gross, before_unvaluable = self._exposure(sid)
            self.desired[sid] = new_book
            rejection = self._check_allocation(
                sid, None if before_unvaluable else before_gross)
            if rejection is not None:
                self.desired[sid] = old
                return rejection
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
    def _owner_gaps(self, sym):
        """Each strategy's outstanding change in `sym`: desired minus FILLED position.

        This is what a new order for `sym` is for. Pending is deliberately excluded — the
        rebalance cancels every working order for the symbol before placing, so the gaps
        and the new order describe the same shares."""
        gaps = {}
        for sid, book in self.desired.items():
            if sid in _BOOKKEEPING:
                continue
            want = book.get(sym, 0.0)
            have = self.ex.ledger.strategy_positions.get(sid, {}).get(sym, 0.0)
            if abs(want - have) > _EPS:
                gaps[sid] = want - have
        return gaps

    def _register(self, oid, sym, gaps, delta):
        """Freeze who an order is for, at the moment it is placed."""
        if oid is None:
            return
        self.order_owners[oid] = {"symbol": sym, "gaps": dict(gaps),
                                  "delta": float(delta), "filled": 0.0}
        status = getattr(self.ex, "order_status", None)
        if isinstance(status, dict) and oid in status:
            status[oid]["owners"] = dict(gaps)            # visible on /orders

    def _working_orders(self, sym):
        """Registered orders for `sym` that can still fill."""
        status = getattr(self.ex, "order_status", None)
        out = []
        for oid, rec in self.order_owners.items():
            if rec["symbol"] != sym or abs(rec["filled"]) >= abs(rec["delta"]) - _EPS:
                continue
            if isinstance(status, dict):
                st = status.get(oid) or {}
                if st.get("ack") == "rejected" or st.get("status") not in (
                        "PreSubmitted", "Submitted", "PendingSubmit"):
                    continue
            out.append(oid)
        return out

    @staticmethod
    def _is_exact_offset(gaps):
        return (any(g > 0 for g in gaps.values()) and any(g < 0 for g in gaps.values())
                and abs(sum(gaps.values())) < _EPS)

    def _place_offsetting_legs(self, sym, gaps, urgent):
        """Legs that cancel out exactly still need real fills. One net order for zero shares
        places nothing, so neither strategy's position could ever move — send the buyers
        and the sellers to the broker as two orders, each owned by its own side."""
        placed = []
        buys = {s: g for s, g in gaps.items() if g > 0}
        sells = {s: g for s, g in gaps.items() if g < 0}
        for side in (buys, sells):
            qty = sum(side.values())
            oid = self.ex.place_net_order(sym, qty, self.instrument.get(sym),
                                          self.ref_price.get(sym), urgent=urgent)
            self._register(oid, sym, side, qty)
            placed.append({"symbol": sym, "delta": qty, "order_id": oid, "offset_leg": True})
        logger.info("Offsetting legs for %s sent as two orders: %+g / %+g",
                    sym, sum(buys.values()), sum(sells.values()))
        return placed

    def _rebalance(self, symbols, urgent: bool = False):
        """Send each symbol's net change to IB as one order, owned by the strategies it is for.

        Positions do NOT move here. They move in attribute_fill when the broker reports a
        fill, and only for the strategies that owned the order that filled. Opposing legs
        still net into a single order; they settle when it fills, at the real price. Internal
        crossing — booking them against each other at the reference price with no fill — runs
        only if the coordinator was built with internal_crossing=True.

        Takes the lock itself: the flatten and kill paths call this directly while fills can
        be arriving on the IB thread, and both touch order_owners.
        urgent=True skips ATR (used by flatten / kill_switch).
        Returns {"orders": [...], "internal_crosses": [...]}."""
        with self._lock:
            crosses = self._internal_cross(symbols) if self.internal_crossing else []
            net = self.net()
            placed = []
            for sym in symbols:
                target = net.get(sym, 0.0)
                if abs(target - self.ex.ledger.effective_position(sym)) < _EPS:
                    # Covered by filled + working orders — unless strategies offset each
                    # other exactly and nothing is working to settle them.
                    gaps = self._owner_gaps(sym)
                    if self._is_exact_offset(gaps) and not self._working_orders(sym):
                        placed += self._place_offsetting_legs(sym, gaps, urgent)
                    continue
                self.ex._cancel_open_orders_for_symbol(sym)      # cancel stale in-flight first
                delta = target - self.ex.ledger.effective_position(sym)
                gaps = self._owner_gaps(sym)
                if abs(delta) < _EPS:
                    if self._is_exact_offset(gaps):
                        placed += self._place_offsetting_legs(sym, gaps, urgent)
                    continue
                oid = self.ex.place_net_order(sym, delta, self.instrument.get(sym),
                                              self.ref_price.get(sym), urgent=urgent)
                self._register(oid, sym, gaps, delta)
                placed.append({"symbol": sym, "delta": delta, "order_id": oid})
            return {"orders": placed, "internal_crosses": crosses}

    # ---------------- fill attribution ----------------
    def attribute_fill(self, symbol, filled_signed, price, order_id=None):
        """Book a fill to the strategies that OWNED the order that filled.

        Owners and their shares were frozen when the order was placed, so a fill cannot be
        claimed by a strategy that merely has an open gap in the same symbol when it arrives.
        That was the old behaviour: owners were re-derived from the live desired book, so
        strategy B's fill could land on strategy A's fresh target — A's position moved
        before A's own order had filled, and a futures strategy once collected 10 shares of
        MSFT that way.

        Partial fills split pro-rata across the owners; the record is dropped once the
        order's full quantity has filled. An order with no record (placed before a restart,
        or a caller that does not pass order_id) falls back to the live desired book.
        Returns a list of (strategy_id, attributed_qty) tuples for DB logging."""
        with self._lock:
            rec = self.order_owners.get(order_id) if order_id is not None else None
            if rec is not None and rec["symbol"] == symbol:
                gaps = rec["gaps"]
                total = sum(gaps.values())
                rec["filled"] += filled_signed
                if abs(rec["filled"]) >= abs(rec["delta"]) - _EPS:
                    del self.order_owners[order_id]
                if not gaps or abs(total) < _EPS:
                    # an order nobody owned, e.g. closing a position no strategy holds
                    self.ex.ledger.apply_attributed_fill(symbol, filled_signed, price, self.NET_SID)
                    return [(self.NET_SID, filled_signed)]
                scale = filled_signed / total
                attributed = []
                for sid, gap in gaps.items():
                    sub_qty = gap * scale
                    self.ex.ledger.apply_attributed_fill(symbol, sub_qty, price, sid)
                    attributed.append((sid, sub_qty))
                return attributed

            if order_id is not None:
                logger.warning("fill for order %s (%s) has no recorded owners — placed before a "
                               "restart or outside the coordinator; attributing by the current "
                               "desired book instead", order_id, symbol)
            changes = {}
            for sid, book in self.desired.items():
                if sid in _BOOKKEEPING:
                    continue
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
