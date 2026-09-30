"""execution/chains.py — chained two-leg orders on the netting pool.

A chain moves one strategy from its current position to a new target in TWO symbols at once
(a pair, a stock and its hedge). Both legs rest as limit orders priced by the ATR layer —
buys at price − f·ATR, sells at price + f·ATR — and the first leg to fill LEADS:

  * the other leg's resting limit is pulled and it is completed through the pool in
    proportion to the lead's progress (30 of 100 lead shares -> 30% of the hedge), urgent,
    so it nets against other strategies and the rest goes at market;
  * the lead keeps resting at its ATR price for the remainder, and every further fill tops
    the hedge up to match;
  * at expiry (the caller's TTL, and never later than the MOC cutoff) whatever still rests
    is cancelled. Nothing is sent at market on expiry: the pair is left smaller but
    balanced — or untouched, if nothing filled.

Nothing here places an order directly. The legs are the strategy's DESIRED targets with a
"limit" execution style, so every existing mechanism applies unchanged: allocation caps,
owner-frozen fill attribution, cancel-and-replace when another strategy re-nets the symbol
(the leg is re-placed at the same limit), and pending accounting. The hedge is an ordinary
urgent set_target. Fills reach `on_fill` from the coordinator's attribute_fill, so a chain
reacts to the same event that moves the strategy's position.

State lives in the coordinator's netting.json with the desired books, so a restart does not
leave a leg resting at the broker with nothing waiting to hedge it.
"""
from __future__ import annotations

import logging
import time
import uuid

logger = logging.getLogger("executor.chains")

_EPS = 1e-9
ACTIVE = ("working", "legging")


class ChainError(ValueError):
    """A chain that cannot be placed as asked."""


def _progress(leg: dict, pos: float) -> float:
    """Fraction of the leg's change that has filled (0 = none, 1 = all)."""
    return max(0.0, (pos - leg["start"]) / leg["delta"])


def _ahead(a: float, b: float, direction: float) -> bool:
    """Is `a` further than `b` in the leg's direction of travel?"""
    return (a - b) * direction > _EPS


class ChainManager:
    def __init__(self, coordinator, clock=time.time):
        self.co = coordinator
        self.clock = clock
        #: chain_id -> record (see submit). Terminal chains are kept for inspection.
        self.chains = {}

    # ------------------------------------------------------------------ helpers
    def _pos(self, sid, sym) -> float:
        return float(self.co.ex.ledger.strategy_positions.get(sid, {}).get(sym, 0.0) or 0.0)

    def active_on(self, sid, sym):
        for c in self.chains.values():
            if (c["strategy_id"] == sid and c["status"] in ACTIVE
                    and any(l["symbol"] == sym for l in c["legs"])):
                return c
        return None

    def _set(self, c, leg, qty, urgent=False):
        """Move one leg's desired target through the ordinary pooled path."""
        return self.co.set_target(c["strategy_id"], leg["symbol"], qty,
                                  instrument=leg["instrument"], price=leg["expected_price"],
                                  urgent=urgent, chain_id=c["id"])

    # ------------------------------------------------------------------ placing
    def submit(self, sid, legs, expires_at, fraction=None):
        """Validate and place a chain. `legs`: two dicts with instrument, target_quantity,
        expected_price and limit_price (already ATR-priced by the caller). Returns the
        coordinator-shaped result: {"accepted": ..., "chain": ..., "orders": ...}."""
        co = self.co
        with co._lock:
            if not co.ex.risk_manager.is_active(sid):
                return {"accepted": False, "reason": f"{sid} not active"}
            problems, built = [], []
            if len(legs) != 2:
                problems.append(f"a chain has exactly two legs, got {len(legs)}")
            syms = [(l.get("instrument") or {}).get("symbol") for l in legs]
            if len(set(syms)) != len(syms):
                problems.append("the two legs must be different symbols")
            for leg, sym in zip(legs, syms):
                pos = self._pos(sid, sym)
                target = float(leg["target_quantity"])
                desired = co.desired.get(sid, {}).get(sym, 0.0)
                if abs(target - pos) < _EPS:
                    problems.append(f"{sym}: already at {target:g} — a chain leg must trade")
                if abs(desired - pos) > _EPS:
                    # an unfilled change already outstanding: the chain measures its fills
                    # from the position, so it would count someone else's shares as its own
                    problems.append(f"{sym}: {desired - pos:+g} still outstanding from an "
                                    "earlier target — wait for it to fill or cancel it first")
                if self.active_on(sid, sym):
                    problems.append(f"{sym}: already in a working chain")
                built.append({"symbol": sym, "instrument": dict(leg["instrument"]),
                              "start": pos, "target": target, "delta": target - pos,
                              "expected_price": float(leg["expected_price"]),
                              "limit_price": float(leg["limit_price"]),
                              "atr": leg.get("atr")})
            if problems:
                return {"accepted": False, "reason": "chain refused, nothing placed: "
                        + "; ".join(problems)}

            cid = f"chain-{uuid.uuid4().hex[:12]}"
            c = {"id": cid, "strategy_id": sid, "status": "working", "lead": None,
                 "hedged_to": None, "legs": built, "atr_fraction": fraction,
                 "created_at": self.clock(), "expires_at": float(expires_at), "reason": None}

            book = co.desired.setdefault(sid, {})
            before_gross, before_unvaluable = co._exposure(sid)
            prev = {l["symbol"]: book.get(l["symbol"]) for l in built}
            prev_style = {l["symbol"]: co.exec_style.get(sid, {}).get(l["symbol"]) for l in built}
            for l in built:
                co.instrument[l["symbol"]] = l["instrument"]
                co._set_ref_price(l["symbol"], l["expected_price"])
                if l["target"]:
                    book[l["symbol"]] = l["target"]
                else:
                    book.pop(l["symbol"], None)
                co.exec_style.setdefault(sid, {})[l["symbol"]] = {
                    "type": "limit", "price": l["limit_price"], "chain": cid,
                    "side": 1 if l["delta"] > 0 else -1}
            rejection = co._check_allocation(sid, None if before_unvaluable else before_gross)
            if rejection is not None:
                for sym, q in prev.items():
                    if q is None:
                        book.pop(sym, None)
                    else:
                        book[sym] = q
                    co._set_style(sid, sym, prev_style[sym])
                return rejection
            self.chains[cid] = c
            co._save()
            rebal = co._rebalance({l["symbol"] for l in built})
            logger.info("chain %s (%s): %s", cid, sid, ", ".join(
                f"{l['symbol']} {l['delta']:+g} lmt {l['limit_price']:g}" for l in built))
            return {"accepted": True, "chain": self.view(c), **rebal}

    # ------------------------------------------------------------------ reacting
    def on_fill(self, symbol):
        """A fill moved some strategy's position in `symbol`: advance every chain on it.
        Runs for finished chains too — a cancel can lose the race to a fill, and that
        late fill still has to be hedged."""
        for c in list(self.chains.values()):
            if c["status"] == "done" or not any(l["symbol"] == symbol for l in c["legs"]):
                continue
            try:
                self._advance(c)
            except Exception as e:
                logger.critical("CHAIN %s NOT HEDGED — %s filled but the other leg could not "
                                "be sent: %s", c["id"], symbol, e)

    def _advance(self, c):
        sid, legs = c["strategy_id"], c["legs"]
        pos = [self._pos(sid, l["symbol"]) for l in legs]
        prog = [_progress(l, p) for l, p in zip(legs, pos)]
        if c["lead"] is None:
            if max(prog) <= _EPS:
                return
            c["lead"] = 0 if prog[0] >= prog[1] else 1
            if c["status"] == "working":
                c["status"] = "legging"
            logger.info("chain %s: %s filled first — hedging %s", c["id"],
                        legs[c["lead"]]["symbol"], legs[1 - c["lead"]]["symbol"])
        lead, hedge = legs[c["lead"]], legs[1 - c["lead"]]
        h_pos = pos[1 - c["lead"]]
        # A chain that is no longer working has had its lead remainder cancelled; a late
        # fill past that must be kept, or the pool trades it straight back out.
        l_pos = pos[c["lead"]]
        if c["status"] not in ACTIVE:
            desired = self.co.desired.get(sid, {}).get(lead["symbol"], 0.0)
            if _ahead(l_pos, desired, lead["delta"]):
                self._set(c, lead, l_pos)
        need = hedge["start"] + round(min(prog[c["lead"]], 1.0) * hedge["delta"])
        if _ahead(h_pos, need, hedge["delta"]):
            need = h_pos                      # never unwind shares the hedge already has
        if c["hedged_to"] is None or _ahead(need, c["hedged_to"], hedge["delta"]):
            c["hedged_to"] = need
            self._set(c, hedge, need, urgent=True)
        if (c["status"] in ACTIVE and prog[c["lead"]] >= 1 - _EPS
                and abs(h_pos - hedge["target"]) < _EPS):
            c["status"] = "done"
            self._drop_styles(c)
            logger.info("chain %s done", c["id"])
        self.co._save()

    # ------------------------------------------------------------------ ending
    def expire(self, now=None):
        """Cancel every chain past its expiry. Returns the chains expired."""
        now = self.clock() if now is None else now
        out = []
        with self.co._lock:
            for c in list(self.chains.values()):
                if c["status"] in ACTIVE and now >= c["expires_at"]:
                    self._stop(c, "expired", "expired")
                    out.append(self.view(c))
        return out

    def cancel(self, cid, reason="cancelled by request"):
        with self.co._lock:
            c = self.chains.get(cid)
            if c is None:
                return None
            if c["status"] in ACTIVE:
                self._stop(c, "cancelled", reason)
            return self.view(c)

    def _stop(self, c, status, reason, only=None):
        """Pull what still rests and leave the pair balanced: nothing filled -> both legs
        back to where they started; legging -> the lead stays at what filled (the hedge
        already matches it). Nothing is sent at market. `only` limits the unwinding to
        those legs (the others were taken over by the caller)."""
        with self.co._lock:
            c["status"], c["reason"] = status, reason
            sid = c["strategy_id"]
            self._drop_styles(c)
            for i, leg in enumerate(c["legs"]):
                if only is not None and leg["symbol"] not in only:
                    continue
                if c["lead"] is None:
                    self._set(c, leg, leg["start"])
                elif i == c["lead"]:
                    self._set(c, leg, self._pos(sid, leg["symbol"]))
            self.co._save()
        logger.warning("chain %s %s (%s)", c["id"], status, reason)

    def supersede(self, sid, symbols=None, unwind=True):
        """Something other than the chain has taken over its symbols — a /targets book, an
        exit, a halt. The chain stops managing them. With `unwind`, legs the caller did not
        touch are stopped as on expiry; without, the caller is about to restate them all."""
        for c in list(self.chains.values()):
            if c["strategy_id"] != sid or c["status"] not in ACTIVE:
                continue
            legs = {l["symbol"] for l in c["legs"]}
            if symbols is not None and not legs & set(symbols):
                continue
            if unwind:
                untouched = legs - set(symbols or ())
                self._stop(c, "superseded", f"taken over: {sorted(legs - untouched)}",
                           only=untouched)
            else:
                c["status"], c["reason"] = "superseded", "replaced by a new book"
                self._drop_styles(c)

    def _drop_styles(self, c):
        styles = self.co.exec_style.get(c["strategy_id"], {})
        for leg in c["legs"]:
            st = styles.get(leg["symbol"])
            if isinstance(st, dict) and st.get("chain") == c["id"]:
                styles.pop(leg["symbol"], None)
        if not styles:
            self.co.exec_style.pop(c["strategy_id"], None)

    # ------------------------------------------------------------------ inspection
    def view(self, c):
        sid = c["strategy_id"]
        legs = []
        for l in c["legs"]:
            p = self._pos(sid, l["symbol"])
            legs.append({**l, "position": p, "filled": p - l["start"],
                         "progress": round(_progress(l, p), 6)})
        return {**c, "legs": legs,
                "lead_symbol": None if c["lead"] is None else c["legs"][c["lead"]]["symbol"]}

    def snapshot(self, strategy_id=None):
        return [self.view(c) for c in self.chains.values()
                if strategy_id is None or c["strategy_id"] == strategy_id]

    def dump(self):
        return self.chains

    def load(self, raw):
        self.chains = dict(raw or {})
