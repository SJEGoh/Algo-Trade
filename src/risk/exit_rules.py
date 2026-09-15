"""risk/exit_rules.py — per-position stop-loss, take-profit and trailing stops.

Opt-in per strategy AND per name. An intent may carry an `exits` object with any
combination of:

    stop_price | stop_pct              close when price moves against the position
    take_profit_price | take_profit_pct close when price moves in its favour
    trail_pct | trail_amount           close when price retraces from its best level

`*_pct` values are fractions (0.05 = 5%) measured from the strategy's own average cost, so
"2% below my fill" can be sent with the entry. A name with no `exits` has no rules; a
strategy that never sends them is untouched.

Each submission REPLACES that name's rules (the book is authoritative, and so is what it
says about exits): resend a stop to keep it, change it to move it, leave it out to drop it.

Enforcement is executor-side, not resting orders at IB. A resting stop would be a working
order, and a new target cancels every working order on its symbol before sizing — so the
next rebalance would silently delete it; and a stop on a pooled position would close other
strategies' shares. Instead a dedicated loop re-prices every armed name from IB every
GLOBAL['exit_check_sec'] (30s), and a hit sets that strategy's target for that name to zero
— pooled names re-net, direct names get a closing order. Either way the close is a MARKET
order that skips ATR and every other execution layer: an exit resting as a limit, waiting
for a pullback, is not an exit. The trade-offs, stated so nobody assumes otherwise: it reacts
on that cadence, not tick by tick, and trailing extremes are sampled at it too; a mark older
than GLOBAL['exit_mark_max_age_sec'] is never acted on; nothing is protected while the
executor is down.

A name that exits is LOCKED OUT of re-entry in the same direction for the rest of the
session (ET date). Without it, a strategy that still likes the name buys it straight back
on its next run and the stop achieves nothing.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger("executor.exits")

ET = ZoneInfo("America/New_York")

#: kind -> (absolute-price field, relative field). One of each pair at most.
KINDS = {
    "stop": ("stop_price", "stop_pct"),
    "take_profit": ("take_profit_price", "take_profit_pct"),
    "trail": ("trail_amount", "trail_pct"),
}
FIELDS = frozenset(f for pair in KINDS.values() for f in pair)

#: Checked in this order: when a single mark crosses two levels, the loss-side exit names it.
_CHECK_ORDER = ("stop", "trail", "take_profit")


class ExitSpecError(ValueError):
    """An exits object that cannot be enforced as written."""


def parse_exits(raw, target_quantity, expected_price=None):
    """Validate an intent's `exits`. Returns the cleaned spec, or None for no rules.

    Fails closed: a stop the executor cannot enforce as written must refuse the submission,
    not arm something different from what the strategy believes it has. A flat target has
    nothing to protect, so its exits are ignored rather than refused."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ExitSpecError("exits must be an object")
    unknown = sorted(set(raw) - FIELDS)
    if unknown:
        raise ExitSpecError(f"unknown exit field(s) {unknown} — allowed: {sorted(FIELDS)}")
    spec, problems = {}, []
    for key, value in raw.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{key} must be a number, got {value!r}")
            continue
        if not math.isfinite(value) or value <= 0:
            problems.append(f"{key} must be positive, got {value!r}")
            continue
        spec[key] = float(value)
    for price_field, pct_field in KINDS.values():
        if price_field in spec and pct_field in spec:
            problems.append(f"set {price_field} or {pct_field}, not both")
    if not spec and not problems:
        return None
    if not target_quantity:
        return None

    long = target_quantity > 0
    for key in ("stop_pct", "trail_pct") + (() if long else ("take_profit_pct",)):
        if spec.get(key, 0) >= 1:
            problems.append(f"{key} is a fraction (0.05 = 5%); {spec[key]:g} would never "
                            "trigger")
    px = expected_price
    if isinstance(px, (int, float)) and math.isfinite(px) and px > 0:
        side = "long" if long else "short"
        stop, tp = spec.get("stop_price"), spec.get("take_profit_price")
        if stop is not None and (stop >= px if long else stop <= px):
            problems.append(f"stop_price {stop:g} is on the wrong side of {px:g} for a "
                            f"{side} — it would trigger immediately")
        if tp is not None and (tp <= px if long else tp >= px):
            problems.append(f"take_profit_price {tp:g} is on the wrong side of {px:g} for a "
                            f"{side} — it would trigger immediately")
    if problems:
        raise ExitSpecError("; ".join(problems))
    return spec


def levels(rule: dict, avg_cost=None) -> dict:
    """The price each armed exit fires at right now. A `*_pct` exit has no level until the
    strategy holds the name at a known cost; a trail has none until it has an extreme."""
    d, spec, out = rule["direction"], rule["spec"], {}
    cost = avg_cost if avg_cost and avg_cost > 0 else None
    if "stop_price" in spec:
        out["stop"] = spec["stop_price"]
    elif "stop_pct" in spec and cost:
        out["stop"] = cost * (1 - d * spec["stop_pct"])
    if "take_profit_price" in spec:
        out["take_profit"] = spec["take_profit_price"]
    elif "take_profit_pct" in spec and cost:
        out["take_profit"] = cost * (1 + d * spec["take_profit_pct"])
    extreme = rule.get("extreme")
    if extreme is not None:
        if "trail_pct" in spec:
            out["trail"] = extreme * (1 - d * spec["trail_pct"])
        elif "trail_amount" in spec:
            out["trail"] = extreme - d * spec["trail_amount"]
    return out


def _has_trail(spec: dict) -> bool:
    return "trail_pct" in spec or "trail_amount" in spec


class ExitManager:
    def __init__(self, executor, alert=None, clock=time.time, max_mark_age: float = None):
        self.ex = executor
        #: {strategy_id: {symbol: {spec, direction, sec_type, extreme, updated_at}}}
        self.rules = {}
        #: {strategy_id: {symbol: {direction, kind, level, mark, quantity, route, session, at}}}
        self.lockouts = {}
        self._alert = alert
        self._clock = clock
        #: oldest mark an exit may act on; None defers to the executor's own staleness limit
        self.max_mark_age = max_mark_age
        #: the marks the last check ran on — what GET /exits shows beside each level
        self.last_marks = {}
        self._lock = threading.RLock()
        self._load()

    # ------------------------------------------------------------------ time
    def session(self) -> str:
        return datetime.fromtimestamp(self._clock(), ET).date().isoformat()

    def _now_iso(self) -> str:
        return datetime.fromtimestamp(self._clock(), timezone.utc).isoformat()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        load = getattr(getattr(self.ex, "logger_db", None), "load_exit_state", None)
        if load is None:
            return
        try:
            raw = load()
            if raw:
                state = json.loads(raw)
                self.rules = state.get("rules", {})
                self.lockouts = state.get("lockouts", {})
        except Exception as e:
            logger.critical("exit rules NOT restored — stops, take-profits and trailing "
                            "stops from before the restart are not being enforced: %s", e)

    def _save(self) -> None:
        save = getattr(getattr(self.ex, "logger_db", None), "save_exit_state", None)
        if save is None:
            return
        try:
            save(json.dumps({"rules": self.rules, "lockouts": self.lockouts}))
        except Exception as e:
            logger.error("exit rules not persisted — they will not survive a restart: %s", e)

    # ------------------------------------------------------------------ registration
    def set(self, sid: str, symbol: str, target: float, spec, sec_type: str = "STK") -> None:
        """Replace one name's rules. No spec, or a flat target, removes them."""
        with self._lock:
            self._set(sid, symbol, target, spec, sec_type)
            self._save()

    def _set(self, sid, symbol, target, spec, sec_type) -> None:
        book = self.rules.setdefault(sid, {})
        if not spec or not target:
            book.pop(symbol, None)
            if not book:
                self.rules.pop(sid, None)
            return
        direction = 1 if target > 0 else -1
        prev = book.get(symbol)
        # A resent trail must keep its high-water mark — strategies resend the book every
        # run, and resetting the extreme each time would make the trail never trail.
        keep = (prev is not None and prev["direction"] == direction
                and _has_trail(prev["spec"]) and _has_trail(spec))
        book[symbol] = {"spec": dict(spec), "direction": direction,
                        "sec_type": sec_type or "STK",
                        "extreme": prev.get("extreme") if keep else None,
                        "updated_at": self._now_iso()}

    def replace_book(self, sid: str, plan: dict) -> None:
        """Authoritative: `plan` is {symbol: (target, spec, sec_type)} for the whole book.
        Rules on names the book no longer carries are dropped."""
        with self._lock:
            for symbol in list(self.rules.get(sid, {})):
                if symbol not in plan:
                    self._set(sid, symbol, 0, None, None)
            for symbol, (target, spec, sec_type) in plan.items():
                self._set(sid, symbol, target, spec, sec_type)
            self._save()

    def clear(self, sid: str, symbol: str) -> dict:
        with self._lock:
            rule = self.rules.get(sid, {}).pop(symbol, None)
            lock = self.lockouts.get(sid, {}).pop(symbol, None)
            for table in (self.rules, self.lockouts):
                if sid in table and not table[sid]:
                    table.pop(sid)
            self._save()
        return {"removed_rule": rule is not None, "cleared_lockout": lock is not None}

    def blocked(self, sid: str, symbol: str, direction: int):
        """The lockout blocking a position in `direction` (+1 long, -1 short), or None.
        Reducing or going flat is never blocked; nor is the opposite direction."""
        if not direction:
            return None
        with self._lock:
            lock = self.lockouts.get(sid, {}).get(symbol)
            if lock is None:
                return None
            if lock["session"] != self.session():
                self._expire_lockouts()
                self._save()
                return None
            return dict(lock) if lock["direction"] == (1 if direction > 0 else -1) else None

    def _expire_lockouts(self) -> bool:
        today, changed = self.session(), False
        for sid in list(self.lockouts):
            for symbol, lock in list(self.lockouts[sid].items()):
                if lock["session"] != today:
                    del self.lockouts[sid][symbol]
                    changed = True
            if not self.lockouts[sid]:
                del self.lockouts[sid]
        return changed

    # ------------------------------------------------------------------ enforcement
    def watched_symbols(self) -> set:
        """Names worth pricing this cycle: an armed rule over a position actually held."""
        with self._lock:
            return {symbol for sid, book in self.rules.items() for symbol in book
                    if abs(self._held(sid, symbol)) > 1e-9}

    def check(self, marks: dict, market_open: bool = True) -> list:
        """One enforcement pass. Returns the exits fired this cycle."""
        fired, retries = [], []
        with self._lock:
            self.last_marks.update({k: v for k, v in marks.items() if v})
            changed = self._expire_lockouts()
            # a close that was placed but has not taken the position off — failed, rejected,
            # cancelled — is re-attempted rather than left holding with no rule over it
            for sid, locks in self.lockouts.items():
                for symbol, lock in locks.items():
                    held = self._held(sid, symbol)
                    if held * lock["direction"] > 1e-9 and self.ex.risk_manager.is_active(sid):
                        retries.append((sid, symbol, lock))

            for sid, book in list(self.rules.items()):
                if not self.ex.risk_manager.is_active(sid):
                    continue                          # a halted strategy is ensure_flat's job
                costs = self.ex.ledger.strategy_avg_cost.get(sid, {})
                for symbol, rule in list(book.items()):
                    held = self._held(sid, symbol)
                    if abs(held) < 1e-9:
                        # not filled yet, or closed: a stale extreme would fire the moment
                        # the name is re-entered
                        if rule.get("extreme") is not None:
                            rule["extreme"] = None
                            changed = True
                        continue
                    if (held > 0) != (rule["direction"] > 0):
                        continue                      # set for the other side of the book
                    if rule["sec_type"] != "FUT" and not market_open:
                        continue                      # no exits into a closed market
                    mark = marks.get(symbol)
                    if (mark is None or mark <= 0
                            or not self.ex.mark_is_fresh(symbol, max_age=self.max_mark_age)):
                        continue                      # never exit on a stale price
                    d = rule["direction"]
                    if _has_trail(rule["spec"]):
                        base = rule.get("extreme")
                        if base is None:
                            base = costs.get(symbol) or mark
                        extreme = max(base, mark) if d > 0 else min(base, mark)
                        if extreme != rule.get("extreme"):
                            rule["extreme"] = extreme
                            changed = True
                    hit = self._hit(rule, costs.get(symbol), mark)
                    if hit is None:
                        continue
                    kind, level = hit
                    lock = {"direction": d, "kind": kind, "level": level, "mark": mark,
                            "quantity": held, "route": self._route(sid, symbol),
                            "session": self.session(), "at": self._now_iso(),
                            "spec": rule["spec"]}
                    self.lockouts.setdefault(sid, {})[symbol] = lock
                    del book[symbol]
                    if not book:
                        del self.rules[sid]
                    fired.append((sid, symbol, lock))
                    changed = True
            if changed:
                self._save()

        # Orders go out WITHOUT this lock held: closing takes the coordinator's lock, and a
        # submission holds that one while it is the other way round.
        for sid, symbol, lock in fired:
            self._announce(sid, symbol, lock)
            self._close(sid, symbol, lock)
        for sid, symbol, lock in retries:
            if not self.ex._has_live_order(sid, symbol):
                logger.error("%s %s exited (%s) but is still held — closing again",
                             sid, symbol, lock["kind"])
                self._close(sid, symbol, lock)
        return [{"strategy_id": s, "symbol": y, **l} for s, y, l in fired]

    @staticmethod
    def _hit(rule: dict, avg_cost, mark: float):
        d, lv = rule["direction"], levels(rule, avg_cost)
        for kind in _CHECK_ORDER:
            if kind not in lv:
                continue
            # A level derived from a percentage is not exact in floating point — 100 * 1.10
            # is 110.00000000000001 — and a mark printing exactly at the level must count.
            tol = 1e-9 * abs(lv[kind])
            if kind == "take_profit":
                if (mark - lv[kind]) * d >= -tol:
                    return kind, lv[kind]
            elif (mark - lv[kind]) * d <= tol:
                return kind, lv[kind]
        return None

    def _held(self, sid: str, symbol: str) -> float:
        return float(self.ex.ledger.strategy_positions.get(sid, {}).get(symbol, 0.0) or 0.0)

    def _route(self, sid: str, symbol: str) -> str:
        co = getattr(self.ex, "coordinator", None)
        return "pooled" if co is not None and symbol in co.desired.get(sid, {}) else "direct"

    def _close(self, sid: str, symbol: str, lock: dict) -> None:
        try:
            if lock["route"] == "pooled":
                # zero the DESIRED book, not just the position — otherwise the next re-net
                # trades straight back to the old target. urgent: at market, past ATR.
                self.ex.coordinator.set_target(sid, symbol, 0, urgent=True)
            elif not self.ex._has_live_order(sid, symbol):
                # place_order directly — the market order goes out as built, no layer applied
                self.ex._flatten_direct(sid, {symbol})
        except Exception as e:
            logger.critical("EXIT NOT PLACED — %s %s hit its %s at %g but the close failed: "
                            "%s. Retrying next cycle.", sid, symbol, lock["kind"],
                            lock["level"], e)

    def _announce(self, sid: str, symbol: str, lock: dict) -> None:
        label = {"stop": "stop-loss", "take_profit": "take-profit",
                 "trail": "trailing stop"}[lock["kind"]]
        side = "long" if lock["direction"] > 0 else "short"
        summary = (f"{label} hit — {sid} {symbol} ({side} {abs(lock['quantity']):g}): mark "
                   f"{lock['mark']:g} crossed {lock['level']:g}; closing, re-entry "
                   f"{side} blocked for the rest of the session")
        logger.warning(summary)
        try:
            self.ex.logger_db.log_decision(sid, "exit", summary,
                                           detail=json.dumps(lock, default=str),
                                           symbols=[symbol])
        except Exception as e:
            logger.error("exit not journalled for %s %s: %s", sid, symbol, e)
        if self._alert is not None:
            try:
                self._alert(f"\U0001f6d1 {summary}", topic="orders")
            except Exception:
                pass

    # ------------------------------------------------------------------ inspection
    def snapshot(self, marks: dict = None, strategy_id: str = None) -> dict:
        marks = marks or {}
        with self._lock:
            self._expire_lockouts()
            rules = {}
            for sid, book in self.rules.items():
                if strategy_id and sid != strategy_id:
                    continue
                costs = self.ex.ledger.strategy_avg_cost.get(sid, {})
                rules[sid] = {
                    symbol: {**rule, "held": self._held(sid, symbol),
                             "avg_cost": costs.get(symbol), "mark": marks.get(symbol),
                             "levels": levels(rule, costs.get(symbol))}
                    for symbol, rule in book.items()}
            lockouts = {sid: dict(locks) for sid, locks in self.lockouts.items()
                        if not strategy_id or sid == strategy_id}
        return {"rules": rules, "lockouts": lockouts, "session": self.session()}
