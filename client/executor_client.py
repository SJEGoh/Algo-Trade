"""
client/executor_client.py — talk to the executor from somewhere else.

Single file, `requests` is the only dependency: copy the `client/` directory onto the box
running your strategy, `pip install requests`, set two environment variables, and go.

Why this exists rather than another `requests.post` in each runner
-----------------------------------------------------------------
On the executor's own host, a failed submission is obvious — you are looking at the box.
From somewhere else it is not: the existing runners swallow a RequestException into
`{"accepted": False, ...}` and print it, so a network blip silently becomes "no rebalance
today". Nothing alerts, because the Telegram watchdog lives next to the executor and only
knows that the EXECUTOR is fine.

So this client:
  * retries what is worth retrying (connection errors, timeouts, 502/503/504) and does NOT
    retry a 4xx — a rejected order is a decision, not a glitch;
  * is safe to retry, because the executor dedups on `client_order_id` and `/targets` takes
    ABSOLUTE targets, so resending the identical payload can't double a position;
  * fails LOUDLY: when it gives up it raises, and it can alert Telegram directly — the one
    channel that still works when the executor is the thing that's down.

Stops, take-profits and trailing stops
--------------------------------------
Optional, per name. Put an `exits` object on the intent with any combination of

    stop_price | stop_pct   take_profit_price | take_profit_pct   trail_amount | trail_pct

(one of each pair; `*_pct` is a fraction of the strategy's average cost, 0.03 = 3%):

    {"instrument": {...}, "target_quantity": 78, "expected_price": 319.97,
     "exits": {"stop_pct": 0.03, "trail_pct": 0.05}}

The executor enforces them — re-pricing every 30s and closing at market — and each
submission replaces that name's exits. `RemoteStrategy.intent(..., stop_pct=0.03)` builds
the object for you; `exits()` shows what is armed. The full rules are in client/README.md.

Environment
-----------
    EXECUTOR_URL        e.g. http://127.0.0.1:8000 (through a tunnel) — required
    EXECUTOR_API_KEY    the X-API-Key value — required for anything that writes
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   optional; used only to shout when unreachable
    TELEGRAM_THREAD_ERRORS                 optional topic for those alerts
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone

import requests

log = logging.getLogger("executor-client")

RETRY_STATUS = {502, 503, 504}          # gateway/restart blips: worth another go
RETRY_EXCEPTIONS = (requests.ConnectionError, requests.Timeout)


class ExecutorError(RuntimeError):
    """Base class for every failure this client raises."""


class ExecutorUnreachable(ExecutorError):
    """The executor could not be reached (or kept failing) after every retry.

    Treat this as "my orders did NOT go in" — never as "probably fine"."""


class ExecutorRejected(ExecutorError):
    """The executor answered, and said no: bad key, unknown strategy, malformed intent.
    Retrying will not help; fix the caller."""

    def __init__(self, message, status_code=None, detail=None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class ExecutorClient:
    def __init__(self, base_url: str = None, api_key: str = None, strategy_id: str = None,
                 timeout: float = 20.0, retries: int = 3, backoff: float = 1.5,
                 alert_on_failure: bool = True, session: requests.Session = None):
        self.base_url = (base_url or os.environ.get("EXECUTOR_URL")
                         or "http://127.0.0.1:8000").rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("EXECUTOR_API_KEY", "")
        self.strategy_id = strategy_id
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.backoff = backoff
        self.alert_on_failure = alert_on_failure
        self._session = session or requests.Session()

    # ------------------------------------------------------------------ plumbing
    def _request(self, method: str, path: str, *, auth: bool = False, **kwargs):
        """One call, with bounded retries. The payload is identical on every attempt, which
        is what makes the retry safe: the executor dedups `client_order_id` and treats
        `/targets` as an absolute book."""
        url = f"{self.base_url}{path}"
        headers = {"X-API-Key": self.api_key} if auth else {}
        last = None

        for attempt in range(1, self.retries + 1):
            try:
                r = self._session.request(method, url, headers=headers,
                                          timeout=self.timeout, **kwargs)
                if r.status_code in RETRY_STATUS:
                    last = f"HTTP {r.status_code}"
                    log.warning("%s %s -> %s (attempt %d/%d)", method, path,
                                r.status_code, attempt, self.retries)
                elif 400 <= r.status_code < 500:
                    detail = self._detail(r)
                    raise ExecutorRejected(
                        f"{method} {path} rejected: {r.status_code} {detail}",
                        status_code=r.status_code, detail=detail)
                else:
                    r.raise_for_status()
                    return r.json() if r.content else {}
            except RETRY_EXCEPTIONS as e:
                last = str(e)
                log.warning("%s %s failed (attempt %d/%d): %s", method, path,
                            attempt, self.retries, e)

            if attempt < self.retries:
                time.sleep(self.backoff * attempt)          # linear is plenty here

        message = (f"executor unreachable at {self.base_url} after {self.retries} attempts "
                   f"({last}) — {method} {path} did NOT go through")
        log.critical(message)
        if self.alert_on_failure:
            self._alert(f"\U0001f6a8 {self.strategy_id or 'strategy'}: {message}")
        raise ExecutorUnreachable(message)

    @staticmethod
    def _detail(response) -> str:
        try:
            return str(response.json().get("detail", ""))[:300]
        except Exception:
            return (response.text or "")[:300]

    def _alert(self, text: str) -> None:
        """Tell Telegram directly. The executor's own alerter can't report the executor
        being unreachable, so this bypasses it entirely. Best effort, never raises."""
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat = os.environ.get("TELEGRAM_CHAT_ID")
        if not token or not chat:
            return
        payload = {"chat_id": chat, "text": text}
        thread = os.environ.get("TELEGRAM_THREAD_ERRORS")
        if thread:
            payload["message_thread_id"] = int(thread)
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json=payload, timeout=5)
        except Exception as e:                              # pragma: no cover - best effort
            log.warning("could not send the unreachable alert: %s", e)

    def _sid(self, strategy_id: str = None) -> str:
        sid = strategy_id or self.strategy_id
        if not sid:
            raise ValueError("strategy_id is required (pass it here or to the constructor)")
        return sid

    # ------------------------------------------------------------------ reads
    def health(self) -> dict:
        return self._request("GET", "/health")

    def positions(self) -> dict:
        return self._request("GET", "/positions")

    def pnl(self) -> dict:
        return self._request("GET", "/pnl")

    def equity(self) -> dict:
        return self._request("GET", "/equity")

    def book(self, strategy_id: str = None) -> dict:
        return self._request("GET", f"/strategies/{self._sid(strategy_id)}/book")

    def allocation(self, strategy_id: str = None) -> dict:
        return self._request("GET", f"/strategies/{self._sid(strategy_id)}/allocation")

    def resolve_front(self, symbol: str, exchange: str = "NYMEX") -> dict:
        return self._request("GET", f"/resolve_front/{symbol}", params={"exchange": exchange})

    def fills(self, limit: int = 50) -> dict:
        """GET /fills — recent fills, each with its commission once IB has reported it."""
        return self._request("GET", "/fills", params={"limit": limit})

    def orders(self) -> dict:
        return self._request("GET", "/orders")

    def pending(self) -> dict:
        """GET /pending — shares the executor still expects, and whether working orders
        explain them. `unexplained` should be empty."""
        return self._request("GET", "/pending")

    def exposure(self, **policy) -> dict:
        """GET /exposure — bucketed exposure, and the hedge that would be placed."""
        return self._request("GET", "/exposure", params=policy or None)

    def orphans(self) -> dict:
        return self._request("GET", "/positions/orphans")

    def strategies(self) -> dict:
        return self._request("GET", "/strategies")

    def strategy_status(self, strategy_id: str = None) -> dict:
        return self._request("GET", f"/strategies/{self._sid(strategy_id)}/status")

    def net(self) -> dict:
        return self._request("GET", "/net")

    def holdings(self, strategy_id: str = None) -> dict:
        """symbol -> quantity this strategy actually HOLDS, cash left out.

        Built from the strategy's book, which the executor moves only when an order owned by
        this strategy fills — so this is the fill-confirmed position, not what was asked for."""
        rows = (self.book(strategy_id) or {}).get("book", [])
        return {r["symbol"]: float(r.get("quantity") or 0.0) for r in rows
                if not r.get("is_cash") and float(r.get("quantity") or 0.0) != 0.0}

    def strategy_pnl(self, strategy_id: str = None) -> dict:
        """This strategy's realized P&L. `realized` is NET of commissions — the executor
        deducts each fee as IB reports it — and `gross` adds them back."""
        sid = self._sid(strategy_id)
        body = self.pnl() or {}
        realized = float((body.get("realized_pnl") or {}).get(sid, 0.0))
        fees = float((body.get("fees") or {}).get(sid, 0.0))
        return {"realized": realized, "fees": fees, "gross": realized + fees}

    def preflight(self) -> dict:
        """Check before generating a book: is the executor there, connected, and accepting?

        Fail here rather than half-way through a submission loop — a strategy that submits
        three of eight legs and then dies leaves a lopsided book."""
        health = self.health()
        if not health.get("connected"):
            raise ExecutorUnreachable("executor is up but NOT connected to IB")
        if health.get("killed"):
            raise ExecutorRejected("kill switch is active — orders will be refused")
        if health.get("startup_degraded"):
            raise ExecutorRejected("executor started degraded (no broker reconciliation) — "
                                   "run /reconcile and /unkill before trading")
        return health

    # ------------------------------------------------------------------ writes
    def submit_order(self, intent: dict) -> dict:
        """POST /orders. A domain rejection comes back as {"accepted": false, "reason": ...}
        with HTTP 200 — that is the executor deciding, so it is RETURNED, not raised.

        `exits` may ride on an absolute-target intent (`target_quantity`) to arm a stop /
        take-profit / trail for that name; the executor refuses them on a side+quantity
        delta, which has no target to protect."""
        intent = dict(intent)
        intent.setdefault("strategy_id", self._sid(intent.get("strategy_id")))
        intent.setdefault("client_order_id", self.new_client_order_id(
            intent["strategy_id"], (intent.get("instrument") or {}).get("symbol", "x")))
        intent.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        intent.setdefault("schema_version", "1.0")
        # The executor's schema REQUIRES intent_type and order_type, and RemoteStrategy.intent()
        # sets neither — so every order posted in orders mode failed validation and came back
        # as a quiet {"accepted": false}. A target_quantity means an absolute target.
        intent.setdefault("intent_type",
                          "target_position" if "target_quantity" in intent else "delta")
        intent.setdefault("order_type", "market")
        intent.setdefault("time_in_force", "day")
        result = self._request("POST", "/orders", auth=True, json=intent)
        if not result.get("accepted", True) and not self.is_noop(result):
            log.warning("order rejected: %s", result.get("reason"))
        return result

    @staticmethod
    def is_noop(result: dict) -> bool:
        """A target the book already meets is refused as `no-op: ...` — nothing to do, which
        is not the executor saying no."""
        return (not result.get("accepted", True)
                and str(result.get("reason", "")).startswith("no-op"))

    def submit_orders(self, intents: list) -> dict:
        """Submit many intents, returning a summary instead of stopping at the first no.

        An unreachable executor still raises — that is not a per-order outcome, it means
        the rest of the book will not go in either."""
        submitted, rejected, noop = [], [], []
        for intent in intents:
            result = self.submit_order(intent)
            symbol = (intent.get("instrument") or {}).get("symbol")
            bucket = (noop if self.is_noop(result)
                      else submitted if result.get("accepted", True) else rejected)
            bucket.append({"symbol": symbol, **result})
        if rejected:
            log.warning("%d of %d intents were rejected", len(rejected), len(intents))
        return {"submitted": submitted, "rejected": rejected, "noop": noop,
                "ok": len(submitted), "refused": len(rejected)}

    def set_target(self, symbol: str, quantity: float, instrument: dict = None,
                   price: float = None, strategy_id: str = None) -> dict:
        """POST /target — one symbol's ABSOLUTE target. Exit with quantity 0."""
        return self._request("POST", "/target", auth=True, json={
            "strategy_id": self._sid(strategy_id), "symbol": symbol,
            "quantity": quantity, "instrument": instrument, "price": price})

    def submit_book(self, intents: list, strategy_id: str = None) -> dict:
        """POST /targets — the authoritative whole book. Any name you stop mentioning gets
        closed, so this self-heals drift and is the right call for a remote strategy: one
        request, absolute targets, safe to repeat.

        Authoritative for exits too: an intent's `exits` replace that name's rules, and a
        name sent without them has none. One unenforceable exit refuses the whole book."""
        return self._request("POST", "/targets", auth=True, json={
            "strategy_id": self._sid(strategy_id),
            "intents": [self._book_entry(i) for i in intents]})

    @staticmethod
    def _book_entry(intent: dict) -> dict:
        # Built field by field so nothing unexpected rides along — which once meant `exits`
        # was dropped here, and a strategy's stops never reached the executor.
        entry = {"instrument": intent["instrument"],
                 "target_quantity": intent["target_quantity"],
                 "expected_price": intent.get("expected_price")}
        if intent.get("exits"):
            entry["exits"] = intent["exits"]
        return entry

    def exits(self, strategy_id: str = None) -> dict:
        """GET /exits — this strategy's armed stop / take-profit / trailing rules with their
        current trigger levels, and any names locked out of re-entry today."""
        return self._request("GET", "/exits", params={"strategy_id": self._sid(strategy_id)})

    def clear_exit(self, symbol: str, strategy_id: str = None) -> dict:
        """DELETE /exits/{id}/{symbol} — drop a name's exit rules and lift its lockout."""
        return self._request("DELETE", f"/exits/{self._sid(strategy_id)}/{symbol}", auth=True)

    # ------------------------------------------------------------------ acknowledgement
    @staticmethod
    def order_ids(result: dict) -> list:
        """Pull the order ids out of whatever submit_book / submit_orders returned."""
        if not isinstance(result, dict):
            return []
        ids = [o.get("order_id") for o in (result.get("orders") or [])]
        for entry in (result.get("submitted") or []):
            ids.append(entry.get("order_id"))
        return [int(i) for i in ids if i is not None]

    def acks(self, order_ids: list) -> dict:
        """GET /orders/acks — has IB actually taken these orders?"""
        if not order_ids:
            return {"acks": {}, "pending": 0, "rejected": 0, "unknown_order_ids": []}
        return self._request("GET", "/orders/acks",
                             params={"ids": ",".join(str(i) for i in order_ids)})

    def wait_for_acks(self, order_ids: list, timeout: float = 30.0,
                      poll: float = 2.0) -> dict:
        """Block until IB has accepted or refused every order, or `timeout` elapses.

        A submission returning `accepted: true` only means the EXECUTOR took the order — it
        has been handed to the socket and nothing more. Whether IB accepted it is a separate
        question with a separate answer, and a strategy that never asks records a position
        it may not have: a read-only gateway rejects every order while the submission still
        comes back clean.

        Returns {"live": [...], "rejected": [...], "pending": [...], "acks": {...}} —
        `pending` means IB never answered within the timeout, which is not the same as a
        refusal and should not be reported as one.
        """
        deadline = time.time() + timeout
        result = {"live": [], "rejected": [], "pending": list(order_ids), "acks": {}}
        while True:
            result["acks"] = (self.acks(order_ids) or {}).get("acks", {})
            buckets = {"live": [], "rejected": [], "pending": []}
            for oid in order_ids:
                state = (result["acks"].get(str(oid)) or {}).get("ack", "pending")
                buckets.get(state, buckets["pending"]).append(oid)
            result.update(buckets)
            if not buckets["pending"] or time.time() >= deadline:
                break
            time.sleep(min(poll, max(0.0, deadline - time.time())))

        for oid in result["rejected"]:
            err = (result["acks"].get(str(oid)) or {}).get("last_error") or {}
            log.error("order %s REJECTED by IB: %s %s", oid,
                      err.get("code", ""), err.get("message", ""))
        for oid in result["pending"]:
            log.error("order %s was never acknowledged by IB after %.0fs — it may not "
                      "exist at the broker", oid, timeout)
        return result

    #: Order statuses that can no longer produce a fill.
    FINAL_STATUSES = frozenset({"Filled", "Cancelled", "ApiCancelled", "Inactive",
                                "Reconciled_Stale"})

    def wait_for_fills(self, targets: dict, order_ids: list = (), authoritative: bool = True,
                       timeout: float = 60.0, poll: float = 2.0,
                       strategy_id: str = None) -> dict:
        """Block until this strategy's HOLDINGS match `targets`, or it is clear they won't.

        Acknowledgement is not a fill. An order IB has accepted can rest unfilled, be cancelled
        (a later rebalance replaces it; the end-of-day sweep removes it) or fill only in part —
        and a cancelled order still reads as acknowledged. So this watches the one thing that
        answers "did it trade": the strategy's own book, which the executor moves only when an
        order owned by this strategy fills. A pooled order's `filled` count would be the wrong
        thing to watch — it covers every strategy the order was for.

        `authoritative=True` (book mode): a name held but missing from `targets` must reach
        zero, because /targets closes it. Stops early once no order is left working, since
        waiting longer cannot change the answer.

        Returns {"filled": [...], "unfilled": {symbol: {"target", "held"}}, "reason",
        "working_order_ids"}; `reason` is None when everything reached target.
        """
        sid = self._sid(strategy_id)
        order_ids = list(order_ids or [])
        deadline = time.time() + timeout
        reason, working, want, unfilled = None, [], {}, {}
        while True:
            held = self.holdings(sid)
            want = {sym: float(q) for sym, q in targets.items()}
            if authoritative:
                for sym in held:
                    want.setdefault(sym, 0.0)
            unfilled = {sym: {"target": q, "held": held.get(sym, 0.0)}
                        for sym, q in want.items() if abs(held.get(sym, 0.0) - q) > 1e-6}
            if not unfilled:
                reason, working = None, []
                break
            if not order_ids:
                reason = "no order was placed to close the gap"
                break
            acks = (self.acks(order_ids) or {}).get("acks", {})
            working = [oid for oid in order_ids
                       if (acks.get(str(oid)) or {}).get("ack") != "rejected"
                       and (acks.get(str(oid)) or {}).get("status") not in self.FINAL_STATUSES]
            if not working:
                reason = "every order ended without reaching the target"
                break
            if time.time() >= deadline:
                reason = f"not filled within {timeout:.0f}s"
                break
            time.sleep(min(poll, max(0.0, deadline - time.time())))

        for sym, gap in unfilled.items():
            log.error("%s %s: holding %g, target %g — %s",
                      sid, sym, gap["held"], gap["target"], reason)
        return {"filled": sorted(s for s in want if s not in unfilled), "unfilled": unfilled,
                "reason": reason, "working_order_ids": working}

    def add_strategy(self, strategy_id: str, capital_allocation: float, max_drawdown: float,
                     starting_cash: float = None, created_by: str = "client") -> dict:
        """POST /strategies — put a new strategy id on the executor's allowlist at runtime."""
        body = {"strategy_id": strategy_id, "capital_allocation": capital_allocation,
                "max_drawdown": max_drawdown, "created_by": created_by}
        if starting_cash is not None:
            body["starting_cash"] = starting_cash
        return self._request("POST", "/strategies", auth=True, json=body)

    def remove_strategy(self, strategy_id: str) -> dict:
        """DELETE /strategies/{id} — the executor refuses while the strategy holds anything."""
        return self._request("DELETE", f"/strategies/{strategy_id}", auth=True)

    def journal(self, event_type: str, summary: str, detail: str = "",
                symbols: list = None, strategy_id: str = None) -> dict:
        """Leave a note in the decision journal — what you decided and why. Worth doing on
        every run: it is the only record of a strategy that decided to do NOTHING."""
        return self._request("POST", "/journal", auth=True, json={
            "strategy_id": self._sid(strategy_id), "event_type": event_type,
            "summary": summary, "detail": detail, "symbols": symbols or []})

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def new_client_order_id(strategy_id: str, symbol: str) -> str:
        """Unique per intent. The executor dedups on this, so a RETRY must reuse it — which
        happens naturally, because a retry resends the identical payload rather than
        building a new one."""
        return f"{strategy_id}-{symbol}-{uuid.uuid4().hex[:12]}"
