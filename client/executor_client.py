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
        with HTTP 200 — that is the executor deciding, so it is RETURNED, not raised."""
        intent = dict(intent)
        intent.setdefault("strategy_id", self._sid(intent.get("strategy_id")))
        intent.setdefault("client_order_id", self.new_client_order_id(
            intent["strategy_id"], (intent.get("instrument") or {}).get("symbol", "x")))
        intent.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        intent.setdefault("schema_version", "1.0")
        result = self._request("POST", "/orders", auth=True, json=intent)
        if not result.get("accepted", True):
            log.warning("order rejected: %s", result.get("reason"))
        return result

    def submit_orders(self, intents: list) -> dict:
        """Submit many intents, returning a summary instead of stopping at the first no.

        An unreachable executor still raises — that is not a per-order outcome, it means
        the rest of the book will not go in either."""
        submitted, rejected = [], []
        for intent in intents:
            result = self.submit_order(intent)
            symbol = (intent.get("instrument") or {}).get("symbol")
            (submitted if result.get("accepted", True) else rejected).append(
                {"symbol": symbol, **result})
        if rejected:
            log.warning("%d of %d intents were rejected", len(rejected), len(intents))
        return {"submitted": submitted, "rejected": rejected,
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
        request, absolute targets, safe to repeat."""
        return self._request("POST", "/targets", auth=True, json={
            "strategy_id": self._sid(strategy_id),
            "intents": [{"instrument": i["instrument"],
                         "target_quantity": i["target_quantity"],
                         "expected_price": i.get("expected_price")} for i in intents]})

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
