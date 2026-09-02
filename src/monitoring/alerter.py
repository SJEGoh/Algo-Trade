# src/monitoring/alerter.py
from dotenv import load_dotenv
load_dotenv()  # must be first — before any other project imports that read env vars

import os
import logging
import queue
import threading
import time

import requests

logger = logging.getLogger("executor")


class Alerter:
    """Fire-and-forget Telegram alerter. send() enqueues and returns immediately so a slow or
    unreachable Telegram NEVER blocks the trading/critical-log path. A background worker posts
    the messages; after repeated failures it self-mutes for a cooldown (so it stops spamming
    timeouts and log noise). Alerting failure never propagates to callers."""

    def __init__(self, bot_token: str = None, chat_id: str = None,
                 timeout: float = 4.0, max_failures: int = 3, cooldown_sec: float = 300.0,
                 max_queue: int = 200):
        self._token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self._chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self._enabled = bool(self._token and self._chat_id)
        self._timeout = timeout
        self._max_failures = max_failures
        self._cooldown = cooldown_sec
        self._q: "queue.Queue[str]" = queue.Queue(maxsize=max_queue)
        self._worker = None
        self._lock = threading.Lock()
        self._fail_streak = 0
        self._muted_until = 0.0
        if not self._enabled:
            logger.warning("Alerter disabled — no TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID configured")

    def send(self, message: str) -> None:
        """Non-blocking: enqueue for the background worker. Drops silently if disabled or the
        queue is full (better to drop an alert than to block or grow unbounded)."""
        if not self._enabled:
            return
        try:
            self._q.put_nowait(message)
        except queue.Full:
            return
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, daemon=True)
                self._worker.start()

    def _run(self) -> None:
        while True:
            try:
                message = self._q.get(timeout=30.0)
            except queue.Empty:
                return  # idle -> exit; re-spawned on the next send
            if time.time() < self._muted_until:
                continue  # muted after failures -> drop rather than hang
            if self._post(message):
                self._fail_streak = 0
            else:
                self._fail_streak += 1
                if self._fail_streak >= self._max_failures:
                    self._muted_until = time.time() + self._cooldown
                    self._fail_streak = 0
                    logger.error("Alerter muted for %.0fs — Telegram unreachable "
                                 "(check network/firewall; alerts will resume after cooldown)",
                                 self._cooldown)

    def _post(self, message: str) -> bool:
        try:
            requests.post(
                f"https://api.telegram.org/bot{self._token}/sendMessage",
                json={"chat_id": self._chat_id, "text": message},
                timeout=self._timeout,
            )
            return True
        except Exception as e:
            logger.warning("Alert send failed: %s", e)
            return False


class AlertingHandler(logging.Handler):
    """A logging handler that sends CRITICAL records to Telegram (via the fire-and-forget
    Alerter, so logging.critical never blocks). Attach once to the root logger at startup."""

    def __init__(self, alerter: Alerter, level=logging.CRITICAL):
        super().__init__(level=level)
        self._alerter = alerter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = f"🚨 {record.levelname} [{record.name}]: {record.getMessage()}"
            self._alerter.send(message)
        except Exception:
            self.handleError(record)


if __name__ == "__main__":
    alerter = Alerter()
