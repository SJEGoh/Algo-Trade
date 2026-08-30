# src/monitoring/alerter.py
from dotenv import load_dotenv
load_dotenv()  # must be first — before any other project imports that read env vars

import os
import logging
import requests

logger = logging.getLogger("executor")


class Alerter:
    def __init__(self, bot_token: str = None, chat_id: str = None):
        self._token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self._chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self._enabled = bool(self._token and self._chat_id)
        if not self._enabled:
            logger.warning("Alerter disabled — no TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID configured")

    def send(self, message: str) -> None:
        if not self._enabled:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self._token}/sendMessage",
                json={"chat_id": self._chat_id, "text": message},
                timeout=5,
            )
        except Exception as e:
            # alerting failure must NEVER propagate into the trading path
            logger.error("Alert send failed: %s", e)


class AlertingHandler(logging.Handler):
    """A logging handler that sends CRITICAL records to Telegram.
    Attach this to the root logger once at startup — any logger.critical(...)
    call anywhere in the app then automatically triggers an alert."""

    def __init__(self, alerter: Alerter, level=logging.CRITICAL):
        super().__init__(level=level)
        self._alerter = alerter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = f"🚨 {record.levelname} [{record.name}]: {record.getMessage()}"
            self._alerter.send(message)
        except Exception:
            # a broken handler must never raise, or it can break the whole logging pipeline
            self.handleError(record)

if __name__ == "__main__":
    alerter = Alerter()
