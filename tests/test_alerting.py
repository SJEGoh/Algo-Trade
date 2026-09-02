"""tests/test_alerting.py — the alerter wiring: AlertingHandler pages on CRITICAL, a
disabled Alerter is a safe no-op, and the executor's connectionClosed logs CRITICAL on an
unexpected disconnect (which the handler turns into an alert) but stays quiet during shutdown."""
import logging
import time

import pytest

from monitoring.alerter import Alerter, AlertingHandler


def test_alerting_handler_pages_only_on_critical():
    sent = []
    a = Alerter(bot_token="t", chat_id="c")     # enabled (token+chat present)
    a.send = lambda m: sent.append(m)           # capture instead of hitting Telegram
    lg = logging.getLogger("test_alerting_probe")
    lg.addHandler(AlertingHandler(a))
    lg.setLevel(logging.DEBUG)
    lg.warning("below threshold")
    lg.critical("boom")
    assert len(sent) == 1 and "boom" in sent[0]


def test_disabled_alerter_is_safe_noop(monkeypatch):
    # clear ambient creds so this tests the disabled path regardless of the machine's .env
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    a = Alerter(bot_token=None, chat_id=None)
    assert a._enabled is False
    a.send("must not raise and must not send")   # no exception


def test_connection_closed_unexpected_is_critical(caplog):
    pytest.importorskip("ibapi")
    from execution.central_execution import CentralExecutor
    import os; os.environ.setdefault("EXECUTOR_API_KEY", "x")
    ex = CentralExecutor.__new__(CentralExecutor); CentralExecutor.__init__(ex)
    ex._shutting_down = False
    with caplog.at_level(logging.CRITICAL, logger="executor"):
        ex.connectionClosed()
    assert any(r.levelno == logging.CRITICAL for r in caplog.records)


def test_connection_closed_during_shutdown_is_quiet(caplog):
    pytest.importorskip("ibapi")
    from execution.central_execution import CentralExecutor
    import os; os.environ.setdefault("EXECUTOR_API_KEY", "x")
    ex = CentralExecutor.__new__(CentralExecutor); CentralExecutor.__init__(ex)
    ex._shutting_down = True
    with caplog.at_level(logging.DEBUG, logger="executor"):
        ex.connectionClosed()
    assert not any(r.levelno == logging.CRITICAL for r in caplog.records)


def test_send_is_nonblocking_and_worker_posts():
    a = Alerter(bot_token="t", chat_id="c")
    posted = []
    a._post = lambda m: (posted.append(m) or True)   # stand in for the HTTP call
    a.send("hi")                                       # must return immediately
    for _ in range(100):
        if posted:
            break
        time.sleep(0.02)
    assert posted == ["hi"]


def test_alerter_self_mutes_after_repeated_failures():
    a = Alerter(bot_token="t", chat_id="c", max_failures=2, cooldown_sec=999)
    a._post = lambda m: False                          # every send fails
    for i in range(3):
        a.send(f"m{i}")
    for _ in range(150):
        if a._muted_until > 0:
            break
        time.sleep(0.02)
    assert a._muted_until > 0                          # muted after the failure streak
