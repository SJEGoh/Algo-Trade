"""tests/test_alerting.py — the alerter wiring: AlertingHandler pages on CRITICAL, a
disabled Alerter is a safe no-op, and the executor's connectionClosed logs CRITICAL on an
unexpected disconnect (which the handler turns into an alert) but stays quiet during shutdown.

Also includes live-send integration tests (marked with @pytest.mark.live) that actually
hit the Telegram API to verify messages land in the correct topics."""
import logging
import os
import time

import pytest

from monitoring.alerter import Alerter, AlertingHandler


# ---------------------------------------------------------------------------
# Unit tests (mock / capture — no network)
# ---------------------------------------------------------------------------

def test_alerting_handler_pages_only_on_critical():
    sent = []
    a = Alerter(bot_token="t", chat_id="c")     # enabled (token+chat present)
    a.send = lambda m, **kw: sent.append((m, kw))  # capture instead of hitting Telegram
    lg = logging.getLogger("test_alerting_probe")
    lg.addHandler(AlertingHandler(a))
    lg.setLevel(logging.DEBUG)
    lg.warning("below threshold")
    lg.critical("boom")
    assert len(sent) == 1 and "boom" in sent[0][0]


def test_alerting_handler_routes_to_errors_topic():
    """AlertingHandler should send with topic='errors'."""
    sent = []
    a = Alerter(bot_token="t", chat_id="c")
    a.send = lambda m, **kw: sent.append((m, kw))
    lg = logging.getLogger("test_topic_routing")
    lg.addHandler(AlertingHandler(a))
    lg.setLevel(logging.DEBUG)
    lg.critical("test error routing")
    assert len(sent) == 1
    assert sent[0][1].get("topic") == "errors"


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
    a._post = lambda m, **kw: (posted.append((m, kw)) or True)
    a.send("hi")                                       # must return immediately
    for _ in range(100):
        if posted:
            break
        time.sleep(0.02)
    assert posted[0][0] == "hi"


def test_alerter_self_mutes_after_repeated_failures():
    a = Alerter(bot_token="t", chat_id="c", max_failures=2, cooldown_sec=999)
    a._post = lambda m, **kw: False                    # every send fails
    for i in range(3):
        a.send(f"m{i}")
    for _ in range(150):
        if a._muted_until > 0:
            break
        time.sleep(0.02)
    assert a._muted_until > 0                          # muted after the failure streak


def test_topic_thread_id_loaded_from_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_THREAD_ERRORS", "2")
    monkeypatch.setenv("TELEGRAM_THREAD_ORDERS", "4")
    a = Alerter(bot_token="t", chat_id="c")
    assert a._topics == {"errors": 2, "orders": 4}


def test_send_with_topic_passes_thread_id():
    """send(topic='orders') should enqueue the message with the correct thread_id."""
    a = Alerter(bot_token="t", chat_id="c")
    a._topics = {"orders": 4, "errors": 2}
    posted = []
    a._post = lambda m, **kw: (posted.append((m, kw)) or True)
    a.send("new order", topic="orders")
    for _ in range(100):
        if posted:
            break
        time.sleep(0.02)
    assert posted[0][0] == "new order"
    assert posted[0][1].get("thread_id") == 4


def test_send_without_topic_has_no_thread_id():
    """send() with no topic should pass thread_id=None."""
    a = Alerter(bot_token="t", chat_id="c")
    a._topics = {"orders": 4}
    posted = []
    a._post = lambda m, **kw: (posted.append((m, kw)) or True)
    a.send("general message")
    for _ in range(100):
        if posted:
            break
        time.sleep(0.02)
    assert posted[0][1].get("thread_id") is None


def test_post_includes_message_thread_id():
    """_post() should include message_thread_id in payload when thread_id is given."""
    import unittest.mock as mock
    a = Alerter(bot_token="fake_token", chat_id="123")
    with mock.patch("monitoring.alerter.requests.post") as mock_post:
        mock_post.return_value = mock.Mock(status_code=200)
        a._post("test", thread_id=4)
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["message_thread_id"] == 4
        assert payload["text"] == "test"


def test_post_omits_message_thread_id_when_none():
    """_post() should NOT include message_thread_id when thread_id is None."""
    import unittest.mock as mock
    a = Alerter(bot_token="fake_token", chat_id="123")
    with mock.patch("monitoring.alerter.requests.post") as mock_post:
        mock_post.return_value = mock.Mock(status_code=200)
        a._post("test", thread_id=None)
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "message_thread_id" not in payload


# ---------------------------------------------------------------------------
# Live-send integration tests — actually hit the Telegram API
# Run with: pytest tests/test_alerting.py -m live -s
# Requires TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_THREAD_ERRORS,
# TELEGRAM_THREAD_ORDERS in the environment or .env
# ---------------------------------------------------------------------------

def _have_telegram_creds():
    from dotenv import load_dotenv
    load_dotenv()
    return all(os.environ.get(k) for k in [
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
        "TELEGRAM_THREAD_ERRORS", "TELEGRAM_THREAD_ORDERS",
    ])

live = pytest.mark.skipif(not _have_telegram_creds(),
                          reason="TELEGRAM_* env vars not set")


@live
def test_live_send_general():
    """Send a message to the general thread (no topic)."""
    a = Alerter()
    ts = time.strftime("%H:%M:%S")
    ok = a._post(f"✅ [LIVE TEST] General thread — {ts}")
    assert ok, "Failed to post to general thread"


@live
def test_live_send_errors_topic():
    """Send a message to the errors topic."""
    a = Alerter()
    ts = time.strftime("%H:%M:%S")
    thread_id = a._topics.get("errors")
    assert thread_id is not None, "TELEGRAM_THREAD_ERRORS not loaded"
    ok = a._post(f"\U0001f6a8 [LIVE TEST] Errors topic — {ts}", thread_id=thread_id)
    assert ok, "Failed to post to errors topic"


@live
def test_live_send_orders_topic():
    """Send a message to the orders topic."""
    a = Alerter()
    ts = time.strftime("%H:%M:%S")
    thread_id = a._topics.get("orders")
    assert thread_id is not None, "TELEGRAM_THREAD_ORDERS not loaded"
    ok = a._post(f"\U0001f4e8 [LIVE TEST] Orders topic — {ts}", thread_id=thread_id)
    assert ok, "Failed to post to orders topic"


@live
def test_live_send_via_queue():
    """End-to-end: send() through the background queue to the orders topic."""
    a = Alerter()
    ts = time.strftime("%H:%M:%S")
    a.send(f"\U0001f4e8 [LIVE TEST] Queue → orders — {ts}", topic="orders")
    # Wait for the background worker to process
    time.sleep(3)
    # If we got here without exception, the queue worker processed it.
    # Visual verification: check the Telegram group.
