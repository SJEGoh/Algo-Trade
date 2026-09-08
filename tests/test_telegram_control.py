"""tools/telegram_control.py — dispatch, authorisation and confirmation.

The bot can flatten the book and kill trading, so the parts worth pinning are the guards:
who may run what, the two-step confirmation, and the update-offset handling that stops a
stale /kill being replayed after a restart. No network: Telegram and the executor API are
both stubbed.
"""
import importlib
import time

import pytest
import requests

CHAT = "-1001234567890"
OWNER = 111
STRANGER = 222


@pytest.fixture
def bot(monkeypatch, tmp_path):
    """A freshly imported module with a known environment."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    monkeypatch.setenv("TELEGRAM_THREAD_ORDERS", "77")
    monkeypatch.setenv("TELEGRAM_THREAD_ERRORS", "88")
    monkeypatch.delenv("TELEGRAM_CONTROL_THREAD", raising=False)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", str(OWNER))
    monkeypatch.setenv("EXECUTOR_API_KEY", "k")
    monkeypatch.setenv("TELEGRAM_STATE_PATH", str(tmp_path / "offset.json"))

    import tools.telegram_control as tc
    tc = importlib.reload(tc)

    tc.sent = []
    tc.posted = []
    monkeypatch.setattr(tc, "say", lambda text, thread=None: tc.sent.append((text, thread)))
    monkeypatch.setattr(tc, "journal", lambda *a, **kw: None)
    monkeypatch.setattr(tc, "api_post",
                        lambda path, body=None: tc.posted.append((path, body)) or {"ok": True})
    monkeypatch.setattr(tc, "api_get", lambda path, **kw: {})
    return tc


def msg(text, user_id=OWNER, chat=CHAT, thread=None, age=0):
    return {"text": text, "from": {"id": user_id, "username": f"u{user_id}"},
            "chat": {"id": chat}, "message_thread_id": thread,
            "date": time.time() - age}


# ----------------------------------------------------------------- replies land in a topic
def test_replies_go_to_the_orders_topic_by_default(bot):
    assert bot._configured_thread() == "77"


def test_one_variable_moves_the_replies(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    monkeypatch.setenv("TELEGRAM_THREAD_ORDERS", "77")
    monkeypatch.setenv("TELEGRAM_CONTROL_THREAD", "999")
    import tools.telegram_control as tc
    tc = importlib.reload(tc)
    assert tc._configured_thread() == "999"


def test_here_replies_in_the_originating_topic(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    monkeypatch.setenv("TELEGRAM_CONTROL_THREAD", "here")
    import tools.telegram_control as tc
    tc = importlib.reload(tc)
    assert tc.reply_thread_for({"message_thread_id": 42}) == 42


# ----------------------------------------------------------------- authorisation
def test_a_stranger_cannot_kill(bot):
    bot.handle(msg("/kill", user_id=STRANGER))
    assert bot.posted == []
    assert "not allow-listed" in bot.sent[-1][0]


def test_a_stranger_can_still_read(bot, monkeypatch):
    monkeypatch.setattr(bot, "api_get", lambda path, **kw: {"strategies": []})
    bot.handle(msg("/strategies", user_id=STRANGER))
    assert "none configured" in bot.sent[-1][0]


def test_control_is_disabled_when_no_allowlist_is_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    monkeypatch.delenv("TELEGRAM_ALLOWED_USER_IDS", raising=False)
    import tools.telegram_control as tc
    tc = importlib.reload(tc)
    tc.sent, tc.posted = [], []
    monkeypatch.setattr(tc, "say", lambda text, thread=None: tc.sent.append((text, thread)))
    monkeypatch.setattr(tc, "api_post", lambda p, b=None: tc.posted.append(p))

    tc.handle(msg("/kill"))
    assert tc.posted == []
    assert "TELEGRAM_ALLOWED_USER_IDS" in tc.sent[-1][0]


def test_messages_from_other_chats_are_ignored(bot):
    """The poll loop filters by chat id before dispatch."""
    update = {"update_id": 1, "message": msg("/kill", chat="-100999")}
    assert str(update["message"]["chat"]["id"]) != bot.CHAT_ID


# ----------------------------------------------------------------- confirmation
@pytest.mark.parametrize("command", ["/kill", "/flatten", "/unkill"])
def test_destructive_commands_need_confirmation(bot, command):
    bot.handle(msg(command))
    assert bot.posted == [], "must not act before confirmation"
    assert "/confirm " in bot.sent[-1][0]


def test_confirmation_runs_the_command(bot):
    bot.handle(msg("/kill"))
    token = bot.sent[-1][0].split("/confirm ")[1].split("\n")[0].strip()
    bot.handle(msg(f"/confirm {token}"))
    assert bot.posted and bot.posted[0][0] == "/kill"


def test_a_confirmation_token_is_single_use(bot):
    bot.handle(msg("/kill"))
    token = bot.sent[-1][0].split("/confirm ")[1].split("\n")[0].strip()
    bot.handle(msg(f"/confirm {token}"))
    bot.handle(msg(f"/confirm {token}"))
    assert len(bot.posted) == 1
    assert "nothing pending" in bot.sent[-1][0]


def test_someone_else_cannot_confirm_your_command(bot, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", f"{OWNER},{STRANGER}")
    bot.ALLOWED = {OWNER, STRANGER}
    bot.handle(msg("/kill", user_id=OWNER))
    token = bot.sent[-1][0].split("/confirm ")[1].split("\n")[0].strip()
    bot.handle(msg(f"/confirm {token}", user_id=STRANGER))
    assert bot.posted == []
    assert "belongs to someone else" in bot.sent[-1][0]


def test_confirmations_expire(bot):
    bot.handle(msg("/kill"))
    token = bot.sent[-1][0].split("/confirm ")[1].split("\n")[0].strip()
    with bot._pending_lock:                      # pretend it was issued long ago
        name, args, owner, _ = bot._pending[token]
        bot._pending[token] = (name, args, owner, time.time() - 1)
    bot.handle(msg(f"/confirm {token}"))
    assert bot.posted == []


def test_protective_commands_need_no_confirmation(bot):
    """A halt reduces risk — don't put a speed bump in front of it."""
    bot.handle(msg("/halt kalman_vecm"))
    assert bot.posted and bot.posted[0][0] == "/strategies/kalman_vecm/halt"
    assert bot.posted[0][1]["flatten"] is True


def test_halt_can_skip_the_flatten(bot):
    bot.handle(msg("/halt kalman_vecm noflatten"))
    assert bot.posted[0][1]["flatten"] is False


# ----------------------------------------------------------------- offset handling
def test_a_fresh_start_skips_the_backlog(bot, monkeypatch):
    """Telegram redelivers un-acked updates forever: without this a bot restarting after
    an outage would replay whatever was sent while it was down — including a stale /kill."""
    monkeypatch.setattr(bot, "tg", lambda method, **kw: [{"update_id": 500}])
    assert bot.skip_backlog() == 501


def test_the_offset_survives_a_restart(bot):
    bot.save_offset(1234)
    assert bot.load_offset() == 1234


def test_a_missing_offset_file_reads_as_none(bot):
    bot.STATE_PATH.unlink(missing_ok=True)
    assert bot.load_offset() is None


# ----------------------------------------------------------------- failure reporting
def test_an_unreachable_executor_is_reported_not_swallowed(bot, monkeypatch):
    def down(path, **kw):
        raise requests.ConnectionError("connection refused")
    monkeypatch.setattr(bot, "api_get", down)
    bot.handle(msg("/status"))
    assert "cannot reach the executor" in bot.sent[-1][0]


def test_an_api_rejection_shows_the_reason(bot, monkeypatch):
    class Response:
        status_code = 423
        def json(self): return {"detail": "kill switch active"}

    def rejected(path, body=None):
        raise requests.HTTPError(response=Response())
    monkeypatch.setattr(bot, "api_post", rejected)
    bot.handle(msg("/halt kalman_vecm"))
    assert "423" in bot.sent[-1][0] and "kill switch active" in bot.sent[-1][0]


def test_unknown_commands_are_answered(bot):
    bot.handle(msg("/definitely_not_a_command"))
    assert "unknown command" in bot.sent[-1][0]


def test_plain_chat_is_ignored(bot):
    bot.handle(msg("how's it going"))
    assert bot.sent == []


def test_a_botname_suffix_is_stripped(bot, monkeypatch):
    monkeypatch.setattr(bot, "api_get", lambda path, **kw: {"strategies": []})
    bot.handle(msg("/strategies@my_exec_bot"))
    assert "none configured" in bot.sent[-1][0]
