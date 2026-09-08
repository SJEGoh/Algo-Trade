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


# ----------------------------------------------------------------- /allocate
PLAN = {
    "strategy_id": "ovn_volsurge", "method": "pro_rata",
    "allocation_before": 100_000.0, "allocation_after": 60_000.0,
    "cash_before": 20_000.0, "cash_after": -20_000.0, "cash_shortfall": 20_000.0,
    "liquidations": [{"symbol": "AAPL", "from_quantity": 600.0, "to_quantity": 450.0,
                      "dollars": 15_000.0, "freed": 15_000.0},
                     {"symbol": "MSFT", "from_quantity": 200.0, "to_quantity": 150.0,
                      "dollars": 5_000.0, "freed": 5_000.0}],
    "orders": [], "valued_at_cost": [],
}


@pytest.fixture
def alloc_bot(bot, monkeypatch):
    """Record every allocation call and hand back a fixed plan."""
    bot.calls = []

    def api_post(path, body=None):
        bot.calls.append((path, body))
        return dict(PLAN)
    monkeypatch.setattr(bot, "api_post", api_post)
    return bot


def test_allocate_previews_the_plan_before_asking_to_confirm(alloc_bot):
    """The whole point of confirming a money move is seeing what it will sell."""
    alloc_bot.handle(msg("/allocate ovn_volsurge 60k"))

    path, body = alloc_bot.calls[0]
    assert path == "/strategies/ovn_volsurge/allocation"
    assert body["dry_run"] is True                       # nothing executed yet
    reply = alloc_bot.sent[-1][0]
    assert "PLAN (nothing changed yet)" in reply
    assert "AAPL 600 -> 450" in reply and "MSFT 200 -> 150" in reply
    assert "/confirm " in reply


def test_allocate_executes_only_after_confirmation(alloc_bot):
    alloc_bot.handle(msg("/allocate ovn_volsurge 60k"))
    token = alloc_bot.sent[-1][0].split("/confirm ")[1].split("\n")[0].strip()
    alloc_bot.handle(msg(f"/confirm {token}"))

    assert [b["dry_run"] for _, b in alloc_bot.calls] == [True, False]
    assert alloc_bot.calls[1][1]["capital_allocation"] == 60_000.0
    assert "DONE" in alloc_bot.sent[-1][0]


def test_a_rejected_plan_never_becomes_a_confirmation_token(bot, monkeypatch):
    """If the executor would refuse it, don't offer to confirm it."""
    class Response:
        status_code = 422
        def json(self): return {"detail": "cannot withdraw 99,999: its NAV is only 96,000"}

    def rejected(path, body=None):
        raise requests.HTTPError(response=Response())
    monkeypatch.setattr(bot, "api_post", rejected)

    bot.handle(msg("/allocate ovn_volsurge 1"))
    assert "NAV is only" in bot.sent[-1][0]
    assert "/confirm" not in bot.sent[-1][0]
    with bot._pending_lock:
        assert bot._pending == {}


@pytest.mark.parametrize("text, expected", [
    ("150000", {"capital_allocation": 150_000.0}),
    ("150k", {"capital_allocation": 150_000.0}),
    ("1.2m", {"capital_allocation": 1_200_000.0}),
    ("$150,000", {"capital_allocation": 150_000.0}),
    ("-40k", {"delta": -40_000.0}),          # a sign means "change it by"
    ("+25000", {"delta": 25_000.0}),
])
def test_amount_forms(alloc_bot, text, expected):
    alloc_bot.handle(msg(f"/allocate ovn_volsurge {text}"))
    _, body = alloc_bot.calls[0]
    key = next(iter(expected))
    assert body[key] == expected[key]


def test_method_defaults_to_pro_rata_and_can_be_overridden(alloc_bot):
    alloc_bot.handle(msg("/allocate ovn_volsurge -40k"))
    assert alloc_bot.calls[-1][1]["method"] == "pro_rata"
    alloc_bot.handle(msg("/allocate ovn_volsurge -40k equal"))
    assert alloc_bot.calls[-1][1]["method"] == "equal"


@pytest.mark.parametrize("command", [
    "/allocate",                       # no arguments
    "/allocate ovn_volsurge",          # no amount
    "/allocate ovn_volsurge lots",     # unreadable amount
    "/allocate ovn_volsurge 60k sideways",   # unknown method
])
def test_bad_allocate_arguments_are_explained_not_executed(alloc_bot, command):
    alloc_bot.handle(msg(command))
    assert alloc_bot.calls == []
    assert "/confirm" not in alloc_bot.sent[-1][0]


def test_allocate_is_allow_listed(alloc_bot):
    alloc_bot.handle(msg("/allocate ovn_volsurge 60k", user_id=STRANGER))
    assert alloc_bot.calls == []
    assert "not allow-listed" in alloc_bot.sent[-1][0]


def test_an_increase_reads_as_a_cash_move(alloc_bot, monkeypatch):
    plan = dict(PLAN, liquidations=[], cash_shortfall=0.0,
                allocation_after=140_000.0, cash_after=60_000.0)
    monkeypatch.setattr(alloc_bot, "api_post", lambda path, body=None: dict(plan))
    alloc_bot.handle(msg("/allocate ovn_volsurge +40k"))
    reply = alloc_bot.sent[-1][0]
    assert "+$40,000" in reply
    assert "selling" not in reply


def test_a_decrease_covered_by_cash_says_nothing_is_sold(alloc_bot, monkeypatch):
    plan = dict(PLAN, liquidations=[], cash_shortfall=0.0, cash_after=-0.0)
    monkeypatch.setattr(alloc_bot, "api_post", lambda path, body=None: dict(plan))
    alloc_bot.handle(msg("/allocate ovn_volsurge 60k"))
    assert "cash covers it — nothing to sell" in alloc_bot.sent[-1][0]


def test_negative_money_reads_naturally(bot):
    assert bot.money(-50_000) == "-$50,000"
    assert bot.money(50_000) == "$50,000"
    assert bot.money(None) == "—"


# ----------------------------------------------------------------- /addstrategy
# Adding an id to the allowlist is what lets a strategy trade at all, so it carries the same
# gates as moving money: allow-listed users only, and a /confirm token. The preview matters
# as much as the gate — a typo'd id CREATES a second strategy rather than failing, and the
# strategy sending that id then trades against an entry nobody meant to make.

def test_addstrategy_is_restricted(bot):
    bot.handle(msg("/addstrategy pairs_v2 150k", user_id=OWNER + 1))
    assert not bot.posted
    assert "not allow-listed" in bot.sent[-1][0]


def test_addstrategy_needs_confirmation(bot, monkeypatch):
    monkeypatch.setattr(bot, "api_get", lambda path, **kw: {"strategies": []})
    bot.handle(msg("/addstrategy pairs_v2 150k"))
    assert not bot.posted, "created the strategy without a confirmation"
    assert "/confirm" in bot.sent[-1][0]


def test_addstrategy_preview_shows_what_will_be_created(bot, monkeypatch):
    monkeypatch.setattr(bot, "api_get", lambda path, **kw: {"strategies": []})
    bot.handle(msg("/addstrategy pairs_v2 150k 10%"))

    prompt = bot.sent[-1][0]
    assert "pairs_v2" in prompt
    assert "150,000" in prompt or "150k" in prompt.lower()
    assert "10%" in prompt


def test_addstrategy_posts_after_confirm(bot, monkeypatch):
    monkeypatch.setattr(bot, "api_get", lambda path, **kw: {"strategies": []})
    monkeypatch.setattr(bot, "api_post", lambda path, body=None: bot.posted.append(
        (path, body)) or {"strategy_id": "pairs_v2", "capital_allocation": 150_000.0,
                          "max_drawdown": 0.10, "starting_cash": 150_000.0})
    bot.handle(msg("/addstrategy pairs_v2 150k 10%"))
    token = bot.sent[-1][0].split("/confirm ")[1].split()[0].strip("`*_ \n")

    bot.handle(msg(f"/confirm {token}"))
    path, body = bot.posted[-1]
    assert path == "/strategies"
    assert body["strategy_id"] == "pairs_v2"
    assert body["capital_allocation"] == 150_000.0
    assert body["max_drawdown"] == 0.10


def test_addstrategy_refuses_an_existing_id(bot, monkeypatch):
    """/allocate re-capitalises an existing strategy — it knows how to raise cash by
    selling. Routing that through /addstrategy would move the cap under an open book."""
    monkeypatch.setattr(bot, "api_get",
                        lambda path, **kw: {"strategies": [{"strategy_id": "ovn_volsurge"}]})
    bot.handle(msg("/addstrategy ovn_volsurge 150k"))
    assert not bot.posted
    assert "already exists" in bot.sent[-1][0]


def test_addstrategy_without_an_amount_shows_usage(bot):
    bot.handle(msg("/addstrategy pairs_v2"))
    assert not bot.posted
    assert "usage:" in bot.sent[-1][0]


def test_addstrategy_defaults_the_drawdown_halt(bot, monkeypatch):
    """Omitting the drawdown must not mean 'no halt' — that is the one default that has to
    be present rather than absent."""
    monkeypatch.setattr(bot, "api_get", lambda path, **kw: {"strategies": []})
    body = bot._addstrategy_body(["pairs_v2", "150k"])
    assert body["max_drawdown"] == bot.DEFAULT_MAX_DRAWDOWN > 0
