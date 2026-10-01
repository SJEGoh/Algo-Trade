"""tests/test_telegram_trade.py — hand-placed trades from Telegram: /buy, /sell, /close.

What is worth pinning down:

  * no strategy named means `discretionary`, so a manual position always has an owner —
    it never lands in __net__ with nobody to close it;
  * the command becomes an ABSOLUTE target built on the strategy's working target, so two
    quick /buy 10s end at 20 rather than 10;
  * nothing trades before /confirm, and a plan the bot can't make (unknown or halted
    strategy, nothing to close, bad quantity) never becomes a confirmation token.
"""
import importlib
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

CHAT, OWNER, STRANGER = "-1001234567890", 111, 222


@pytest.fixture
def bot(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", str(OWNER))
    monkeypatch.setenv("EXECUTOR_API_KEY", "k")
    monkeypatch.setenv("TELEGRAM_STATE_PATH", str(tmp_path / "offset.json"))
    import tools.telegram_control as tc
    tc = importlib.reload(tc)

    tc.sent, tc.posted = [], []
    tc.state = {
        "strategies": {"discretionary": True, "pair_break_fade": True, "halted_one": False},
        "books": {"pair_break_fade": {"AAPL": 20.0}},
        "desired": {},
        "price": 230.0,
    }

    def api_get(path, **kw):
        st = tc.state
        if path == "/strategies":
            return {"strategies": [{"strategy_id": s, "active": a, "capital_allocation": 10_000.0}
                                   for s, a in st["strategies"].items()]}
        if path.startswith("/strategies/") and path.endswith("/book"):
            sid = path.split("/")[2]
            return {"book": [{"symbol": "CASH", "quantity": 1.0, "is_cash": True}]
                    + [{"symbol": s, "quantity": q} for s, q in st["books"].get(sid, {}).items()]}
        if path == "/net":
            return {"desired": st["desired"]}
        if path.startswith("/price/"):
            return {"symbol": path.split("/")[2], "price": st["price"]}
        return {}

    def api_post(path, body=None):
        tc.posted.append((path, body))
        return {"accepted": True, "orders": [{"order_id": 501}], "internal_crosses": []}

    monkeypatch.setattr(tc, "say", lambda text, thread=None: tc.sent.append(text))
    monkeypatch.setattr(tc, "journal", lambda *a, **kw: None)
    monkeypatch.setattr(tc, "api_get", api_get)
    monkeypatch.setattr(tc, "api_post", api_post)
    return tc


def msg(text, user_id=OWNER):
    return {"text": text, "from": {"id": user_id, "username": f"u{user_id}"},
            "chat": {"id": CHAT}, "date": time.time()}


def _confirm(bot):
    token = bot.sent[-1].split("/confirm ")[1].split("\n")[0].strip()
    bot.handle(msg(f"/confirm {token}"))


def test_buy_with_no_strategy_is_discretionary(bot):
    bot.handle(msg("/buy aapl 10"))
    assert "BUY 10 AAPL  (discretionary)" in bot.sent[-1]
    assert "position 0 -> 10" in bot.sent[-1]
    assert bot.posted == []                                # nothing before /confirm
    _confirm(bot)
    path, body = bot.posted[-1]
    assert path == "/target"
    assert (body["strategy_id"], body["symbol"], body["quantity"], body["price"]) == \
        ("discretionary", "AAPL", 10.0, 230.0)
    assert "#501" in bot.sent[-1]


def test_sell_for_a_named_strategy_reduces_its_position(bot):
    bot.handle(msg("/sell AAPL 5 pair_break_fade"))
    assert "position 20 -> 15" in bot.sent[-1]
    assert "may undo this" in bot.sent[-1]                 # its next run restates the book
    _confirm(bot)
    assert bot.posted[-1][1]["quantity"] == 15.0


def test_sell_can_go_short(bot):
    bot.handle(msg("/sell TSLA 3"))
    _confirm(bot)
    assert bot.posted[-1][1]["quantity"] == -3.0


def test_buy_builds_on_a_target_still_working(bot):
    bot.state["desired"] = {"discretionary": {"AAPL": 10.0}}   # unfilled earlier /buy 10
    bot.handle(msg("/buy AAPL 10"))
    assert "position 10 -> 20" in bot.sent[-1]
    _confirm(bot)
    assert bot.posted[-1][1]["quantity"] == 20.0


def test_close_takes_the_strategy_flat(bot):
    bot.handle(msg("/close AAPL pair_break_fade"))
    assert "SELL 20 AAPL" in bot.sent[-1]
    _confirm(bot)
    assert bot.posted[-1][1]["quantity"] == 0.0


@pytest.mark.parametrize("text, expected", [
    ("/close MSFT", "holds no MSFT"),
    ("/buy AAPL 10 typo_strategy", "unknown strategy"),
    ("/buy AAPL 10 halted_one", "halted"),
    ("/buy AAPL -5", "whole number"),
    ("/buy AAPL 2.5", "whole number"),
    ("/buy AAPL lots", "usage"),
    ("/buy AAPL", "usage"),
])
def test_bad_trades_never_get_a_token(bot, text, expected):
    bot.handle(msg(text))
    assert expected in bot.sent[-1]
    assert "/confirm" not in bot.sent[-1]
    assert bot.posted == []


def test_a_stranger_cannot_trade(bot):
    bot.handle(msg("/buy AAPL 10", user_id=STRANGER))
    assert "not allow-listed" in bot.sent[-1]
    assert bot.posted == []


# ------------------------------------------------------------------ GET /price

def test_price_prefers_ib_and_falls_back_to_the_last_mark(monkeypatch):
    import api.server as server
    fake = SimpleNamespace(isConnected=lambda: True, fetch_price=lambda sym: None)
    monkeypatch.setattr(server, "executor", fake)
    monkeypatch.setitem(server._last_equity, "marks", {"AAPL": 229.5})
    client = TestClient(server.app)

    assert client.get("/price/aapl").json() == {"symbol": "AAPL", "price": 229.5,
                                                "source": "last_mark"}
    fake.fetch_price = lambda sym: 231.0
    assert client.get("/price/AAPL").json()["source"] == "ib"
    fake.fetch_price = lambda sym: None
    assert client.get("/price/MSFT").status_code == 503      # no IB price, no mark
