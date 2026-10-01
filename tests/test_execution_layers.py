"""tests/test_execution_layers.py — switching execution layers (ATR, and any added later) on
and off and choosing their strategies at runtime: POST /execution/{layer}, Telegram /execset.

The properties worth pinning down are the ones whose failure is silent:

  * an EMPTY strategy list is refused — a layer reads [] as "every strategy", so a list
    emptied by accident would rework every order instead of none;
  * it PERSISTS BEFORE it takes effect, and the saved row is what startup restores — a
    setting that reverts at the next restart looks applied until the day it isn't;
  * an unknown strategy id is refused rather than stored as a name that never matches,
    and an unknown layer is a 404 rather than a setting nothing reads.
"""
import importlib
import time

import pytest
from fastapi.testclient import TestClient

import api.server as server
from api.server import app
from config import ATR_EXECUTION, CONFIG
from execution.atr_execution import AtrPullbackLayer
from logger.event_logger import EventLogger

API_KEY = "test-key-123"
AUTH = {"X-API-Key": API_KEY}
KNOWN = next(iter(CONFIG))               # any strategy id that exists


def _atr(enabled=True, strategies=("orb_breakout",)):
    return AtrPullbackLayer({**ATR_EXECUTION, "enabled": enabled,
                             "strategies": list(strategies)})


class FakeLoggerDB:
    def __init__(self):
        self.saved = {}
        self.decisions = []
        self.fail_save = False

    def save_execution_settings(self, layer, enabled, strategies):
        if self.fail_save:
            raise RuntimeError("disk is full")
        self.saved[layer] = {"enabled": enabled, "strategies": list(strategies)}

    def log_decision(self, *a, **kw):
        self.decisions.append((a, kw))


class FakeExecutor:
    def __init__(self):
        self.atr_layer = _atr()
        self.execution_layers = {"atr": self.atr_layer}
        self.logger_db = FakeLoggerDB()


@pytest.fixture
def ex(monkeypatch):
    fake = FakeExecutor()
    monkeypatch.setattr(server, "executor", fake)
    monkeypatch.setattr(server, "EXECUTOR_API_KEY", API_KEY)
    monkeypatch.setattr(server, "_alert", lambda *a, **kw: None)
    return fake


@pytest.fixture
def client(ex):
    return TestClient(app)


def test_requires_api_key(client, ex):
    assert client.post("/execution/atr", json={"strategies": "all"}).status_code == 401
    assert ex.atr_layer.strategies == ["orb_breakout"]


def test_all_applies_to_every_strategy(client, ex):
    r = client.post("/execution/atr", json={"strategies": "all"}, headers=AUTH)
    assert r.status_code == 200
    assert ex.atr_layer.strategies == []
    assert ex.logger_db.saved == {"atr": {"enabled": True, "strategies": []}}
    assert r.json()["strategies"] == []


def test_a_list_replaces_the_strategies(client, ex):
    client.post("/execution/atr", json={"strategies": [KNOWN, KNOWN]}, headers=AUTH)
    assert ex.atr_layer.strategies == [KNOWN]            # de-duplicated


def test_omitted_fields_keep_their_value(client, ex):
    client.post("/execution/atr", json={"enabled": False}, headers=AUTH)
    assert ex.atr_layer.enabled is False
    assert ex.atr_layer.strategies == ["orb_breakout"]


def test_an_empty_list_is_refused(client, ex):
    r = client.post("/execution/atr", json={"strategies": []}, headers=AUTH)
    assert r.status_code == 422
    assert ex.atr_layer.strategies == ["orb_breakout"]
    assert ex.logger_db.saved == {}


def test_an_unknown_strategy_is_refused(client, ex):
    r = client.post("/execution/atr", json={"strategies": ["no_such_strategy"]}, headers=AUTH)
    assert r.status_code == 422
    assert "no_such_strategy" in r.json()["detail"]
    assert ex.logger_db.saved == {}


def test_an_unknown_layer_is_404(client, ex):
    r = client.post("/execution/vwap", json={"strategies": "all"}, headers=AUTH)
    assert r.status_code == 404
    assert "atr" in r.json()["detail"]                   # says what does exist


def test_a_failed_save_changes_nothing(client, ex):
    ex.logger_db.fail_save = True
    r = client.post("/execution/atr", json={"strategies": "all"}, headers=AUTH)
    assert r.status_code == 500
    assert ex.atr_layer.strategies == ["orb_breakout"]


def test_every_layer_is_listed(client, ex):
    ex.execution_layers["other"] = _atr(enabled=False, strategies=())
    layers = {l["layer"]: l for l in client.get("/execution").json()["layers"]}
    assert layers["atr"]["enabled"] is True and layers["atr"]["strategies"] == ["orb_breakout"]
    assert layers["other"]["enabled"] is False and layers["other"]["strategies"] == []


# ------------------------------------------------------------------ persistence

def test_settings_round_trip_through_sqlite(tmp_path):
    db = EventLogger(db_path=tmp_path / "executor.db")
    try:
        assert db.load_execution_settings() == {}         # never set -> config.py applies
        db.save_execution_settings("atr", True, [])
        db.save_execution_settings("atr", False, ["a", "b"])
        db.save_execution_settings("vwap", True, [])
        assert db.load_execution_settings() == {
            "atr": {"enabled": False, "strategies": ["a", "b"]},
            "vwap": {"enabled": True, "strategies": []},
        }
    finally:
        db.close()


def test_startup_restores_the_saved_setting(tmp_path):
    from execution.central_execution import CentralExecutor

    class Holder:                                         # just what the restore touches
        _restore_persistent_state = CentralExecutor._restore_persistent_state

    db = EventLogger(db_path=tmp_path / "executor.db")
    try:
        db.save_execution_settings("atr", True, [])
        db.save_execution_settings("retired_layer", True, [])   # ignored, not fatal
        h = Holder()
        h.logger_db = db
        h.atr_layer = _atr(enabled=False)
        h.execution_layers = {"atr": h.atr_layer}
        h.ledger = type("L", (), {"restore_state": lambda self, db: None})()
        h.risk_manager = type("R", (), {"_active_strategies": set()})()
        h._restore_persistent_state()
        assert h.atr_layer.enabled is True
        assert h.atr_layer.strategies == []
    finally:
        db.close()


# ------------------------------------------------------------------ Telegram /execset

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
    tc.layer = {"layer": "atr", "enabled": True, "strategies": ["orb_breakout"],
                "pending_orders": 0}
    strategies = ["orb_breakout", "cross_sectional_momentum", "kalman_vecm"]

    def api_get(path, **kw):
        if path == "/execution":
            return {"layers": [dict(tc.layer)]}
        if path == "/strategies":
            return {"strategies": [{"strategy_id": s} for s in strategies]}
        return {}

    def api_post(path, body=None):
        tc.posted.append((path, body))
        return {"layer": "atr", "enabled": body["enabled"],
                "strategies": [] if body["strategies"] == "all" else body["strategies"]}

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


def test_exec_shows_every_layer(bot):
    bot.handle(msg("/exec"))
    assert "atr: ON for orb_breakout" in bot.sent[-1]


def test_all_previews_then_applies_after_confirmation(bot):
    bot.handle(msg("/execset atr all"))
    assert "now   ON for orb_breakout" in bot.sent[-1]
    assert "after ON for all strategies" in bot.sent[-1]
    assert bot.posted == []                               # nothing changed yet
    _confirm(bot)
    assert bot.posted == [("/execution/atr", {"enabled": True, "strategies": "all",
                                              "changed_by": f"u{OWNER}"})]


def test_add_extends_the_list(bot):
    bot.handle(msg("/execset atr add cross_sectional_momentum"))
    _confirm(bot)
    assert bot.posted[-1][1]["strategies"] == ["orb_breakout", "cross_sectional_momentum"]


def test_removing_the_last_strategy_is_refused(bot):
    """An empty list means ALL strategies to the layer — the opposite of intended."""
    bot.handle(msg("/execset atr remove orb_breakout"))
    assert "use /execset atr off" in bot.sent[-1]
    assert "/confirm" not in bot.sent[-1]


def test_removing_from_all_spells_out_the_rest(bot):
    bot.layer["strategies"] = []
    bot.handle(msg("/execset atr remove kalman_vecm"))
    assert "one added later will not be on atr" in bot.sent[-1]
    _confirm(bot)
    assert bot.posted[-1][1]["strategies"] == ["orb_breakout", "cross_sectional_momentum"]


def test_an_unknown_strategy_or_layer_never_gets_a_token(bot):
    bot.handle(msg("/execset atr only typo_strategy"))
    assert "unknown strategy" in bot.sent[-1] and "/confirm" not in bot.sent[-1]
    bot.handle(msg("/execset vwap all"))
    assert "unknown execution layer" in bot.sent[-1] and "/confirm" not in bot.sent[-1]


def test_a_stranger_cannot_change_execution(bot):
    bot.handle(msg("/execset atr all", user_id=STRANGER))
    assert "not allow-listed" in bot.sent[-1]
    assert bot.posted == []
