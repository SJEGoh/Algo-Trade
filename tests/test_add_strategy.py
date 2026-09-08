"""tests/test_add_strategy.py — POST /strategies puts a strategy on the fail-closed
allowlist at runtime, and DELETE takes it off again.

The properties worth pinning down are the ones whose failure is silent:

  * it PERSISTS BEFORE it mutates CONFIG — a strategy that trades today and is missing from
    the database tomorrow has its orders rejected mid-session with nothing to explain it;
  * it refuses an id that already exists, rather than overwriting a live strategy's cap;
  * the new id is added to the risk manager's ACTIVE set — CONFIG membership alone is not
    enough, and an inactive strategy has every intent rejected as "not active";
  * DELETE refuses while the strategy still holds something, which would orphan positions
    at the broker with no allowlist entry to flatten them through.
"""
import pytest
from fastapi.testclient import TestClient

import api.server as server
from api.server import app
from config import CONFIG

API_KEY = "test-key-123"
AUTH = {"X-API-Key": API_KEY}
NEW = {"strategy_id": "runtime_added", "capital_allocation": 150_000.0,
       "max_drawdown": 0.10}


class FakeRM:
    def __init__(self):
        self._active_strategies = set()

    def is_active(self, sid):
        return sid in self._active_strategies


class FakeLedger:
    def __init__(self):
        self.strategy_positions = {}

    def _basis(self, sid):
        return CONFIG[sid].get("starting_cash", CONFIG[sid]["capital_allocation"])


class FakeLoggerDB:
    def __init__(self):
        self.saved = {}
        self.decisions = []
        self.fail_save = False

    def save_runtime_strategy(self, sid, alloc, dd, cash=None, by=""):
        if self.fail_save:
            raise RuntimeError("disk is full")
        self.saved[sid] = {"capital_allocation": alloc, "max_drawdown": dd}
        if cash is not None:
            self.saved[sid]["starting_cash"] = cash

    def load_runtime_strategies(self):
        return dict(self.saved)

    def delete_runtime_strategy(self, sid):
        self.saved.pop(sid, None)

    def log_decision(self, *a, **kw):
        self.decisions.append((a, kw))


class FakeExecutor:
    def __init__(self):
        self.risk_manager = FakeRM()
        self.ledger = FakeLedger()
        self.logger_db = FakeLoggerDB()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "executor", FakeExecutor())
    monkeypatch.setattr(server, "EXECUTOR_API_KEY", API_KEY)
    monkeypatch.setattr(server, "_alert", lambda *a, **kw: None)
    yield TestClient(app)
    CONFIG.pop(NEW["strategy_id"], None)        # CONFIG is module-level and shared


# ------------------------------------------------------------------ auth and validation
def test_requires_api_key(client):
    assert client.post("/strategies", json=NEW).status_code == 401


def test_rejects_existing_id(client):
    body = dict(NEW, strategy_id="test_suite")   # already in config.py
    r = client.post("/strategies", json=body, headers=AUTH)
    assert r.status_code == 409
    assert "allocation" in r.json()["detail"]    # points at the right endpoint


@pytest.mark.parametrize("bad, field", [
    ({"capital_allocation": 0}, "allocation must be > 0"),
    ({"capital_allocation": -5}, "negative allocation"),
    ({"max_drawdown": 0}, "a zero drawdown halt is not a halt"),
    ({"max_drawdown": 1.5}, "drawdown over 100%"),
    ({"strategy_id": "has spaces"}, "id must be identifier-safe"),
    ({"strategy_id": ""}, "empty id"),
])
def test_rejects_bad_input(client, bad, field):
    r = client.post("/strategies", json=dict(NEW, **bad), headers=AUTH)
    assert r.status_code == 422, f"{field}: {r.status_code} {r.text}"
    assert NEW["strategy_id"] not in CONFIG


# ------------------------------------------------------------------ the happy path
def test_adds_to_config_and_active_set(client):
    r = client.post("/strategies", json=NEW, headers=AUTH)
    assert r.status_code == 200, r.text
    sid = NEW["strategy_id"]

    assert CONFIG[sid]["capital_allocation"] == 150_000.0
    assert CONFIG[sid]["max_drawdown"] == 0.10
    # CONFIG membership alone would leave every intent rejected as "not active"
    assert sid in server.executor.risk_manager._active_strategies
    assert server.executor.logger_db.saved[sid]["capital_allocation"] == 150_000.0
    assert r.json()["starting_cash"] == 150_000.0


def test_explicit_starting_cash_is_kept(client):
    r = client.post("/strategies", json=dict(NEW, starting_cash=25_000.0), headers=AUTH)
    assert r.status_code == 200
    assert r.json()["starting_cash"] == 25_000.0
    assert CONFIG[NEW["strategy_id"]]["starting_cash"] == 25_000.0


def test_shows_up_in_list_strategies(client):
    client.post("/strategies", json=NEW, headers=AUTH)
    listed = {s["strategy_id"] for s in client.get("/strategies").json()["strategies"]}
    assert NEW["strategy_id"] in listed


# ------------------------------------------------------------------ the failure that matters
def test_failed_persist_does_not_add_the_strategy(client):
    """A strategy live in CONFIG but absent from the database is the worst outcome here: it
    trades until the next restart and is then silently rejected. Nothing must be mutated."""
    server.executor.logger_db.fail_save = True
    r = client.post("/strategies", json=NEW, headers=AUTH)

    assert r.status_code == 500
    assert "NOT added" in r.json()["detail"]
    assert NEW["strategy_id"] not in CONFIG
    assert NEW["strategy_id"] not in server.executor.risk_manager._active_strategies


# ------------------------------------------------------------------ removal
def test_delete_removes_a_runtime_strategy(client):
    client.post("/strategies", json=NEW, headers=AUTH)
    r = client.delete(f"/strategies/{NEW['strategy_id']}", headers=AUTH)
    assert r.status_code == 200 and r.json()["removed"] is True
    assert NEW["strategy_id"] not in CONFIG
    assert NEW["strategy_id"] not in server.executor.logger_db.saved


def test_delete_refuses_while_positions_are_open(client):
    client.post("/strategies", json=NEW, headers=AUTH)
    server.executor.ledger.strategy_positions[NEW["strategy_id"]] = {"AAPL": 10}

    r = client.delete(f"/strategies/{NEW['strategy_id']}", headers=AUTH)
    assert r.status_code == 409
    assert "AAPL" in r.json()["detail"]
    assert NEW["strategy_id"] in CONFIG          # still tradeable, still flattenable


def test_delete_refuses_a_config_py_strategy(client):
    """config.py is the source of truth for those — deleting one from memory would come
    back at the next restart anyway."""
    r = client.delete("/strategies/test_suite", headers=AUTH)
    assert r.status_code == 409
    assert "test_suite" in CONFIG


def test_delete_unknown_is_404(client):
    assert client.delete("/strategies/nope", headers=AUTH).status_code == 404
