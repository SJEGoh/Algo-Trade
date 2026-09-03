"""tests/test_pool_guard.py — the pooled endpoints (/target, /targets) enforce the same
protections as /orders: API key, kill-switch, and the market-hours gate for equities (with
a futures bypass). HTTP-level, fake executor + coordinator; no TWS. Mirrors test_server.py."""
import pytest
from fastapi.testclient import TestClient

import api.server as server
from api.server import app

API_KEY = "test-key-123"
AUTH = {"X-API-Key": API_KEY}


class FakeCoordinator:
    def __init__(self):
        self.set_calls = []
        self.book_calls = []

    def set_target(self, sid, symbol, qty, instrument=None, price=None):
        self.set_calls.append((sid, symbol, qty))
        return {"accepted": True, "orders": [{"symbol": symbol, "delta": qty, "order_id": 1}]}

    def submit_book(self, sid, intents):
        self.book_calls.append((sid, len(intents)))
        return {"accepted": True, "orders": []}


class FakeLoggerDB:
    def log_decision(self, *a, **kw): pass


class FakeExecutor:
    def __init__(self):
        self._killed = False
        self._enforce_market_hours = True
        self.coordinator = FakeCoordinator()
        self.logger_db = FakeLoggerDB()


@pytest.fixture
def fake(monkeypatch):
    fx = FakeExecutor()
    monkeypatch.setattr(server, "executor", fx)
    monkeypatch.setattr(server, "EXECUTOR_API_KEY", API_KEY)
    return fx


@pytest.fixture
def client(fake):
    return TestClient(app)


def _open(monkeypatch, is_open):
    monkeypatch.setattr(server, "is_market_open", lambda *a, **k: is_open)


STK = {"symbol": "MSFT", "asset_class": "equity", "exchange": "SMART", "sec_type": "STK"}
FUT = {"symbol": "CL", "asset_class": "future", "sec_type": "FUT", "exchange": "NYMEX",
       "multiplier": 1000.0}


def _target_body(inst, qty=10):
    return {"strategy_id": "s1", "symbol": inst["symbol"], "quantity": qty,
            "instrument": inst, "price": 100.0}


def _book_body(inst):
    return {"strategy_id": "s1", "intents": [
        {"instrument": inst, "target_quantity": 10, "expected_price": 100.0}]}


# --- auth ---
def test_target_requires_key(client):
    assert client.post("/target", json=_target_body(STK)).status_code == 401


def test_targets_requires_key(client):
    assert client.post("/targets", json=_book_body(STK)).status_code == 401


# --- kill switch ---
def test_targets_blocked_when_killed(client, fake, monkeypatch):
    _open(monkeypatch, True)
    fake._killed = True
    assert client.post("/targets", json=_book_body(STK), headers=AUTH).status_code == 423
    assert fake.coordinator.book_calls == []          # never reached the coordinator


# --- market-hours gate for equities ---
def test_equity_targets_blocked_when_market_closed(client, fake, monkeypatch):
    _open(monkeypatch, False)
    assert client.post("/targets", json=_book_body(STK), headers=AUTH).status_code == 409
    assert fake.coordinator.book_calls == []


def test_equity_target_allowed_when_market_open(client, fake, monkeypatch):
    _open(monkeypatch, True)
    r = client.post("/target", json=_target_body(STK), headers=AUTH)
    assert r.status_code == 200 and r.json()["accepted"] is True
    assert fake.coordinator.set_calls == [("s1", "MSFT", 10)]


# --- futures bypass the market-hours gate (as in process_intent) ---
def test_futures_target_allowed_when_market_closed(client, fake, monkeypatch):
    _open(monkeypatch, False)
    r = client.post("/target", json=_target_body(FUT, qty=1), headers=AUTH)
    assert r.status_code == 200 and r.json()["accepted"] is True


def test_futures_only_book_allowed_when_market_closed(client, fake, monkeypatch):
    _open(monkeypatch, False)
    r = client.post("/targets", json=_book_body(FUT), headers=AUTH)
    assert r.status_code == 200
    assert len(fake.coordinator.book_calls) == 1
