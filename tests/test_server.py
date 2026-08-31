"""
tests/test_server.py

HTTP-level tests for api/server.py with a fake executor injected in place of
the real CentralExecutor. Uses TestClient WITHOUT its context manager on
purpose: that skips the lifespan handler, so executor.start() (which connects
to IB) never runs. No TWS/Gateway required.

Run:
    source venv_algotrade/bin/activate
    pip install httpx           # Starlette's TestClient needs it
    PYTHONPATH=src pytest tests/test_server.py -v
"""

import pytest
from fastapi.testclient import TestClient

import api.server as server
from api.server import app

API_KEY = "test-key-123"
AUTH = {"X-API-Key": API_KEY}

# a strategy_id that actually exists in config.py, for the /strategies routes
KNOWN_STRAT = "cross_sectional_momentum"


# ---------------------------------------------------------------------------
# Fake executor — only the surface area the routes touch.
# ---------------------------------------------------------------------------
class FakeLedger:
    def __init__(self):
        self.current_positions = {"AAPL": 100.0, "MSFT": -50.0}
        self.strategy_positions = {KNOWN_STRAT: {"AAPL": 100.0, "MSFT": -50.0}}
        self.strategy_realized_pnl = {KNOWN_STRAT: 1234.5}


class FakeRiskManager:
    def __init__(self):
        self._active = {KNOWN_STRAT}

    def is_active(self, strategy_id):
        return strategy_id in self._active


class FakeDB:
    def __init__(self):
        self.orders = {}  # order_id -> dict

    def get_order(self, order_id):
        return self.orders.get(order_id)


class FakeExecutor:
    def __init__(self):
        self.ledger = FakeLedger()
        self.risk_manager = FakeRiskManager()
        self.logger_db = FakeDB()
        self.order_status = {}
        self._killed = False
        self._connected = True

        # call recorders
        self.last_intent = None
        self.kill_calls = []

    # --- methods the routes call ---
    def process_intent(self, intent):
        self.last_intent = intent
        # default: accept. Tests override for the rejection case.
        return {"accepted": True, "order_id": 42}

    def isConnected(self):
        return self._connected

    def kill_switch(self, flatten=True):
        self.kill_calls.append(flatten)
        self._killed = True


@pytest.fixture
def fake(monkeypatch):
    fx = FakeExecutor()
    monkeypatch.setattr(server, "executor", fx)
    monkeypatch.setattr(server, "EXECUTOR_API_KEY", API_KEY)
    return fx


@pytest.fixture
def client(fake):
    # no `with` — lifespan (and executor.start / IB connect) never runs
    return TestClient(app)


# ---------------------------------------------------------------------------
# Import canary — fails to even collect if the line-50 typo isn't fixed.
# ---------------------------------------------------------------------------
def test_module_imports():
    assert server.app is not None


# ---------------------------------------------------------------------------
# POST /orders — auth + pass-through + domain rejection semantics
# ---------------------------------------------------------------------------
def test_post_orders_requires_key(client):
    r = client.post("/orders", json={"foo": "bar"})
    assert r.status_code == 401


def test_post_orders_wrong_key(client):
    r = client.post("/orders", json={"foo": "bar"}, headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_post_orders_accepted_passthrough(client, fake):
    intent = {"strategy_id": KNOWN_STRAT, "client_order_id": "abc"}
    r = client.post("/orders", json=intent, headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"accepted": True, "order_id": 42}
    # the raw dict reached process_intent unchanged
    assert fake.last_intent == intent


def test_post_orders_rejection_is_200_with_reason(client, fake):
    fake.process_intent = lambda intent: {"accepted": False, "reason": "risk breach"}
    r = client.post("/orders", json={"strategy_id": KNOWN_STRAT}, headers=AUTH)
    assert r.status_code == 200          # domain rejection, not an HTTP error
    body = r.json()
    assert body["accepted"] is False
    assert body["reason"] == "risk breach"


# ---------------------------------------------------------------------------
# GET /orders/{id} — memory first, DB fallback, 404
# ---------------------------------------------------------------------------
def test_get_order_from_memory(client, fake):
    fake.order_status[7] = {"status": "Filled", "filled": 100, "remaining": 0}
    r = client.get("/orders/7")
    assert r.status_code == 200
    body = r.json()
    assert body["order_id"] == 7
    assert body["status"] == "Filled"
    assert body["filled"] == 100


def test_get_order_db_fallback(client, fake):
    fake.logger_db.orders[9] = {"order_id": 9, "client_order_id": "x", "status": "Submitted"}
    r = client.get("/orders/9")
    assert r.status_code == 200
    assert r.json()["order_id"] == 9


def test_get_order_unknown_404(client):
    r = client.get("/orders/999")
    assert r.status_code == 404


def test_get_orders_is_unauthenticated(client, fake):
    # read routes carry no key by design
    fake.order_status[1] = {"status": "Submitted"}
    assert client.get("/orders/1").status_code == 200


# ---------------------------------------------------------------------------
# GET /positions, /pnl, /health
# ---------------------------------------------------------------------------
def test_positions(client):
    body = client.get("/positions").json()
    assert body["current_positions"]["AAPL"] == 100.0
    assert body["strategy_positions"][KNOWN_STRAT]["MSFT"] == -50.0


def test_pnl(client):
    body = client.get("/pnl").json()
    assert body["realized_pnl"][KNOWN_STRAT] == 1234.5


def test_health(client, fake, monkeypatch):
    # pin market status so the test doesn't depend on the wall clock
    monkeypatch.setattr(server, "is_market_open", lambda *a, **k: True)

    fake._connected = True
    fake._killed = False
    body = client.get("/health").json()
    assert body == {"connected": True, "killed": False, "market_open": True}

    fake._connected = False
    fake._killed = True
    body = client.get("/health").json()
    assert body == {"connected": False, "killed": True, "market_open": True}


# ---------------------------------------------------------------------------
# POST /kill — auth + flatten flag plumbing
# ---------------------------------------------------------------------------
def test_kill_requires_key(client):
    assert client.post("/kill", json={}).status_code == 401


def test_kill_default_flatten_true(client, fake):
    r = client.post("/kill", json={}, headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"killed": True, "flattened": True}
    assert fake.kill_calls == [True]


def test_kill_flatten_false(client, fake):
    r = client.post("/kill", json={"flatten": False}, headers=AUTH)
    assert r.status_code == 200
    assert r.json()["flattened"] is False
    assert fake.kill_calls == [False]


# ---------------------------------------------------------------------------
# GET /strategies/{id}/status and /allocation
# ---------------------------------------------------------------------------
def test_strategy_status_active(client):
    body = client.get(f"/strategies/{KNOWN_STRAT}/status").json()
    assert body == {"strategy_id": KNOWN_STRAT, "status": "active"}


def test_strategy_status_halted(client, fake):
    fake.risk_manager._active.discard(KNOWN_STRAT)
    body = client.get(f"/strategies/{KNOWN_STRAT}/status").json()
    assert body["status"] == "halted"


def test_strategy_status_unknown_404(client):
    assert client.get("/strategies/does_not_exist/status").status_code == 404


def test_strategy_allocation_known(client):
    body = client.get(f"/strategies/{KNOWN_STRAT}/allocation").json()
    assert body["strategy_id"] == KNOWN_STRAT
    assert body["capital_allocation"] == 100_000.0
    assert body["max_drawdown"] == 0.15


def test_strategy_allocation_unknown_404(client):
    assert client.get("/strategies/does_not_exist/allocation").status_code == 404
