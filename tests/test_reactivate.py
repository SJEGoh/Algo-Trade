"""tests/test_reactivate.py — POST /strategies/{id}/reactivate clears a halt: auth-gated,
404 for unknown ids, and calls RiskManager.reactivate_strategy for a known one."""
import pytest
from fastapi.testclient import TestClient

import api.server as server
from api.server import app

API_KEY = "test-key-123"
AUTH = {"X-API-Key": API_KEY}


class FakeRM:
    def __init__(self):
        self.reactivated = []

    def reactivate_strategy(self, sid):
        self.reactivated.append(sid)


class FakeExecutor:
    def __init__(self):
        self.risk_manager = FakeRM()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "executor", FakeExecutor())
    monkeypatch.setattr(server, "EXECUTOR_API_KEY", API_KEY)
    return TestClient(app)


def test_reactivate_requires_key(client):
    assert client.post("/strategies/halt_test_1/reactivate").status_code == 401


def test_reactivate_unknown_strategy_404(client):
    assert client.post("/strategies/does_not_exist/reactivate", headers=AUTH).status_code == 404


def test_reactivate_known_strategy(client):
    r = client.post("/strategies/halt_test_1/reactivate", headers=AUTH)
    assert r.status_code == 200 and r.json()["status"] == "active"
    assert server.executor.risk_manager.reactivated == ["halt_test_1"]
