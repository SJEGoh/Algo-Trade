"""Telling "we sent it" apart from "the broker took it".

`place_order` writes `status: "Submitted"` when the order goes to the socket, and IB's own
callback later writes the identical string. Nothing could distinguish the two, so orders IB
never accepted appeared on the dashboard as working, and a submission returned
`accepted: true` to a strategy whose orders were being refused. During the read-only-gateway
incident that produced 27 phantom orders that looked live and a strategy that would have
reported success while holding nothing.

`ack` is the distinction: pending (we sent it, IB has said nothing), live (IB has it),
rejected (IB refused it, with the reason).
"""
import time

import pytest
from fastapi.testclient import TestClient

import api.server as server
from api.server import app

API_KEY = "test-key-123"
AUTH = {"X-API-Key": API_KEY}


class FakeLedger:
    strategy_positions = {}


class FakeExecutor:
    def __init__(self):
        self.order_status = {}
        self.coordinator = None
        self.ledger = FakeLedger()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "executor", FakeExecutor())
    monkeypatch.setattr(server, "EXECUTOR_API_KEY", API_KEY)
    return TestClient(app)


def order(ex, oid, symbol="ARM", ack="pending", age=0.0, **kw):
    ex.order_status[oid] = {
        "symbol": symbol, "status": "Submitted", "strategy_id": "orb_breakout",
        "pending_qty": 8.0, "filled": 0, "remaining": 8.0,
        "ack": ack, "sent_at": time.time() - age, **kw}


# ------------------------------------------------------------------ the dashboard split
def test_unacknowledged_orders_are_kept_out_of_the_main_list(client):
    ex = server.executor
    order(ex, 1, ack="live")
    order(ex, 2, ack="pending")

    body = client.get("/orders").json()
    assert [o["order_id"] for o in body["orders"]] == [1]
    assert [o["order_id"] for o in body["unacknowledged"]] == [2]
    assert body["unacknowledged_count"] == 1


def test_they_are_shown_separately_not_hidden(client):
    """Hiding them would have made the incident invisible: an empty dashboard and no
    explanation for why nothing traded. The signal is "sent, but not acknowledged"."""
    ex = server.executor
    for oid in range(1, 9):
        order(ex, oid, ack="pending", age=45)

    body = client.get("/orders").json()
    assert body["orders"] == []
    assert len(body["unacknowledged"]) == 8
    assert "not acknowledged" in body["warning"]


def test_a_briefly_pending_order_is_not_alarming(client):
    """A moment between sending and acknowledgement is normal; only a stuck one warrants
    a warning, or the dashboard cries wolf on every order."""
    order(server.executor, 1, ack="pending", age=0.5)
    assert client.get("/orders").json()["warning"] is None


def test_rejected_orders_carry_ibs_reason(client):
    order(server.executor, 1, ack="rejected",
          last_error={"code": 321, "message": "read-only API"})
    body = client.get("/orders").json()
    assert body["orders"][0]["last_error"]["code"] == 321


def test_age_is_reported(client):
    order(server.executor, 1, ack="pending", age=30)
    assert client.get("/orders").json()["unacknowledged"][0]["age_sec"] == pytest.approx(30, abs=2)


# ------------------------------------------------------------------ the strategy's answer
def test_acks_endpoint_reports_per_order_state(client):
    ex = server.executor
    order(ex, 1, "ARM", ack="live")
    order(ex, 2, "ANET", ack="rejected", last_error={"code": 321, "message": "read-only"})
    order(ex, 3, "GLD", ack="pending")

    body = client.get("/orders/acks", params={"ids": "1,2,3"}).json()
    assert body["acks"]["1"]["ack"] == "live"
    assert body["acks"]["2"]["ack"] == "rejected"
    assert body["acks"]["2"]["last_error"]["code"] == 321
    assert body["acks"]["3"]["ack"] == "pending"
    assert body["pending"] == 1 and body["rejected"] == 1


def test_acks_endpoint_reports_ids_it_does_not_know(client):
    """An id the executor has never heard of is not the same as a pending one, and a
    strategy must not read it as "still working"."""
    order(server.executor, 1, ack="live")
    body = client.get("/orders/acks", params={"ids": "1,99"}).json()
    assert body["unknown_order_ids"] == [99]


def test_acks_endpoint_rejects_junk_ids(client):
    assert client.get("/orders/acks", params={"ids": "1,abc"}).status_code == 422


def test_acks_endpoint_needs_no_api_key(client):
    """A strategy holds the key, but this is a read and the dashboard uses it too."""
    order(server.executor, 1, ack="live")
    assert client.get("/orders/acks", params={"ids": "1"}).status_code == 200


# ------------------------------------------------------------------ the client half
from client.executor_client import ExecutorClient          # noqa: E402
from client.remote_strategy import RemoteStrategy          # noqa: E402


class FakeClient(ExecutorClient):
    """An ExecutorClient whose HTTP layer is a scripted sequence of ack responses."""
    def __init__(self, ack_sequence, **kw):
        super().__init__(base_url="http://x", api_key="k", strategy_id="s1",
                         alert_on_failure=False, **kw)
        self.ack_sequence = list(ack_sequence)
        self.submitted = None
        self.journalled = []

    def _request(self, method, path, **kw):
        if path == "/orders/acks":
            return self.ack_sequence.pop(0) if len(self.ack_sequence) > 1 \
                else self.ack_sequence[0]
        raise AssertionError(f"unexpected {method} {path}")

    def preflight(self): return {"market_open": True}
    def allocation(self, strategy_id=None): return {"capital_allocation": 10_000.0}
    def submit_book(self, intents, strategy_id=None):
        self.submitted = intents
        return {"orders": [{"symbol": "AAPL", "delta": 10, "order_id": 7}]}
    def journal(self, *a, **kw):
        self.journalled.append((a, kw))
        return {"logged": True}


class Strat(RemoteStrategy):
    strategy_id = "s1"
    ack_timeout = 0.2

    def generate_book(self, capital):
        return [self.intent("AAPL", 10, 100.0)]


def acks_body(state, err=None):
    entry = {"ack": state, "status": "Submitted", "symbol": "AAPL"}
    if err:
        entry["last_error"] = err
    return {"acks": {"7": entry}, "pending": int(state == "pending"),
            "rejected": int(state == "rejected"), "unknown_order_ids": []}


def test_order_ids_are_pulled_from_either_submission_shape():
    assert ExecutorClient.order_ids({"orders": [{"order_id": 7}, {"order_id": 8}]}) == [7, 8]
    assert ExecutorClient.order_ids({"submitted": [{"order_id": 3}]}) == [3]
    assert ExecutorClient.order_ids({}) == []


def test_a_confirmed_order_exits_ok():
    c = FakeClient([acks_body("live")])
    assert Strat(client=c).run() == RemoteStrategy.EXIT_OK


def test_a_rejected_order_does_not_report_success():
    """The whole point: the executor accepted it, the broker refused it. Exiting 0 here is
    how a strategy records a position it does not have."""
    c = FakeClient([acks_body("rejected", {"code": 321, "message": "read-only API"})])
    assert Strat(client=c).run() == RemoteStrategy.EXIT_NOT_ACKED


def test_an_unacknowledged_order_does_not_report_success():
    c = FakeClient([acks_body("pending")])
    assert Strat(client=c).run() == RemoteStrategy.EXIT_NOT_ACKED


def test_the_broker_outcome_reaches_the_journal():
    """A run that was refused must not read like one that worked."""
    c = FakeClient([acks_body("rejected", {"code": 321, "message": "read-only API"})])
    Strat(client=c).run()

    detail = c.journalled[-1][1]["detail"]
    assert "1 rejected" in detail
    assert "321" in detail and "read-only" in detail


def test_confirmation_can_be_turned_off():
    class NoConfirm(Strat):
        confirm_with_broker = False
    c = FakeClient([acks_body("pending")])
    assert NoConfirm(client=c).run() == RemoteStrategy.EXIT_OK


def test_waiting_stops_as_soon_as_every_order_is_answered():
    c = FakeClient([acks_body("live")])
    started = time.time()
    r = c.wait_for_acks([7], timeout=5.0, poll=1.0)
    assert r["live"] == [7] and r["pending"] == []
    assert time.time() - started < 1.0, "kept polling after the answer arrived"


def test_waiting_gives_up_after_the_timeout():
    c = FakeClient([acks_body("pending")])
    r = c.wait_for_acks([7], timeout=0.2, poll=0.05)
    assert r["pending"] == [7] and r["live"] == []


def test_no_orders_means_no_polling():
    """An empty book submits nothing, so there is nothing to confirm — it must not block."""
    c = FakeClient([])
    assert c.wait_for_acks([], timeout=5.0)["pending"] == []
