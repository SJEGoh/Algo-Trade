"""client/executor_client.py — the behaviour that matters when a strategy is off-box.

The signal logic is the strategy's problem. What this client owes you is that a submission
either happens or is impossible to miss: retry what is transient, refuse to retry what is
a decision, and shout when it gives up. No network — requests is stubbed.
"""
import pytest
import requests

from client.executor_client import (ExecutorClient, ExecutorRejected, ExecutorUnreachable)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text
        self.content = b"x"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 500:
            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    """Replays a scripted list of responses (or exceptions) and records every call."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        item = self.script.pop(0) if self.script else FakeResponse()
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr("client.executor_client.time.sleep", lambda *_: None)


@pytest.fixture(autouse=True)
def no_telegram(monkeypatch):
    """Catch the alert instead of posting it."""
    sent = []
    monkeypatch.setattr(ExecutorClient, "_alert", lambda self, text: sent.append(text))
    return sent


def make(session, **kw):
    return ExecutorClient(base_url="http://executor:8000", api_key="k",
                          strategy_id="s1", session=session, **kw)


# ------------------------------------------------------------------ retries
def test_a_transient_connection_error_is_retried(no_telegram):
    session = FakeSession(requests.ConnectionError("refused"),
                          FakeResponse(200, {"accepted": True, "order_id": 7}))
    client = make(session)
    assert client.submit_order({"instrument": {"symbol": "AAPL"}})["order_id"] == 7
    assert len(session.calls) == 2
    assert no_telegram == []


def test_a_timeout_is_retried(no_telegram):
    session = FakeSession(requests.Timeout("slow"), FakeResponse(200, {"ok": True}))
    assert make(session).health() == {"ok": True}


@pytest.mark.parametrize("status", [502, 503, 504])
def test_gateway_errors_are_retried(status, no_telegram):
    session = FakeSession(FakeResponse(status), FakeResponse(200, {"ok": True}))
    assert make(session).health() == {"ok": True}


def test_a_retry_reuses_the_same_client_order_id():
    """The executor dedups on it, which is the whole reason retrying is safe — a retry that
    minted a fresh id would place the order twice."""
    session = FakeSession(requests.ConnectionError("refused"),
                          FakeResponse(200, {"accepted": True}))
    make(session).submit_order({"instrument": {"symbol": "AAPL"}})
    first, second = session.calls[0]["json"], session.calls[1]["json"]
    assert first["client_order_id"] == second["client_order_id"]


# ------------------------------------------------------------------ giving up
def test_giving_up_raises_and_alerts(no_telegram):
    session = FakeSession(*[requests.ConnectionError("refused")] * 3)
    with pytest.raises(ExecutorUnreachable, match="did NOT go through"):
        make(session, retries=3).submit_order({"instrument": {"symbol": "AAPL"}})
    assert len(session.calls) == 3
    assert no_telegram and "unreachable" in no_telegram[0]


def test_the_alert_can_be_turned_off(no_telegram):
    session = FakeSession(*[requests.ConnectionError("x")] * 2)
    with pytest.raises(ExecutorUnreachable):
        make(session, retries=2, alert_on_failure=False).health()
    assert no_telegram == []


# ------------------------------------------------------------------ not retried
@pytest.mark.parametrize("status, detail", [
    (401, "invalid or missing API key"),
    (404, "unknown strategy"),
    (422, "strategy_id required"),
    (423, "kill switch active"),
])
def test_a_4xx_is_a_decision_not_a_glitch(status, detail, no_telegram):
    """Retrying a bad key or a malformed intent just makes the same mistake three times."""
    session = FakeSession(FakeResponse(status, {"detail": detail}))
    with pytest.raises(ExecutorRejected) as excinfo:
        make(session).submit_order({"instrument": {"symbol": "AAPL"}})
    assert excinfo.value.status_code == status
    assert detail in str(excinfo.value)
    assert len(session.calls) == 1
    assert no_telegram == []


def test_a_domain_rejection_is_returned_not_raised():
    """The executor answers HTTP 200 with accepted:false when RISK says no — that is a
    normal outcome the strategy should see and carry on from."""
    session = FakeSession(FakeResponse(200, {"accepted": False,
                                             "reason": "order would exceed allocation"}))
    result = make(session).submit_order({"instrument": {"symbol": "AAPL"}})
    assert result["accepted"] is False and "allocation" in result["reason"]


# ------------------------------------------------------------------ preflight
def test_preflight_passes_when_healthy():
    session = FakeSession(FakeResponse(200, {"connected": True, "killed": False}))
    assert make(session).preflight()["connected"] is True


@pytest.mark.parametrize("health, expected, error", [
    ({"connected": False}, "NOT connected to IB", ExecutorUnreachable),
    ({"connected": True, "killed": True}, "kill switch", ExecutorRejected),
    ({"connected": True, "startup_degraded": True}, "degraded", ExecutorRejected),
])
def test_preflight_refuses_to_trade_into_a_broken_executor(health, expected, error):
    """Fail before generating a book, not three legs into submitting one."""
    with pytest.raises(error, match=expected):
        make(FakeSession(FakeResponse(200, health))).preflight()


# ------------------------------------------------------------------ payloads
def test_submit_book_sends_an_authoritative_book():
    session = FakeSession(FakeResponse(200, {"orders": []}))
    make(session).submit_book([
        {"instrument": {"symbol": "AAPL"}, "target_quantity": 10, "expected_price": 100.0},
        {"instrument": {"symbol": "MSFT"}, "target_quantity": 0, "expected_price": 400.0}])
    body = session.calls[0]["json"]
    assert body["strategy_id"] == "s1"
    assert [i["target_quantity"] for i in body["intents"]] == [10, 0]
    assert session.calls[0]["headers"]["X-API-Key"] == "k"


def test_reads_do_not_send_the_api_key():
    session = FakeSession(FakeResponse(200, {}))
    make(session).positions()
    assert session.calls[0]["headers"] == {}


def test_submit_orders_summarises_instead_of_stopping_at_the_first_no():
    session = FakeSession(
        FakeResponse(200, {"accepted": True, "order_id": 1}),
        FakeResponse(200, {"accepted": False, "reason": "exceeds allocation"}),
        FakeResponse(200, {"accepted": True, "order_id": 3}))
    result = make(session).submit_orders([
        {"instrument": {"symbol": s}} for s in ("AAPL", "MSFT", "GOOGL")])
    assert result["ok"] == 2 and result["refused"] == 1
    assert result["rejected"][0]["symbol"] == "MSFT"


def test_an_unreachable_executor_still_aborts_a_multi_order_submission(no_telegram):
    """One rejected order is a per-order outcome; an unreachable executor means the rest of
    the book will not go in either, so it must not be swallowed into the summary."""
    session = FakeSession(FakeResponse(200, {"accepted": True}),
                          *[requests.ConnectionError("gone")] * 3)
    with pytest.raises(ExecutorUnreachable):
        make(session, retries=3).submit_orders([
            {"instrument": {"symbol": s}} for s in ("AAPL", "MSFT")])


def test_strategy_id_is_required_somewhere():
    client = ExecutorClient(base_url="http://x", api_key="k", session=FakeSession())
    with pytest.raises(ValueError, match="strategy_id is required"):
        client.submit_book([])


def test_client_order_ids_are_unique_per_intent():
    ids = {ExecutorClient.new_client_order_id("s1", "AAPL") for _ in range(100)}
    assert len(ids) == 100
