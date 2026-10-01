"""tests/test_account_summary.py — IB's own NAV next to the strategy ledger's.

Pinned down:
  * the executor's one-shot reqAccountSummary collects every tag per account, returns on
    accountSummaryEnd, and always cancels the subscription (IB allows two open at once);
  * GET /account caches for 30s, so the dashboard's 3-second poll doesn't become a stream
    of IB requests, and reports `unallocated` = IB NAV - strategies' NAV;
  * when IB doesn't answer it says `available: false` instead of erroring — an HTTP error
    would mark the whole dashboard stale.
"""
import threading

import pytest
from fastapi.testclient import TestClient

import api.server as server
from execution.central_execution import CentralExecutor


class Holder:
    """Just what fetch_account_summary and its callbacks touch."""
    fetch_account_summary = CentralExecutor.fetch_account_summary
    accountSummary = CentralExecutor.accountSummary
    accountSummaryEnd = CentralExecutor.accountSummaryEnd
    ACCOUNT_TAGS = CentralExecutor.ACCOUNT_TAGS

    def __init__(self, answer=True):
        self._acct_req_id = 1_000_000_000
        self._acct_lock = threading.Lock()
        self._acct_rows, self._acct_events = {}, {}
        self.answer, self.cancelled = answer, []

    def reqAccountSummary(self, req_id, group, tags):
        if not self.answer:
            return
        self.accountSummary(req_id, "DU123", "NetLiquidation", "1012345.67", "USD")
        self.accountSummary(req_id, "DU123", "BuyingPower", "4000000", "USD")
        self.accountSummaryEnd(req_id)

    def cancelAccountSummary(self, req_id):
        self.cancelled.append(req_id)


def test_fetch_collects_tags_and_cancels():
    h = Holder()
    out = h.fetch_account_summary()
    assert out == {"DU123": {"currency": "USD", "NetLiquidation": 1012345.67,
                             "BuyingPower": 4_000_000.0}}
    assert h.cancelled == [1_000_000_001]
    assert h._acct_rows == {} and h._acct_events == {}


def test_fetch_times_out_to_none_and_still_cancels():
    h = Holder(answer=False)
    assert h.fetch_account_summary(timeout=0.05) is None
    assert h.cancelled == [1_000_000_001]


class FakeExecutor:
    def __init__(self, summary):
        self.summary, self.calls = summary, 0

    def isConnected(self):
        return True

    def fetch_account_summary(self):
        self.calls += 1
        return self.summary


@pytest.fixture
def setup(monkeypatch):
    def make(summary):
        fake = FakeExecutor(summary)
        monkeypatch.setattr(server, "executor", fake)
        monkeypatch.setattr(server, "_account_cache", {"ts": 0.0, "accounts": None})
        monkeypatch.setitem(server._last_equity, "totals", {"nav": 550_000.0})
        return fake, TestClient(server.app)
    return make


def test_account_reports_ib_nav_and_the_unallocated_gap(setup):
    fake, client = setup({"DU123": {"currency": "USD", "NetLiquidation": 1_000_000.0,
                                    "BuyingPower": 4_000_000.0}})
    body = client.get("/account").json()
    assert body["available"] is True
    assert body["net_liquidation"] == 1_000_000.0
    assert body["strategies_nav"] == 550_000.0
    assert body["unallocated"] == 450_000.0
    client.get("/account")
    assert fake.calls == 1                                  # second call served from cache


def test_account_says_unavailable_instead_of_erroring(setup):
    _, client = setup(None)
    r = client.get("/account")
    assert r.status_code == 200
    assert r.json()["available"] is False
