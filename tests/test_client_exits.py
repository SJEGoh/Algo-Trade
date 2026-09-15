"""The client side of exits: they are built with the entry, validated before they leave,
actually reach the executor, and a locked-out name is expected flat rather than reported as
an order that failed to fill."""
import pytest

from client.executor_client import ExecutorClient
from client.remote_strategy import RemoteStrategy, StrategyError


def test_intent_carries_only_the_exits_given():
    plain = RemoteStrategy.intent("AAPL", 10, 100.0)
    assert "exits" not in plain
    trail = RemoteStrategy.intent("AAPL", 10, 100.0, trail_pct=0.05)
    assert trail["exits"] == {"trail_pct": 0.05}


class Sent(ExecutorClient):
    def __init__(self):
        super().__init__(base_url="http://x", api_key="k", strategy_id="s1",
                         alert_on_failure=False)
        self.calls = []

    def _request(self, method, path, **kw):
        self.calls.append((method, path, kw))
        return {"accepted": True, "orders": []}


def test_submit_book_sends_the_exits():
    """It rebuilt each entry from three fields, so exits were silently dropped."""
    c = Sent()
    c.submit_book([RemoteStrategy.intent("AAPL", 10, 100.0, stop_pct=0.02),
                   RemoteStrategy.intent("MSFT", 5, 400.0)])
    sent = c.calls[-1][2]["json"]["intents"]
    assert sent[0]["exits"] == {"stop_pct": 0.02}
    assert "exits" not in sent[1]


def test_exits_endpoints():
    c = Sent()
    c.exits()
    c.clear_exit("AAPL")
    assert c.calls[0][1] == "/exits" and c.calls[0][2]["params"] == {"strategy_id": "s1"}
    assert c.calls[1][:2] == ("DELETE", "/exits/s1/AAPL") and c.calls[1][2]["auth"]


class Strat(RemoteStrategy):
    strategy_id = "s1"
    entries = []

    def generate_book(self, capital):
        return self.entries


@pytest.mark.parametrize("exits, message", [
    ({"stop_loss": 95}, "unknown exit field"),
    ({"stop_price": 95, "stop_pct": 0.05}, "not both"),
    ({"trail_pct": 5}, "fraction"),
    ({"stop_price": 105}, "wrong side"),
    ({"take_profit_price": -1}, "positive"),
])
def test_a_bad_exit_never_leaves_the_strategy(exits, message):
    e = RemoteStrategy.intent("AAPL", 10, 100.0)
    e["exits"] = exits
    with pytest.raises(StrategyError, match=message):
        Strat(client=object()).validate([e])


class FakeClient:
    base_url = "http://x"
    order_ids = staticmethod(ExecutorClient.order_ids)

    def __init__(self, result):
        self.result = result
        self.fill_calls = []
        self.journalled = []

    def preflight(self):
        return {"market_open": True}

    def allocation(self):
        return {"capital_allocation": 10_000.0}

    def submit_book(self, book, strategy_id=None):
        return self.result

    def wait_for_acks(self, order_ids, timeout=30.0, poll=2.0):
        return {"live": list(order_ids), "rejected": [], "pending": [], "acks": {}}

    def wait_for_fills(self, targets, order_ids=(), authoritative=True, timeout=60.0,
                       poll=2.0, strategy_id=None):
        self.fill_calls.append(dict(targets))
        return {"filled": sorted(targets), "unfilled": {}, "reason": None,
                "working_order_ids": []}

    def journal(self, event_type, summary, detail="", symbols=None, strategy_id=None):
        self.journalled.append(detail)


def test_a_locked_out_name_is_expected_flat():
    c = FakeClient({"accepted": True, "orders": [{"symbol": "MSFT", "order_id": 1}],
                    "exits": {"armed": [], "blocked": {"AAPL": {"kind": "stop"}}}})
    Strat.entries = [RemoteStrategy.intent("AAPL", 10, 100.0),
                     RemoteStrategy.intent("MSFT", 5, 400.0)]
    assert Strat(client=c).run() == RemoteStrategy.EXIT_OK
    assert c.fill_calls[-1] == {"AAPL": 0, "MSFT": 5}
    assert "held flat by exit lockout: AAPL (stop)" in c.journalled[-1]


def test_a_refused_book_exits_1_without_waiting_for_fills():
    """It used to fall through to waiting for fills that could never come, and exit 4."""
    c = FakeClient({"accepted": False, "reason": "allocation cap exceeded"})
    Strat.entries = [RemoteStrategy.intent("AAPL", 10, 100.0)]
    assert Strat(client=c).run() == RemoteStrategy.EXIT_REFUSED
    assert c.fill_calls == []
    assert "allocation cap exceeded" in c.journalled[-1]
