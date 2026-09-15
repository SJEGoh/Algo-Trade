"""The client knows whether its orders TRADED, not just whether they were accepted.

Three things have to be true after a strategy submits, and the client used to stop at the
first two:

  * the EXECUTOR accepted the orders — the submission's accepted:true;
  * the BROKER took them — /orders/acks says `live`;
  * they FILLED — the strategy's own book reached its targets.

`live` does not mean filled: a resting order is live, and so is one a later rebalance or the
end-of-day sweep cancelled. The fill check reads the strategy's book because the executor
moves a strategy's position only when an order that strategy owns fills; a pooled order's
`filled` count covers every strategy the order was for.

Also pinned: orders mode never worked. The executor's schema requires intent_type and
order_type, the client sent neither, every order failed validation as HTTP 200
accepted:false, and the run exited 0.
"""
import pytest

from client.executor_client import ExecutorClient
from client.remote_strategy import RemoteStrategy


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr("client.executor_client.time.sleep", lambda *_: None)


def book_rows(**held):
    return {"strategy_id": "s1",
            "book": [{"symbol": "CASH", "quantity": 9_000.0, "is_cash": True}]
            + [{"symbol": sym, "quantity": q, "is_cash": False} for sym, q in held.items()]}


def acks_body(states):
    return {"acks": {str(oid): {"ack": ack, "status": status}
                     for oid, (ack, status) in states.items()}}


class Router(ExecutorClient):
    """An ExecutorClient answering from a table. A list is served one element per call (the
    last repeats); a callable receives the request kwargs."""

    def __init__(self, routes):
        super().__init__(base_url="http://x", api_key="k", strategy_id="s1",
                         alert_on_failure=False)
        self.routes = routes
        self.calls = []

    def _request(self, method, path, **kw):
        self.calls.append((method, path, kw))
        key = (method, path) if (method, path) in self.routes else path
        if key not in self.routes:
            raise AssertionError(f"unexpected {method} {path}")
        value = self.routes[key]
        if isinstance(value, list):
            return value.pop(0) if len(value) > 1 else value[0]
        return value(kw) if callable(value) else value

    def count(self, path):
        return sum(1 for _m, p, _k in self.calls if p == path)


TARGET = {"instrument": {"symbol": "AAPL", "asset_class": "equity"},
          "target_quantity": 10, "expected_price": 100.0}


# ------------------------------------------------------------------ orders mode
def test_an_order_carries_the_fields_the_executor_requires():
    c = Router({("POST", "/orders"): {"accepted": True, "order_id": 1}})
    c.submit_order(dict(TARGET))
    sent = c.calls[-1][2]["json"]
    assert sent["intent_type"] == "target_position"
    assert sent["order_type"] == "market"
    assert sent["time_in_force"] == "day"


def test_a_side_and_quantity_intent_is_typed_as_a_delta():
    c = Router({("POST", "/orders"): {"accepted": True, "order_id": 1}})
    c.submit_order({"instrument": {"symbol": "AAPL", "asset_class": "equity"},
                    "side": "buy", "quantity": 5, "expected_price": 100.0})
    assert c.calls[-1][2]["json"]["intent_type"] == "delta"


def test_the_callers_own_order_type_is_kept():
    c = Router({("POST", "/orders"): {"accepted": True, "order_id": 1}})
    c.submit_order(dict(TARGET, order_type="limit", limit_price=99.0))
    sent = c.calls[-1][2]["json"]
    assert sent["order_type"] == "limit" and sent["limit_price"] == 99.0


def test_a_target_already_met_is_a_noop_not_a_refusal():
    c = Router({("POST", "/orders"): {"accepted": False, "reason":
                "no-op: target already covered by position + working orders"}})
    r = c.submit_orders([dict(TARGET)])
    assert r["refused"] == 0 and len(r["noop"]) == 1


def test_a_real_refusal_is_still_counted():
    c = Router({("POST", "/orders"): {"accepted": False, "reason": "strategy s1 is not active"}})
    assert c.submit_orders([dict(TARGET)])["refused"] == 1


# ------------------------------------------------------------------ holdings and P&L
def test_holdings_leave_out_cash_and_flat_names():
    c = Router({"/strategies/s1/book": book_rows(AAPL=10.0, MSFT=0.0)})
    assert c.holdings() == {"AAPL": 10.0}


def test_strategy_pnl_separates_fees_from_trading():
    c = Router({"/pnl": {"realized_pnl": {"s1": -3.0}, "fees": {"s1": 2.0}}})
    assert c.strategy_pnl() == {"realized": -3.0, "fees": 2.0, "gross": -1.0}


# ------------------------------------------------------------------ waiting for fills
def test_waits_until_holdings_reach_the_target():
    c = Router({"/strategies/s1/book": [book_rows(AAPL=4.0), book_rows(AAPL=10.0)],
                "/orders/acks": acks_body({7: ("live", "Submitted")})})
    r = c.wait_for_fills({"AAPL": 10}, [7], timeout=60)
    assert r["unfilled"] == {} and r["filled"] == ["AAPL"] and r["reason"] is None
    assert c.count("/strategies/s1/book") == 2


def test_book_mode_expects_names_left_out_to_close():
    """/targets closes any name the book omits, so holding it is not the book being in place."""
    c = Router({"/strategies/s1/book": book_rows(AAPL=10.0, MSFT=5.0),
                "/orders/acks": acks_body({7: ("live", "Submitted")})})
    r = c.wait_for_fills({"AAPL": 10}, [7], authoritative=True, timeout=0)
    assert r["unfilled"] == {"MSFT": {"target": 0.0, "held": 5.0}}


def test_orders_mode_ignores_names_it_did_not_submit():
    c = Router({"/strategies/s1/book": book_rows(AAPL=10.0, MSFT=5.0)})
    r = c.wait_for_fills({"AAPL": 10}, [7], authoritative=False, timeout=0)
    assert r["unfilled"] == {}


def test_stops_early_once_every_order_has_ended():
    """A cancelled order still reads as acknowledged. Once nothing is working, waiting out
    the whole timeout cannot change the answer."""
    c = Router({"/strategies/s1/book": book_rows(AAPL=4.0),
                "/orders/acks": acks_body({7: ("live", "Cancelled")})})
    r = c.wait_for_fills({"AAPL": 10}, [7], timeout=600)
    assert r["reason"] == "every order ended without reaching the target"
    assert c.count("/strategies/s1/book") == 1


def test_a_rejected_order_counts_as_ended():
    c = Router({"/strategies/s1/book": book_rows(),
                "/orders/acks": acks_body({7: ("rejected", "Submitted")})})
    r = c.wait_for_fills({"AAPL": 10}, [7], timeout=600)
    assert r["reason"] == "every order ended without reaching the target"


def test_gives_up_at_the_timeout_while_an_order_is_still_working():
    c = Router({"/strategies/s1/book": book_rows(AAPL=4.0),
                "/orders/acks": acks_body({7: ("live", "Submitted")})})
    r = c.wait_for_fills({"AAPL": 10}, [7], timeout=0)
    assert r["reason"].startswith("not filled within")
    assert r["working_order_ids"] == [7]
    assert r["unfilled"] == {"AAPL": {"target": 10.0, "held": 4.0}}


def test_a_gap_with_no_order_behind_it_is_reported_straight_away():
    c = Router({"/strategies/s1/book": book_rows()})
    r = c.wait_for_fills({"AAPL": 10}, [], timeout=600)
    assert r["reason"] == "no order was placed to close the gap"
    assert c.count("/orders/acks") == 0


# ------------------------------------------------------------------ the rest of the API
@pytest.mark.parametrize("call, path", [
    (lambda c: c.pending(), "/pending"),
    (lambda c: c.exposure(), "/exposure"),
    (lambda c: c.orphans(), "/positions/orphans"),
    (lambda c: c.strategies(), "/strategies"),
    (lambda c: c.strategy_status(), "/strategies/s1/status"),
    (lambda c: c.net(), "/net"),
    (lambda c: c.fills(), "/fills"),
    (lambda c: c.orders(), "/orders"),
])
def test_reads_hit_their_endpoint_without_the_key(call, path):
    c = Router({path: {}})
    call(c)
    assert c.calls[-1][1] == path
    assert not c.calls[-1][2].get("auth")


def test_adding_and_removing_a_strategy_send_the_key():
    c = Router({("POST", "/strategies"): {"strategy_id": "new"},
                ("DELETE", "/strategies/new"): {"removed": True}})
    c.add_strategy("new", 10_000.0, 0.1)
    c.remove_strategy("new")
    assert all(kw.get("auth") for _m, _p, kw in c.calls)
    assert c.calls[0][2]["json"]["max_drawdown"] == 0.1


# ------------------------------------------------------------------ RemoteStrategy
class FakeClient:
    base_url = "http://executor:8000"
    order_ids = staticmethod(ExecutorClient.order_ids)

    def __init__(self, ack="live", fill="filled", refused=(), pnl_error=False):
        self._ack, self._fill = ack, fill
        self._refused, self._pnl_error = list(refused), pnl_error
        self.fill_calls, self.journalled = [], []

    def preflight(self):
        return {"connected": True, "market_open": True}

    def allocation(self):
        return {"capital_allocation": 10_000.0}

    def submit_book(self, book, strategy_id=None):
        return {"orders": [{"symbol": i["instrument"]["symbol"], "order_id": n + 1}
                           for n, i in enumerate(book) if i["target_quantity"]]}

    def submit_orders(self, book):
        names = {r["symbol"] for r in self._refused}
        submitted = [{"symbol": i["instrument"]["symbol"], "accepted": True, "order_id": n + 1}
                     for n, i in enumerate(book) if i["instrument"]["symbol"] not in names]
        return {"submitted": submitted, "rejected": self._refused, "noop": [],
                "ok": len(submitted), "refused": len(self._refused)}

    def wait_for_acks(self, order_ids, timeout=30.0, poll=2.0):
        bucket = {"live": [], "rejected": [], "pending": []}
        bucket[self._ack] = list(order_ids)
        return {**bucket, "acks": {}}

    def wait_for_fills(self, targets, order_ids=(), authoritative=True, timeout=60.0,
                       poll=2.0, strategy_id=None):
        self.fill_calls.append({"targets": dict(targets), "authoritative": authoritative})
        if self._fill == "filled":
            return {"filled": sorted(targets), "unfilled": {}, "reason": None,
                    "working_order_ids": []}
        return {"filled": [],
                "unfilled": {s: {"target": float(q), "held": 0.0} for s, q in targets.items()},
                "reason": "not filled within 60s", "working_order_ids": list(order_ids)}

    def journal(self, event_type, summary, detail="", symbols=None, strategy_id=None):
        self.journalled.append(detail)
        return {"logged": True}

    def strategy_pnl(self, strategy_id=None):
        if self._pnl_error:
            raise RuntimeError("could not read /pnl")
        return {"realized": -1.0, "fees": 1.0, "gross": 0.0}


class Book(RemoteStrategy):
    strategy_id = "s1"

    def generate_book(self, capital):
        return [self.intent("AAPL", 10, 100.0), self.intent("MSFT", 0, 400.0)]


class Orders(Book):
    mode = "orders"


def test_a_filled_book_exits_ok():
    assert Book(client=FakeClient()).run() == RemoteStrategy.EXIT_OK


def test_acknowledged_but_not_filled_exits_4():
    """A resting or cancelled order is acknowledged. Reporting success on that is how a run
    ends looking clean while holding nothing."""
    assert Book(client=FakeClient(fill="unfilled")).run() == RemoteStrategy.EXIT_NOT_FILLED


def test_the_fill_outcome_reaches_the_journal():
    c = FakeClient(fill="unfilled")
    Book(client=c).run()
    assert "fills: 0 at target" in c.journalled[-1]
    assert "AAPL: holding 0, target 10" in c.journalled[-1]


def test_book_mode_asks_for_dropped_names_to_close():
    c = FakeClient()
    Book(client=c).run()
    assert c.fill_calls[-1]["authoritative"] is True
    assert c.fill_calls[-1]["targets"] == {"AAPL": 10, "MSFT": 0}


def test_orders_mode_checks_only_what_it_submitted():
    c = FakeClient()
    Orders(client=c).run()
    assert c.fill_calls[-1]["authoritative"] is False


def test_a_refusal_in_orders_mode_is_a_refusal_not_a_success():
    """Refusals come back as HTTP 200 accepted:false, and used to be logged then ignored."""
    c = FakeClient(refused=[{"symbol": "MSFT", "accepted": False,
                             "reason": "strategy s1 is not active"}])
    assert Orders(client=c).run() == RemoteStrategy.EXIT_REFUSED
    assert "MSFT" not in c.fill_calls[-1]["targets"], "waited for a fill on a refused order"
    assert "refused: MSFT" in c.journalled[-1]


def test_fills_are_not_awaited_when_the_broker_refused():
    c = FakeClient(ack="rejected")
    assert Book(client=c).run() == RemoteStrategy.EXIT_NOT_ACKED
    assert c.fill_calls == []


def test_fill_confirmation_can_be_turned_off():
    class NoFills(Book):
        confirm_fills = False

    c = FakeClient(fill="unfilled")
    assert NoFills(client=c).run() == RemoteStrategy.EXIT_OK
    assert c.fill_calls == []


def test_a_client_that_cannot_confirm_does_not_crash():
    """`acks` stayed None on this path and was indexed anyway — a TypeError, not a run."""
    class Minimal:
        base_url = "http://x"
        def preflight(self): return {"market_open": True}
        def allocation(self): return {"capital_allocation": 10_000.0}
        def submit_book(self, book, strategy_id=None): return {"orders": []}
        def journal(self, *a, **kw): return {"logged": True}

    assert Book(client=Minimal()).run() == RemoteStrategy.EXIT_OK


def test_a_failed_pnl_read_never_changes_the_outcome():
    assert Book(client=FakeClient(pnl_error=True)).run() == RemoteStrategy.EXIT_OK
