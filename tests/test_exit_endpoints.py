"""Exits through the API: they arm only when the submission is taken, an unenforceable one
refuses the whole submission, and a locked-out name is held flat whatever the book says."""
import pytest
from fastapi.testclient import TestClient

import api.server as server
from api.server import app
from risk.exit_rules import ExitManager

API_KEY = "test-key"
AUTH = {"X-API-Key": API_KEY}


class Ledger:
    def __init__(self):
        self.strategy_positions = {}
        self.strategy_avg_cost = {}


class Risk:
    def is_active(self, sid):
        return True


class DB:
    def __init__(self):
        self.decisions = []

    def log_decision(self, sid, kind, summary, detail="", symbols=None):
        self.decisions.append((sid, kind, summary))


class Coordinator:
    def __init__(self):
        self.desired = {}
        self.books = []
        self.targets = []

    def submit_book(self, sid, intents):
        self.books.append((sid, [dict(i) for i in intents]))
        self.desired[sid] = {i["instrument"]["symbol"]: float(i["target_quantity"])
                             for i in intents if i["target_quantity"]}
        return {"accepted": True, "orders": [], "internal_crosses": []}

    def set_target(self, sid, symbol, qty, instrument=None, price=None, urgent=False):
        self.targets.append((sid, symbol, qty))
        return {"accepted": True, "orders": []}


class FakeExecutor:
    def __init__(self, with_exits=True):
        self.ledger = Ledger()
        self.risk_manager = Risk()
        self.logger_db = DB()
        self.coordinator = Coordinator()
        self._killed = False
        self._enforce_market_hours = False
        self.intents = []
        self.priced = []
        if with_exits:
            self.exit_manager = ExitManager(self)

    def process_intent(self, intent):
        self.intents.append(dict(intent))
        return {"accepted": True, "order_id": 1}

    def get_marks(self, symbols):
        self.priced.append(set(symbols))
        return {s: 90.0 for s in symbols}

    def mark_is_fresh(self, symbol, max_age=None):
        return True

    def _has_live_order(self, sid, symbol):
        return False

    def _flatten_direct(self, sid, symbols=None):
        pass


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "executor", FakeExecutor())
    monkeypatch.setattr(server, "EXECUTOR_API_KEY", API_KEY)
    monkeypatch.setattr(server, "_alert", lambda *a, **kw: None)
    return TestClient(app)


def book(*entries):
    return {"strategy_id": "s1", "intents": list(entries)}


def entry(symbol, qty, price=100.0, **exits):
    e = {"instrument": {"symbol": symbol, "asset_class": "equity", "sec_type": "STK"},
         "target_quantity": qty, "expected_price": price}
    if exits:
        e["exits"] = exits
    return e


def lock_out(symbol, direction=1):
    em = server.executor.exit_manager
    em.lockouts.setdefault("s1", {})[symbol] = {
        "direction": direction, "kind": "stop", "level": 95.0, "mark": 94.0, "quantity": 10,
        "route": "pooled", "session": em.session(), "at": "now", "spec": {}}


# ------------------------------------------------------------------ /targets
def test_a_book_arms_only_the_names_that_carry_exits(client):
    r = client.post("/targets", headers=AUTH, json=book(
        entry("AAPL", 10, stop_pct=0.05), entry("MSFT", 5), entry("NVDA", 3, trail_pct=0.1)))
    assert r.json()["exits"] == {"armed": ["AAPL", "NVDA"], "blocked": {}}
    assert set(server.executor.exit_manager.rules["s1"]) == {"AAPL", "NVDA"}


def test_one_unenforceable_exit_refuses_the_whole_book(client):
    r = client.post("/targets", headers=AUTH, json=book(
        entry("AAPL", 10, stop_pct=0.05), entry("MSFT", 5, stop_price=120.0)))
    assert r.json()["accepted"] is False and "MSFT" in r.json()["reason"]
    assert server.executor.coordinator.books == [], "nothing may be submitted"
    assert server.executor.exit_manager.rules == {}


def test_leaving_a_names_exits_out_of_the_next_book_drops_them(client):
    client.post("/targets", headers=AUTH, json=book(entry("AAPL", 10, stop_pct=0.05)))
    client.post("/targets", headers=AUTH, json=book(entry("AAPL", 10)))
    assert server.executor.exit_manager.rules == {}


def test_a_locked_out_name_is_submitted_flat(client):
    lock_out("AAPL")
    r = client.post("/targets", headers=AUTH, json=book(
        entry("AAPL", 10, stop_pct=0.05), entry("MSFT", 5)))
    sent = {i["instrument"]["symbol"]: i["target_quantity"]
            for i in server.executor.coordinator.books[-1][1]}
    assert sent == {"AAPL": 0, "MSFT": 5}
    assert list(r.json()["exits"]["blocked"]) == ["AAPL"]
    assert "exit lockout" in server.executor.logger_db.decisions[-1][2]


def test_a_short_is_not_blocked_by_a_long_lockout(client):
    lock_out("AAPL", direction=1)
    client.post("/targets", headers=AUTH, json=book(entry("AAPL", -10)))
    assert server.executor.coordinator.books[-1][1][0]["target_quantity"] == -10


def test_exits_without_an_exit_manager_are_refused_not_ignored(client, monkeypatch):
    monkeypatch.setattr(server, "executor", FakeExecutor(with_exits=False))
    r = client.post("/targets", headers=AUTH, json=book(entry("AAPL", 10, stop_pct=0.05)))
    assert r.json()["accepted"] is False and "not be enforced" in r.json()["reason"]


# ------------------------------------------------------------------ /target and /orders
def test_single_target_arms_its_exit(client):
    r = client.post("/target", headers=AUTH, json={
        "strategy_id": "s1", "symbol": "AAPL", "quantity": 10, "price": 100.0,
        "exits": {"take_profit_pct": 0.2}})
    assert r.json()["accepted"] is True
    assert server.executor.exit_manager.rules["s1"]["AAPL"]["spec"] == {"take_profit_pct": 0.2}


def order(**kw):
    return {"strategy_id": "s1", "client_order_id": "c1", "timestamp": "t",
            "schema_version": "1.0", "order_type": "market", "expected_price": 100.0,
            "instrument": {"symbol": "AAPL", "asset_class": "equity"}, **kw}


def test_an_order_arms_its_exit_and_the_executor_never_sees_the_field(client):
    client.post("/orders", headers=AUTH, json=order(
        intent_type="target_position", target_quantity=10, exits={"stop_pct": 0.03}))
    assert "exits" not in server.executor.intents[-1]
    assert "AAPL" in server.executor.exit_manager.rules["s1"]


def test_exits_on_a_delta_order_are_refused(client):
    r = client.post("/orders", headers=AUTH, json=order(
        intent_type="delta", side="buy", quantity=10, exits={"stop_pct": 0.03}))
    assert r.json()["accepted"] is False and "absolute target" in r.json()["reason"]
    assert server.executor.intents == []


def test_a_delta_buy_cannot_sneak_back_into_a_locked_out_long(client):
    lock_out("AAPL")
    r = client.post("/orders", headers=AUTH, json=order(
        intent_type="delta", side="buy", quantity=10))
    assert r.json()["accepted"] is False and "lockout" in r.json()["reason"]
    assert server.executor.intents == []


def test_a_locked_out_target_order_is_sent_flat(client):
    lock_out("AAPL")
    r = client.post("/orders", headers=AUTH, json=order(
        intent_type="target_position", target_quantity=10))
    assert server.executor.intents[-1]["target_quantity"] == 0
    assert "AAPL" in r.json()["exits_blocked"]


# ------------------------------------------------------------------ /exits
def test_exits_can_be_inspected_and_cleared(client):
    client.post("/targets", headers=AUTH, json=book(entry("AAPL", 10, stop_price=95.0)))
    body = client.get("/exits", params={"strategy_id": "s1"}).json()
    assert body["enabled"] and body["rules"]["s1"]["AAPL"]["levels"] == {"stop": 95.0}
    assert client.delete("/exits/s1/AAPL").status_code == 401
    assert client.delete("/exits/s1/AAPL", headers=AUTH).json()["removed_rule"] is True
    assert client.delete("/exits/s1/AAPL", headers=AUTH).status_code == 404


# ------------------------------------------------------------------ the exit loop
def test_the_exit_loop_prices_only_armed_held_names_and_closes_urgently(client, monkeypatch):
    monkeypatch.setattr(server, "is_market_open", lambda: True)
    ex = server.executor
    ex.ledger.strategy_positions["s1"] = {"AAPL": 10.0}
    ex.ledger.strategy_avg_cost["s1"] = {"AAPL": 100.0}
    ex.coordinator.desired["s1"] = {"AAPL": 10.0}
    ex.exit_manager.set("s1", "AAPL", 10, {"stop_price": 95.0})
    ex.exit_manager.set("s1", "MSFT", 10, {"stop_price": 95.0})     # not held
    fired = server._check_exits()
    assert ex.priced == [{"AAPL"}]
    assert [f["symbol"] for f in fired] == ["AAPL"]
    assert ex.coordinator.targets == [("s1", "AAPL", 0)]


def test_the_exit_loop_runs_even_with_nothing_to_price(client, monkeypatch):
    monkeypatch.setattr(server, "is_market_open", lambda: True)
    assert server._check_exits() == []
    assert server.executor.priced == []


def test_exit_cadence_is_configured_at_thirty_seconds():
    from config import GLOBAL
    assert GLOBAL["exit_check_sec"] == 30.0
    assert GLOBAL["exit_mark_max_age_sec"] < GLOBAL["mark_staleness_sec"]
