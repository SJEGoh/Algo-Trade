"""tests/test_position_repair.py — fixing positions that ended up booked to nobody.

POST /positions/{symbol}/close_unowned trades a position only an internal bucket holds back
to zero; POST /positions/transfer re-books shares between buckets without trading. The
properties worth pinning down:

  * close_unowned books its fill to the bucket that held the position, so that bucket ends
    at zero — not flat at the broker with two internal buckets holding opposite lots;
  * it refuses whenever the close could move shares a real strategy owns, while an order in
    the symbol is still working, or when the records for the symbol don't add up;
  * a transfer is zero-sum: the account position, total cash and the summed strategy
    positions are unchanged, and it can only move shares the source actually holds.
"""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api.server as server
from api.server import app
from ledger.position_ledger import PositionLedger

API_KEY = "test-key-123"
AUTH = {"X-API-Key": API_KEY}
STRAT = "pair_break_fade"


class FakeExecutor:
    INTERNAL_SIDS = frozenset({"__net__", "flatten_all", "kill_switch"})

    def __init__(self):
        self.ledger = PositionLedger(None, {STRAT: {"capital_allocation": 100_000.0,
                                                    "max_drawdown": 0.1}})
        self.order_status = {}
        self._instruments = {}
        self._ref_value = {}
        self._enforce_market_hours = False
        self.logger_db = MagicMock()
        self.placed = []
        self.connected = True

    def isConnected(self):
        return self.connected

    def place_order(self, intent):
        self.placed.append(intent)
        return 900 + len(self.placed)


def _fill(ex, sym, qty, price, sid):
    ex.ledger.record_fill(sym, qty, price, sid)


@pytest.fixture
def ex(monkeypatch):
    fake = FakeExecutor()
    monkeypatch.setattr(server, "executor", fake)
    monkeypatch.setattr(server, "EXECUTOR_API_KEY", API_KEY)
    monkeypatch.setattr(server, "_alert", lambda *a, **kw: None)
    monkeypatch.setitem(server.CONFIG, STRAT, {"capital_allocation": 100_000.0,
                                               "max_drawdown": 0.1})
    return fake


@pytest.fixture
def client(ex):
    return TestClient(app)


# ------------------------------------------------------------------ close_unowned

def test_close_unowned_books_the_fill_to_the_bucket_that_held_it(client, ex):
    _fill(ex, "SO", -18, 82.98, "__net__")                # short 18 nobody owns
    r = client.post("/positions/SO/close_unowned", headers=AUTH)
    assert r.status_code == 200, r.text
    intent = ex.placed[-1]
    assert intent["strategy_id"] == "__net__"
    assert (intent["side"], intent["quantity"], intent["order_type"]) == ("buy", 18, "market")

    _fill(ex, "SO", 18, 83.05, intent["strategy_id"])     # the fill, as execDetails books it
    assert ex.ledger.strategy_positions["__net__"]["SO"] == 0
    assert ex.ledger.current_positions["SO"] == 0


def test_close_unowned_refuses_when_a_strategy_also_holds_the_symbol(client, ex):
    _fill(ex, "CVX", 7, 206.28, "__net__")
    _fill(ex, "CVX", -14, 206.33, STRAT)
    r = client.post("/positions/CVX/close_unowned", headers=AUTH)
    assert r.status_code == 409 and "transfer" in r.json()["detail"]
    assert ex.placed == []


def test_close_unowned_refuses_while_an_order_is_working(client, ex):
    _fill(ex, "SO", -18, 82.98, "__net__")
    ex.order_status[51] = {"symbol": "SO", "status": "PendingSubmit"}
    r = client.post("/positions/SO/close_unowned", headers=AUTH)
    assert r.status_code == 409 and "51" in r.json()["detail"]
    assert ex.placed == []


def test_close_unowned_refuses_when_the_records_disagree_with_the_account(client, ex):
    _fill(ex, "SO", -18, 82.98, "__net__")
    ex.ledger.current_positions["SO"] = -10.0            # broker says otherwise
    r = client.post("/positions/SO/close_unowned", headers=AUTH)
    assert r.status_code == 409 and "reconcile" in r.json()["detail"]


def test_close_unowned_refuses_when_disconnected(client, ex):
    _fill(ex, "SO", -18, 82.98, "__net__")
    ex.connected = False
    assert client.post("/positions/SO/close_unowned", headers=AUTH).status_code == 503


def test_close_unowned_needs_the_api_key(client, ex):
    _fill(ex, "SO", -18, 82.98, "__net__")
    assert client.post("/positions/SO/close_unowned").status_code == 401
    assert ex.placed == []


# ------------------------------------------------------------------ transfer

def _totals(ex, sym):
    led = ex.ledger
    return (led.current_positions.get(sym, 0.0),
            sum(p.get(sym, 0.0) for p in led.strategy_positions.values()),
            sum(led.strategy_cash.values()))


def test_transfer_fixes_the_cvx_split_without_trading(client, ex):
    """Account short 7; records said strategy -14 and __net__ +7."""
    _fill(ex, "CVX", 7, 206.28, "__net__")
    _fill(ex, "CVX", -14, 206.33, STRAT)
    before = _totals(ex, "CVX")

    r = client.post("/positions/transfer", headers=AUTH, json={
        "symbol": "CVX", "from_strategy": "__net__", "to_strategy": STRAT, "quantity": 7})
    assert r.status_code == 200, r.text

    assert ex.ledger.strategy_positions["__net__"]["CVX"] == 0
    assert ex.ledger.strategy_positions[STRAT]["CVX"] == -7
    assert ex.ledger.strategy_avg_cost[STRAT]["CVX"] == pytest.approx(206.33)
    assert _totals(ex, "CVX") == pytest.approx(before)    # account, positions, cash unchanged
    assert ex.placed == []                                 # nothing traded
    ex.logger_db.save_strategy_positions.assert_called()   # persisted


@pytest.mark.parametrize("qty", [-7, 8, 0])
def test_transfer_only_moves_what_the_source_holds(client, ex, qty):
    _fill(ex, "CVX", 7, 206.28, "__net__")
    r = client.post("/positions/transfer", headers=AUTH, json={
        "symbol": "CVX", "from_strategy": "__net__", "to_strategy": STRAT, "quantity": qty})
    assert r.status_code == 409
    assert ex.ledger.strategy_positions["__net__"]["CVX"] == 7


def test_transfer_refuses_unknown_strategies(client, ex):
    _fill(ex, "CVX", 7, 206.28, "__net__")
    r = client.post("/positions/transfer", headers=AUTH, json={
        "symbol": "CVX", "from_strategy": "__net__", "to_strategy": "typo", "quantity": 7})
    assert r.status_code == 404
