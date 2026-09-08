"""Positions no strategy claims — the ones every flatten path used to step over.

A position becomes an orphan when its only ledger attribution is an internal bookkeeping
bucket (`__net__`, `flatten_all`, `kill_switch`) or nothing at all: a flatten that left a
residue, a fill attributed to the net pool, a reconcile that adopted a broker position.

They were invisible AND unreachable. `/flatten` and `kill_switch` both build their symbol
set by walking `strategy_positions` and skipping the internal buckets, so an orphan survived
both — the emergency brake could not close a position nobody owned — while the equity
sampler valued it at zero because NAV is computed from strategy attribution.
"""
import pytest

import execution.central_execution as ce
from risk.risk_manager import RiskManager

CFG = {"s1": {"capital_allocation": 100_000.0, "max_drawdown": 0.10}}


class FakeDB:
    def __init__(self, *a, **kw): pass
    def log_fill(self, *a, **kw): pass
    def log_equity(self, *a, **kw): pass
    def log_decision(self, *a, **kw): pass
    def log_risk_event(self, *a, **kw): pass
    def log_reconciliation(self, *a, **kw): pass
    def save_strategy_positions(self, *a, **kw): pass
    def save_realized_pnl(self, *a, **kw): pass
    def save_multipliers(self, *a, **kw): pass
    def save_strategy_cash(self, *a, **kw): pass
    def save_halted_strategies(self, *a, **kw): pass
    def load_strategy_positions(self): return {}, {}
    def load_realized_pnl(self): return {}
    def load_multipliers(self): return {}
    def load_strategy_cash(self): return {}, {}
    def load_halted_strategies(self): return set()
    def close(self): pass


class FakeCoordinator:
    """Records what the flatten path asked it to rebalance."""
    def __init__(self):
        self.desired = {}
        self.rebalanced = None
        self.urgent = None

    def _save(self): pass

    def _rebalance(self, symbols, urgent=False):
        self.rebalanced = set(symbols)
        self.urgent = urgent
        return {"orders": [], "internal_crosses": []}


@pytest.fixture
def ex(monkeypatch):
    monkeypatch.setattr(ce, "EventLogger", FakeDB)
    x = ce.CentralExecutor()
    x.logger_db = FakeDB()
    x.risk_manager = RiskManager(x.ledger, CFG, {})
    x.placed, x.cancelled = [], []
    x.place_order = lambda intent: (x.placed.append(intent), len(x.placed))[1]
    x.cancelOrder = lambda oid, *a, **kw: x.cancelled.append(oid)
    x._alerter = None
    return x


# ------------------------------------------------------------------ identifying an orphan
def test_position_claimed_by_a_strategy_is_not_an_orphan(ex):
    ex.ledger.current_positions["AAPL"] = 100.0
    ex.ledger.strategy_positions["s1"] = {"AAPL": 100.0}
    assert ex.orphaned_positions() == {}


def test_position_only_in_an_internal_bucket_is_an_orphan(ex):
    """This is the GLD case: a flatten left the quantity attributed to `flatten_all`, which
    every flatten path skips, so it sat at the broker through repeated flattens."""
    ex.ledger.current_positions["GLD"] = 43.0
    ex.ledger.strategy_positions["flatten_all"] = {"GLD": 43.0}
    assert ex.orphaned_positions() == {"GLD": 43.0}


def test_position_attributed_to_nobody_is_an_orphan(ex):
    ex.ledger.current_positions["MCL"] = 1.0
    assert ex.orphaned_positions() == {"MCL": 1.0}


def test_flat_positions_are_not_orphans(ex):
    ex.ledger.current_positions.update({"AAPL": 0.0, "MSFT": 0.0})
    assert ex.orphaned_positions() == {}


def test_a_short_orphan_is_found_too(ex):
    ex.ledger.current_positions["VNQ"] = -1.0
    ex.ledger.strategy_positions["__net__"] = {"VNQ": -1.0}
    assert ex.orphaned_positions() == {"VNQ": -1.0}


def test_a_symbol_a_strategy_holds_is_claimed_even_if_a_bucket_also_lists_it(ex):
    """Both `flatten_all` and a real strategy name it — the real strategy owns it, and its
    own flatten will close it. Treating it as an orphan too would double the closing order."""
    ex.ledger.current_positions["ANET"] = 10.0
    ex.ledger.strategy_positions["s1"] = {"ANET": 10.0}
    ex.ledger.strategy_positions["flatten_all"] = {"ANET": 10.0}
    assert ex.orphaned_positions() == {}


# ------------------------------------------------------------------ the kill switch
def test_kill_switch_closes_orphans_via_the_coordinator(ex):
    """The regression that mattered: the emergency brake left them open."""
    ex.coordinator = FakeCoordinator()
    ex.ledger.current_positions.update({"GLD": 43.0, "MCL": 1.0})
    ex.ledger.strategy_positions["flatten_all"] = {"GLD": 43.0}

    ex.kill_switch(flatten=True)

    assert ex.coordinator.rebalanced == {"GLD", "MCL"}
    assert ex.coordinator.urgent is True, "a kill must close at market, not rest a limit"


def test_kill_switch_closes_orphans_without_a_coordinator(ex):
    ex.coordinator = None
    ex.ledger.current_positions["GLD"] = 43.0
    ex.ledger.strategy_positions["flatten_all"] = {"GLD": 43.0}

    ex.kill_switch(flatten=True)

    assert [(i["side"], i["quantity"], i["instrument"]["symbol"]) for i in ex.placed] \
        == [("sell", 43.0, "GLD")]


def test_kill_switch_still_closes_real_strategy_books(ex):
    """The orphan sweep must not displace what the kill switch already did."""
    ex.coordinator = FakeCoordinator()
    ex.ledger.current_positions.update({"AAPL": 100.0, "GLD": 43.0})
    ex.ledger.strategy_positions["s1"] = {"AAPL": 100.0}
    ex.ledger.strategy_positions["flatten_all"] = {"GLD": 43.0}

    ex.kill_switch(flatten=True)
    assert ex.coordinator.rebalanced == {"AAPL", "GLD"}


# ------------------------------------------------------------------ safety on derivatives
def test_an_unknown_derivative_is_reported_not_guessed(ex):
    """No contract spec and a non-unit multiplier means we cannot build the contract. Sending
    STK/SMART would be an order for an instrument that does not exist, so it is logged and
    left for a human instead."""
    ex.coordinator = None
    ex.ledger.current_positions["MCL"] = 1.0
    ex.ledger.multipliers["MCL"] = 100.0

    closed = ex._flatten_orphans()

    assert closed == []
    assert ex.placed == [], "guessed a contract for a futures position"


def test_a_known_derivative_uses_its_recorded_instrument(ex):
    ex.coordinator = None
    ex.ledger.current_positions["MCL"] = 1.0
    ex.ledger.multipliers["MCL"] = 100.0
    ex._instruments["MCL"] = {"symbol": "MCL", "sec_type": "FUT",
                              "exchange": "NYMEX", "multiplier": 100}

    ex._flatten_orphans()

    assert len(ex.placed) == 1
    assert ex.placed[0]["instrument"]["sec_type"] == "FUT"
    assert ex.placed[0]["side"] == "sell"


# ------------------------------------------------------------------ the API surface
from fastapi.testclient import TestClient          # noqa: E402

import api.server as server                        # noqa: E402
from api.server import app                         # noqa: E402

API_KEY = "test-key-123"
AUTH = {"X-API-Key": API_KEY}


class FakeLedger:
    def __init__(self):
        self.current_positions = {}
        self.strategy_positions = {}
        self.multipliers = {}

    def save_state(self, db): pass
    def write_off_position(self, sid, sym): return None


class ApiExecutor:
    """Only what /flatten and /positions/orphans touch."""
    INTERNAL_SIDS = ce.CentralExecutor.INTERNAL_SIDS
    orphaned_positions = ce.CentralExecutor.orphaned_positions

    def __init__(self):
        self.ledger = FakeLedger()
        self.coordinator = FakeCoordinator()
        self.order_status = {}
        self.logger_db = FakeDB()

    def cancelOrder(self, oid, *a, **kw): pass


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "executor", ApiExecutor())
    monkeypatch.setattr(server, "EXECUTOR_API_KEY", API_KEY)
    monkeypatch.setattr(server, "_alert", lambda *a, **kw: None)
    monkeypatch.setattr(server, "is_market_open", lambda: True)
    return TestClient(app)


def test_flatten_includes_orphans(client):
    ex = server.executor
    ex.ledger.current_positions.update({"AAPL": 100.0, "GLD": 43.0})
    ex.ledger.strategy_positions["s1"] = {"AAPL": 100.0}
    ex.ledger.strategy_positions["flatten_all"] = {"GLD": 43.0}

    r = client.post("/flatten", headers=AUTH)
    assert r.status_code == 200

    assert ex.coordinator.rebalanced == {"AAPL", "GLD"}
    assert ex.coordinator.urgent is True
    # and it is REPORTED — a flatten that silently skips a position is how this went
    # unnoticed in the first place
    reported = {(e["symbol"], e["strategy_id"]) for e in r.json()["flattened_positions"]}
    assert ("GLD", None) in reported


def test_flatten_reports_orphans_it_closed(client):
    ex = server.executor
    ex.ledger.current_positions["MCL"] = 1.0

    r = client.post("/flatten", headers=AUTH)
    assert [e["symbol"] for e in r.json()["flattened_positions"]] == ["MCL"]


def test_orphans_endpoint_lists_them_with_notional(client):
    ex = server.executor
    ex.ledger.current_positions.update({"GLD": 43.0, "AAPL": 10.0})
    ex.ledger.strategy_positions["s1"] = {"AAPL": 10.0}
    ex.ledger.multipliers["GLD"] = 1.0
    monkey = server._last_equity
    monkey["marks"] = {"GLD": 403.49}

    r = client.get("/positions/orphans")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    row = body["orphans"][0]
    assert row["symbol"] == "GLD" and row["quantity"] == 43.0
    assert row["notional"] == pytest.approx(43 * 403.49)


def test_orphans_endpoint_applies_the_multiplier(client):
    """A futures orphan understated by its multiplier is exactly the mis-valuation that
    hides the size of the problem."""
    ex = server.executor
    ex.ledger.current_positions["MCL"] = 1.0
    ex.ledger.multipliers["MCL"] = 100.0
    server._last_equity["marks"] = {"MCL": 92.75}

    row = client.get("/positions/orphans").json()["orphans"][0]
    assert row["notional"] == pytest.approx(9275.0)


def test_orphans_endpoint_empty_when_everything_is_claimed(client):
    ex = server.executor
    ex.ledger.current_positions["AAPL"] = 10.0
    ex.ledger.strategy_positions["s1"] = {"AAPL": 10.0}
    assert client.get("/positions/orphans").json() == {"orphans": [], "count": 0}


def test_a_position_a_strategy_still_wants_is_not_an_orphan(ex):
    """A pooled fill is attributed to `__net__`, not to the strategy that asked for it. If
    that alone made a position an orphan, every coordinator-traded holding would be
    reported as unowned while its strategy was actively running it."""
    ex.coordinator = FakeCoordinator()
    ex.coordinator.desired = {"orb_breakout": {"ANET": 10.0}}
    ex.ledger.current_positions["ANET"] = 10.0
    ex.ledger.strategy_positions["flatten_all"] = {"ANET": 10.0}

    assert ex.orphaned_positions() == {}


def test_a_position_nobody_wants_is_still_an_orphan(ex):
    ex.coordinator = FakeCoordinator()
    ex.coordinator.desired = {"orb_breakout": {"ANET": 10.0}}
    ex.ledger.current_positions.update({"ANET": 10.0, "GLD": 43.0})
    ex.ledger.strategy_positions["flatten_all"] = {"ANET": 10.0, "GLD": 43.0}

    assert ex.orphaned_positions() == {"GLD": 43.0}
