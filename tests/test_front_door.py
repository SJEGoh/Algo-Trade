"""tests/test_front_door.py — the single front door: process_intent (i.e. POST /orders)
routes target_position EQUITY intents into the NettingCoordinator, while delta intents and
FUTURES intents take the direct path. metadata.pool overrides in both directions.

Uses the real CentralExecutor with IB calls stubbed. Requires ibapi."""
import os

import pytest

pytest.importorskip("ibapi")

from execution.central_execution import CentralExecutor
from execution.netting import NettingCoordinator


class _NullLog:
    def __getattr__(self, name):
        return lambda *a, **k: None


def _stk(symbol):
    return {"symbol": symbol, "asset_class": "equity", "exchange": "SMART", "sec_type": "STK"}


def _fut(symbol):
    return {"symbol": symbol, "asset_class": "future", "sec_type": "FUT",
            "exchange": "NYMEX", "multiplier": 1000.0, "last_trade_date": "20261120"}


def _intent(**kw):
    base = {
        "strategy_id": "s1",
        "client_order_id": f"cid-{os.urandom(4).hex()}",
        "timestamp": "2026-09-01T14:00:00Z",
        "schema_version": "1.0",
        "order_type": "market",
        "expected_price": 500.0,
        # keep the market-hours gate happy for equities without needing a live clock
        "metadata": {"allow_when_closed": True},
    }
    base.update(kw)
    return base


def _make():
    os.environ.setdefault("EXECUTOR_API_KEY", "x")
    ex = CentralExecutor.__new__(CentralExecutor)
    CentralExecutor.__init__(ex)
    cfg = {"s1": {"capital_allocation": 1e9, "max_drawdown": 0.5}}
    ex.risk_manager._config = cfg
    ex.risk_manager._active_strategies = set(cfg)
    ex.logger_db = _NullLog()
    ex._oid = 0
    ex.get_next_order_id = lambda: (ex.__setattr__("_oid", ex._oid + 1) or ex._oid)
    ex.placeOrder = lambda *a, **k: None
    ex.cancelOrder = lambda *a, **k: None

    co = NettingCoordinator(ex, cfg)
    ex.coordinator = co

    # spy on the DIRECT path
    ex._direct = []
    _orig = ex.place_order
    def _spy(resolved):
        ex._direct.append(resolved)
        return _orig(resolved)
    ex.place_order = _spy
    return ex, co


def test_equity_target_position_is_pooled():
    ex, co = _make()
    r = ex.process_intent(_intent(instrument=_stk("MSFT"),
                                  intent_type="target_position", target_quantity=100))
    assert r["accepted"] and r.get("pooled") is True
    assert co.desired["s1"]["MSFT"] == 100          # landed in the pool
    assert ex._direct == []                          # did NOT take the direct path


def test_delta_intent_goes_direct():
    ex, co = _make()
    r = ex.process_intent(_intent(instrument=_stk("MSFT"),
                                  intent_type="delta", side="buy", quantity=10))
    assert r["accepted"]
    assert len(ex._direct) == 1                       # direct path taken
    assert "MSFT" not in co.desired.get("s1", {})     # coordinator untouched


def test_futures_target_position_goes_direct():
    ex, co = _make()
    r = ex.process_intent(_intent(strategy_id="s1", instrument=_fut("CL"),
                                  intent_type="target_position", target_quantity=1,
                                  expected_price=68.5))
    assert r["accepted"]
    assert len(ex._direct) == 1                       # futures stay direct (disjoint)
    assert "CL" not in co.desired.get("s1", {})


def test_metadata_pool_false_forces_equity_direct():
    ex, co = _make()
    r = ex.process_intent(_intent(instrument=_stk("MSFT"),
                                  intent_type="target_position", target_quantity=100,
                                  metadata={"allow_when_closed": True, "pool": False}))
    assert r["accepted"]
    assert len(ex._direct) == 1                       # override sent it direct
    assert "MSFT" not in co.desired.get("s1", {})


def test_metadata_pool_true_forces_futures_pooled():
    ex, co = _make()
    r = ex.process_intent(_intent(instrument=_fut("CL"),
                                  intent_type="target_position", target_quantity=1,
                                  expected_price=68.5,
                                  metadata={"pool": True}))
    assert r["accepted"] and r.get("pooled") is True
    assert co.desired["s1"]["CL"] == 1               # override pooled the future
    assert ex._direct == []
