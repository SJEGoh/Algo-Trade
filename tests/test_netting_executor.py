"""tests/test_netting_executor.py — integration of the NettingCoordinator with the REAL
CentralExecutor (not the FakeExecutor used in test_netting.py). Exercises the code paths
that only exist on the executor: place_net_order, execDetails net-fill routing, and the
net-aware pending reversal in _cancel_open_orders_for_symbol.

Requires ibapi (Order/Contract/Execution + EClient base). Skipped where ibapi is absent
(e.g. a bare system-python driver). It runs under the project venv and in the container."""
import os
import sys

import pytest

pytest.importorskip("ibapi")  # skip cleanly when ibapi isn't installed

from ibapi.contract import Contract
from ibapi.execution import Execution

from execution.central_execution import CentralExecutor
from execution.netting import NettingCoordinator

INST = {"symbol": "MSFT", "asset_class": "equity", "exchange": "SMART"}


class _NullLog:
    def __getattr__(self, name):
        return lambda *a, **k: None


def _make_executor(cfg):
    """Real CentralExecutor with the IB-facing calls stubbed (no socket)."""
    os.environ.setdefault("EXECUTOR_API_KEY", "x")
    ex = CentralExecutor.__new__(CentralExecutor)
    CentralExecutor.__init__(ex)
    ex.risk_manager._config = cfg
    ex.risk_manager._active_strategies = set(cfg)
    ex._oid = 0

    def _next():
        ex._oid += 1
        return ex._oid

    ex.get_next_order_id = _next
    ex._cancelled = set()
    ex.placeOrder = lambda oid, c, o: None
    ex.cancelOrder = lambda oid, *a, **k: ex._cancelled.add(oid)
    ex.logger_db = _NullLog()
    co = NettingCoordinator(ex, cfg)
    ex.coordinator = co
    return ex, co


def _fill(ex, oid, price=500.0):
    st = ex.order_status[oid]
    q = st["pending_qty"]
    c = Contract(); c.symbol = st["symbol"]
    e = Execution()
    e.orderId = oid; e.execId = f"e{oid}"
    e.shares = abs(q); e.side = "BOT" if q > 0 else "SLD"; e.price = price
    ex.execDetails(1, c, e)


def _live_net_orders(ex):
    return [oid for oid, st in ex.order_status.items()
            if st.get("net") and oid not in ex._cancelled]


def test_place_net_order_records_net_pending_and_fills_to_target():
    cfg = {"s1": {"capital_allocation": 1e9, "max_drawdown": 0.2}}
    ex, co = _make_executor(cfg)
    r = co.set_target("s1", "MSFT", 100, instrument=INST, price=500)
    assert r["accepted"]
    assert ex.ledger.pending_deltas["MSFT"] == 100          # net pending set by place_net_order
    live = _live_net_orders(ex)
    assert len(live) == 1
    _fill(ex, live[0])
    assert ex.ledger.current_positions["MSFT"] == 100
    assert ex.ledger.strategy_positions["s1"]["MSFT"] == 100
    assert abs(ex.ledger.pending_deltas["MSFT"]) < 1e-9     # net pending reversed on the fill


def test_second_target_cancels_stale_net_order_without_phantom_pending():
    """The regression this guards: cancelling a stale NET order must reverse its pending via
    record_net_pending, NOT record_pending — otherwise a phantom strategy_pending['__net__']
    is left behind and net pending is double-counted."""
    cfg = {"s1": {"capital_allocation": 1e9, "max_drawdown": 0.2},
           "s2": {"capital_allocation": 1e9, "max_drawdown": 0.2}}
    ex, co = _make_executor(cfg)
    co.set_target("s1", "MSFT", 100, instrument=INST, price=500)   # order1, pending +100
    co.set_target("s2", "MSFT", -60, instrument=INST, price=500)   # cancels order1, live net = 40
    assert abs(ex.ledger.pending_deltas["MSFT"] - 40) < 1e-9
    assert ex.ledger.strategy_pending.get("__net__", {}).get("MSFT", 0) == 0  # no phantom
    live = _live_net_orders(ex)
    assert len(live) == 1
    assert abs(ex.order_status[live[0]]["pending_qty"] - 40) < 1e-9


def test_net_fill_decomposes_offsetting_legs_over_the_real_executor():
    cfg = {"s1": {"capital_allocation": 1e9, "max_drawdown": 0.2},
           "s2": {"capital_allocation": 1e9, "max_drawdown": 0.2}}
    ex, co = _make_executor(cfg)
    co.set_target("s1", "MSFT", 100, instrument=INST, price=500)
    co.set_target("s2", "MSFT", -60, instrument=INST, price=500)
    _fill(ex, _live_net_orders(ex)[0])
    assert ex.ledger.current_positions["MSFT"] == 40           # only 40 shares actually traded
    assert ex.ledger.strategy_positions["s1"]["MSFT"] == 100   # but each strategy books its side
    assert ex.ledger.strategy_positions["s2"]["MSFT"] == -60
    net = sum(ex.ledger.strategy_positions[s].get("MSFT", 0) for s in ("s1", "s2"))
    assert net == ex.ledger.current_positions["MSFT"]          # invariant
    assert abs(ex.ledger.pending_deltas["MSFT"]) < 1e-9
