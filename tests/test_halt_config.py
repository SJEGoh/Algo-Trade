"""tests/test_halt_config.py — the 1% halt-test strategies exercise the full live drawdown
halt path through the REAL executor: a losing round-trip -> execDetails -> check_drawdown ->
halt_strategy -> is_active False -> subsequent orders rejected by check_order.

Deterministic (simulated fills, no market/notional dependence). Requires ibapi."""
import os

import pytest

pytest.importorskip("ibapi")

from ibapi.contract import Contract
from ibapi.execution import Execution

from execution.central_execution import CentralExecutor
from config import CONFIG


class _NullLog:
    def __getattr__(self, name):
        return lambda *a, **k: None


def _ex():
    os.environ.setdefault("EXECUTOR_API_KEY", "x")
    ex = CentralExecutor.__new__(CentralExecutor)
    CentralExecutor.__init__(ex)          # risk_manager built from the real CONFIG
    ex.logger_db = _NullLog()
    return ex


def _fill(ex, sid, sym, signed_qty, price, oid):
    """Route a fill through execDetails exactly as a real IB fill would (non-net order)."""
    ex.order_status[oid] = {"strategy_id": sid, "symbol": sym, "expected_price": price}
    c = Contract(); c.symbol = sym
    e = Execution()
    e.orderId = oid; e.execId = f"e{oid}"
    e.shares = abs(signed_qty); e.side = "BOT" if signed_qty > 0 else "SLD"; e.price = price
    ex.execDetails(1, c, e)


def test_all_three_halt_test_strategies_are_one_percent():
    for sid in ("halt_test_1", "halt_test_2", "halt_test_3"):
        assert CONFIG[sid]["max_drawdown"] == 0.01
        assert CONFIG[sid]["capital_allocation"] == 10_000.0


def test_one_percent_loss_halts_and_blocks_new_orders():
    ex = _ex()
    sid, sym = "halt_test_1", "ZZZ"
    assert ex.risk_manager.is_active(sid)

    _fill(ex, sid, sym, +100, 100.0, 1)          # open long 100 @ 100
    assert ex.risk_manager.is_active(sid)          # no realized loss yet -> still active

    _fill(ex, sid, sym, -100, 98.0, 2)           # close @ 98 -> realized -200 = 2% of 10k
    assert ex.ledger.strategy_realized_pnl[sid] == pytest.approx(-200.0)
    assert not ex.risk_manager.is_active(sid)      # breached 1% -> HALTED

    # a subsequent order for the halted strategy is rejected at the risk check
    intent = {"strategy_id": sid, "instrument": {"symbol": sym}}
    r = ex.risk_manager.check_order(intent, 100, 100.0)
    assert r["approved"] is False and "not active" in r["reason"]


def test_sub_one_percent_loss_does_not_halt():
    ex = _ex()
    sid, sym = "halt_test_2", "YYY"
    _fill(ex, sid, sym, +100, 100.0, 1)
    _fill(ex, sid, sym, -100, 99.95, 2)          # loss = 5.0 = 0.05% of 10k -> under 1%
    assert ex.ledger.strategy_realized_pnl[sid] == pytest.approx(-5.0)
    assert ex.risk_manager.is_active(sid)          # still active
