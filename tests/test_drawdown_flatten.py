"""tests/test_drawdown_flatten.py — max_drawdown now (a) is dollar/multiplier-aware,
(b) triggers on TOTAL P&L incl. unrealized mark-to-market, and (c) HALTS + FLATTENS the
strategy (direct -> closing orders; pooled -> coordinator.halt zeroes the desired book).
Real executor + risk + ledger, IB calls stubbed. Requires ibapi."""
import os

import pytest

pytest.importorskip("ibapi")

from ibapi.contract import Contract
from ibapi.execution import Execution

from execution.central_execution import CentralExecutor
from execution.netting import NettingCoordinator


class _NullLog:
    def __getattr__(self, name):
        return lambda *a, **k: None


def _ex():
    os.environ.setdefault("EXECUTOR_API_KEY", "x")
    ex = CentralExecutor.__new__(CentralExecutor)
    CentralExecutor.__init__(ex)
    ex.logger_db = _NullLog()
    ex._oid = 0
    ex.get_next_order_id = lambda: (setattr(ex, "_oid", ex._oid + 1) or ex._oid)
    ex._placed = []
    ex.placeOrder = lambda oid, c, o: ex._placed.append(
        {"symbol": c.symbol, "action": o.action, "qty": o.totalQuantity})
    ex.cancelOrder = lambda *a, **k: None
    return ex


def _fut(sym, mult=100.0):
    return {"symbol": sym, "asset_class": "future", "sec_type": "FUT",
            "exchange": "NYMEX", "multiplier": mult, "last_trade_date": "20261120"}


def _fill(ex, sid, sym, signed_qty, price, oid, mult=None):
    ex.order_status[oid] = {"strategy_id": sid, "symbol": sym, "expected_price": price}
    c = Contract(); c.symbol = sym
    if mult is not None:
        c.multiplier = str(int(mult))
    e = Execution()
    e.orderId = oid; e.execId = f"e{oid}"
    e.shares = abs(signed_qty); e.side = "BOT" if signed_qty > 0 else "SLD"; e.price = price
    ex.execDetails(1, c, e)


# --- (a) multiplier-aware P&L -------------------------------------------------
def test_realized_pnl_is_multiplier_aware():
    ex = _ex(); sid, sym = "halt_test_1", "MCL"
    ex.ledger.multipliers[sym] = 100.0
    ex.ledger.record_fill(sym, +1, 68.0, sid)         # long 1 @ 68
    ex.ledger.record_fill(sym, -1, 67.0, sid)         # close @ 67: -1 pt * 100 = -$100
    assert ex.ledger.strategy_realized_pnl[sid] == pytest.approx(-100.0)


def test_unrealized_is_multiplier_aware():
    ex = _ex(); sid, sym = "halt_test_1", "MCL"
    ex.ledger.multipliers[sym] = 100.0
    ex.ledger.record_fill(sym, +1, 68.0, sid)
    snap = ex.ledger.equity_snapshot({sym: 66.0})     # (66-68)*1*100 = -$200
    assert snap[sid]["unrealized"] == pytest.approx(-200.0)


# --- (b)+(c) total-equity drawdown halts AND flattens ------------------------
def test_unrealized_drawdown_halts_and_flattens_direct():
    ex = _ex(); sid, sym = "halt_test_1", "MCL"       # alloc 10k, max_dd 1% -> $100
    ex.ledger.multipliers[sym] = 100.0
    ex._instruments[sym] = _fut(sym)
    ex.ledger.record_fill(sym, +1, 68.0, sid)         # hold 1 long, no realized loss
    assert ex.risk_manager.is_active(sid)

    equity = ex.ledger.equity_snapshot({sym: 66.0})[sid]["equity"]  # -$200 = 2%
    ex.enforce_drawdown(sid, equity, "total")
    assert not ex.risk_manager.is_active(sid)          # HALTED on unrealized
    assert any(p["symbol"] == sym and p["action"] == "SELL" and p["qty"] == 1
               for p in ex._placed)                    # FLATTENED (sold the long)


def test_realized_drawdown_via_execdetails_flattens_remaining():
    ex = _ex(); sid, sym = "halt_test_2", "MCL"
    ex.ledger.multipliers[sym] = 100.0
    ex._instruments[sym] = _fut(sym)
    _fill(ex, sid, sym, +2, 68.0, 1, mult=100.0)      # long 2
    assert ex.risk_manager.is_active(sid)
    _fill(ex, sid, sym, -1, 66.0, 2, mult=100.0)      # close 1 @ 66: -$200 realized (>1%)
    assert not ex.risk_manager.is_active(sid)          # fast-path halt on the fill
    assert any(p["action"] == "SELL" and p["qty"] == 1 for p in ex._placed)  # flatten remaining long


def test_sub_threshold_total_pnl_does_not_halt():
    ex = _ex(); sid, sym = "halt_test_3", "MCL"
    ex.ledger.multipliers[sym] = 100.0
    ex.ledger.record_fill(sym, +1, 68.0, sid)
    equity = ex.ledger.equity_snapshot({sym: 67.9})[sid]["equity"]  # (67.9-68)*100 = -$10 = 0.1%
    ex.enforce_drawdown(sid, equity, "total")
    assert ex.risk_manager.is_active(sid)              # under 1% -> still active
    assert ex._placed == []                            # nothing flattened


# --- (c) pooled flatten routes through the coordinator -----------------------
def test_pooled_drawdown_flattens_via_coordinator():
    ex = _ex()
    co = NettingCoordinator(ex, ex.risk_manager._config)
    ex.coordinator = co
    sid = "halt_test_1"
    co.set_target(sid, "AAPL", 50, instrument={"symbol": "AAPL", "asset_class": "equity",
                                               "exchange": "SMART", "sec_type": "STK"}, price=50)
    assert sid in co.desired and co.desired[sid].get("AAPL") == 50
    ex.enforce_drawdown(sid, -500.0, "total")          # -500/10000 = 5% > 1%
    assert not ex.risk_manager.is_active(sid)
    assert co.desired.get(sid) == {}                   # coordinator zeroed the book (won't re-open)


# --- predicate ---------------------------------------------------------------
def test_drawdown_status_predicate():
    ex = _ex()
    assert ex.risk_manager.drawdown_status("halt_test_1", -50.0)["breached"] is False   # 0.5%
    assert ex.risk_manager.drawdown_status("halt_test_1", -150.0)["breached"] is True    # 1.5%
    assert ex.risk_manager.drawdown_status("halt_test_1", 500.0)["breached"] is False    # profit
