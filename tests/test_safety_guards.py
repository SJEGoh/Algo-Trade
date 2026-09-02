"""tests/test_safety_guards.py — the portfolio/safety guards added in Phase 6.7:
global gross-exposure cap, portfolio circuit breaker, fill-price sanity guard, stale-mark
guard (mark_is_fresh), and the pre-trade whatIf margin check. Real executor, IB stubbed."""
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("ibapi")

from ibapi.contract import Contract
from ibapi.execution import Execution

import execution.central_execution as ce
from execution.central_execution import CentralExecutor


class _NullLog:
    def __getattr__(self, name):
        return lambda *a, **k: None


@pytest.fixture
def G():
    """Mutate the shared GLOBAL config for a test, then restore it."""
    saved = dict(ce.GLOBAL)
    try:
        yield ce.GLOBAL
    finally:
        ce.GLOBAL.clear(); ce.GLOBAL.update(saved)


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


def _fut(sym):
    return {"symbol": sym, "asset_class": "future", "sec_type": "FUT",
            "exchange": "NYMEX", "multiplier": 100.0, "last_trade_date": "20261120"}


# --- 1) global gross-exposure cap -------------------------------------------
def test_global_gross_cap_rejects(G):
    G["max_gross_exposure"] = 100_000.0
    ex = _ex()
    ex.ledger.strategy_positions["test_suite"] = {"AAPL": 1000.0}   # 1000 * 100 = 100k gross
    ex._ref_value["AAPL"] = 100.0
    intent = {"strategy_id": "test_suite", "instrument": {"symbol": "MSFT"}}
    r = ex.risk_manager.check_order(intent, 100, 100.0, multiplier=1.0, ref_values=ex._ref_value)
    assert not r["approved"] and "GLOBAL gross" in r["reason"]


def test_global_gross_cap_allows_under(G):
    G["max_gross_exposure"] = 1_000_000.0
    ex = _ex()
    ex.ledger.strategy_positions["test_suite"] = {"AAPL": 100.0}
    ex._ref_value["AAPL"] = 100.0
    r = ex.risk_manager.check_order({"strategy_id": "test_suite", "instrument": {"symbol": "MSFT"}},
                                    10, 100.0, multiplier=1.0, ref_values=ex._ref_value)
    assert r["approved"]


# --- 2) portfolio circuit breaker -------------------------------------------
def test_circuit_breaker_trips_flattens_and_kills(G):
    G["max_daily_loss"] = 500.0
    ex = _ex()
    ex._instruments["MCL"] = _fut("MCL"); ex.ledger.multipliers["MCL"] = 100.0
    ex.ledger.record_fill("MCL", +1, 68.0, "halt_test_1")     # a live long to be flattened
    ex.enforce_daily_loss(0.0)                                 # baseline = 0
    ex.enforce_daily_loss(-600.0)                             # loss 600 >= 500 -> trip
    assert ex._circuit_broken and ex._killed
    assert not ex.risk_manager.is_active("halt_test_1")
    assert any(p["symbol"] == "MCL" and p["action"] == "SELL" for p in ex._placed)


def test_circuit_breaker_not_tripped_under_limit(G):
    G["max_daily_loss"] = 500.0
    ex = _ex()
    ex.enforce_daily_loss(0.0)
    ex.enforce_daily_loss(-100.0)                             # loss 100 < 500
    assert not ex._circuit_broken and not ex._killed


# --- 3) fill-price sanity guard ---------------------------------------------
def test_fill_sanity_halts_beyond_hard_threshold(G):
    G["fill_slippage_alert_pct"] = 0.05
    G["fill_slippage_halt_pct"] = 0.20
    ex = _ex()
    ex._instruments["X"] = {"symbol": "X", "asset_class": "equity", "sec_type": "STK", "exchange": "SMART"}
    ex.ledger.strategy_positions["halt_test_1"] = {"X": 10.0}
    ex._check_fill_sanity("halt_test_1", "X", 90.0, 68.0)     # 32% off > 20% -> halt+flatten
    assert not ex.risk_manager.is_active("halt_test_1")
    assert any(p["symbol"] == "X" and p["action"] == "SELL" for p in ex._placed)


def test_fill_sanity_alert_only_does_not_halt(G):
    G["fill_slippage_alert_pct"] = 0.05
    G["fill_slippage_halt_pct"] = None
    ex = _ex()
    ex._check_fill_sanity("halt_test_1", "X", 72.0, 68.0)     # ~5.9% off > alert, but halt disabled
    assert ex.risk_manager.is_active("halt_test_1")


# --- 4) stale-mark guard ----------------------------------------------------
def test_mark_is_fresh(G):
    import time
    G["mark_staleness_sec"] = 100.0
    ex = _ex()
    ex._mark_cache["X"] = 10.0; ex._mark_ts["X"] = time.time()
    assert ex.mark_is_fresh("X")
    ex._mark_ts["X"] = time.time() - 200                      # older than 100s
    assert not ex.mark_is_fresh("X")
    assert not ex.mark_is_fresh("NEVER_MARKED")


# --- 5) pre-trade whatIf margin check ---------------------------------------
def _wire_whatif(ex, init_margin):
    def _place(oid, c, o):
        if getattr(o, "whatIf", False):
            ex.openOrder(oid, c, o, SimpleNamespace(
                initMarginChange=str(init_margin), maintMarginChange=str(init_margin),
                commission="1.0"))
        else:
            ex._placed.append({"symbol": c.symbol, "action": o.action, "qty": o.totalQuantity})
    ex.placeOrder = _place


def _fut_target_intent(sym="MCL", qty=1):
    return {
        "strategy_id": "test_suite", "client_order_id": f"wi-{sym}-{qty}",
        "timestamp": "2026-09-01T14:00:00Z", "schema_version": "1.0",
        "instrument": _fut(sym), "intent_type": "target_position",
        "target_quantity": qty, "order_type": "market", "expected_price": 68.0,
    }


def test_pretrade_margin_rejects_over_cap(G):
    G["pretrade_margin_check"] = True
    G["max_order_init_margin"] = 1000.0
    ex = _ex(); _wire_whatif(ex, 5000.0)                      # margin 5000 > cap 1000
    r = ex.process_intent(_fut_target_intent())
    assert not r["accepted"] and "margin" in r["reason"]
    assert ex._placed == []                                   # no real order placed


def test_pretrade_margin_allows_under_cap(G):
    G["pretrade_margin_check"] = True
    G["max_order_init_margin"] = 10_000.0
    ex = _ex(); _wire_whatif(ex, 5000.0)                      # margin 5000 < cap 10000
    r = ex.process_intent(_fut_target_intent())
    assert r["accepted"]
    assert any(p["symbol"] == "MCL" for p in ex._placed)     # real order went through
