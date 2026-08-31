"""
tests/test_risk_safety.py

Regression tests for the two safety fixes made after the double-submit incident:

  1. Allocation check counts PENDING exposure (not just filled positions), so a
     stack of individually-legal orders that collectively breach is rejected.
  2. A drawdown breach actually HALTS the strategy (check_drawdown -> halt_strategy),
     and a halt blocks subsequent orders.

Executor-free: PositionLedger(None) + RiskManager over a controlled config.
Run:  PYTHONPATH=src pytest tests/test_risk_safety.py -v
"""

import pytest

from ledger.position_ledger import PositionLedger
from risk.risk_manager import RiskManager

STRAT = "strat"


def make(alloc=100_000.0, max_dd=0.15):
    ledger = PositionLedger(None)
    config = {STRAT: {"capital_allocation": alloc, "max_drawdown": max_dd}}
    return ledger, RiskManager(ledger, config)


def intent(symbol, strat=STRAT):
    # check_order only reads strategy_id and instrument.symbol from the intent
    return {"strategy_id": strat,
            "instrument": {"symbol": symbol, "asset_class": "equity", "exchange": "SMART"}}


# ---------------------------------------------------------------------------
# Allocation check — basic bounds
# ---------------------------------------------------------------------------
def test_order_within_allocation_approved():
    _, rm = make(alloc=100_000)
    assert rm.check_order(intent("AAA"), resolved_delta=500, price=100.0)["approved"] is True  # 50k


def test_order_over_allocation_rejected():
    _, rm = make(alloc=100_000)
    r = rm.check_order(intent("AAA"), resolved_delta=1500, price=100.0)  # 150k
    assert r["approved"] is False
    assert "allocation" in r["reason"]


def test_missing_price_skips_notional_check():
    _, rm = make(alloc=1_000)  # tiny cap
    # no reference price -> notional check skipped (schema normally guarantees one)
    assert rm.check_order(intent("AAA"), resolved_delta=99_999, price=None)["approved"] is True


# ---------------------------------------------------------------------------
# Allocation check — THE INCIDENT: pending must consume the budget
# ---------------------------------------------------------------------------
def test_stacked_pending_breaches_even_though_each_order_passes():
    """4 orders of 30k each against a 100k cap. Mirror process_intent's
    check-then-record discipline: record pending only for orders that pass.
    The 4th (which pushes projected gross to 120k) must be rejected."""
    ledger, rm = make(alloc=100_000)
    price, qty = 500.0, 60            # 60 * 500 = 30k per order

    approved = []
    for i in range(4):
        sym = f"S{i}"
        r = rm.check_order(intent(sym), resolved_delta=-qty, price=price)  # short
        approved.append(r["approved"])
        if r["approved"]:
            ledger.record_pending(sym, -qty, STRAT)   # only if it would have been placed

    assert approved == [True, True, True, False]


def test_cancel_releases_budget():
    ledger, rm = make(alloc=100_000)
    price = 500.0
    for i in range(4):
        ledger.record_pending(f"S{i}", -50, STRAT)    # 25k each -> 100k pending

    assert rm.check_order(intent("S9"), -50, price)["approved"] is False   # 125k -> reject
    ledger.record_pending("S0", +50, STRAT)           # cancel reversal frees 25k
    assert rm.check_order(intent("S9"), -50, price)["approved"] is True    # 100k -> ok


def test_fill_does_not_double_count():
    """pending -> filled transition must not double-count exposure."""
    ledger, rm = make(alloc=100_000)
    ledger.record_pending("AAA", -60, STRAT)
    assert ledger.strategy_effective_positions(STRAT)["AAA"] == -60
    ledger.record_fill("AAA", -60, 500.0, STRAT)      # converts pending -> filled
    assert ledger.strategy_effective_positions(STRAT)["AAA"] == -60   # not -120


def test_reducing_order_passes_when_at_cap():
    """Projected post-trade gross means a position-reducing order LOWERS gross
    and should pass even at the cap (the old additive check falsely rejected it)."""
    ledger, rm = make(alloc=100_000)
    price = 500.0
    ledger.record_pending("AAA", +200, STRAT)
    ledger.record_fill("AAA", +200, price, STRAT)     # 200 * 500 = 100k, at cap

    assert rm.check_order(intent("AAA"), resolved_delta=-50, price=price)["approved"] is True   # 75k
    assert rm.check_order(intent("AAA"), resolved_delta=+50, price=price)["approved"] is False  # 125k


# ---------------------------------------------------------------------------
# Drawdown halt — breach must halt, halt must block
# ---------------------------------------------------------------------------
def test_drawdown_breach_halts_strategy():
    ledger, rm = make(alloc=100_000, max_dd=0.10)
