# tests/test_risk_manager.py
import pytest
import sys
import os

from risk.risk_manager import RiskManager
from ledger.position_ledger import PositionLedger


@pytest.fixture
def config():
    return {
        "s1": {"capital_allocation": 50000, "max_drawdown": 0.15},
        "s2": {"capital_allocation": 10000, "max_drawdown": 0.10},
    }


@pytest.fixture
def ledger():
    return PositionLedger(executor=None)


@pytest.fixture
def risk(ledger, config):
    return RiskManager(ledger, config)


def _intent(strategy_id, symbol="AAPL"):
    return {"strategy_id": strategy_id, "instrument": {"symbol": symbol}}


# --- allowlist behaviour ---

def test_configured_strategy_is_active(risk):
    result = risk.check_order(_intent("s1"), resolved_delta=10, price=100.0)
    assert result["approved"] is True


def test_unconfigured_strategy_is_rejected(risk):
    # fail-closed: a strategy not in config was never added to the allowlist
    result = risk.check_order(_intent("unknown_strategy"), resolved_delta=10, price=100.0)
    assert result["approved"] is False
    assert "not active" in result["reason"]


# --- allocation limit ---

def test_order_within_allocation_approved(risk):
    # s1 has 50k allocation, 10 shares @ 100 = 1000 notional, well under
    result = risk.check_order(_intent("s1"), resolved_delta=10, price=100.0)
    assert result["approved"] is True


def test_order_exceeding_allocation_rejected(risk):
    # s2 has 10k allocation; 200 shares @ 100 = 20000 notional, over the limit
    result = risk.check_order(_intent("s2"), resolved_delta=200, price=100.0)
    assert result["approved"] is False
    assert "exceed allocation" in result["reason"]


def test_allocation_accounts_for_existing_position(risk, ledger):
    # s1 already holds 400 AAPL @ 100 = 40000 notional; a further 200 @ 100 = 20000
    # would bring total to 60000, over the 50000 allocation
    ledger.strategy_positions["s1"] = {"AAPL": 400.0}
    result = risk.check_order(_intent("s1"), resolved_delta=200, price=100.0)
    assert result["approved"] is False


def test_allocation_boundary_exactly_at_limit(risk, ledger):
    # exactly at allocation should pass (limit is "> alloc", not ">=")
    # s2: 100 shares @ 100 = 10000, exactly the 10000 allocation
    result = risk.check_order(_intent("s2"), resolved_delta=100, price=100.0)
    assert result["approved"] is True


# --- halt / reactivate ---

def test_halt_removes_from_active(risk):
    risk.halt_strategy("s1", "test halt")
    result = risk.check_order(_intent("s1"), resolved_delta=10, price=100.0)
    assert result["approved"] is False
    assert "not active" in result["reason"]


def test_reactivate_restores_active(risk):
    risk.halt_strategy("s1", "test halt")
    risk.reactivate_strategy("s1")
    result = risk.check_order(_intent("s1"), resolved_delta=10, price=100.0)
    assert result["approved"] is True


# --- drawdown check ---

def test_drawdown_breach_halts_strategy(risk, ledger):
    # s2: 10k allocation, 10% max drawdown -> breach at -1000 realized P&L
    ledger.strategy_realized_pnl["s2"] = -1500.0  # -15%, past the 10% limit
    risk.check_drawdown("s2")
    result = risk.check_order(_intent("s2"), resolved_delta=1, price=100.0)
    assert result["approved"] is False  # should now be halted


def test_drawdown_within_limit_does_not_halt(risk, ledger):
    ledger.strategy_realized_pnl["s1"] = -1000.0  # -2% on 50k, well within 15%
    risk.check_drawdown("s1")
    result = risk.check_order(_intent("s1"), resolved_delta=10, price=100.0)
    assert result["approved"] is True


def test_positive_pnl_never_triggers_drawdown(risk, ledger):
    ledger.strategy_realized_pnl["s1"] = 5000.0  # profit
    risk.check_drawdown("s1")
    result = risk.check_order(_intent("s1"), resolved_delta=10, price=100.0)
    assert result["approved"] is True


def test_drawdown_exactly_at_limit_halts(risk, ledger):
    # s2: 10% limit, exactly -1000 on 10k = -10%
    ledger.strategy_realized_pnl["s2"] = -1000.0
    risk.check_drawdown("s2")
    result = risk.check_order(_intent("s2"), resolved_delta=1, price=100.0)
    assert result["approved"] is False  # >= means exactly-at-limit breaches


def test_missing_drawdown_config_does_not_crash(risk, ledger):
    # a strategy with no max_drawdown_pct configured should be skipped, not crash
    risk._config["s3"] = {"capital_allocation": 5000}  # no max_drawdown_pct
    ledger.strategy_realized_pnl["s3"] = -9999.0
    risk.check_drawdown("s3")  # should silently return, not raise
    # s3 was never in the allowlist anyway, so this mainly checks no exception
