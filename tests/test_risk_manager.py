"""RiskManager — the per-order gate: allowlist, allocation cap, drawdown halt.

Executor-free: PositionLedger(None) + RiskManager over a controlled config. Includes the
regressions from the double-submit incident: the allocation check counts PENDING exposure,
so a stack of individually legal orders that collectively breach is rejected; and a
drawdown breach actually halts the strategy.
"""
import pytest

from ledger.position_ledger import PositionLedger
from risk.risk_manager import RiskManager


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


def make(alloc=100_000.0, max_dd=0.15):
    ledger = PositionLedger(None)
    return ledger, RiskManager(ledger, {"s": {"capital_allocation": alloc, "max_drawdown": max_dd}})


# ---------------------------------------------------------------------------
# allowlist
# ---------------------------------------------------------------------------
def test_unconfigured_strategy_is_rejected(risk):
    # fail-closed: a strategy not in config was never added to the allowlist
    result = risk.check_order(_intent("unknown_strategy"), resolved_delta=10, price=100.0)
    assert result["approved"] is False
    assert "not active" in result["reason"]


# ---------------------------------------------------------------------------
# allocation cap
# ---------------------------------------------------------------------------
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


@pytest.mark.parametrize("bad_price", [None, 0, -5.0, float("nan"), float("inf"), "abc"])
def test_unpriceable_order_is_rejected_not_waved_through(bad_price):
    """An order that can't be valued must FAIL CLOSED.

    This used to approve unconditionally ("notional check skipped"), which waives the
    allocation cap entirely — the way an oversized order gets past a small allocation.
    NaN matters too: every comparison against a limit is False, so a NaN gross passes."""
    _, rm = make(alloc=1_000)  # tiny cap
    r = rm.check_order(_intent("s", "AAA"), resolved_delta=99_999, price=bad_price)
    assert r["approved"] is False
    assert "reference price" in r["reason"] or "not a number" in r["reason"]


def test_nan_multiplier_cannot_hide_a_position():
    """A NaN unit value makes projected gross NaN, and NaN > alloc is False."""
    _, rm = make(alloc=1_000)
    r = rm.check_order(_intent("s", "AAA"), resolved_delta=99_999, price=100.0,
                       multiplier=float("nan"))
    assert r["approved"] is False
    assert "not a number" in r["reason"]


# ---------------------------------------------------------------------------
# THE INCIDENT: pending must consume the budget
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
        r = rm.check_order(_intent("s", sym), resolved_delta=-qty, price=price)  # short
        approved.append(r["approved"])
        if r["approved"]:
            ledger.record_pending(sym, -qty, "s")     # only if it would have been placed

    assert approved == [True, True, True, False]


def test_cancel_releases_budget():
    ledger, rm = make(alloc=100_000)
    price = 500.0
    for i in range(4):
        ledger.record_pending(f"S{i}", -50, "s")      # 25k each -> 100k pending

    assert rm.check_order(_intent("s", "S9"), -50, price)["approved"] is False   # 125k
    ledger.record_pending("S0", +50, "s")             # cancel reversal frees 25k
    assert rm.check_order(_intent("s", "S9"), -50, price)["approved"] is True    # 100k


def test_fill_does_not_double_count():
    """pending -> filled transition must not double-count exposure."""
    ledger, _ = make(alloc=100_000)
    ledger.record_pending("AAA", -60, "s")
    assert ledger.strategy_effective_positions("s")["AAA"] == -60
    ledger.record_fill("AAA", -60, 500.0, "s")        # converts pending -> filled
    assert ledger.strategy_effective_positions("s")["AAA"] == -60     # not -120


def test_reducing_order_passes_when_at_cap():
    """Projected post-trade gross means a position-reducing order LOWERS gross
    and should pass even at the cap (the old additive check falsely rejected it)."""
    ledger, rm = make(alloc=100_000)
    price = 500.0
    ledger.record_pending("AAA", +200, "s")
    ledger.record_fill("AAA", +200, price, "s")       # 200 * 500 = 100k, at cap

    assert rm.check_order(_intent("s", "AAA"), resolved_delta=-50, price=price)["approved"] is True
    assert rm.check_order(_intent("s", "AAA"), resolved_delta=+50, price=price)["approved"] is False


# ---------------------------------------------------------------------------
# futures: multiplier-aware notional
# ---------------------------------------------------------------------------
def test_futures_notional_uses_multiplier():
    """1 CL contract at $75 x mult 1000 = $75k notional, not $75."""
    _, rm = make(alloc=100_000)
    assert rm.check_order(_intent("s", "CL"), resolved_delta=1, price=75.0,
                          multiplier=1000)["approved"] is True
    # 2 contracts -> 150k > 100k; without multiplier awareness this would wrongly pass at $150
    assert rm.check_order(_intent("s", "CL"), resolved_delta=2, price=75.0,
                          multiplier=1000)["approved"] is False


def test_futures_gross_values_each_leg_by_its_own_unit():
    """ref_values lets a mixed-multiplier book be valued per leg, not all at one price."""
    ledger, rm = make(alloc=1_000_000)
    ledger.record_pending("RB", 1, "s")                    # RBOB leg already working
    ref = {"RB": 2.3 * 42000}                              # its own unit value (~96.6k)
    r = rm.check_order(_intent("s", "CL"), resolved_delta=1, price=75.0, multiplier=1000,
                       ref_values=ref)
    assert r["approved"] is True                           # 75k + 96.6k = 171.6k < 1M


# ---------------------------------------------------------------------------
# halt / reactivate / drawdown
# ---------------------------------------------------------------------------
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
    # a strategy with no max_drawdown configured should be skipped, not crash
    risk._config["s3"] = {"capital_allocation": 5000}
    ledger.strategy_realized_pnl["s3"] = -9999.0
    risk.check_drawdown("s3")  # should silently return, not raise
