"""POST /strategies/{id}/allocation — moving capital in and out of a strategy.

Rules being pinned:
  * an increase lands entirely in cash;
  * a decrease comes out of cash first;
  * whatever cash can't cover is raised by SELLING positions — pro_rata (every position
    shrinks by the same fraction) or equal (even dollar split, redistributing what a small
    position can't cover);
  * capital moves never show up as P&L: cash and the capital basis move together, so
    `nav == starting_cash + realized + unrealized` still holds afterwards.
"""
import pytest

import execution.central_execution as ce
from ledger.position_ledger import PositionLedger
from risk.risk_manager import RiskManager

CFG = {"s1": {"capital_allocation": 100_000.0, "max_drawdown": 0.15}}


class FakeDB:
    def __init__(self, *a, **kw): self.allocations = {}
    def log_fill(self, *a, **kw): pass
    def log_equity(self, *a, **kw): pass
    def log_decision(self, *a, **kw): pass
    def log_risk_event(self, *a, **kw): pass
    def log_reconciliation(self, *a, **kw): pass
    def save_strategy_positions(self, *a, **kw): pass
    def save_realized_pnl(self, *a, **kw): pass
    def save_multipliers(self, *a, **kw): pass
    def save_strategy_cash(self, *a, **kw): pass
    def save_allocation(self, sid, alloc): self.allocations[sid] = alloc
    def load_allocations(self): return dict(self.allocations)
    def load_strategy_positions(self): return {}, {}
    def load_realized_pnl(self): return {}
    def load_multipliers(self): return {}
    def load_strategy_cash(self): return {}, {}
    def load_halted_strategies(self): return set()
    def save_halted_strategies(self, *a, **kw): pass
    def close(self): pass


@pytest.fixture
def ex(monkeypatch):
    monkeypatch.setattr(ce, "EventLogger", FakeDB)
    monkeypatch.setattr(ce, "CONFIG", {k: dict(v) for k, v in CFG.items()})
    x = ce.CentralExecutor()
    x.logger_db = FakeDB()
    x.ledger = PositionLedger(x, ce.CONFIG)
    x.risk_manager = RiskManager(x.ledger, ce.CONFIG, {})
    x.placed = []
    x.place_order = lambda intent: (x.placed.append(intent), len(x.placed))[1]
    x.get_marks = lambda symbols, **kw: {s: x.marks.get(s) for s in symbols}
    x.marks = {}
    return x


def hold(ex, book, marks, cost=None, cash=None):
    """Buy the book through the ledger so CASH moves like it would on a real fill."""
    costs = cost or marks
    for sym, qty in book.items():
        ex.ledger.record_fill(sym, qty, costs[sym], "s1")
    ex.marks = dict(marks)
    if cash is not None:
        ex.ledger.set_cash("s1", cash)


def nav_identity_holds(ex):
    snap = ex.ledger.equity_snapshot(ex.marks)["s1"]
    return abs(snap["nav"] - (snap["starting_cash"] + snap["equity"])) < 1e-6


# ------------------------------------------------------------------ increases
def test_an_increase_goes_entirely_to_cash(ex):
    r = ex.rebalance_allocation("s1", 150_000.0)
    assert r["cash_after"] == pytest.approx(150_000.0)      # was 100k, all cash
    assert r["liquidations"] == [] and ex.placed == []
    assert ce.CONFIG["s1"]["capital_allocation"] == 150_000.0


def test_funding_is_not_profit(ex):
    hold(ex, {"AAPL": 100.0}, {"AAPL": 60.0}, cost={"AAPL": 50.0})   # +$1,000 unrealized
    before = ex.ledger.equity_snapshot(ex.marks)["s1"]["equity"]
    ex.rebalance_allocation("s1", 150_000.0)
    after = ex.ledger.equity_snapshot(ex.marks)["s1"]
    assert after["equity"] == pytest.approx(before)          # P&L untouched
    assert after["starting_cash"] == pytest.approx(150_000.0)
    assert nav_identity_holds(ex)


# ------------------------------------------------------------------ decrease, cash covers
def test_a_decrease_comes_out_of_cash_when_cash_is_enough(ex):
    hold(ex, {"AAPL": 100.0}, {"AAPL": 50.0})                # $5k invested, $95k cash
    r = ex.rebalance_allocation("s1", 60_000.0)              # withdraw $40k
    assert r["cash_shortfall"] == 0
    assert r["cash_after"] == pytest.approx(55_000.0)        # 95k - 40k
    assert r["liquidations"] == [] and ex.placed == []       # nothing sold
    assert nav_identity_holds(ex)


def test_withdrawal_does_not_move_pnl(ex):
    hold(ex, {"AAPL": 100.0}, {"AAPL": 60.0}, cost={"AAPL": 50.0})
    before = ex.ledger.equity_snapshot(ex.marks)["s1"]["equity"]
    ex.rebalance_allocation("s1", 60_000.0)
    assert ex.ledger.equity_snapshot(ex.marks)["s1"]["equity"] == pytest.approx(before)
    assert nav_identity_holds(ex)


# ------------------------------------------------------------------ decrease, must sell
def test_shortfall_is_raised_from_positions_pro_rata(ex):
    """$20k cash, $80k of stock split 60/20. Withdraw $40k: cash covers $20k, the other
    $20k comes off the book — each position shrinking by the same 25%."""
    hold(ex, {"AAPL": 600.0, "MSFT": 200.0}, {"AAPL": 100.0, "MSFT": 100.0},
         cash=20_000.0)

    r = ex.rebalance_allocation("s1", 60_000.0)

    assert r["cash_shortfall"] == pytest.approx(20_000.0)
    cuts = {c["symbol"]: c for c in r["liquidations"]}
    assert cuts["AAPL"]["to_quantity"] == pytest.approx(450.0)   # 600 - 25%
    assert cuts["MSFT"]["to_quantity"] == pytest.approx(150.0)   # 200 - 25%
    assert sum(c["freed"] for c in r["liquidations"]) == pytest.approx(20_000.0)
    # the withdrawal is booked immediately; the sales settle into cash on the fills
    assert r["cash_after"] == pytest.approx(-20_000.0)
    assert {i["instrument"]["symbol"]: i["side"] for i in ex.placed} == \
        {"AAPL": "sell", "MSFT": "sell"}


def test_equal_method_splits_dollars_not_proportions(ex):
    hold(ex, {"AAPL": 600.0, "MSFT": 200.0}, {"AAPL": 100.0, "MSFT": 100.0}, cash=0.0)

    r = ex.rebalance_allocation("s1", 80_000.0, method="equal")   # raise $20k of $80k

    cuts = {c["symbol"]: c["freed"] for c in r["liquidations"]}
    assert cuts["AAPL"] == pytest.approx(10_000.0)
    assert cuts["MSFT"] == pytest.approx(10_000.0)


def test_equal_method_redistributes_what_a_small_position_cannot_cover(ex):
    """MSFT is only worth $3k, so it gives all of it and AAPL covers the rest."""
    hold(ex, {"AAPL": 600.0, "MSFT": 30.0}, {"AAPL": 100.0, "MSFT": 100.0}, cash=0.0)

    r = ex.rebalance_allocation("s1", 80_000.0, method="equal")   # raise $20k of $63k

    cuts = {c["symbol"]: c["freed"] for c in r["liquidations"]}
    assert cuts["MSFT"] == pytest.approx(3_000.0)                 # everything it had
    assert cuts["AAPL"] == pytest.approx(17_000.0)
    assert sum(cuts.values()) == pytest.approx(20_000.0)


def test_pro_rata_scales_shorts_too_and_the_cash_still_nets_out(ex):
    """Covering a short SPENDS cash, so shorts can't fund a withdrawal — but pro_rata
    still scales them, because the point is to keep the book's shape. The longs raise
    more than the shortfall and the short-covering gives part of it back; net = shortfall."""
    hold(ex, {"AAPL": 600.0, "MSFT": -200.0}, {"AAPL": 100.0, "MSFT": 100.0},
         cash=10_000.0)                              # NAV = 10k + 60k - 20k = 50k
    r = ex.rebalance_allocation("s1", 70_000.0)      # withdraw 30k, cash covers 10k

    assert r["cash_shortfall"] == pytest.approx(20_000.0)
    cuts = {c["symbol"]: c for c in r["liquidations"]}
    assert cuts["AAPL"]["to_quantity"] == pytest.approx(300.0)    # halved
    assert cuts["MSFT"]["to_quantity"] == pytest.approx(-100.0)   # halved, still short
    assert sum(c["freed"] for c in r["liquidations"]) == pytest.approx(20_000.0)
    assert {i["instrument"]["symbol"]: i["side"] for i in ex.placed} == \
        {"AAPL": "sell", "MSFT": "buy"}              # sell the long, buy back the short


def test_equal_leaves_shorts_alone(ex):
    """Buying back a short consumes cash, so it cannot contribute to an even split."""
    hold(ex, {"AAPL": 600.0, "MSFT": -200.0}, {"AAPL": 100.0, "MSFT": 100.0}, cash=0.0)
    r = ex.rebalance_allocation("s1", 80_000.0, method="equal")   # raise 20k
    assert [c["symbol"] for c in r["liquidations"]] == ["AAPL"]


def test_futures_are_reduced_in_whole_contracts(ex):
    ex.ledger.multipliers["CL"] = 1000.0
    hold(ex, {"CL": 10.0}, {"CL": 68.5}, cash=0.0)                          # $685k of notional
    r = ex.rebalance_allocation("s1", 99_000.0)                   # raise ~$1k
    if r["liquidations"]:                                          # a contract is $68.5k
        assert float(r["liquidations"][0]["to_quantity"]).is_integer()


# ------------------------------------------------------------------ guards
def test_cannot_withdraw_more_than_the_strategy_is_worth(ex):
    """The book is down: 100 AAPL bought at $50 now marks at $10, so NAV is $96k — you
    cannot take out the $99,999 the allocation still nominally says is there."""
    hold(ex, {"AAPL": 100.0}, {"AAPL": 10.0}, cost={"AAPL": 50.0})
    with pytest.raises(ValueError, match="NAV is only"):
        ex.rebalance_allocation("s1", 1.0)                        # withdraw $99,999


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_allocation_must_be_a_positive_number(ex, bad):
    with pytest.raises(ValueError):
        ex.rebalance_allocation("s1", bad)


def test_unknown_strategy_is_rejected(ex):
    with pytest.raises(ValueError, match="unknown strategy"):
        ex.rebalance_allocation("nope", 1000.0)


def test_a_sell_down_cannot_be_sized_without_prices(ex):
    """Fail closed rather than guess: an unpriced leg would make the plan arbitrary."""
    ex.ledger.set_cash("s1", 0.0)
    ex.ledger.strategy_positions["s1"] = {"AAPL": 100.0}
    ex.ledger.strategy_avg_cost["s1"] = {}          # no cost basis either
    ex.marks = {}
    with pytest.raises(ValueError, match="no price"):
        ex.rebalance_allocation("s1", 50_000.0)


def test_dry_run_changes_nothing(ex):
    hold(ex, {"AAPL": 800.0}, {"AAPL": 100.0}, cash=20_000.0)
    r = ex.rebalance_allocation("s1", 60_000.0, dry_run=True)

    assert r["liquidations"] and r["cash_shortfall"] == pytest.approx(20_000.0)
    assert ex.placed == []
    assert ex.ledger.cash("s1") == 20_000.0
    assert ce.CONFIG["s1"]["capital_allocation"] == 100_000.0


def test_no_change_is_a_no_op(ex):
    r = ex.rebalance_allocation("s1", 100_000.0)
    assert r["note"] == "allocation unchanged"
    assert ex.placed == []


# ------------------------------------------------------------------ persistence
def test_the_new_allocation_is_persisted(ex):
    ex.rebalance_allocation("s1", 150_000.0)
    assert ex.logger_db.load_allocations()["s1"] == 150_000.0


def test_a_restart_restores_the_rebalanced_allocation(ex):
    """Without this the drawdown denominator and the cap snap back to the config default."""
    ex.rebalance_allocation("s1", 150_000.0)
    ce.CONFIG["s1"]["capital_allocation"] = 100_000.0     # as if freshly imported
    ex._restore_persistent_state()
    assert ce.CONFIG["s1"]["capital_allocation"] == 150_000.0


# ------------------------------------------------------------------ plan arithmetic
def test_pro_rata_plan_shares_the_burden_by_size():
    plan = ce.CentralExecutor._reduction_plan({"A": 75_000.0, "B": 25_000.0}, 20_000.0, "pro_rata")
    assert plan["A"] == pytest.approx(15_000.0)
    assert plan["B"] == pytest.approx(5_000.0)


def test_equal_plan_caps_at_what_each_position_holds():
    plan = ce.CentralExecutor._reduction_plan({"A": 75_000.0, "B": 1_000.0}, 20_000.0, "equal")
    assert plan["B"] == pytest.approx(1_000.0)
    assert plan["A"] == pytest.approx(19_000.0)


def test_a_plan_never_asks_for_more_than_exists():
    plan = ce.CentralExecutor._reduction_plan({"A": 5_000.0}, 999_000.0, "pro_rata")
    assert sum(plan.values()) == pytest.approx(5_000.0)
