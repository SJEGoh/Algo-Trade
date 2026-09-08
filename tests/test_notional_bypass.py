"""Ways an order used to get past the allocation cap — each one is now a rejection.

THE INCIDENT: a strategy allocated $100,000 placed a ~$700,000 order. Every case below
was reproducible against the real RiskManager / NettingCoordinator, and they share one
root cause: a leg the risk check COULDN'T VALUE was treated as costing nothing, so the
cap it was measured against was effectively infinite. Valuation failures must fail closed.

The pooled path matters most: equity `target_position` intents are routed by
`_should_pool` to NettingCoordinator, which never calls RiskManager.check_order — its
allocation gate is the only thing standing between a strategy and the broker.
"""
import math

import pytest

from execution.netting import NettingCoordinator
from ledger.position_ledger import PositionLedger
from risk.risk_manager import RiskManager

ALLOC = 100_000.0
CFG = {"xs_mom": {"capital_allocation": ALLOC, "max_drawdown": 0.15}}


class FakeLoggerDB:
    def log_fill(self, *a, **kw): pass
    def log_decision(self, *a, **kw): pass
    def save_strategy_positions(self, *a, **kw): pass
    def save_realized_pnl(self, *a, **kw): pass
    def save_multipliers(self, *a, **kw): pass
    def save_strategy_cash(self, *a, **kw): pass


class FakeExecutor:
    """Records what would have reached the broker."""

    def __init__(self, config=CFG, global_config=None):
        self.ledger = PositionLedger(None, config)
        self.risk_manager = RiskManager(self.ledger, config, global_config or {})
        self.logger_db = FakeLoggerDB()
        self.placed = []

    def place_net_order(self, sym, delta, instrument, price, urgent=False):
        self.placed.append({"symbol": sym, "delta": delta, "price": price})
        return len(self.placed)

    def _cancel_open_orders_for_symbol(self, sym):
        pass


def book_intent(sym, qty, price=None, sec_type="STK", multiplier=None):
    it = {"instrument": {"symbol": sym, "sec_type": sec_type, "exchange": "SMART",
                         "multiplier": multiplier},
          "target_quantity": qty}
    if price is not None:
        it["expected_price"] = price
    return it


@pytest.fixture
def co():
    ex = FakeExecutor()
    return NettingCoordinator(ex, CFG), ex


# ---------------------------------------------------------------- unpriced legs
def test_book_with_no_price_is_rejected(co):
    """THE SMOKING GUN. `expected_price or limit_price or 0.0` stored 0.0 for a missing
    price; 2,294 * 0.0 = $0 of gross, which passes any cap — and the net order then went
    to the broker with a reference price of 0.0."""
    coord, ex = co
    r = coord.submit_book("xs_mom", [book_intent("GOOGL", 2294)])       # ~$776k, no price
    assert r["accepted"] is False
    assert "cannot value" in r["reason"] and "GOOGL" in r["reason"]
    assert ex.placed == []                       # nothing reached the broker
    assert "GOOGL" not in coord.ref_price        # and no 0.0 was cached to poison later checks
    assert coord.desired.get("xs_mom", {}) == {}


@pytest.mark.parametrize("bad", [0.0, -12.0, float("nan"), float("inf"), "n/a", None])
def test_no_unusable_price_is_ever_treated_as_free(co, bad):
    coord, ex = co
    r = coord.submit_book("xs_mom", [book_intent("GOOGL", 2294, price=bad)])
    assert r["accepted"] is False
    assert ex.placed == []


def test_set_target_without_a_price_is_rejected(co):
    coord, ex = co
    r = coord.set_target("xs_mom", "GOOGL", 2294, instrument={"symbol": "GOOGL"}, price=None)
    assert r["accepted"] is False
    assert ex.placed == []


def test_a_stale_price_still_values_the_leg(co):
    """A previously-seen price is better than none: the check uses it rather than
    skipping the leg."""
    coord, ex = co
    coord.ref_price["GOOGL"] = 338.0
    r = coord.set_target("xs_mom", "GOOGL", 2294, instrument={"symbol": "GOOGL"}, price=None)
    assert r["accepted"] is False
    assert "exceeds allocation" in r["reason"]


# ---------------------------------------------------------------- futures multiplier
def test_futures_leg_without_a_multiplier_is_rejected(co):
    """multiplier or 1.0 understated CL by 1,000x: $685,000 counted as $685."""
    coord, ex = co
    r = coord.submit_book("xs_mom", [book_intent("CL", 10, price=68.5, sec_type="FUT")])
    assert r["accepted"] is False
    assert "cannot value" in r["reason"]
    assert ex.placed == []


def test_multiplier_learned_from_ib_is_used_when_the_intent_omits_it(co):
    """The ledger learns multipliers from IB contracts; the cap should use them."""
    coord, ex = co
    ex.ledger.multipliers["CL"] = 1000.0
    r = coord.submit_book("xs_mom", [book_intent("CL", 10, price=68.5, sec_type="FUT")])
    assert r["accepted"] is False
    assert "exceeds allocation" in r["reason"]    # $685,000 vs $100,000, correctly valued
    assert ex.placed == []


def test_correctly_sized_futures_book_still_passes(co):
    coord, ex = co
    r = coord.submit_book("xs_mom", [book_intent("CL", 1, price=68.5, sec_type="FUT",
                                                 multiplier=1000.0)])   # $68,500
    assert r["accepted"] is True
    assert [p["symbol"] for p in ex.placed] == ["CL"]


# ---------------------------------------------------------------- held vs desired
def test_new_exposure_on_top_of_untracked_holdings_is_rejected(co):
    """A desired book reset (lost netting.json, fresh container volume, restart before the
    first resync) looks empty while the strategy still carries the position — so checking
    the book alone let it book a fresh allocation ON TOP of what it already owned."""
    coord, ex = co
    ex.ledger.strategy_positions["xs_mom"] = {"GOOGL": 1700.0}     # ~$574k already held
    ex.ledger.strategy_avg_cost["xs_mom"] = {"GOOGL": 338.0}
    ex.ledger.current_positions["GOOGL"] = 1700.0

    # with no price for the held leg it fails closed: unvaluable, so unknown exposure
    r = coord.submit_book("xs_mom", [book_intent("MSFT", 294, price=338.0)])  # "only $99k"
    assert r["accepted"] is False
    assert "cannot value" in r["reason"] and "GOOGL" in r["reason"]

    # and once the held leg CAN be valued, on the cap itself: $574k + $99k vs $100k
    coord.ref_price["GOOGL"] = 338.0
    coord.instrument["GOOGL"] = {"symbol": "GOOGL", "sec_type": "STK"}
    r = coord.submit_book("xs_mom", [book_intent("MSFT", 294, price=338.0)])
    assert r["accepted"] is False
    assert "exceeds allocation" in r["reason"]
    assert ex.placed == []


def test_booking_less_of_a_name_you_already_hold_is_allowed(co):
    """The other half of the same scenario: asking for LESS of an oversized position is a
    sell-down, and the cap must not block it — otherwise a strategy that is already over
    its allocation (or whose allocation was just lowered) can never get back under it."""
    coord, ex = co
    ex.ledger.strategy_positions["xs_mom"] = {"GOOGL": 1700.0}     # ~$574k, over the cap
    ex.ledger.strategy_avg_cost["xs_mom"] = {"GOOGL": 338.0}
    ex.ledger.current_positions["GOOGL"] = 1700.0

    r = coord.submit_book("xs_mom", [book_intent("GOOGL", 294, price=338.0)])
    assert r["accepted"] is True
    assert ex.placed[0]["symbol"] == "GOOGL" and ex.placed[0]["delta"] == -1406.0


def test_reducing_an_over_cap_position_is_allowed(co):
    """Exits must never be blocked — the strategy has to be able to get smaller."""
    coord, ex = co
    ex.ledger.strategy_positions["xs_mom"] = {"GOOGL": 1700.0}
    ex.ledger.current_positions["GOOGL"] = 1700.0
    coord.ref_price["GOOGL"] = 338.0
    coord.instrument["GOOGL"] = {"symbol": "GOOGL", "sec_type": "STK"}

    r = coord.set_target("xs_mom", "GOOGL", 0, price=338.0)         # full exit
    assert r["accepted"] is True
    assert ex.placed and ex.placed[0]["delta"] == -1700.0


# ---------------------------------------------------------------- portfolio cap
def test_pooled_books_respect_the_global_gross_cap():
    """/orders enforced GLOBAL max_gross_exposure; pooled books never consulted it."""
    ex = FakeExecutor(global_config={"max_gross_exposure": 50_000.0})
    coord = NettingCoordinator(ex, CFG)
    r = coord.submit_book("xs_mom", [book_intent("GOOGL", 290, price=338.0)])  # $98k
    assert r["accepted"] is False
    assert "max_gross_exposure" in r["reason"]
    assert ex.placed == []


def test_global_cap_counts_every_strategy():
    cfg = {"a": {"capital_allocation": ALLOC, "max_drawdown": 0.15},
           "b": {"capital_allocation": ALLOC, "max_drawdown": 0.15}}
    ex = FakeExecutor(config=cfg, global_config={"max_gross_exposure": 120_000.0})
    coord = NettingCoordinator(ex, cfg)
    assert coord.submit_book("a", [book_intent("GOOGL", 200, price=338.0)])["accepted"]  # $67.6k
    r = coord.submit_book("b", [book_intent("MSFT", 200, price=338.0)])                  # +$67.6k
    assert r["accepted"] is False
    assert "portfolio gross" in r["reason"]


# ---------------------------------------------------------------- the happy path
def test_a_book_inside_its_allocation_is_accepted(co):
    coord, ex = co
    r = coord.submit_book("xs_mom", [book_intent("GOOGL", 290, price=338.0)])   # $98k < $100k
    assert r["accepted"] is True
    assert [p["symbol"] for p in ex.placed] == ["GOOGL"]


def test_exposure_reports_what_it_cannot_value(co):
    coord, _ = co
    coord.desired["xs_mom"] = {"GOOGL": 100.0, "MSFT": 50.0}
    coord.ref_price["GOOGL"] = 338.0
    gross, unvaluable = coord._exposure("xs_mom")
    assert gross == pytest.approx(33_800.0)
    assert unvaluable == ["MSFT"]
    assert math.isfinite(gross)


# ---------------------------------------------------------------- stale price cache
def test_existing_holdings_are_not_valued_at_the_new_orders_price():
    """The executor's price cache is in-memory and starts EMPTY after a restart, while
    positions restore from the database. Valuing those holdings at whatever the incoming
    order happens to cost made a $338k position look like $1,000."""
    led = PositionLedger(None, CFG)
    rm = RiskManager(led, CFG, {})
    led.strategy_positions["xs_mom"] = {"GOOGL": 1000.0}       # $338k held
    led.strategy_avg_cost["xs_mom"] = {"GOOGL": 338.0}

    r = rm.check_order({"strategy_id": "xs_mom", "instrument": {"symbol": "PENNY"}},
                       resolved_delta=100, price=1.0, multiplier=1.0, ref_values={})
    assert r["approved"] is False
    assert "exceed allocation" in r["reason"]


def test_futures_holdings_keep_their_multiplier_when_unpriced():
    led = PositionLedger(None, CFG)
    rm = RiskManager(led, CFG, {})
    led.strategy_positions["xs_mom"] = {"CL": 5.0}
    led.strategy_avg_cost["xs_mom"] = {"CL": 68.5}
    led.multipliers["CL"] = 1000.0                              # $342.5k held

    r = rm.check_order({"strategy_id": "xs_mom", "instrument": {"symbol": "PENNY"}},
                       resolved_delta=1, price=1.0, multiplier=1.0, ref_values={})
    assert r["approved"] is False


def test_a_traded_reference_price_still_wins_over_cost():
    """ref_values carries live traded prices; those beat the historical average cost."""
    led = PositionLedger(None, CFG)
    rm = RiskManager(led, CFG, {})
    led.strategy_positions["xs_mom"] = {"GOOGL": 100.0}
    led.strategy_avg_cost["xs_mom"] = {"GOOGL": 338.0}          # cost basis $33.8k

    r = rm.check_order({"strategy_id": "xs_mom", "instrument": {"symbol": "GOOGL"}},
                       resolved_delta=100, price=400.0, multiplier=1.0,
                       ref_values={"GOOGL": 400.0})             # marked $80k for 200 shares
    assert r["approved"] is True
