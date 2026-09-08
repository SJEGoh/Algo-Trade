"""client/remote_strategy.py — the framework every off-box strategy inherits.

A strategy author should only have to write `generate_book`. Everything else here is a
mistake the framework must make for them: refusing to trade into a broken executor, refusing
to send a book that can't be priced, journalling the run, and returning an exit code the
scheduler can act on.
"""
import math

import pytest

from client.executor_client import ExecutorRejected, ExecutorUnreachable
from client.remote_strategy import RemoteStrategy, StrategyError


class FakeClient:
    """Stands in for ExecutorClient, recording what the framework asked it to do."""

    def __init__(self, health=None, capital=100_000.0, fail=None):
        self.base_url = "http://executor:8000"
        self._health = health or {"connected": True, "killed": False, "market_open": True}
        self._capital = capital
        self._fail = fail
        self.submitted = []
        self.journalled = []
        self.preflighted = False

    def preflight(self):
        self.preflighted = True
        if self._fail:
            raise self._fail
        return self._health

    def allocation(self):
        return {"capital_allocation": self._capital}

    def submit_book(self, book, strategy_id=None):
        self.submitted.append(("book", book))
        return {"orders": [{"symbol": i["instrument"]["symbol"]} for i in book
                           if i["target_quantity"]]}

    def submit_orders(self, book):
        self.submitted.append(("orders", book))
        return {"ok": len(book), "refused": 0}

    def journal(self, event_type, summary, detail="", symbols=None, strategy_id=None):
        self.journalled.append({"event_type": event_type, "summary": summary,
                                "detail": detail, "symbols": symbols})
        return {"logged": True}


class Simple(RemoteStrategy):
    strategy_id = "s1"

    def generate_book(self, capital):
        self.capital_seen = capital
        return [self.intent("AAPL", 10, 100.0), self.intent("MSFT", 0, 400.0)]


def make(cls=Simple, client=None, **kw):
    client = client or FakeClient()
    return cls(client=client, **kw), client


# ------------------------------------------------------------------ the contract
def test_generate_book_is_abstract():
    """The one thing a strategy must write — and the only thing."""
    with pytest.raises(TypeError, match="abstract"):
        RemoteStrategy(strategy_id="s1", client=FakeClient())


def test_a_subclass_needs_only_generate_book():
    strategy, client = make()
    assert strategy.run() == RemoteStrategy.EXIT_OK
    assert client.submitted[0][0] == "book"


def test_strategy_id_must_come_from_somewhere():
    class Nameless(Simple):
        strategy_id = None
    with pytest.raises(ValueError, match="strategy_id"):
        Nameless(client=FakeClient())


# ------------------------------------------------------------------ run order
def test_it_preflights_before_generating_anything():
    """A broken executor must be found before a half-book exists."""
    class Tracker(Simple):
        def generate_book(self, capital):
            assert self.client.preflighted, "generated a book before preflighting"
            return [self.intent("AAPL", 1, 10.0)]
    strategy, _ = make(Tracker)
    assert strategy.run() == RemoteStrategy.EXIT_OK


def test_capital_comes_from_the_executor_not_a_constant():
    """So the strategy follows /allocate instead of a number someone has to remember."""
    strategy, _ = make(client=FakeClient(capital=250_000.0))
    strategy.run()
    assert strategy.capital_seen == 250_000.0


def test_dry_run_touches_nothing():
    strategy, client = make(dry_run=True)
    assert strategy.run() == RemoteStrategy.EXIT_OK
    assert client.submitted == [] and client.journalled == []
    assert client.preflighted is False          # works before the tunnel is even up
    assert strategy.capital_seen == RemoteStrategy.dry_run_capital


def test_orders_mode_posts_individually():
    class Individual(Simple):
        mode = "orders"
    strategy, client = make(Individual)
    strategy.run()
    assert client.submitted[0][0] == "orders"


# ------------------------------------------------------------------ should_run
def test_a_closed_market_skips_cleanly():
    class MarketHours(Simple):
        require_market_open = True
    strategy, client = make(MarketHours, client=FakeClient(
        health={"connected": True, "market_open": False}))
    assert strategy.run() == RemoteStrategy.EXIT_OK      # skipped, not failed
    assert client.submitted == []


def test_should_run_can_be_overridden():
    class Never(Simple):
        def should_run(self, health):
            return False
    strategy, client = make(Never)
    assert strategy.run() == RemoteStrategy.EXIT_OK
    assert client.submitted == []


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize("price, why", [
    (0, "positive"), (-5.0, "positive"), (float("nan"), "positive"),
    (float("inf"), "positive"), (None, "must be a number"), ("100", "must be a number"),
])
def test_an_unpriceable_book_never_leaves_the_strategy(price, why):
    """The executor values the book with expected_price when it applies the allocation cap,
    so a bad price mis-sizes the order AND the limit meant to contain it. Catch it here."""
    class BadPrice(Simple):
        def generate_book(self, capital):
            return [{"instrument": {"symbol": "AAPL", "sec_type": "STK"},
                     "target_quantity": 10, "expected_price": price}]
    strategy, client = make(BadPrice)
    with pytest.raises(StrategyError, match=why):
        strategy.run()
    assert client.submitted == []


def test_validation_reports_every_problem_at_once():
    class Messy(Simple):
        def generate_book(self, capital):
            return [{"instrument": {"symbol": ""}, "target_quantity": 1, "expected_price": 1.0},
                    {"instrument": {"symbol": "AAPL"}, "target_quantity": float("nan"),
                     "expected_price": 1.0},
                    {"instrument": {"symbol": "MSFT"}, "target_quantity": 1,
                     "expected_price": 0}]
    strategy, _ = make(Messy)
    with pytest.raises(StrategyError) as excinfo:
        strategy.run()
    message = str(excinfo.value)
    assert "instrument.symbol" in message
    assert "target_quantity" in message
    assert "expected_price" in message


def test_a_duplicated_symbol_is_caught():
    class Duplicate(Simple):
        def generate_book(self, capital):
            return [self.intent("AAPL", 10, 100.0), self.intent("AAPL", 20, 100.0)]
    strategy, _ = make(Duplicate)
    with pytest.raises(StrategyError, match="more than once"):
        strategy.run()


def test_a_futures_leg_without_a_multiplier_is_caught():
    """Same failure the executor fails closed on: without it the notional is understated
    by the multiplier — 1,000x on CL."""
    class Futures(Simple):
        def generate_book(self, capital):
            return [self.intent("CL", 1, 68.5, sec_type="FUT", exchange="NYMEX")]
    strategy, _ = make(Futures)
    with pytest.raises(StrategyError, match="multiplier"):
        strategy.run()

    class WithMultiplier(Futures):
        def generate_book(self, capital):
            return [self.intent("CL", 1, 68.5, sec_type="FUT", exchange="NYMEX",
                                multiplier=1000.0)]
    strategy, client = make(WithMultiplier)
    assert strategy.run() == RemoteStrategy.EXIT_OK


def test_returning_none_is_an_error_not_an_empty_book():
    class Forgetful(Simple):
        def generate_book(self, capital):
            return None
    strategy, _ = make(Forgetful)
    with pytest.raises(StrategyError, match="returned None"):
        strategy.run()


def test_an_empty_book_is_legitimate():
    """Deciding to hold nothing is a decision, and closes everything."""
    class Flat(Simple):
        def generate_book(self, capital):
            return []
    strategy, client = make(Flat)
    assert strategy.run() == RemoteStrategy.EXIT_OK
    assert client.submitted[0][1] == []


# ------------------------------------------------------------------ journalling
def test_every_run_is_journalled_including_a_flat_one():
    class Flat(Simple):
        def generate_book(self, capital):
            return [self.intent("AAPL", 0, 100.0)]
    strategy, client = make(Flat)
    strategy.run()
    assert client.journalled and "flat" in client.journalled[0]["summary"]
    assert client.journalled[0]["symbols"] == []


def test_journal_records_only_the_names_actually_held():
    strategy, client = make()
    strategy.run()
    assert client.journalled[0]["symbols"] == ["AAPL"]      # MSFT had a 0 target


def test_a_failed_journal_does_not_undo_a_submission():
    """The orders are in. Losing the note about them is a warning, not a failure."""
    client = FakeClient()
    client.journal = lambda *a, **kw: (_ for _ in ()).throw(
        ExecutorRejected("journal is down"))
    strategy, _ = make(client=client)
    assert strategy.run() == RemoteStrategy.EXIT_OK
    assert client.submitted


# ------------------------------------------------------------------ exit codes
def test_unreachable_exits_2_so_a_scheduler_cannot_miss_it():
    client = FakeClient(fail=ExecutorUnreachable("executor unreachable"))
    assert Simple.cli([], client=client) == RemoteStrategy.EXIT_UNREACHABLE


def test_a_refusal_exits_1():
    client = FakeClient(fail=ExecutorRejected("kill switch is active"))
    assert Simple.cli([], client=client) == RemoteStrategy.EXIT_REFUSED


def test_a_bad_book_exits_1():
    class BadPrice(Simple):
        def generate_book(self, capital):
            return [self.intent("AAPL", 10, 0.0)]
    assert BadPrice.cli([], client=FakeClient()) == RemoteStrategy.EXIT_REFUSED


def test_a_good_run_exits_0():
    assert Simple.cli([], client=FakeClient()) == RemoteStrategy.EXIT_OK


def test_cli_dry_run_flag():
    client = FakeClient()
    assert Simple.cli(["--dry-run"], client=client) == RemoteStrategy.EXIT_OK
    assert client.submitted == []


def test_cli_can_override_the_strategy_id():
    client = FakeClient()
    Simple.cli(["--strategy", "other_book", "--dry-run"], client=client)
    # nothing submitted on a dry run, but the override must reach the instance
    assert Simple(strategy_id="other_book", client=client).strategy_id == "other_book"


# ------------------------------------------------------------------ intent helper
def test_intent_builds_the_shape_the_executor_expects():
    i = RemoteStrategy.intent("AAPL", 10, 319.97)
    assert i == {"instrument": {"symbol": "AAPL", "asset_class": "equity",
                                "sec_type": "STK", "exchange": "SMART"},
                 "target_quantity": 10, "expected_price": 319.97}


def test_intent_carries_a_futures_multiplier():
    i = RemoteStrategy.intent("CL", 1, 68.5, sec_type="FUT", exchange="NYMEX",
                              asset_class="future", multiplier=1000.0)
    assert i["instrument"]["multiplier"] == 1000.0
    assert math.isfinite(i["expected_price"])
