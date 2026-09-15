"""Stop-losses, take-profits and trailing stops — opt-in per strategy and per name.

The failures that would be silent, and that these pin:

  * a trailing stop whose high-water mark resets every time the strategy resends its book,
    so it never trails anything;
  * a stop that fires on a stale mark, into a closed market, or on a position that has not
    filled yet;
  * a stop that fires and closes the position, then the strategy buys it straight back on
    its next run — so the stop achieved nothing;
  * a close that fails and leaves the position held with no rule over it;
  * an exit that goes out through ATR as a resting limit instead of closing at market;
  * a strategy that never sent exits being touched at all.
"""
import json
from datetime import datetime, timezone

import pytest

from risk.exit_rules import ExitManager, ExitSpecError, levels, parse_exits

T0 = datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc).timestamp()     # 10:00 ET


class Ledger:
    def __init__(self):
        self.strategy_positions = {}
        self.strategy_avg_cost = {}


class Risk:
    def __init__(self, active):
        self.active = set(active)

    def is_active(self, sid):
        return sid in self.active


class Coordinator:
    def __init__(self):
        self.desired = {}
        self.calls = []
        self.urgent = []
        self.fail = False

    def set_target(self, sid, symbol, qty, instrument=None, price=None, urgent=False):
        if self.fail:
            raise RuntimeError("socket gone")
        self.calls.append((sid, symbol, qty))
        self.urgent.append(urgent)
        self.desired.get(sid, {}).pop(symbol, None)
        return {"accepted": True}


class DB:
    def __init__(self):
        self.state = None
        self.decisions = []

    def save_exit_state(self, state):
        self.state = state

    def load_exit_state(self):
        return self.state

    def log_decision(self, sid, kind, summary, detail="", symbols=None):
        self.decisions.append((sid, kind, summary, symbols))


class Ex:
    def __init__(self):
        self.ledger = Ledger()
        self.risk_manager = Risk({"s1", "s2"})
        self.coordinator = Coordinator()
        self.logger_db = DB()
        self.stale = set()
        self.mark_age = {}
        self.live = set()
        self.flattened = []

    def mark_is_fresh(self, symbol, max_age=None):
        if symbol in self.stale:
            return False
        return self.mark_age.get(symbol, 0.0) <= (120.0 if max_age is None else max_age)

    def _has_live_order(self, sid, symbol):
        return (sid, symbol) in self.live

    def _flatten_direct(self, sid, symbols=None):
        self.flattened.append((sid, set(symbols)))


@pytest.fixture
def now():
    return [T0]


@pytest.fixture
def ex():
    return Ex()


@pytest.fixture
def em(ex, now):
    return ExitManager(ex, clock=lambda: now[0])


def hold(ex, sid, symbol, qty, cost, pooled=True):
    ex.ledger.strategy_positions.setdefault(sid, {})[symbol] = qty
    ex.ledger.strategy_avg_cost.setdefault(sid, {})[symbol] = cost
    if pooled:
        ex.coordinator.desired.setdefault(sid, {})[symbol] = qty


# ------------------------------------------------------------------ the spec
def test_any_single_kind_is_enough():
    assert parse_exits({"trail_pct": 0.05}, 10, 100.0) == {"trail_pct": 0.05}
    assert parse_exits({"take_profit_price": 120}, 10, 100.0) == {"take_profit_price": 120.0}


def test_no_exits_means_no_rules():
    assert parse_exits(None, 10, 100.0) is None
    assert parse_exits({"stop_pct": None}, 10, 100.0) is None


def test_a_flat_target_ignores_its_exits_rather_than_refusing_the_book():
    assert parse_exits({"stop_pct": 0.05}, 0, 100.0) is None


@pytest.mark.parametrize("raw, qty, message", [
    ({"stop_loss": 95}, 10, "unknown exit field"),
    ({"stop_price": 95, "stop_pct": 0.05}, 10, "not both"),
    ({"trail_pct": 5}, 10, "fraction"),
    ({"stop_pct": -0.05}, 10, "positive"),
    ({"stop_pct": True}, 10, "number"),
    ({"stop_price": 105}, 10, "wrong side"),           # a long's stop above the price
    ({"take_profit_price": 95}, 10, "wrong side"),     # a long's target below it
    ({"stop_price": 95}, -10, "wrong side"),           # a short's stop below it
    ({"take_profit_pct": 1.2}, -10, "fraction"),       # a short can't gain more than 100%
])
def test_an_exit_that_cannot_be_enforced_as_written_is_refused(raw, qty, message):
    with pytest.raises(ExitSpecError, match=message):
        parse_exits(raw, qty, 100.0)


def test_a_long_may_target_more_than_double():
    assert parse_exits({"take_profit_pct": 1.5}, 10, 100.0) == {"take_profit_pct": 1.5}


# ------------------------------------------------------------------ firing
def test_a_strategy_without_exits_is_never_touched(ex, em):
    hold(ex, "s1", "AAPL", 10, 100.0)
    assert em.check({"AAPL": 1.0}) == []
    assert ex.coordinator.calls == []


def test_stop_pct_fires_at_its_level_and_not_before(ex, em):
    hold(ex, "s1", "AAPL", 10, 100.0)
    em.set("s1", "AAPL", 10, {"stop_pct": 0.05})
    assert em.check({"AAPL": 95.01}) == []
    fired = em.check({"AAPL": 95.0})
    assert [(f["symbol"], f["kind"]) for f in fired] == [("AAPL", "stop")]
    assert ex.coordinator.calls == [("s1", "AAPL", 0)]
    assert "AAPL" not in em.rules.get("s1", {})
    assert ex.logger_db.decisions[-1][1] == "exit"


def test_a_short_stop_fires_when_price_rises(ex, em):
    hold(ex, "s1", "TSLA", -10, 200.0)
    em.set("s1", "TSLA", -10, {"stop_price": 210.0})
    assert em.check({"TSLA": 209.0}) == []
    assert em.check({"TSLA": 210.5})[0]["kind"] == "stop"


def test_take_profit_fires_for_longs_and_shorts(ex, em):
    hold(ex, "s1", "AAPL", 10, 100.0)
    hold(ex, "s1", "TSLA", -10, 200.0)
    em.set("s1", "AAPL", 10, {"take_profit_pct": 0.10})
    em.set("s1", "TSLA", -10, {"take_profit_price": 180.0})
    fired = em.check({"AAPL": 110.0, "TSLA": 179.0})
    assert sorted((f["symbol"], f["kind"]) for f in fired) == [
        ("AAPL", "take_profit"), ("TSLA", "take_profit")]


def test_a_trailing_stop_follows_the_high_and_fires_on_the_retrace(ex, em):
    hold(ex, "s1", "AAPL", 10, 100.0)
    em.set("s1", "AAPL", 10, {"trail_pct": 0.05})
    assert em.check({"AAPL": 100.0}) == []
    assert em.check({"AAPL": 120.0}) == []
    assert em.rules["s1"]["AAPL"]["extreme"] == 120.0
    assert em.check({"AAPL": 115.0}) == []          # level is 114
    assert em.rules["s1"]["AAPL"]["extreme"] == 120.0, "a pullback must not lower the high"
    fired = em.check({"AAPL": 114.0})
    assert fired[0]["kind"] == "trail" and fired[0]["level"] == pytest.approx(114.0)


def test_a_short_trail_in_dollars(ex, em):
    hold(ex, "s1", "TSLA", -10, 200.0)
    em.set("s1", "TSLA", -10, {"trail_amount": 5.0})
    em.check({"TSLA": 180.0})
    assert em.check({"TSLA": 184.0}) == []
    assert em.check({"TSLA": 185.0})[0]["kind"] == "trail"


def test_resending_the_same_trail_keeps_its_high(ex, em):
    """Strategies resend their book every run. A trail that reset each time would sit at
    the entry price forever."""
    hold(ex, "s1", "AAPL", 10, 100.0)
    em.set("s1", "AAPL", 10, {"trail_pct": 0.05})
    em.check({"AAPL": 130.0})
    em.set("s1", "AAPL", 12, {"trail_pct": 0.04})
    assert em.rules["s1"]["AAPL"]["extreme"] == 130.0


def test_flipping_direction_starts_a_new_trail(ex, em):
    hold(ex, "s1", "AAPL", 10, 100.0)
    em.set("s1", "AAPL", 10, {"trail_pct": 0.05})
    em.check({"AAPL": 130.0})
    em.set("s1", "AAPL", -10, {"trail_pct": 0.05})
    assert em.rules["s1"]["AAPL"]["extreme"] is None


def test_an_unfilled_position_never_fires(ex, em):
    """Rules are set when the target is, and a target is not a position."""
    em.set("s1", "AAPL", 10, {"stop_price": 95.0})
    assert em.check({"AAPL": 50.0}) == []


def test_a_flat_position_forgets_its_trail(ex, em):
    hold(ex, "s1", "AAPL", 10, 100.0)
    em.set("s1", "AAPL", 10, {"trail_pct": 0.05})
    em.check({"AAPL": 150.0})
    ex.ledger.strategy_positions["s1"]["AAPL"] = 0.0
    em.check({"AAPL": 150.0})
    assert em.rules["s1"]["AAPL"]["extreme"] is None, \
        "a stale high would fire the moment the name is re-entered"


def test_a_position_on_the_other_side_is_not_this_rules_business(ex, em):
    hold(ex, "s1", "AAPL", -10, 100.0)
    em.set("s1", "AAPL", 10, {"stop_price": 95.0})
    assert em.check({"AAPL": 50.0}) == []


def test_never_exits_on_a_stale_mark(ex, em):
    hold(ex, "s1", "AAPL", 10, 100.0)
    em.set("s1", "AAPL", 10, {"stop_price": 95.0})
    ex.stale.add("AAPL")
    assert em.check({"AAPL": 50.0}) == []


def test_equities_wait_for_the_open_but_futures_do_not(ex, em):
    hold(ex, "s1", "AAPL", 10, 100.0)
    hold(ex, "s1", "CL", 1, 70.0, pooled=False)
    em.set("s1", "AAPL", 10, {"stop_price": 95.0})
    em.set("s1", "CL", 1, {"stop_price": 65.0}, sec_type="FUT")
    fired = em.check({"AAPL": 50.0, "CL": 60.0}, market_open=False)
    assert [f["symbol"] for f in fired] == ["CL"]


def test_a_halted_strategy_is_left_to_ensure_flat(ex, em):
    hold(ex, "s1", "AAPL", 10, 100.0)
    em.set("s1", "AAPL", 10, {"stop_price": 95.0})
    ex.risk_manager.active.discard("s1")
    assert em.check({"AAPL": 50.0}) == []


def test_a_pooled_exit_goes_out_urgent_so_it_skips_atr(ex, em):
    hold(ex, "s1", "AAPL", 10, 100.0)
    em.set("s1", "AAPL", 10, {"stop_price": 95.0})
    em.check({"AAPL": 90.0})
    assert ex.coordinator.urgent == [True]


def test_an_exit_never_acts_on_a_mark_older_than_its_limit(ex, now):
    """A failed fetch carries the last price forward. The executor's general limit (120s)
    is too loose for a stop re-priced every 30s."""
    em = ExitManager(ex, clock=lambda: now[0], max_mark_age=65.0)
    hold(ex, "s1", "AAPL", 10, 100.0)
    em.set("s1", "AAPL", 10, {"stop_price": 95.0})
    ex.mark_age["AAPL"] = 90.0
    assert em.check({"AAPL": 50.0}) == []
    ex.mark_age["AAPL"] = 10.0
    assert em.check({"AAPL": 50.0})[0]["kind"] == "stop"


def test_only_held_names_with_rules_are_priced(ex, em):
    hold(ex, "s1", "AAPL", 10, 100.0)
    hold(ex, "s1", "MSFT", 10, 100.0)
    em.set("s1", "AAPL", 10, {"stop_pct": 0.05})
    em.set("s1", "NVDA", 10, {"stop_pct": 0.05})          # target set, not filled
    assert em.watched_symbols() == {"AAPL"}


def test_a_direct_position_gets_a_closing_order_for_that_name_only(ex, em):
    hold(ex, "s1", "CL", 2, 70.0, pooled=False)
    em.set("s1", "CL", 2, {"stop_price": 65.0}, sec_type="FUT")
    em.check({"CL": 60.0})
    assert ex.flattened == [("s1", {"CL"})]
    assert ex.coordinator.calls == []


def test_a_close_that_did_not_take_is_retried(ex, em):
    hold(ex, "s1", "AAPL", 10, 100.0)
    em.set("s1", "AAPL", 10, {"stop_price": 95.0})
    em.check({"AAPL": 90.0})
    ex.live.add(("s1", "AAPL"))                     # the close is still working
    em.check({"AAPL": 90.0})
    assert len(ex.coordinator.calls) == 1
    ex.live.clear()                                 # ...and then it was cancelled
    em.check({"AAPL": 90.0})
    assert len(ex.coordinator.calls) == 2


def test_a_close_that_throws_keeps_the_lockout_and_pages(ex, em, caplog):
    hold(ex, "s1", "AAPL", 10, 100.0)
    em.set("s1", "AAPL", 10, {"stop_price": 95.0})
    ex.coordinator.fail = True
    em.check({"AAPL": 90.0})
    assert any(r.levelname == "CRITICAL" and "EXIT NOT PLACED" in r.message
               for r in caplog.records)
    assert em.lockouts["s1"]["AAPL"]["kind"] == "stop"
    ex.coordinator.fail = False
    em.check({"AAPL": 90.0})
    assert ex.coordinator.calls == [("s1", "AAPL", 0)]


# ------------------------------------------------------------------ re-entry
def test_a_stopped_name_cannot_be_reentered_the_same_way_today(ex, em, now):
    hold(ex, "s1", "AAPL", 10, 100.0)
    em.set("s1", "AAPL", 10, {"stop_price": 95.0})
    em.check({"AAPL": 90.0})
    assert em.blocked("s1", "AAPL", 10)["kind"] == "stop"
    assert em.blocked("s1", "AAPL", -10) is None, "the other direction is a new trade"
    assert em.blocked("s1", "AAPL", 0) is None, "flat is never blocked"
    assert em.blocked("s2", "AAPL", 10) is None, "lockouts belong to one strategy"
    now[0] += 24 * 3600
    assert em.blocked("s1", "AAPL", 10) is None
    assert em.lockouts == {}


def test_clear_lifts_a_lockout_by_hand(ex, em):
    hold(ex, "s1", "AAPL", 10, 100.0)
    em.set("s1", "AAPL", 10, {"stop_price": 95.0})
    em.check({"AAPL": 90.0})
    assert em.clear("s1", "AAPL") == {"removed_rule": False, "cleared_lockout": True}
    assert em.blocked("s1", "AAPL", 10) is None


# ------------------------------------------------------------------ the book is authoritative
def test_a_book_replaces_every_rule_it_mentions_and_drops_the_rest(em):
    em.replace_book("s1", {"AAPL": (10, {"stop_pct": 0.05}, "STK"),
                           "MSFT": (5, {"trail_pct": 0.1}, "STK")})
    em.replace_book("s1", {"AAPL": (10, None, "STK")})
    assert em.rules == {}


def test_one_strategys_book_leaves_anothers_rules_alone(em):
    em.set("s2", "AAPL", 10, {"stop_pct": 0.05})
    em.replace_book("s1", {})
    assert "AAPL" in em.rules["s2"]


# ------------------------------------------------------------------ restart
def test_rules_highs_and_lockouts_survive_a_restart(ex, em, now):
    hold(ex, "s1", "AAPL", 10, 100.0)
    hold(ex, "s1", "MSFT", 10, 100.0)
    em.set("s1", "AAPL", 10, {"trail_pct": 0.05})
    em.set("s1", "MSFT", 10, {"stop_price": 95.0})
    em.check({"AAPL": 140.0, "MSFT": 90.0})
    again = ExitManager(ex, clock=lambda: now[0])
    assert again.rules["s1"]["AAPL"]["extreme"] == 140.0
    assert again.blocked("s1", "MSFT", 10) is not None
    assert json.loads(ex.logger_db.state)["rules"]["s1"]["AAPL"]["spec"] == {"trail_pct": 0.05}


def test_the_real_event_logger_round_trips_the_state(tmp_path):
    from logger.event_logger import EventLogger
    db = EventLogger(str(tmp_path / "exits.db"))
    assert db.load_exit_state() is None
    db.save_exit_state('{"rules": {}, "lockouts": {}}')
    db.save_exit_state('{"rules": {"s1": {}}, "lockouts": {}}')
    assert json.loads(db.load_exit_state()) == {"rules": {"s1": {}}, "lockouts": {}}


# ------------------------------------------------------------------ inspection
def test_levels_show_where_each_exit_fires():
    rule = {"direction": 1, "extreme": 120.0,
            "spec": {"stop_pct": 0.05, "take_profit_price": 130.0, "trail_amount": 3.0}}
    assert levels(rule, 100.0) == {"stop": 95.0, "take_profit": 130.0, "trail": 117.0}
    assert "stop" not in levels(rule, None), "a percentage stop needs a known cost"
