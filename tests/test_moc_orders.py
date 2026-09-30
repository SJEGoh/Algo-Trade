"""tests/test_moc_orders.py — market-on-close orders, through /targets and /orders.

A book can ask for a name's change to be traded in the closing auction (order_type "moc")
rather than now. The failures worth pinning down are the silent ones:

  * an MOC request that quietly trades at MARKET instead — a pooled order is built in one
    place, and it used to hard-code "market";
  * mixed styles on one symbol: the close-auction shares and the immediate shares must go
    out as separate orders, EACH OWNED BY ITS OWN STRATEGIES, or one strategy's fill lands
    on the other's book;
  * past the cutoff the exchange refuses to cancel an MOC. A rebalance that cancels and
    re-sizes (the normal path) would then trade those shares twice — once at the close,
    once now. A working MOC must be left alone and the new order sized around it;
  * a late MOC must be refused up front, not accepted and then rejected by the exchange
    after the rest of the book has already traded;
  * urgent exits (stops, take-profits) and halts close NOW, whatever style the book asked for.
"""
from datetime import datetime

import pytest
import pytz

pytest.importorskip("ibapi")

import execution.central_execution as ce
from execution.central_execution import CentralExecutor, OrderIntent
from tests.fixtures.executor import fill as _fill, held, inst, make_executor as _make_executor

ET = pytz.timezone("America/New_York")


@pytest.fixture
def before_cutoff(monkeypatch):
    monkeypatch.setattr(ce, "moc_cutoff_passed", lambda *a, **k: False)


def _pass_cutoff(monkeypatch):
    monkeypatch.setattr(ce, "moc_cutoff_passed", lambda *a, **k: True)


def _book(*entries):
    return [{"instrument": inst(sym), "target_quantity": qty, "expected_price": 100.0,
             **({"order_type": ot} if ot else {})} for sym, qty, ot in entries]


# ------------------------------------------------------------------ the order itself
def test_moc_intent_builds_an_ib_moc_day_order():
    o = CentralExecutor.build_order({"side": "buy", "quantity": 10, "order_type": "moc",
                                     "time_in_force": "day"})
    assert (o.orderType, o.tif, o.action) == ("MOC", "DAY", "BUY")


def _intent(**kw):
    base = {"strategy_id": "s1", "client_order_id": "c1", "timestamp": "t",
            "schema_version": "1.0", "instrument": inst("AAPL"), "intent_type": "delta",
            "side": "buy", "quantity": 10, "order_type": "moc", "expected_price": 100.0}
    return {**base, **kw}


def test_moc_intent_needs_a_reference_price_like_market():
    with pytest.raises(Exception, match="expected_price"):
        OrderIntent(**_intent(expected_price=None))


def test_moc_intent_refuses_gtc():
    with pytest.raises(Exception, match="time_in_force"):
        OrderIntent(**_intent(time_in_force="gtc"))


# ------------------------------------------------------------------ the cutoff clock
@pytest.mark.parametrize("when, passed", [
    ("2026-09-30 15:49:59", False),     # normal session, cutoff 15:50
    ("2026-09-30 15:50:00", True),
    ("2026-09-30 10:00:00", False),
    ("2025-11-28 12:49:00", False),     # day after Thanksgiving: 13:00 close
    ("2025-11-28 12:50:00", True),
    ("2026-10-03 11:00:00", True),      # a Saturday: no close to trade into
])
def test_moc_cutoff_follows_the_session_close(when, passed):
    now = ET.localize(datetime.strptime(when, "%Y-%m-%d %H:%M:%S"))
    assert ce.moc_cutoff_passed(now=now) is passed


# ------------------------------------------------------------------ pooled books
def test_book_moc_goes_out_as_an_moc_order_and_fills_to_its_owner(before_cutoff):
    ex, co = _make_executor()
    r = co.submit_book("s1", _book(("AAPL", 100, "moc")))
    (o,) = r["orders"]
    assert o["order_type"] == "moc"
    assert ex.sent[o["order_id"]].orderType == "MOC"
    assert ex.order_status[o["order_id"]]["order_type"] == "moc"
    _fill(ex, o["order_id"])
    assert held(ex, "AAPL", "s1") == 100


def test_book_without_order_type_is_still_market(before_cutoff):
    ex, co = _make_executor()
    (o,) = co.submit_book("s1", _book(("AAPL", 100, None)))["orders"]
    assert ex.sent[o["order_id"]].orderType == "MKT"


def test_mixed_styles_on_one_symbol_split_into_owned_orders(before_cutoff):
    ex, co = _make_executor()
    co.submit_book("s1", _book(("SPY", -100, "moc")))
    ex._cancelled.clear()
    r = co.submit_book("s2", _book(("SPY", 40, None)))
    # s2's change re-nets SPY: s1's MOC is cancelled and re-placed alongside s2's market leg
    by_type = {ex.sent[o["order_id"]].orderType: o for o in r["orders"]}
    assert set(by_type) == {"MOC", "MKT"}
    assert by_type["MOC"]["delta"] == -100 and by_type["MKT"]["delta"] == 40
    assert co.order_owners[by_type["MOC"]["order_id"]]["gaps"] == {"s1": -100}
    assert co.order_owners[by_type["MKT"]["order_id"]]["gaps"] == {"s2": 40}
    _fill(ex, by_type["MKT"]["order_id"])
    assert held(ex, "SPY", "s2") == 40 and held(ex, "SPY", "s1") == 0
    _fill(ex, by_type["MOC"]["order_id"])
    assert held(ex, "SPY", "s1") == -100


def test_name_sent_at_zero_with_moc_closes_at_the_close(before_cutoff):
    ex, co = _make_executor()
    (o,) = co.submit_book("s1", _book(("AAPL", 100, None)))["orders"]
    _fill(ex, o["order_id"])
    (o,) = co.submit_book("s1", _book(("AAPL", 0, "moc")))["orders"]
    assert ex.sent[o["order_id"]].orderType == "MOC" and o["delta"] == -100


def test_name_dropped_from_the_book_closes_at_market(before_cutoff):
    ex, co = _make_executor()
    (o,) = co.submit_book("s1", _book(("AAPL", 100, "moc")))["orders"]
    _fill(ex, o["order_id"])
    orders = co.submit_book("s1", _book(("MSFT", 10, "moc")))["orders"]
    (o,) = [x for x in orders if x["symbol"] == "AAPL"]
    assert ex.sent[o["order_id"]].orderType == "MKT" and o["delta"] == -100


def test_urgent_exit_closes_now_even_on_an_moc_name(before_cutoff):
    ex, co = _make_executor()
    (o,) = co.submit_book("s1", _book(("AAPL", 100, "moc")))["orders"]
    _fill(ex, o["order_id"])
    (o,) = co.set_target("s1", "AAPL", 0, urgent=True)["orders"]
    assert ex.sent[o["order_id"]].orderType == "MKT"


def test_halt_unwinds_at_market(before_cutoff):
    ex, co = _make_executor()
    (o,) = co.submit_book("s1", _book(("AAPL", 100, "moc")))["orders"]
    _fill(ex, o["order_id"])
    (o,) = co.halt("s1")["orders"]
    assert ex.sent[o["order_id"]].orderType == "MKT"


def test_moc_placed_before_cutoff_is_not_cancelled_after_it(monkeypatch, before_cutoff):
    ex, co = _make_executor()
    (moc,) = co.submit_book("s1", _book(("SPY", -100, "moc")))["orders"]
    _pass_cutoff(monkeypatch)
    r = co.submit_book("s2", _book(("SPY", 30, None)))
    assert moc["order_id"] not in ex._cancelled          # left working
    (o,) = r["orders"]                                   # only s2's shares go out now
    assert o["delta"] == 30 and ex.sent[o["order_id"]].orderType == "MKT"
    assert co.order_owners[o["order_id"]]["gaps"] == {"s2": 30}
    _fill(ex, o["order_id"])
    _fill(ex, moc["order_id"])                           # the auction prints
    assert held(ex, "SPY", "s1") == -100 and held(ex, "SPY", "s2") == 30
    assert ex.ledger.current_positions["SPY"] == -70     # nothing traded twice


def test_resubmitting_the_same_moc_book_after_cutoff_places_nothing(monkeypatch, before_cutoff):
    ex, co = _make_executor()
    (moc,) = co.submit_book("s1", _book(("SPY", -100, "moc")))["orders"]
    _pass_cutoff(monkeypatch)
    co.submit_book("s1", _book(("SPY", -100, None)))     # e.g. a retry without order_type
    assert moc["order_id"] not in ex._cancelled
    assert len(ex.sent) == 1


def test_moc_gap_uncovered_after_cutoff_falls_back_to_market(monkeypatch, before_cutoff):
    ex, co = _make_executor()
    co.exec_style["s1"] = {"AAPL": "moc"}                # asked for the close, too late
    _pass_cutoff(monkeypatch)
    co.desired["s1"] = {"AAPL": 50}
    (o,) = co._rebalance({"AAPL"})["orders"]
    assert ex.sent[o["order_id"]].orderType == "MKT"


# ------------------------------------------------------------------ front doors
def test_direct_moc_refused_after_cutoff(monkeypatch):
    ex, _ = _make_executor()
    ex._enforce_market_hours = False
    _pass_cutoff(monkeypatch)
    r = ex.process_intent(_intent())
    assert not r["accepted"] and "MOC cutoff" in r["reason"]


def test_direct_moc_accepted_before_cutoff(before_cutoff):
    ex, _ = _make_executor()
    ex._enforce_market_hours = False
    r = ex.process_intent(_intent())
    assert r["accepted"] and ex.sent[r["order_id"]].orderType == "MOC"


def test_pooled_target_position_moc_keeps_its_style(before_cutoff):
    ex, _ = _make_executor()
    ex._enforce_market_hours = False
    r = ex.process_intent(_intent(intent_type="target_position", side=None, quantity=None,
                                  target_quantity=25))
    assert r["pooled"] and ex.sent[r["order_id"]].orderType == "MOC"


class _StubExecutor:
    MOC_CLOSED_REASON = CentralExecutor.MOC_CLOSED_REASON

    def __init__(self, closed):
        self.closed = closed

    def moc_closed(self):
        return self.closed


def test_targets_refuses_unknown_order_type_before_placing(monkeypatch):
    import api.server as server
    monkeypatch.setattr(server, "executor", _StubExecutor(False))
    r = server._check_book_order_types(_book(("AAPL", 1, "limit")))
    assert not r["accepted"] and "AAPL: 'limit'" in r["reason"]


def test_targets_refuses_a_late_moc_book(monkeypatch):
    import api.server as server
    monkeypatch.setattr(server, "executor", _StubExecutor(True))
    r = server._check_book_order_types(_book(("AAPL", 1, "moc"), ("MSFT", 1, None)))
    assert not r["accepted"] and "MOC cutoff" in r["reason"]
    # a market-only book is untouched by the cutoff
    assert server._check_book_order_types(_book(("AAPL", 1, None))) is None


def test_a_stop_on_a_name_whose_moc_exit_is_working_closes_now(before_cutoff):
    """Same quantity, different urgency: the MOC exit already covers target 0, but a stop
    firing must replace it with a market order rather than wait for the close."""
    ex, co = _make_executor()
    (o,) = co.submit_book("s1", _book(("AAPL", 100, None)))["orders"]
    _fill(ex, o["order_id"])
    (moc,) = co.submit_book("s1", _book(("AAPL", 0, "moc")))["orders"]
    (now,) = co.set_target("s1", "AAPL", 0, urgent=True)["orders"]
    assert moc["order_id"] in ex._cancelled
    assert ex.sent[now["order_id"]].orderType == "MKT" and now["delta"] == -100
