"""tests/test_chains.py — chained two-leg orders (execution/chains.py, POST /chains).

Both legs rest as ATR-priced limits; the first to fill leads and the other is hedged in
proportion, urgently, through the pool. The failures that would be silent:

  * a lead fills and the hedge never goes out — the pair is left naked with nothing saying so;
  * the hedge is sized to the whole leg on a PARTIAL lead fill, overshooting the pair;
  * the hedge leg's own resting limit is left working after the lead fills, so it can fill
    too and double the hedge;
  * another strategy re-nets a leg's symbol and the leg comes back at MARKET, not its limit;
  * expiry sends the remainder at market (it must not), or cancels a lead and then lets the
    pool trade a late fill straight back out;
  * a restart forgets the chain, leaving a resting leg with nothing waiting to hedge it;
  * an unavailable ATR quietly turns a leg into a market order.
"""
import pytest

pytest.importorskip("ibapi")

from fastapi.testclient import TestClient

import execution.central_execution as ce
from tests.fixtures import executor as harness
from tests.fixtures.executor import fill as _fill, held, inst

ATR = 2.0
T0 = 1_000_000.0


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


def _make_executor(state_path=None):
    """The shared harness, plus a fixed ATR, open market hours and a controllable clock."""
    ex, co = harness.make_executor(state_path=state_path)
    ex.atr_layer._get_atr = lambda sym: ATR
    ex._enforce_market_hours = False
    co.chains.clock = Clock()
    return ex, co


@pytest.fixture(autouse=True)
def _before_cutoff(monkeypatch):
    monkeypatch.setattr(ce, "moc_cutoff_passed", lambda *a, **k: False)


def _leg(sym, qty, px=100.0, sid_held=0.0):
    lp = round(px - 0.5 * ATR, 2) if qty > sid_held else round(px + 0.5 * ATR, 2)
    return {"instrument": inst(sym), "target_quantity": qty, "expected_price": px,
            "limit_price": lp, "atr": ATR}


def _submit(co, legs, sid="s1", ttl=600):
    return co.chains.submit(sid, legs, T0 + ttl)


def _working(ex, sym):
    return [oid for oid, st in ex.order_status.items()
            if st["symbol"] == sym and oid not in ex._cancelled
            and abs(st["exec_filled"]) < abs(st["pending_qty"])]


def _pair(ex, co, a=("AAPL", 100), b=("MSFT", -50)):
    r = _submit(co, [_leg(*a), _leg(*b)])
    assert r["accepted"], r
    oids = {o["symbol"]: o["order_id"] for o in r["orders"]}
    return r, oids


# ------------------------------------------------------------------ placing
def test_both_legs_rest_as_atr_limits_owned_by_the_strategy():
    ex, co = _make_executor()
    r, oids = _pair(ex, co)
    a, m = ex.sent[oids["AAPL"]], ex.sent[oids["MSFT"]]
    assert (a.orderType, a.action, a.lmtPrice) == ("LMT", "BUY", 99.0)     # 100 - 0.5*2
    assert (m.orderType, m.action, m.lmtPrice) == ("LMT", "SELL", 101.0)   # 100 + 0.5*2
    assert co.order_owners[oids["AAPL"]]["gaps"] == {"s1": 100}
    assert co.desired["s1"] == {"AAPL": 100, "MSFT": -50}
    assert r["chain"]["status"] == "working"


# ------------------------------------------------------------------ reacting
def test_a_full_lead_fill_pulls_the_other_limit_and_hedges_it_at_market():
    ex, co = _make_executor()
    _, oids = _pair(ex, co)
    _fill(ex, oids["AAPL"])
    assert oids["MSFT"] in ex._cancelled                  # its own limit can't fill as well
    (h,) = _working(ex, "MSFT")
    assert ex.sent[h].orderType == "MKT" and ex.order_status[h]["pending_qty"] == -50
    assert co.order_owners[h]["gaps"] == {"s1": -50}
    _fill(ex, h)
    (c,) = co.chains.snapshot()
    assert c["status"] == "done" and c["lead_symbol"] == "AAPL"
    assert held(ex, "AAPL") == 100 and held(ex, "MSFT") == -50
    assert "s1" not in co.exec_style                      # nothing left resting as a limit


def test_a_partial_lead_fill_hedges_in_proportion_and_the_lead_keeps_resting():
    ex, co = _make_executor()
    _, oids = _pair(ex, co)
    _fill(ex, oids["AAPL"], shares=30)                    # 30% of the lead
    assert oids["AAPL"] not in ex._cancelled              # rest keeps working at 99
    (h,) = _working(ex, "MSFT")
    assert ex.order_status[h]["pending_qty"] == -15       # 30% of -50
    _fill(ex, h)
    _fill(ex, oids["AAPL"], shares=40)                    # now 70%
    (h2,) = _working(ex, "MSFT")
    assert ex.order_status[h2]["pending_qty"] == -20      # tops up to -35
    _fill(ex, h2)
    _fill(ex, oids["AAPL"])
    (h3,) = _working(ex, "MSFT")
    _fill(ex, h3)
    assert held(ex, "MSFT") == -50 and co.chains.snapshot()[0]["status"] == "done"


def test_the_short_leg_can_lead():
    ex, co = _make_executor()
    _, oids = _pair(ex, co)
    _fill(ex, oids["MSFT"])
    assert oids["AAPL"] in ex._cancelled
    (h,) = _working(ex, "AAPL")
    assert ex.sent[h].orderType == "MKT" and ex.order_status[h]["pending_qty"] == 100


def test_another_strategy_renetting_a_leg_keeps_it_a_limit_at_its_price():
    ex, co = _make_executor()
    _, oids = _pair(ex, co)
    r = co.set_target("s2", "AAPL", 10, instrument=inst("AAPL"), price=100.0)
    assert oids["AAPL"] in ex._cancelled                  # the pool re-nets the symbol...
    by = {ex.sent[o["order_id"]].orderType: o for o in r["orders"]}
    assert ex.sent[by["LMT"]["order_id"]].lmtPrice == 99.0  # ...the leg comes back at 99
    assert co.order_owners[by["LMT"]["order_id"]]["gaps"] == {"s1": 100}
    assert co.order_owners[by["MKT"]["order_id"]]["gaps"] == {"s2": 10}


# ------------------------------------------------------------------ ending
def test_expiry_with_nothing_filled_pulls_both_and_trades_nothing():
    ex, co = _make_executor()
    _, oids = _pair(ex, co)
    co.chains.clock.t = T0 + 601
    (c,) = co.chains.expire()
    assert c["status"] == "expired"
    assert {oids["AAPL"], oids["MSFT"]} <= ex._cancelled
    assert _working(ex, "AAPL") == [] and _working(ex, "MSFT") == []
    assert co.desired.get("s1", {}) == {}


def test_expiry_while_legging_leaves_a_smaller_balanced_pair_and_no_market_order():
    ex, co = _make_executor()
    _, oids = _pair(ex, co)
    _fill(ex, oids["AAPL"], shares=40)
    (h,) = _working(ex, "MSFT")
    _fill(ex, h)
    sent_before = len(ex.sent)
    co.chains.clock.t = T0 + 601
    co.chains.expire()
    assert oids["AAPL"] in ex._cancelled
    assert len(ex.sent) == sent_before                    # nothing at market on expiry
    assert co.desired["s1"] == {"AAPL": 40, "MSFT": -20}


def test_a_late_fill_after_expiry_is_kept_and_hedged_not_traded_back():
    ex, co = _make_executor()
    _, oids = _pair(ex, co)
    _fill(ex, oids["AAPL"], shares=40)
    _fill(ex, _working(ex, "MSFT")[0])
    co.chains.clock.t = T0 + 601
    co.chains.expire()
    _fill(ex, oids["AAPL"], shares=20)                    # the cancel lost the race
    assert _working(ex, "AAPL") == []                     # no sell-back of the late shares
    assert co.desired["s1"]["AAPL"] == 60
    (h,) = _working(ex, "MSFT")
    assert ex.order_status[h]["pending_qty"] == -10       # hedge follows to -30


def test_cancel_is_the_same_as_expiry():
    ex, co = _make_executor()
    r, oids = _pair(ex, co)
    c = co.chains.cancel(r["chain"]["id"])
    assert c["status"] == "cancelled" and {oids["AAPL"], oids["MSFT"]} <= ex._cancelled


def test_an_exit_on_one_leg_takes_it_over_and_stops_the_other():
    ex, co = _make_executor()
    _, oids = _pair(ex, co)
    co.set_target("s1", "AAPL", 0, urgent=True)           # e.g. an exit rule firing
    (c,) = co.chains.snapshot()
    assert c["status"] == "superseded"
    assert oids["MSFT"] in ex._cancelled and "MSFT" not in co.desired.get("s1", {})


def test_a_book_takes_the_chain_over():
    ex, co = _make_executor()
    _, oids = _pair(ex, co)
    co.submit_book("s1", [{"instrument": inst("AAPL"), "target_quantity": 100,
                           "expected_price": 100.0}])
    (c,) = co.chains.snapshot()
    assert c["status"] == "superseded"
    (a,) = _working(ex, "AAPL")
    assert ex.sent[a].orderType == "MKT"                  # the book's style, not the chain's


# ------------------------------------------------------------------ refusals
@pytest.mark.parametrize("legs, message", [
    (lambda: [_leg("AAPL", 100)], "exactly two legs"),
    (lambda: [_leg("AAPL", 100), _leg("AAPL", -5)], "different symbols"),
    (lambda: [_leg("AAPL", 0), _leg("MSFT", -5)], "must trade"),
])
def test_a_malformed_chain_places_nothing(legs, message):
    ex, co = _make_executor()
    r = _submit(co, legs())
    assert not r["accepted"] and message in r["reason"] and ex.sent == {}


def test_a_symbol_with_an_outstanding_target_is_refused():
    ex, co = _make_executor()
    co.set_target("s1", "AAPL", 10, instrument=inst("AAPL"), price=100.0)   # unfilled
    n = len(ex.sent)
    r = _submit(co, [_leg("AAPL", 100), _leg("MSFT", -5)])
    assert not r["accepted"] and "outstanding" in r["reason"] and len(ex.sent) == n


def test_a_symbol_already_in_a_working_chain_is_refused():
    ex, co = _make_executor()
    _pair(ex, co)
    r = _submit(co, [_leg("AAPL", 200), _leg("GOOG", -5)])
    assert not r["accepted"] and "working chain" in r["reason"]


# ------------------------------------------------------------------ restart
def test_a_chain_survives_a_restart_and_still_hedges(tmp_path):
    path = str(tmp_path / "netting.json")
    ex, co = _make_executor(state_path=path)
    _, oids = _pair(ex, co)
    ex2, co2 = _make_executor(state_path=path)
    assert co2.chains.snapshot()[0]["status"] == "working"
    assert co2.exec_style["s1"]["AAPL"]["price"] == 99.0
    # the resting AAPL leg fills after the restart (recovered order, no owner record)
    ex2.ledger.strategy_positions.setdefault("s1", {})["AAPL"] = 100.0
    co2.chains.on_fill("AAPL")
    (h,) = _working(ex2, "MSFT")
    assert ex2.order_status[h]["pending_qty"] == -50


# ------------------------------------------------------------------ the endpoint
@pytest.fixture
def api(monkeypatch):
    import api.server as server
    ex, co = _make_executor()
    ex.exit_manager = None
    monkeypatch.setattr(server, "executor", ex)
    monkeypatch.setattr(server, "EXECUTOR_API_KEY", "k")
    import datetime as dt
    monkeypatch.setattr(server, "moc_cutoff_at",
                        lambda *a, **k: dt.datetime.fromtimestamp(4_000_000_000))
    return TestClient(server.app), ex, co


def _body(**kw):
    return {"strategy_id": "s1", "legs": [
        {"instrument": {"symbol": "AAPL"}, "target_quantity": 100, "expected_price": 200.0},
        {"instrument": {"symbol": "SPY"}, "target_quantity": -30, "expected_price": 600.0}],
        **kw}


def test_endpoint_prices_legs_with_the_atr_layer(api):
    client, ex, co = api
    r = client.post("/chains", json=_body(atr_fraction=1.0, ttl_sec=900),
                    headers={"X-API-Key": "k"}).json()
    assert r["accepted"], r
    lims = {l["symbol"]: l["limit_price"] for l in r["chain"]["legs"]}
    assert lims == {"AAPL": 198.0, "SPY": 602.0}          # ∓ 1.0 x ATR 2.0
    assert client.get("/chains", params={"strategy_id": "s1"}).json()["chains"][0]["id"] \
        == r["chain"]["id"]


def test_endpoint_refuses_when_atr_is_unavailable(api):
    client, ex, co = api
    ex.atr_layer._get_atr = lambda sym: None if sym == "SPY" else ATR
    r = client.post("/chains", json=_body(), headers={"X-API-Key": "k"}).json()
    assert not r["accepted"] and "ATR unavailable for SPY" in r["reason"]
    assert ex.sent == {}


def test_endpoint_refuses_past_the_cutoff(api, monkeypatch):
    import api.server as server
    import datetime as dt
    client, ex, co = api
    monkeypatch.setattr(server, "moc_cutoff_at",
                        lambda *a, **k: dt.datetime.fromtimestamp(1_000))
    r = client.post("/chains", json=_body(), headers={"X-API-Key": "k"}).json()
    assert not r["accepted"] and "cutoff" in r["reason"] and ex.sent == {}


def test_endpoint_cancel(api):
    client, ex, co = api
    r = client.post("/chains", json=_body(), headers={"X-API-Key": "k"}).json()
    c = client.delete(f"/chains/{r['chain']['id']}", headers={"X-API-Key": "k"}).json()
    assert c["status"] == "cancelled"
    assert client.delete("/chains/nope", headers={"X-API-Key": "k"}).status_code == 404
