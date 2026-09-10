"""What an order record has to remember about HOW it was worked.

`order_status` recorded the symbol, the quantity and a status, but never the order type or
the limit. On the dashboard that made a resting ATR limit indistinguishable from a market
order the broker had not filled yet — both simply "Submitted" — so an order sitting away
from the market looked like a slow fill rather than a price that was never going to trade.
"""
import pytest

import execution.central_execution as ce
from risk.risk_manager import RiskManager

CFG = {"s1": {"capital_allocation": 1_000_000.0, "max_drawdown": 0.20}}


class FakeDB:
    def __init__(self, *a, **kw): pass
    def __getattr__(self, _name):            # every log_/save_ call is a no-op
        return lambda *a, **kw: None
    def load_strategy_positions(self): return {}, {}
    def load_realized_pnl(self): return {}
    def load_multipliers(self): return {}
    def load_strategy_cash(self): return {}, {}
    def load_halted_strategies(self): return set()


@pytest.fixture
def ex(monkeypatch):
    """A real executor with only the IB socket stubbed, so the real place_order runs."""
    monkeypatch.setattr(ce, "EventLogger", FakeDB)
    x = ce.CentralExecutor()
    x.logger_db = FakeDB()
    x.risk_manager = RiskManager(x.ledger, CFG, {})
    x._alerter = None
    x._next_id = 100
    x.sent = []

    def next_id(timeout=5.0):
        x._next_id += 1
        return x._next_id
    monkeypatch.setattr(x, "get_next_order_id", next_id)
    monkeypatch.setattr(x, "placeOrder", lambda oid, contract, order: x.sent.append(order))
    monkeypatch.setattr(x, "_paper_subscribe", lambda *a, **kw: None)
    return x


def intent(order_type="market", **kw):
    base = {
        "client_order_id": "c1", "strategy_id": "s1",
        "instrument": {"symbol": "AAPL", "asset_class": "equity",
                       "sec_type": "STK", "exchange": "SMART"},
        "side": "buy", "quantity": 10, "order_type": order_type,
        "time_in_force": "day", "expected_price": 200.0,
    }
    base.update(kw)
    return base


# ------------------------------------------------------------------ direct orders
def test_a_market_order_records_its_type(ex):
    oid = ex.place_order(intent())
    assert ex.order_status[oid]["order_type"] == "market"
    assert ex.order_status[oid]["limit_price"] is None


def test_a_limit_order_records_the_price_it_was_set_to(ex):
    oid = ex.place_order(intent("limit", limit_price=198.25))
    rec = ex.order_status[oid]
    assert rec["order_type"] == "limit"
    assert rec["limit_price"] == pytest.approx(198.25)
    # and the limit actually reached IB, not just the record
    assert ex.sent[-1].orderType == "LMT"
    assert ex.sent[-1].lmtPrice == pytest.approx(198.25)


def test_an_atr_transformed_order_is_identifiable_as_one(ex):
    """The case that motivated this. Without `execution_layer` the dashboard cannot say
    whether a limit came from a strategy or from the ATR layer re-pricing a market order."""
    oid = ex.place_order(intent("limit", limit_price=197.10,
                                metadata={"execution_layer": {"atr": 1.8,
                                                              "reference": 200.0}}))
    rec = ex.order_status[oid]
    assert rec["execution_layer"]["atr"] == 1.8
    assert rec["limit_price"] == pytest.approx(197.10)


def test_the_send_time_is_recorded(ex):
    import time
    before = time.time()
    oid = ex.place_order(intent())
    assert before <= ex.order_status[oid]["sent_at"] <= time.time()


# ------------------------------------------------------------------ pooled orders
def test_a_pooled_market_order_records_its_type(ex):
    oid = ex.place_net_order("AAPL", -10, {"symbol": "AAPL", "sec_type": "STK",
                                           "exchange": "SMART"}, 200.0, urgent=True)
    rec = ex.order_status[oid]
    assert rec["order_type"] == "market"
    assert rec["limit_price"] is None
    assert rec["net"] is True


def test_a_pooled_order_transformed_by_atr_keeps_the_limit(ex, monkeypatch):
    """place_net_order builds its own intent and may hand it to the ATR layer, so the fields
    have to be read back AFTER the transform, not from the intent that went in."""
    def fake_transform(i):
        out = dict(i)
        out["order_type"] = "limit"
        out["limit_price"] = 196.5
        out["metadata"] = {"execution_layer": {"atr": 2.0}}
        return out
    monkeypatch.setattr(ex.atr_layer, "transform", fake_transform)
    monkeypatch.setattr(ex.atr_layer, "strategies", [])       # empty = applies to all

    oid = ex.place_net_order("AAPL", 10, {"symbol": "AAPL", "sec_type": "STK",
                                          "exchange": "SMART"}, 200.0, urgent=False)
    rec = ex.order_status[oid]
    assert rec["order_type"] == "limit"
    assert rec["limit_price"] == pytest.approx(196.5)
    assert rec["execution_layer"]["atr"] == 2.0


def test_an_urgent_pooled_order_is_never_re_priced(ex, monkeypatch):
    """A flatten or a kill must close at market. If ATR could re-price it, the closing order
    would rest away from the market exactly when getting out matters."""
    monkeypatch.setattr(ex.atr_layer, "transform",
                        lambda i: (_ for _ in ()).throw(AssertionError("ATR ran on urgent")))
    oid = ex.place_net_order("AAPL", -10, {"symbol": "AAPL", "sec_type": "STK",
                                           "exchange": "SMART"}, 200.0, urgent=True)
    assert ex.order_status[oid]["order_type"] == "market"
