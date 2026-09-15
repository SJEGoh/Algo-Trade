"""How an exit reaches the broker, on the REAL CentralExecutor with only the IB socket stubbed.

  * A stop closes at market. The ATR layer turns market orders into limits resting below the
    price, waiting for a pullback — for an exit, that pullback is the move it is getting out
    of. Exits go out urgent, past every execution layer.
  * A futures position has to be priced as a future. Every mark used to be requested as a
    stock, so a futures stop could never have fired.
  * An exit acts only on a recent mark, tighter than the executor's general limit.
"""
import os
import time

import pytest

pytest.importorskip("ibapi")

from execution.central_execution import CentralExecutor
from execution.netting import NettingCoordinator

INST = {"symbol": "MSFT", "asset_class": "equity", "sec_type": "STK", "exchange": "SMART"}
CFG = {"s1": {"capital_allocation": 1e9, "max_drawdown": 0.5}}


class _Log:
    def __getattr__(self, name):
        return lambda *a, **k: None


@pytest.fixture
def ex():
    os.environ.setdefault("EXECUTOR_API_KEY", "x")
    x = CentralExecutor.__new__(CentralExecutor)
    CentralExecutor.__init__(x)
    x.risk_manager._config = CFG
    x.risk_manager._active_strategies = set(CFG)
    ids = iter(range(1, 1000))
    x.get_next_order_id = lambda *a, **k: next(ids)
    x.placed = []
    x.placeOrder = lambda oid, contract, order: x.placed.append(order)
    x.cancelOrder = lambda *a, **k: None
    x.logger_db = _Log()
    x.coordinator = NettingCoordinator(x, CFG)
    return x


@pytest.fixture
def atr_calls(ex):
    """ATR switched on for every strategy, turning whatever it sees into a resting limit."""
    calls = []
    ex.atr_layer.strategies = []

    def transform(intent):
        calls.append(intent)
        return {**intent, "order_type": "limit", "limit_price": 1.0,
                "metadata": {"atr_execution": True}}

    ex.atr_layer.transform = transform
    return calls


def holding(ex, qty=10.0):
    ex.ledger.current_positions["MSFT"] = qty
    ex.ledger.strategy_positions["s1"] = {"MSFT": qty}
    ex.coordinator.desired["s1"] = {"MSFT": qty}
    ex.coordinator.instrument["MSFT"] = INST
    ex.coordinator.ref_price["MSFT"] = 400.0


def test_an_urgent_exit_closes_at_market_past_atr(ex, atr_calls):
    holding(ex)
    ex.coordinator.set_target("s1", "MSFT", 0, urgent=True)
    assert atr_calls == []
    assert [o.orderType for o in ex.placed] == ["MKT"]
    assert ex.placed[0].action == "SELL" and ex.placed[0].totalQuantity == 10


def test_an_ordinary_target_change_still_goes_through_atr(ex, atr_calls):
    """The bypass is for exits only — rebalances keep their execution layer."""
    holding(ex)
    ex.coordinator.set_target("s1", "MSFT", 0)
    assert len(atr_calls) == 1
    assert [o.orderType for o in ex.placed] == ["LMT"]


def test_a_future_is_priced_as_a_future(ex):
    seen = {}

    def fetch(symbol, timeout=3.0, contract=None):
        seen[symbol] = contract
        return 70.0

    ex.fetch_price = fetch
    ex._instruments["CL"] = {"symbol": "CL", "sec_type": "FUT", "exchange": "NYMEX",
                             "multiplier": 1000, "last_trade_date": "202611"}
    ex.coordinator.instrument["ES"] = {"symbol": "ES", "sec_type": "FUT", "exchange": "CME",
                                       "multiplier": 50}       # known only from netting state
    marks = ex.get_marks(["CL", "ES", "AAPL"])
    assert marks == {"CL": 70.0, "ES": 70.0, "AAPL": 70.0}
    assert seen["CL"].secType == "FUT" and seen["CL"].exchange == "NYMEX"
    assert seen["CL"].lastTradeDateOrContractMonth == "202611"
    assert seen["ES"].secType == "FUT" and seen["ES"].exchange == "CME"
    assert seen["AAPL"] is None, "stocks keep the stock path"


def test_exits_can_demand_a_fresher_mark_than_the_general_limit(ex):
    ex._mark_cache["MSFT"] = 400.0
    ex._mark_ts["MSFT"] = time.time() - 90
    assert ex.mark_is_fresh("MSFT")                      # within the general 120s
    assert not ex.mark_is_fresh("MSFT", max_age=65.0)    # too old to exit on
