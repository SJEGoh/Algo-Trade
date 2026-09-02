"""tests/test_netting.py — NettingCoordinator. Pure: real PositionLedger + RiskManager +
a fake executor that records net orders and lets the test simulate fills. No IBKR/numpy."""
from ledger.position_ledger import PositionLedger
from risk.risk_manager import RiskManager
from execution.netting import NettingCoordinator

INST = {"symbol": "X", "asset_class": "equity", "exchange": "SMART"}


class _FakeLoggerDB:
    def __init__(self):
        self.fills = []
    def log_fill(self, order_id, exec_id, symbol, side, price, quantity,
                 strategy_id, expected_price=None):
        self.fills.append({"order_id": order_id, "exec_id": exec_id,
                           "symbol": symbol, "side": side, "price": price,
                           "quantity": quantity, "strategy_id": strategy_id})


class FakeExecutor:
    def __init__(self, config):
        self.ledger = PositionLedger(None)
        self.risk_manager = RiskManager(self.ledger, config)
        self.logger_db = _FakeLoggerDB()
        self.placed = []

    def _cancel_open_orders_for_symbol(self, sym):
        pass  # tests fill promptly, so no stale in-flight orders

    def place_net_order(self, sym, delta, instrument, price):
        self.ledger.record_net_pending(sym, delta)
        oid = len(self.placed) + 1
        self.placed.append({"symbol": sym, "delta": delta, "price": price})
        return oid

    def fill_last(self, co, price=None):
        o = self.placed[-1]
        co.attribute_fill(o["symbol"], o["delta"], price if price is not None else o["price"])

    def fill_orders(self, co, orders):
        for o in orders:
            co.attribute_fill(o["symbol"], o["delta"], self.ref(o["symbol"]))

    def ref(self, sym):
        return next(p["price"] for p in reversed(self.placed) if p["symbol"] == sym)


def _cfg(**allocs):
    return {sid: {"capital_allocation": a, "max_drawdown": 0.2} for sid, a in allocs.items()}


def inst(sym):
    return {"symbol": sym, "asset_class": "equity", "exchange": "SMART"}


def test_single_strategy_nets_to_target():
    cfg = _cfg(s1=1e9)
    ex = FakeExecutor(cfg); co = NettingCoordinator(ex, cfg)
    r = co.set_target("s1", "MSFT", 100, instrument=inst("MSFT"), price=500)
    assert r["accepted"] and r["orders"][0]["delta"] == 100
    ex.fill_last(co)
    assert ex.ledger.current_positions["MSFT"] == 100
    assert ex.ledger.strategy_positions["s1"]["MSFT"] == 100
    assert abs(ex.ledger.effective_position("MSFT") - 100) < 1e-9   # net pending reversed


def test_offsetting_legs_net_and_attribute():
    cfg = _cfg(s1=1e9, s2=1e9)
    ex = FakeExecutor(cfg); co = NettingCoordinator(ex, cfg)
    ex.fill_last(co) if False else None
    co.set_target("s1", "MSFT", 100, instrument=inst("MSFT"), price=500); ex.fill_last(co)
    r = co.set_target("s2", "MSFT", -60, instrument=inst("MSFT"), price=500)
    assert r["orders"][0]["delta"] == -60          # net 40 - current 100
    ex.fill_last(co)
    assert ex.ledger.current_positions["MSFT"] == 40
    assert ex.ledger.strategy_positions["s1"]["MSFT"] == 100
    assert ex.ledger.strategy_positions["s2"]["MSFT"] == -60
    # invariant
    net = sum(ex.ledger.strategy_positions[s].get("MSFT", 0) for s in ("s1", "s2"))
    assert net == ex.ledger.current_positions["MSFT"]


def test_rerun_same_target_is_noop():
    cfg = _cfg(s1=1e9)
    ex = FakeExecutor(cfg); co = NettingCoordinator(ex, cfg)
    co.set_target("s1", "MSFT", 100, instrument=inst("MSFT"), price=500); ex.fill_last(co)
    r = co.set_target("s1", "MSFT", 100, instrument=inst("MSFT"), price=500)
    assert r["orders"] == []                        # already at target -> no order


def test_resync_closes_a_stale_position():
    """The stale-exit trap: incremental strategy stops mentioning MSFT; a full-book
    resync WITHOUT MSFT must close it."""
    cfg = _cfg(s1=1e9)
    ex = FakeExecutor(cfg); co = NettingCoordinator(ex, cfg)
    co.set_target("s1", "MSFT", 100, instrument=inst("MSFT"), price=500); ex.fill_last(co)
    # strategy goes flat on MSFT but never sent MSFT=0; it resyncs its full book (AAPL only)
    r = co.submit_book("s1", [{"instrument": inst("AAPL"), "target_quantity": 10, "expected_price": 200}])
    deltas = {o["symbol"]: o["delta"] for o in r["orders"]}
    assert deltas["MSFT"] == -100                    # stale MSFT closed
    assert deltas["AAPL"] == 10
    ex.fill_orders(co, r["orders"])
    assert ex.ledger.current_positions.get("MSFT", 0) == 0
    assert ex.ledger.strategy_positions["s1"].get("MSFT", 0) == 0


def test_halt_unwinds_and_attributes_to_the_strategy():
    cfg = _cfg(s1=1e9, s2=1e9)
    ex = FakeExecutor(cfg); co = NettingCoordinator(ex, cfg)
    co.set_target("s1", "MSFT", 100, instrument=inst("MSFT"), price=500); ex.fill_last(co)
    co.set_target("s2", "MSFT", 50, instrument=inst("MSFT"), price=500); ex.fill_last(co)  # net 150
    result = co.halt("s1")
    orders = result["orders"]
    assert orders[0]["delta"] == -100                # net 50 - current 150
    ex.fill_orders(co, orders)
    assert ex.ledger.current_positions["MSFT"] == 50
    assert ex.ledger.strategy_positions["s1"]["MSFT"] == 0   # s1 unwound on its own book
    assert ex.ledger.strategy_positions["s2"]["MSFT"] == 50


def test_allocation_rejected_and_reverted():
    cfg = _cfg(s1=10_000)                             # tiny cap
    ex = FakeExecutor(cfg); co = NettingCoordinator(ex, cfg)
    r = co.set_target("s1", "MSFT", 100, instrument=inst("MSFT"), price=500)  # 50k > 10k
    assert not r["accepted"]
    assert co.desired.get("s1", {}).get("MSFT") is None      # reverted, no order
    assert ex.placed == []


def test_halted_strategy_cannot_submit():
    cfg = _cfg(s1=1e9)
    ex = FakeExecutor(cfg); co = NettingCoordinator(ex, cfg)
    ex.risk_manager.halt_strategy("s1", "test")
    r = co.set_target("s1", "MSFT", 100, instrument=inst("MSFT"), price=500)
    assert not r["accepted"] and "not active" in r["reason"]


def test_partial_fill_pro_rata():
    cfg = _cfg(s1=1e9, s2=1e9)
    ex = FakeExecutor(cfg); co = NettingCoordinator(ex, cfg)
    co.set_target("s1", "MSFT", 80, instrument=inst("MSFT"), price=500)
    co.set_target("s2", "MSFT", 40, instrument=inst("MSFT"), price=500)  # net 120, one order 120
    # simulate a HALF fill of the last net order (60 of 120)
    co.attribute_fill("MSFT", 60, 500)
    # 60/120 of each strategy's change: s1 +40, s2 +20
    assert abs(ex.ledger.strategy_positions["s1"]["MSFT"] - 40) < 1e-9
    assert abs(ex.ledger.strategy_positions["s2"]["MSFT"] - 20) < 1e-9
    assert abs(ex.ledger.current_positions["MSFT"] - 60) < 1e-9
