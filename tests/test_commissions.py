"""IB commissions reach realized P&L, cash, NAV and the drawdown halts.

Nothing handled IB's commissionReport callback, so every figure the executor produced —
realized P&L, cash, NAV, the P&L the drawdown halts read — was gross of fees. On a tight
limit that is not a rounding error: the halt_test_* strategies stop at -$100, and a day of
round trips can spend a real share of that on commissions alone.

A fee comes out of cash AND realized P&L, which keeps  nav == starting_cash + realized +
unrealized  intact. It is charged to the strategies the fill was booked to, pro rata.
"""
import sqlite3

import pytest
from fastapi.testclient import TestClient
from ibapi.commission_report import CommissionReport
from ibapi.contract import Contract
from ibapi.execution import Execution

import api.server as server
import execution.central_execution as ce
from execution.netting import NettingCoordinator
from ledger.position_ledger import PositionLedger
from logger.event_logger import EventLogger
from risk.risk_manager import RiskManager

UNSET = 1.7976931348623157e308          # IB's "no value yet"
CFG = {s: {"capital_allocation": 1e9, "max_drawdown": 0.5} for s in ("s1", "s2")}
ONE = {"s1": {"capital_allocation": 10_000.0, "max_drawdown": 0.2}}


# ------------------------------------------------------------------ ledger
def test_a_fee_comes_out_of_cash_and_realized_pnl_together():
    led = PositionLedger(None, ONE)
    led.record_pending("AAPL", 10, "s1")
    led.record_fill("AAPL", 10, 100.0, "s1")
    led.apply_fee("s1", 1.5)

    snap = led.equity_snapshot({"AAPL": 105.0})["s1"]
    assert snap["realized"] == pytest.approx(-1.5)
    assert snap["cash"] == pytest.approx(10_000 - 1_000 - 1.5)
    assert snap["nav"] == pytest.approx(
        snap["starting_cash"] + snap["realized"] + snap["unrealized"]), "NAV identity broken"
    assert led.strategy_fees["s1"] == pytest.approx(1.5)


def test_fees_accumulate():
    led = PositionLedger(None, ONE)
    led.apply_fee("s1", 1.0)
    led.apply_fee("s1", 0.25)
    assert led.strategy_fees["s1"] == pytest.approx(1.25)
    assert led.strategy_realized_pnl["s1"] == pytest.approx(-1.25)


def test_a_zero_fee_changes_nothing():
    led = PositionLedger(None, ONE)
    led.apply_fee("s1", 0.0)
    assert "s1" not in led.strategy_fees
    assert led.strategy_realized_pnl.get("s1", 0.0) == 0.0


# ------------------------------------------------------------------ storage
def test_an_existing_fills_table_gains_fee_columns(tmp_path):
    path = tmp_path / "legacy.db"
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE fills (
        fill_id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER NOT NULL,
        exec_id TEXT UNIQUE NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,
        price REAL NOT NULL, expected_price REAL, quantity REAL NOT NULL,
        strategy_id TEXT NOT NULL, filled_at TEXT NOT NULL)""")
    con.execute("INSERT INTO fills (order_id, exec_id, symbol, side, price, quantity, "
                "strategy_id, filled_at) VALUES (1, 'old', 'AAPL', 'BOT', 100, 10, 's1', "
                "'2026-09-01T00:00:00+00:00')")
    con.commit()
    con.close()

    db = EventLogger(db_path=path)
    cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(fills)")}
    assert {"commission", "commission_currency"} <= cols
    # an old row's fee is unknown, not zero
    assert db.get_recent_fills(5)[0]["commission"] is None


def test_a_commission_is_attached_to_its_fill(tmp_path):
    db = EventLogger(db_path=tmp_path / "t.db")
    db.log_fill(7, "e7", "AAPL", "BOT", 100.0, 10, "s1", expected_price=100.0)
    db.log_commission("e7", 1.25, "USD")
    row = db.get_recent_fills(1)[0]
    assert row["commission"] == pytest.approx(1.25)
    assert row["commission_currency"] == "USD"


def test_fee_totals_survive_a_restart(tmp_path):
    path = tmp_path / "t.db"
    led = PositionLedger(None, ONE)
    led.apply_fee("s1", 2.5)
    db = EventLogger(db_path=path)
    led.save_state(db)
    db.close()

    again = PositionLedger(None, ONE)
    again.restore_state(EventLogger(db_path=path))
    assert again.strategy_fees["s1"] == pytest.approx(2.5)
    assert again.strategy_realized_pnl["s1"] == pytest.approx(-2.5)


# ------------------------------------------------------------------ executor
class FakeDB:
    def __init__(self, *a, **kw):
        self.commissions = {}

    def __getattr__(self, _name):
        return lambda *a, **kw: None

    def log_commission(self, exec_id, commission, currency):
        self.commissions[exec_id] = (commission, currency)

    def load_strategy_positions(self): return {}, {}
    def load_realized_pnl(self): return {}
    def load_multipliers(self): return {}
    def load_strategy_cash(self): return {}, {}
    def load_halted_strategies(self): return set()


def make_ex(monkeypatch, cfg=CFG):
    monkeypatch.setattr(ce, "EventLogger", FakeDB)
    x = ce.CentralExecutor()
    x.logger_db = FakeDB()
    x.risk_manager = RiskManager(x.ledger, cfg, {})
    x._alerter = None
    counter = [100]

    def next_id(*a, **k):
        counter[0] += 1
        return counter[0]

    monkeypatch.setattr(x, "get_next_order_id", next_id)
    monkeypatch.setattr(x, "placeOrder", lambda oid, c, o: None)
    monkeypatch.setattr(x, "_paper_subscribe", lambda *a, **kw: None)
    monkeypatch.setattr(x, "cancelOrder", lambda oid, *a, **kw: None)
    return x


def order(x, sid="s1", qty=10, price=100.0, sym="AAPL"):
    return x.place_order({
        "client_order_id": f"c{len(x.order_status)}-{sid}", "strategy_id": sid,
        "instrument": {"symbol": sym, "asset_class": "equity", "sec_type": "STK",
                       "exchange": "SMART"},
        "side": "buy" if qty > 0 else "sell", "quantity": abs(qty),
        "order_type": "market", "time_in_force": "day", "expected_price": price})


def fill(x, oid, shares, price, exec_id):
    st = x.order_status[oid]
    c = Contract(); c.symbol = st["symbol"]
    e = Execution()
    e.orderId = oid; e.execId = exec_id; e.shares = shares; e.price = price
    e.side = "BOT" if st["pending_qty"] > 0 else "SLD"
    x.execDetails(1, c, e)


def report(exec_id, fee, currency="USD"):
    r = CommissionReport()
    r.execId = exec_id; r.commission = fee; r.currency = currency
    return r


def test_a_commission_is_charged_to_the_strategy_that_traded(monkeypatch):
    x = make_ex(monkeypatch)
    oid = order(x)
    fill(x, oid, 10, 100.0, "e1")
    x.commissionReport(report("e1", 1.0))

    assert x.ledger.strategy_realized_pnl["s1"] == pytest.approx(-1.0)
    assert x.ledger.strategy_fees["s1"] == pytest.approx(1.0)
    assert x.logger_db.commissions["e1"] == (1.0, "USD")


def test_a_commission_that_beats_its_fill_is_held_then_charged(monkeypatch):
    x = make_ex(monkeypatch)
    oid = order(x)
    x.commissionReport(report("e1", 1.0))
    assert x.ledger.strategy_fees.get("s1", 0.0) == 0.0
    fill(x, oid, 10, 100.0, "e1")
    assert x.ledger.strategy_fees["s1"] == pytest.approx(1.0)


def test_a_replayed_commission_is_charged_once(monkeypatch):
    x = make_ex(monkeypatch)
    oid = order(x)
    fill(x, oid, 10, 100.0, "e1")
    x.commissionReport(report("e1", 1.0))
    x.commissionReport(report("e1", 1.0))
    assert x.ledger.strategy_fees["s1"] == pytest.approx(1.0)


def test_ibs_placeholder_is_never_charged_but_the_real_report_still_is(monkeypatch):
    x = make_ex(monkeypatch)
    oid = order(x)
    fill(x, oid, 10, 100.0, "e1")
    x.commissionReport(report("e1", UNSET))
    assert x.ledger.strategy_fees.get("s1", 0.0) == 0.0
    assert x.ledger.strategy_realized_pnl.get("s1", 0.0) == 0.0

    x.commissionReport(report("e1", 1.0))
    assert x.ledger.strategy_fees["s1"] == pytest.approx(1.0)


def test_a_non_usd_commission_is_stored_but_not_deducted(monkeypatch):
    """No FX rate here, and deducting euros as dollars would be quietly wrong."""
    x = make_ex(monkeypatch)
    oid = order(x)
    fill(x, oid, 10, 100.0, "e1")
    x.commissionReport(report("e1", 2.0, "EUR"))
    assert x.ledger.strategy_fees.get("s1", 0.0) == 0.0
    assert x.logger_db.commissions["e1"] == (2.0, "EUR")


def test_a_pooled_fill_splits_its_commission_across_the_strategies_it_was_booked_to(monkeypatch):
    """One 40-share execution booked s1 +100 and s2 -60. IB charged for 40 shares; each
    strategy bears it in proportion to what it was booked: 100/160 and 60/160."""
    x = make_ex(monkeypatch)
    x.coordinator = NettingCoordinator(x, CFG)
    inst = {"symbol": "AAPL", "asset_class": "equity", "exchange": "SMART"}
    x.coordinator.set_target("s1", "AAPL", 100, instrument=inst, price=100)
    x.coordinator.set_target("s2", "AAPL", -60, instrument=inst, price=100)
    (oid,) = [o for o, st in x.order_status.items() if st.get("status") == "Submitted"]

    fill(x, oid, 40, 100.0, "n1")
    x.commissionReport(report("n1", 2.0))

    assert x.ledger.strategy_fees["s1"] == pytest.approx(1.25)
    assert x.ledger.strategy_fees["s2"] == pytest.approx(0.75)
    assert x.logger_db.commissions["n1"] == (2.0, "USD")
    assert x.logger_db.commissions["n1-attr-s1-0"][0] == pytest.approx(1.25)
    assert x.logger_db.commissions["n1-attr-s2-1"][0] == pytest.approx(0.75)


def test_a_fee_counts_toward_a_drawdown_halt(monkeypatch):
    """-$95 of trading loss is inside a $100 limit. The $10 commission on the closing trade
    is what crosses it — and a check that ignored fees would never halt."""
    tight = {"s1": {"capital_allocation": 10_000.0, "max_drawdown": 0.01}}
    x = make_ex(monkeypatch, tight)
    buy = order(x, qty=10, price=100.0)
    fill(x, buy, 10, 100.0, "b1")
    sell = order(x, qty=-10, price=90.5)
    fill(x, sell, 10, 90.5, "x1")
    assert x.ledger.strategy_realized_pnl["s1"] == pytest.approx(-95.0)
    assert x.risk_manager.is_active("s1")

    x.commissionReport(report("x1", 10.0))
    assert not x.risk_manager.is_active("s1"), "fees crossed the limit but it was not halted"


def test_pnl_reports_fees_and_net_realized(monkeypatch):
    x = make_ex(monkeypatch)
    oid = order(x)
    fill(x, oid, 10, 100.0, "e1")
    x.commissionReport(report("e1", 1.0))
    monkeypatch.setattr(server, "executor", x)

    body = TestClient(server.app).get("/pnl").json()
    assert body["fees"] == {"s1": 1.0}
    assert body["realized_pnl"]["s1"] == pytest.approx(-1.0)
