"""
tests/test_event_logger.py

Unit tests for EventLogger. Uses pytest's tmp_path fixture so each test gets a
fresh throwaway database — no shared state, no cleanup needed, no dependence on
the real db/ directory or an IB connection.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "execution"))
from event_logger import EventLogger


@pytest.fixture
def db(tmp_path):
    """Fresh EventLogger backed by a temp db file, unique per test."""
    return EventLogger(db_path=tmp_path / "test.db")


def _order_intent(client_order_id="c1", strategy_id="s1", symbol="AAPL",
                  side="buy", quantity=10, order_type="market",
                  limit_price=None, expected_price=100.0):
    return {
        "client_order_id": client_order_id,
        "strategy_id": strategy_id,
        "instrument": {"symbol": symbol},
        "side": side,
        "quantity": quantity,
        "order_type": order_type,
        "limit_price": limit_price,
        "expected_price": expected_price,
    }


# --- schema / construction ---

def test_tables_created(db):
    tables = {row[0] for row in db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert {"orders", "fills", "reconciliations", "risk_events"} <= tables


# --- orders ---

def test_log_order_inserts_row(db):
    db.log_order(1, _order_intent())
    row = db._conn.execute(
        "SELECT order_id, client_order_id, strategy_id, symbol, side, quantity, "
        "order_type, expected_price, status FROM orders WHERE order_id = 1"
    ).fetchone()
    assert row == (1, "c1", "s1", "AAPL", "buy", 10.0, "market", 100.0, "Submitted")


def test_log_order_stores_limit_price(db):
    db.log_order(2, _order_intent(order_type="limit", limit_price=95.0, expected_price=None))
    row = db._conn.execute("SELECT limit_price, expected_price FROM orders WHERE order_id = 2").fetchone()
    assert row == (95.0, None)


def test_duplicate_client_order_id_ignored(db):
    db.log_order(1, _order_intent(client_order_id="dup"))
    db.log_order(2, _order_intent(client_order_id="dup"))  # same client_order_id, different order_id
    count = db._conn.execute("SELECT COUNT(*) FROM orders WHERE client_order_id = 'dup'").fetchone()[0]
    assert count == 1  # INSERT OR IGNORE prevented the second


def test_update_order_status(db):
    db.log_order(1, _order_intent())
    db.update_order_status(1, "Filled")
    status = db._conn.execute("SELECT status FROM orders WHERE order_id = 1").fetchone()[0]
    assert status == "Filled"


def test_update_order_status_touches_updated_at(db):
    db.log_order(1, _order_intent())
    before = db._conn.execute("SELECT submitted_at, updated_at FROM orders WHERE order_id = 1").fetchone()
    db.update_order_status(1, "Filled")
    after = db._conn.execute("SELECT submitted_at, updated_at FROM orders WHERE order_id = 1").fetchone()
    assert after[0] == before[0]        # submitted_at unchanged
    assert after[1] >= before[1]        # updated_at moved forward (ISO strings sort chronologically)


# --- fills ---

def test_log_fill_inserts_row(db):
    db.log_fill(1, "exec1", "AAPL", "BOT", 100.5, 10, "s1", expected_price=100.0)
    row = db._conn.execute(
        "SELECT order_id, exec_id, symbol, side, price, expected_price, quantity, strategy_id "
        "FROM fills WHERE exec_id = 'exec1'"
    ).fetchone()
    assert row == (1, "exec1", "AAPL", "BOT", 100.5, 100.0, 10.0, "s1")


def test_log_fill_without_expected_price(db):
    db.log_fill(1, "exec2", "AAPL", "BOT", 100.0, 10, "s1")  # expected_price defaults to None
    ep = db._conn.execute("SELECT expected_price FROM fills WHERE exec_id = 'exec2'").fetchone()[0]
    assert ep is None


def test_duplicate_exec_id_ignored(db):
    db.log_fill(1, "exec1", "AAPL", "BOT", 100.0, 10, "s1")
    db.log_fill(1, "exec1", "AAPL", "BOT", 100.0, 10, "s1")  # IB occasionally resends a fill
    count = db._conn.execute("SELECT COUNT(*) FROM fills WHERE exec_id = 'exec1'").fetchone()[0]
    assert count == 1


def test_slippage_query(db):
    # the actual point of storing expected + actual price: slippage in bps
    db.log_fill(1, "e1", "AAPL", "BOT", 101.0, 10, "s1", expected_price=100.0)  # +100 bps
    db.log_fill(2, "e2", "AAPL", "BOT", 100.5, 10, "s1", expected_price=100.0)  # +50 bps
    avg_bps = db._conn.execute(
        "SELECT AVG((price - expected_price) / expected_price * 10000) "
        "FROM fills WHERE strategy_id = 's1' AND expected_price IS NOT NULL"
    ).fetchone()[0]
    assert avg_bps == pytest.approx(75.0)  # (100 + 50) / 2


# --- reconciliations ---

def test_log_reconciliation_matched(db):
    db.log_reconciliation(True, {})
    row = db._conn.execute("SELECT matched, discrepancies FROM reconciliations").fetchone()
    assert row[0] == 1          # bool stored as int
    assert row[1] is None       # empty discrepancies stored as NULL


def test_log_reconciliation_mismatch(db):
    disc = {"AAPL": {"internal": 100, "broker": 90}}
    db.log_reconciliation(False, disc)
    row = db._conn.execute("SELECT matched, discrepancies FROM reconciliations").fetchone()
    assert row[0] == 0
    assert "AAPL" in row[1]     # stored as str(dict)


# --- risk events ---

def test_log_risk_event(db):
    db.log_risk_event("s1", "drawdown_breach", "15% >= 10%")
    row = db._conn.execute(
        "SELECT strategy_id, event_type, detail FROM risk_events"
    ).fetchone()
    assert row == ("s1", "drawdown_breach", "15% >= 10%")


# --- failure isolation ---

def test_write_failure_does_not_raise(db):
    # a bad write (missing NOT NULL column) must be swallowed, not raised —
    # logging must never crash the trading path
    db._execute("INSERT INTO orders (order_id) VALUES (?)", (999,))  # missing required cols
    # if we got here without an exception, failure isolation worked
    count = db._conn.execute("SELECT COUNT(*) FROM orders WHERE order_id = 999").fetchone()[0]
    assert count == 0  # the bad insert failed silently, no row written


def test_close_is_safe(db):
    db.log_order(1, _order_intent())
    db.close()  # should not raise
