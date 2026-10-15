"""
tests/test_event_logger.py

Unit tests for EventLogger. Uses pytest's tmp_path fixture so each test gets a
fresh throwaway database — no shared state, no cleanup needed, no dependence on
the real db/ directory or an IB connection.
"""

import pytest
import sys
import os

from logger.event_logger import EventLogger


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


def test_equity_history_downsamples_to_last_snapshot_per_bucket(db):
    # 3 days of per-minute snapshots for two strategies: the shape that OOM-killed /pnl/history
    from datetime import datetime, timedelta, timezone
    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
    n = 3 * 24 * 60
    for i in range(n):
        ts = (t0 + timedelta(minutes=i, seconds=0.5)).isoformat()
        for sid in ("a", "b"):
            db.log_equity(ts, sid, 0.0, float(i), float(i))

    raw = db.get_equity_history()
    assert len(raw) == 2 * n

    rows = db.get_equity_history(max_points=100)
    per = {sid: [r for r in rows if r["strategy_id"] == sid] for sid in ("a", "b")}
    assert all(90 <= len(v) <= 101 for v in per.values())
    assert [r["ts"] for r in rows] == sorted(r["ts"] for r in rows)
    # the newest snapshot always survives, so the chart's right edge is live
    assert per["a"][-1]["equity"] == float(n - 1)

    assert len(db.get_equity_history(strategy_id="a", max_points=100)) == len(per["a"])


def test_compact_equity_snapshots_thins_only_old_rows(db):
    from datetime import datetime, timedelta, timezone
    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
    for i in range(4 * 24 * 60):                        # 4 days, per minute, two strategies
        ts = (t0 + timedelta(minutes=i)).isoformat()
        for sid in ("a", "b"):
            db.log_equity(ts, sid, 0.0, 0.0, float(i))
    cutoff = t0 + timedelta(days=2)

    removed = db.compact_equity_snapshots(cutoff, bucket_sec=3600)
    rows = db.get_equity_history()
    old = [r for r in rows if r["ts"] < cutoff.isoformat()]
    new = [r for r in rows if r["ts"] >= cutoff.isoformat()]
    assert len(old) == 2 * 48                            # one per strategy per hour
    assert len(new) == 2 * 2 * 24 * 60                   # untouched
    assert removed == 2 * 2 * 24 * 60 - len(old)
    # the survivor of each hour is its last minute
    assert {r["equity"] for r in old if r["strategy_id"] == "a"} == {float(h * 60 + 59) for h in range(48)}
    assert db.compact_equity_snapshots(cutoff, bucket_sec=3600) == 0   # idempotent
