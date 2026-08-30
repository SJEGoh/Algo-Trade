import sqlite3
import threading
import logging
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

logger = logging.getLogger("executor")
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = MODULE_DIR / ".." / ".." / "db" / "executor.db"

class EventLogger:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        db_path = Path(db_path).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread = False)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id        INTEGER PRIMARY KEY,
                    client_order_id TEXT UNIQUE NOT NULL,
                    strategy_id     TEXT NOT NULL,
                    symbol          TEXT NOT NULL,
                    side            TEXT NOT NULL,
                    quantity        REAL NOT NULL,
                    order_type      TEXT NOT NULL,
                    limit_price     REAL,
                    expected_price  REAL,
                    status          TEXT NOT NULL,
                    submitted_at    TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS fills (
                    fill_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id       INTEGER NOT NULL,
                    exec_id        TEXT UNIQUE NOT NULL,
                    symbol         TEXT NOT NULL,
                    side           TEXT NOT NULL,
                    price          REAL NOT NULL,
                    expected_price REAL,
                    quantity       REAL NOT NULL,
                    strategy_id    TEXT NOT NULL,
                    filled_at      TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reconciliations (
                    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    matched       INTEGER NOT NULL,
                    discrepancies TEXT,
                    occurred_at   TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS risk_events (
                    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    event_type  TEXT NOT NULL,
                    detail      TEXT,
                    occurred_at TEXT NOT NULL
                );
            """)
            self._conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _execute(self, sql: str, params: tuple) -> None:
        """Single point for all writes: locked, committed, and failure-isolated.
        A DB error is logged but never raised — logging must not break trading."""
        try:
            with self._lock:
                self._conn.execute(sql, params)
                self._conn.commit()
        except Exception as e:
            logger.error("EventLogger write failed: %s | sql=%s", e, sql.split()[0])

    def log_order(self, order_id: int, intent: dict) -> None:
        instrument = intent["instrument"]
        now = self._now()
        self._execute(
            """INSERT OR IGNORE INTO orders
               (order_id, client_order_id, strategy_id, symbol, side, quantity,
                order_type, limit_price, expected_price, status, submitted_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (order_id, intent["client_order_id"], intent["strategy_id"], instrument["symbol"],
             intent["side"], intent["quantity"], intent["order_type"], intent.get("limit_price"),
             intent.get("expected_price"), "Submitted", now, now),
        )

    def update_order_status(self, order_id: int, status: str) -> None:
        self._execute(
            "UPDATE orders SET status = ?, updated_at = ? WHERE order_id = ?",
            (status, self._now(), order_id),
        )

    def log_fill(self, order_id: int, exec_id: str, symbol: str, side: str,
                 price: float, quantity: float, strategy_id: str,
                 expected_price: Optional[float] = None) -> None:
        self._execute(
            """INSERT OR IGNORE INTO fills
               (order_id, exec_id, symbol, side, price, expected_price, quantity, strategy_id, filled_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (order_id, exec_id, symbol, side, price, expected_price, quantity, strategy_id, self._now()),
        )

    def log_reconciliation(self, matched: bool, discrepancies: dict) -> None:
        self._execute(
            "INSERT INTO reconciliations (matched, discrepancies, occurred_at) VALUES (?, ?, ?)",
            (int(matched), str(discrepancies) if discrepancies else None, self._now()),
        )

    def log_risk_event(self, strategy_id: str, event_type: str, detail: str = "") -> None:
        self._execute(
            "INSERT INTO risk_events (strategy_id, event_type, detail, occurred_at) VALUES (?, ?, ?, ?)",
            (strategy_id, event_type, detail, self._now()),
        )

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.close()
        except Exception as e:
            logger.error("EventLogger close failed: %s", e)

    def get_order(self, order_id: int) -> Optional[dict]:
        """Look up a previously logged order by order_id, for recovery after a restart.
        Returns the original client_order_id / strategy_id / etc. that IB's openOrder
        callback can't give back, or None if not found."""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT order_id, client_order_id, strategy_id, symbol, side, "
                    "quantity, order_type, limit_price, expected_price, status "
                    "FROM orders WHERE order_id = ?",
                    (order_id,),
                ).fetchone()
        except Exception as e:
            logger.error("EventLogger get_order failed: %s", e)
            return None

        if row is None:
            return None

        return {
            "order_id": row[0],
            "client_order_id": row[1],
            "strategy_id": row[2],
            "symbol": row[3],
            "side": row[4],
            "quantity": row[5],
            "order_type": row[6],
            "limit_price": row[7],
            "expected_price": row[8],
            "status": row[9],
        }
