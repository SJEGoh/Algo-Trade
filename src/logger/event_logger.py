import sqlite3
import threading
import time
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

    def _init_schema(self, attempts: int = 3, delay: float = 2.0) -> None:
        """Create any missing tables, retrying a locked database.

        `docker compose up` can start the new executor while the old one still holds the
        DB on the shared volume; schema statements take a write lock, so a single attempt
        turns that race into a failed startup ("database is locked" -> container Error)."""
        for attempt in range(1, attempts + 1):
            try:
                self._create_tables()
                break
            except Exception as e:
                logger.warning("schema init attempt %d/%d failed: %s", attempt, attempts, e)
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                if attempt == attempts:
                    raise
                time.sleep(delay)
        # Columns added after the first release — migrated separately because, unlike the
        # CREATE TABLE IF NOT EXISTS statements above, ALTER TABLE must never be able to
        # stop the executor from starting.
        self._has_nav_cols = self._migrate_equity_nav_columns()

    def _create_tables(self) -> None:
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
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    snap_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    realized    REAL NOT NULL,
                    unrealized  REAL NOT NULL,
                    equity      REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots(ts);

                -- Persistent strategy state: survives server restarts so the dashboard
                -- shows positions / avg cost / realized P&L immediately.
                CREATE TABLE IF NOT EXISTS strategy_state (
                    strategy_id TEXT NOT NULL,
                    symbol      TEXT NOT NULL,
                    quantity    REAL NOT NULL,
                    avg_cost    REAL NOT NULL,
                    updated_at  TEXT NOT NULL,
                    PRIMARY KEY (strategy_id, symbol)
                );
                CREATE TABLE IF NOT EXISTS strategy_pnl (
                    strategy_id TEXT PRIMARY KEY,
                    realized    REAL NOT NULL,
                    updated_at  TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS halted_strategies (
                    strategy_id TEXT PRIMARY KEY,
                    halted_at   TEXT NOT NULL,
                    reason      TEXT
                );
                -- Cash held as a position by each strategy (basis = capital at inception)
                CREATE TABLE IF NOT EXISTS strategy_cash (
                    strategy_id   TEXT PRIMARY KEY,
                    cash          REAL NOT NULL,
                    starting_cash REAL NOT NULL,
                    updated_at    TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS strategy_multipliers (
                    symbol     TEXT PRIMARY KEY,
                    multiplier REAL NOT NULL
                );

                -- Trade/decision journal: every signal, weight, and rebalance decision
                CREATE TABLE IF NOT EXISTS decision_journal (
                    entry_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    event_type  TEXT NOT NULL,
                    summary     TEXT,
                    detail      TEXT,
                    symbols     TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_journal_ts ON decision_journal(ts);
                CREATE INDEX IF NOT EXISTS idx_journal_strat ON decision_journal(strategy_id);
            """)
            self._conn.commit()

    NAV_COLUMNS = ("cash", "position_value", "nav")

    def _migrate_equity_nav_columns(self, attempts: int = 3, delay: float = 1.0) -> bool:
        """Add cash / position_value / nav to an existing equity_snapshots table.

        NEVER fatal. A concurrent writer (a still-shutting-down executor, the scheduler,
        a tool) holds the write lock and this raises `database is locked` — losing the NAV
        columns is a degraded dashboard, but failing here would take down trading. On
        failure we fall back to writing the legacy P&L columns (the dashboard rebases rows
        with no NAV onto each strategy's capital basis) and retry on the next restart."""
        for attempt in range(1, attempts + 1):
            try:
                with self._lock:
                    cols = {r[1] for r in self._conn.execute("PRAGMA table_info(equity_snapshots)")}
                    missing = [c for c in self.NAV_COLUMNS if c not in cols]
                    if not missing:
                        return True
                    for col in missing:
                        self._conn.execute(f"ALTER TABLE equity_snapshots ADD COLUMN {col} REAL")
                    self._conn.commit()
                logger.info("equity_snapshots migrated: added %s", ", ".join(missing))
                return True
            except Exception as e:
                logger.warning("equity_snapshots NAV migration attempt %d/%d failed: %s",
                               attempt, attempts, e)
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                if attempt < attempts:
                    time.sleep(delay)
        logger.error("equity_snapshots NAV migration failed — logging P&L columns only "
                     "(dashboard will rebase); retried on next restart")
        return False

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

    def get_recent_fills(self, limit: int = 50) -> list:
        """Most-recent fills first, for the dashboard fills/slippage panel."""
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT order_id, exec_id, symbol, side, price, expected_price, "
                    "quantity, strategy_id, filled_at "
                    "FROM fills ORDER BY fill_id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        except Exception as e:
            logger.error("EventLogger get_recent_fills failed: %s", e)
            return []
        cols = ["order_id", "exec_id", "symbol", "side", "price",
                "expected_price", "quantity", "strategy_id", "filled_at"]
        return [dict(zip(cols, r)) for r in rows]

    def log_equity(self, ts, strategy_id, realized, unrealized, equity,
                   cash=None, position_value=None, nav=None) -> None:
        """`equity` is P&L (realized + unrealized); `nav` is the balance-sheet value
        (cash + position_value). Both are stored so the dashboard can plot either — unless
        the NAV migration couldn't run, in which case only the P&L columns are written."""
        if not self._has_nav_cols:
            self._execute(
                "INSERT INTO equity_snapshots (ts, strategy_id, realized, unrealized, equity) "
                "VALUES (?, ?, ?, ?, ?)",
                (ts, strategy_id, realized, unrealized, equity),
            )
            return
        self._execute(
            "INSERT INTO equity_snapshots "
            "(ts, strategy_id, realized, unrealized, equity, cash, position_value, nav) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, strategy_id, realized, unrealized, equity, cash, position_value, nav),
        )

    # ------------------------------------------------------------------
    # Strategy state persistence (positions, avg cost, realized P&L, halts)
    # ------------------------------------------------------------------
    def save_strategy_positions(self, strategy_positions: dict, strategy_avg_cost: dict) -> None:
        """Atomically snapshot all per-strategy positions + avg costs."""
        now = self._now()
        try:
            with self._lock:
                self._conn.execute("DELETE FROM strategy_state")
                for sid, positions in strategy_positions.items():
                    costs = strategy_avg_cost.get(sid, {})
                    for sym, qty in positions.items():
                        if qty != 0.0:
                            self._conn.execute(
                                "INSERT INTO strategy_state (strategy_id, symbol, quantity, avg_cost, updated_at) "
                                "VALUES (?, ?, ?, ?, ?)",
                                (sid, sym, qty, costs.get(sym, 0.0), now),
                            )
                self._conn.commit()
        except Exception as e:
            logger.error("save_strategy_positions failed: %s", e)

    def save_realized_pnl(self, strategy_realized_pnl: dict) -> None:
        now = self._now()
        try:
            with self._lock:
                self._conn.execute("DELETE FROM strategy_pnl")
                for sid, pnl in strategy_realized_pnl.items():
                    self._conn.execute(
                        "INSERT INTO strategy_pnl (strategy_id, realized, updated_at) VALUES (?, ?, ?)",
                        (sid, pnl, now),
                    )
                self._conn.commit()
        except Exception as e:
            logger.error("save_realized_pnl failed: %s", e)

    def save_strategy_cash(self, strategy_cash: dict, starting_cash: dict) -> None:
        """Snapshot every strategy's cash position + capital basis."""
        now = self._now()
        try:
            with self._lock:
                for sid, cash in strategy_cash.items():
                    self._conn.execute(
                        "INSERT INTO strategy_cash (strategy_id, cash, starting_cash, updated_at) "
                        "VALUES (?, ?, ?, ?) ON CONFLICT(strategy_id) DO UPDATE SET "
                        "cash = excluded.cash, starting_cash = excluded.starting_cash, "
                        "updated_at = excluded.updated_at",
                        (sid, cash, starting_cash.get(sid, cash), now),
                    )
                self._conn.commit()
        except Exception as e:
            logger.error("save_strategy_cash failed: %s", e)

    def load_strategy_cash(self) -> tuple:
        """Returns (strategy_cash, starting_cash) dicts."""
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT strategy_id, cash, starting_cash FROM strategy_cash"
                ).fetchall()
        except Exception as e:
            logger.error("load_strategy_cash failed: %s", e)
            return {}, {}
        return ({sid: cash for sid, cash, _ in rows},
                {sid: basis for sid, _, basis in rows})

    def save_halted_strategies(self, halted: set, active: set, config_keys: set, reason: str = "") -> None:
        """Save which strategies are halted (= in config but NOT in the active set)."""
        now = self._now()
        try:
            with self._lock:
                self._conn.execute("DELETE FROM halted_strategies")
                for sid in config_keys - active:
                    self._conn.execute(
                        "INSERT INTO halted_strategies (strategy_id, halted_at, reason) VALUES (?, ?, ?)",
                        (sid, now, reason),
                    )
                self._conn.commit()
        except Exception as e:
            # CRITICAL (-> Telegram): an unpersisted halt is resurrected on the next
            # restart, and the strategy resumes trading as if it never breached.
            logger.critical("save_halted_strategies FAILED — halt will not survive a "
                            "restart: %s", e)

    def save_multipliers(self, multipliers: dict) -> None:
        try:
            with self._lock:
                self._conn.execute("DELETE FROM strategy_multipliers")
                for sym, mult in multipliers.items():
                    if mult != 1.0:
                        self._conn.execute(
                            "INSERT INTO strategy_multipliers (symbol, multiplier) VALUES (?, ?)",
                            (sym, mult),
                        )
                self._conn.commit()
        except Exception as e:
            logger.error("save_multipliers failed: %s", e)

    def load_strategy_positions(self) -> tuple:
        """Returns (strategy_positions, strategy_avg_cost) dicts."""
        positions = {}
        avg_cost = {}
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT strategy_id, symbol, quantity, avg_cost FROM strategy_state"
                ).fetchall()
        except Exception as e:
            logger.error("load_strategy_positions failed: %s", e)
            return positions, avg_cost
        for sid, sym, qty, cost in rows:
            positions.setdefault(sid, {})[sym] = qty
            avg_cost.setdefault(sid, {})[sym] = cost
        return positions, avg_cost

    def load_realized_pnl(self) -> dict:
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT strategy_id, realized FROM strategy_pnl"
                ).fetchall()
        except Exception as e:
            logger.error("load_realized_pnl failed: %s", e)
            return {}
        return {sid: pnl for sid, pnl in rows}

    def load_halted_strategies(self) -> set:
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT strategy_id FROM halted_strategies"
                ).fetchall()
        except Exception as e:
            logger.error("load_halted_strategies failed: %s", e)
            return set()
        return {row[0] for row in rows}

    def load_multipliers(self) -> dict:
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT symbol, multiplier FROM strategy_multipliers"
                ).fetchall()
        except Exception as e:
            logger.error("load_multipliers failed: %s", e)
            return {}
        return {sym: mult for sym, mult in rows}

    # ------------------------------------------------------------------
    # Trade / decision journal
    # ------------------------------------------------------------------
    def log_decision(self, strategy_id: str, event_type: str,
                     summary: str, detail: str = "", symbols: list = None) -> None:
        """Append one entry to the decision journal.
        event_type examples: 'signal', 'rebalance', 'internal_cross', 'halt', 'reactivate'.
        detail is a JSON string (or free text) with the full context; symbols is a
        comma-separated list of tickers involved."""
        import json as _json
        sym_str = ",".join(symbols) if symbols else None
        # Truncate detail to 10 KB to avoid bloating the DB with huge signal dumps
        if isinstance(detail, dict):
            detail = _json.dumps(detail, default=str)
        if detail and len(detail) > 10_000:
            detail = detail[:10_000] + "...(truncated)"
        self._execute(
            "INSERT INTO decision_journal (ts, strategy_id, event_type, summary, detail, symbols) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (self._now(), strategy_id, event_type, summary, detail, sym_str),
        )

    def get_journal(self, strategy_id: str = None, event_type: str = None,
                    since: str = None, limit: int = 100) -> list:
        """Query the decision journal with optional filters."""
        q = "SELECT entry_id, ts, strategy_id, event_type, summary, detail, symbols FROM decision_journal"
        conds, params = [], []
        if strategy_id: conds.append("strategy_id = ?"); params.append(strategy_id)
        if event_type:  conds.append("event_type = ?");  params.append(event_type)
        if since:       conds.append("ts >= ?");         params.append(since)
        if conds: q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY entry_id DESC LIMIT ?"
        params.append(limit)
        try:
            with self._lock:
                rows = self._conn.execute(q, tuple(params)).fetchall()
        except Exception as e:
            logger.error("get_journal failed: %s", e)
            return []
        cols = ["entry_id", "ts", "strategy_id", "event_type", "summary", "detail", "symbols"]
        return [dict(zip(cols, r)) for r in rows]

    def get_equity_history(self, strategy_id=None, since=None) -> list:
        cols = ["ts", "strategy_id", "realized", "unrealized", "equity"]
        if self._has_nav_cols:
            cols += list(self.NAV_COLUMNS)
        q = f"SELECT {', '.join(cols)} FROM equity_snapshots"
        conds, params = [], []
        if strategy_id: conds.append("strategy_id = ?"); params.append(strategy_id)
        if since:       conds.append("ts >= ?");         params.append(since)
        if conds: q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY ts ASC"
        try:
            with self._lock:
                rows = self._conn.execute(q, tuple(params)).fetchall()
        except Exception as e:
            logger.error("get_equity_history failed: %s", e)
            return []
        return [dict(zip(cols, r)) for r in rows]
