#!/usr/bin/env python3
"""
tools/seed_equity.py — seed synthetic equity_snapshots to test the equity curve.

Writes N minutes of 1-min snapshots ending now, for a couple of demo strategies,
so the dashboard's /pnl/history has something to plot before any real fills exist.

Re-run to replace demo rows (clears strategy_id LIKE 'demo_%' first).
Clear only:   python3 tools/seed_equity.py --clear
"""
import sqlite3, random, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "db" / "executor.db"
N = 180  # minutes of history
STRATS = [
    ("demo_momentum", 12.0, 140.0, 500_000.0),   # id, drift/min, vol, starting cash -> trends up
    ("demo_meanrev", -4.0, 260.0, 500_000.0),    # choppy, slight down
]

DDL = """
CREATE TABLE IF NOT EXISTS equity_snapshots (
    snap_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    realized    REAL NOT NULL,
    unrealized  REAL NOT NULL,
    equity      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots(ts);
"""

# NAV columns (cash as a position) — added after the first release, so seed into an
# existing db the same way EventLogger migrates it.
NAV_COLS = ("cash", "position_value", "nav")

def main():
    conn = sqlite3.connect(str(DB))
    conn.executescript(DDL)
    have = {r[1] for r in conn.execute("PRAGMA table_info(equity_snapshots)")}
    for col in NAV_COLS:
        if col not in have:
            conn.execute(f"ALTER TABLE equity_snapshots ADD COLUMN {col} REAL")
    conn.execute("DELETE FROM equity_snapshots WHERE strategy_id LIKE 'demo_%'")
    if "--clear" in sys.argv:
        conn.commit(); conn.close()
        print("cleared demo_% equity rows"); return

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    rows = []
    for idx, (sid, drift, vol, start_cash) in enumerate(STRATS):
        random.seed(1000 + idx)                      # deterministic
        realized = 0.0
        for i in range(N):
            ts = (now - timedelta(minutes=(N - 1 - i))).isoformat()
            realized += drift + random.gauss(0, vol * 0.4)
            unrealized = random.gauss(0, vol)
            equity = realized + unrealized
            # Balance sheet: cash carries the realized P&L, positions carry the mark-to-market,
            # so nav = cash + position_value = starting cash + equity (same identity as live).
            position_value = round(vol * 20 + unrealized, 2)
            cash = round(start_cash + realized - vol * 20, 2)
            rows.append((ts, sid, round(realized, 2), round(unrealized, 2), round(equity, 2),
                         cash, position_value, round(cash + position_value, 2)))
    conn.executemany(
        "INSERT INTO equity_snapshots "
        "(ts, strategy_id, realized, unrealized, equity, cash, position_value, nav) "
        "VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM equity_snapshots WHERE strategy_id LIKE 'demo_%'").fetchone()[0]
    conn.close()
    print(f"seeded {len(rows)} rows across {len(STRATS)} demo strategies ({total} demo rows in table)")

if __name__ == "__main__":
    main()
