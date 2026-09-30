#!/usr/bin/env python3
"""
tools/reset_history.py — start the executor from scratch, e.g. on a new IB account.

Moves every piece of persisted trading state out of db/ into db/archive/<UTC timestamp>/:

    executor.db (+ -wal / -shm)   orders, fills, P&L, fees, cash, positions, allocations,
                                  halts, exits, journal, equity curve, runtime strategies
    netting.json                  desired books, pooled working orders, chains
    vecm_state.json, rrg_state.json   strategy-side state

Nothing is deleted — to undo, stop the executor and move the files back. telegram_offset.json
stays put: it only tracks which Telegram updates have been read.

Run it with the executor STOPPED. The executor holds all of this in memory and writes it back,
so a reset under a running executor is overwritten at its next save. The tool refuses while
$EXECUTOR_URL/health answers.

Usage:
    python3 tools/reset_history.py --dry-run            # show what would move, change nothing
    python3 tools/reset_history.py                      # asks you to type "reset"
    python3 tools/reset_history.py --keep-strategies    # carry /addstrategy registrations over
    python3 tools/reset_history.py --db-dir /app/db --yes    # inside the container, no prompt

On the next start the executor builds a fresh executor.db and every strategy begins with
config.CONFIG's capital allocation as its starting cash.
"""
import argparse, os, shutil, sqlite3, sys, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_DIR = _ROOT / "db"

DB_FILES = ("executor.db", "executor.db-wal", "executor.db-shm")
STATE_FILES = DB_FILES + ("netting.json", "vecm_state.json", "rrg_state.json")


def executor_running(url: str) -> bool:
    """Any HTTP answer at all means something is serving on that port."""
    try:
        urllib.request.urlopen(url.rstrip("/") + "/health", timeout=3)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def summarize(db_path: Path) -> dict:
    """What is about to be archived, read-only. Empty when there is no database."""
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = {}

    def q(sql, default=None):
        try:
            return conn.execute(sql).fetchall()
        except sqlite3.Error:
            return default

    for table in ("orders", "fills", "decision_journal", "equity_snapshots"):
        rows = q(f"SELECT COUNT(*) FROM {table}")
        if rows:
            out[table] = rows[0][0]
    out["open_positions"] = q(
        "SELECT strategy_id, symbol, quantity FROM strategy_state WHERE quantity != 0 "
        "ORDER BY strategy_id, symbol", []) or []
    out["realized"] = q("SELECT strategy_id, realized FROM strategy_pnl ORDER BY strategy_id", []) or []
    out["runtime_strategies"] = q(
        "SELECT strategy_id, capital_allocation, max_drawdown, starting_cash, created_by "
        "FROM runtime_strategies ORDER BY strategy_id", []) or []
    conn.close()
    return out


def checkpoint(db_path: Path) -> None:
    """Fold the WAL into executor.db so the archived copy is complete on its own."""
    if not db_path.exists():
        return
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def restore_runtime_strategies(db_path: Path, rows: list) -> None:
    sys.path.insert(0, str(_ROOT / "src"))
    from logger.event_logger import EventLogger
    db = EventLogger(db_path=db_path)
    try:
        for sid, alloc, dd, cash, created_by in rows:
            db.save_runtime_strategy(sid, alloc, dd, starting_cash=cash,
                                     created_by=created_by or "reset_history")
    finally:
        db.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Archive all executor history and start fresh.")
    ap.add_argument("--db-dir", default=os.environ.get("EXECUTOR_DB_DIR", str(DEFAULT_DB_DIR)),
                    help="state directory (default: repo db/; /app/db in the container)")
    ap.add_argument("--executor-url", default=os.environ.get("EXECUTOR_URL", "http://127.0.0.1:8000"),
                    help="checked to make sure the executor is stopped")
    ap.add_argument("--keep-strategies", action="store_true",
                    help="re-register strategies added at runtime (/addstrategy) in the new db")
    ap.add_argument("--dry-run", action="store_true", help="report only, move nothing")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--force", action="store_true",
                    help="skip the executor-running check (only if /health is something else)")
    args = ap.parse_args()

    db_dir = Path(args.db_dir).resolve()
    db_path = db_dir / "executor.db"
    present = [name for name in STATE_FILES if (db_dir / name).exists()]
    if not present:
        print(f"nothing to reset in {db_dir}")
        return 0

    if not args.force and executor_running(args.executor_url):
        print(f"REFUSED: the executor is answering at {args.executor_url}. Stop it first "
              "(systemctl stop executor / docker compose stop) — a running executor rewrites "
              "its state from memory and would undo the reset.")
        return 1

    info = summarize(db_path)
    print(f"state directory: {db_dir}")
    for table in ("orders", "fills", "decision_journal", "equity_snapshots"):
        if table in info:
            print(f"  {table:<18} {info[table]:>8} rows")
    for sid, realized in info.get("realized", []):
        print(f"  realized P&L       {sid}: {realized:,.2f}")
    if info.get("runtime_strategies"):
        names = ", ".join(r[0] for r in info["runtime_strategies"])
        verb = "kept" if args.keep_strategies else "DROPPED (use --keep-strategies to keep)"
        print(f"  runtime strategies {names} — {verb}")
    if info.get("open_positions"):
        print("\nWARNING: the ledger still shows open positions. After the reset the executor "
              "will not know any strategy owns them; if the account still holds them they "
              "come back as orphans:")
        for sid, sym, qty in info["open_positions"]:
            print(f"  {sid:<28} {sym:<8} {qty:g}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = db_dir / "archive" / stamp
    print(f"\nwill move to {archive}:")
    for name in present:
        if name not in DB_FILES[1:]:     # the WAL is folded into executor.db first
            print(f"  {name}")

    if args.dry_run:
        print("\n--dry-run: nothing moved")
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            print("REFUSED: not a terminal — pass --yes to confirm")
            return 1
        if input('\ntype "reset" to continue: ').strip() != "reset":
            print("aborted, nothing moved")
            return 1

    checkpoint(db_path)
    archive.mkdir(parents=True, exist_ok=False)
    for name in STATE_FILES:          # re-list: the checkpoint can remove -wal / -shm
        src = db_dir / name
        if src.exists():
            shutil.move(str(src), str(archive / name))

    if args.keep_strategies and info.get("runtime_strategies"):
        restore_runtime_strategies(db_path, info["runtime_strategies"])
        print(f"re-registered {len(info['runtime_strategies'])} runtime strategies in a new executor.db")

    print(f"\narchived to {archive}")
    print("start the executor — it begins with an empty ledger and CONFIG allocations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
