#!/usr/bin/env python3
"""
tools/day_scheduler.py — one-shot intraday scheduler. Start it once at (or just before) the
market open and leave it running; it fires each strategy at the right time through the day,
then exits after the EOD VECM run.

It just calls your existing runners as subprocesses — run_equity.py / run_strat.py /
run_vecm.py — which POST to the server. With the single-front-door change in place, the
equity target_position intents pool through the NettingCoordinator automatically; VECM
futures go direct. So this script adds scheduling only; it changes no trading logic.

Times are derived from TODAY's actual NYSE session (via pandas_market_calendars), so
holidays and half-days are handled. On a non-trading day it just exits.

    uvicorn src.api.server:app --host 127.0.0.1 --port 8000     # (server must be up first)
    python3 tools/day_scheduler.py            # run the day
    python3 tools/day_scheduler.py --dry-run  # print the plan and exit (no server needed)

Ctrl-C stops it cleanly. It is safe to restart mid-day: any event already >GRACE_MIN in the
past is skipped, the rest still fire.

Built for UNATTENDED operation:
  * orb_breakout fires in RESYNC mode — an authoritative full-book submit (/targets) every
    cycle, so any name it no longer holds is closed (no stale-exit trap left open overnight).
  * a periodic RECONCILE (every RECONCILE_EVERY_MIN) pulls broker positions and logs any
    drift between the ledger and the account, so a discrepancy is captured while you sleep.
  * the pooled endpoints are guarded (kill-switch + market-hours), so nothing can misfire
    while the market is closed.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import requests
from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

ET = ZoneInfo("America/New_York")
BASE = os.environ.get("EXECUTOR_URL", "http://127.0.0.1:8000")
KEY = os.environ.get("EXECUTOR_API_KEY")

# ---- what to run (flip any to False to skip) --------------------------------
ENABLE = {
    "momentum":     True,   # cross_sectional_momentum — daily rebalance after the open
    "ovn_volsurge": True,   # overnight vol-surge — exit at the open, enter near the close
    "orb_breakout": True,   # intraday opening-range breakout — every ORB_EVERY_MIN
    "vecm":         True,   # Kalman VECM (futures) — once after the close
}
ORB_EVERY_MIN = 30              # cadence of the ORB runner through the session
ORB_START_AFTER_OPEN_MIN = 30   # wait for the opening range to form before the first ORB
MOMENTUM_AFTER_OPEN_MIN = 60     # let the open settle before the daily momentum rebalance
ENTER_BEFORE_CLOSE_MIN = 2      # ovn_volsurge enter, near the close (today's volume ~complete)
EXIT_AFTER_OPEN_MIN = 60         # ovn_volsurge exit, at the open
VECM_AFTER_CLOSE_MIN = 5        # VECM EOD run, after the close
ORB_STOP_BEFORE_CLOSE_MIN = 15  # last ORB fire this long before the close
RECONCILE_EVERY_MIN = 60        # POST /reconcile this often through the session (0 disables)
GRACE_MIN = 10                  # run an event up to this late; older -> skip


def session_today():
    """(open, close) as tz-aware ET datetimes for today's NYSE session, or (None, None)."""
    import pandas_market_calendars as mcal
    now = datetime.now(ET)
    sched = mcal.get_calendar("NYSE").schedule(start_date=now.date(), end_date=now.date())
    if sched.empty:
        return None, None
    o = sched.iloc[0]["market_open"].tz_convert(ET).to_pydatetime()
    c = sched.iloc[0]["market_close"].tz_convert(ET).to_pydatetime()
    return o, c


def build_events(o, c):
    """List of (when_ET, label, kind, payload) sorted by time.
    kind='run' -> payload is [script, *args] run as a subprocess;
    kind='reconcile' -> payload is None (POST /reconcile)."""
    ev = []
    R = lambda name: str(_ROOT / name)
    if ENABLE["ovn_volsurge"]:
        ev.append((o + timedelta(minutes=EXIT_AFTER_OPEN_MIN),
                   "ovn_volsurge exit (flatten overnight book)", "run",
                   [R("run_equity.py"), "ovn_volsurge", "exit"]))
    if ENABLE["momentum"]:
        ev.append((o + timedelta(minutes=MOMENTUM_AFTER_OPEN_MIN),
                   "momentum rebalance", "run", [R("run_strat.py")]))
    if ENABLE["orb_breakout"]:
        t = o + timedelta(minutes=ORB_START_AFTER_OPEN_MIN)
        last = c - timedelta(minutes=ORB_STOP_BEFORE_CLOSE_MIN)
        while t <= last:
            # resync = authoritative full-book submit (closes stale names each cycle)
            ev.append((t, "orb_breakout resync", "run",
                       [R("run_equity.py"), "orb_breakout", "resync"]))
            t += timedelta(minutes=ORB_EVERY_MIN)
    if ENABLE["ovn_volsurge"]:
        ev.append((c - timedelta(minutes=ENTER_BEFORE_CLOSE_MIN),
                   "ovn_volsurge enter (buy volume-surge names for overnight)", "run",
                   [R("run_equity.py"), "ovn_volsurge", "enter"]))
    if ENABLE["vecm"]:
        ev.append((c + timedelta(minutes=VECM_AFTER_CLOSE_MIN),
                   "VECM EOD run", "run", [R("run_vecm.py")]))
    if RECONCILE_EVERY_MIN > 0:
        t = o + timedelta(minutes=RECONCILE_EVERY_MIN)
        while t < c:
            ev.append((t, "reconcile (broker vs ledger)", "reconcile", None))
            t += timedelta(minutes=RECONCILE_EVERY_MIN)
        ev.append((c + timedelta(minutes=VECM_AFTER_CLOSE_MIN + 2),
                   "reconcile (post-close)", "reconcile", None))
    ev.sort(key=lambda e: e[0])
    return ev


def preflight():
    try:
        h = requests.get(f"{BASE}/health", timeout=10).json()
    except Exception as e:
        sys.exit(f"ABORT: server not reachable at {BASE} — start uvicorn first ({e})")
    if not h.get("connected"):
        sys.exit(f"ABORT: IB not connected — {h}")
    if h.get("killed"):
        sys.exit("ABORT: kill switch is active — refusing to schedule a trading day")
    if RECONCILE_EVERY_MIN > 0 and not KEY:
        print("WARN: EXECUTOR_API_KEY not set — reconcile calls will fail (set it in .env)")
    print(f"server OK at {BASE}  (market_open={h.get('market_open')})")


def run_event(label, cmd):
    stamp = datetime.now(ET).strftime("%H:%M:%S")
    print(f"\n[{stamp} ET] ▶ {label}\n    $ {' '.join(cmd)}")
    try:
        p = subprocess.run([sys.executable, *cmd], cwd=str(_ROOT),
                           capture_output=True, text=True, timeout=300)
        for ln in (p.stdout or "").strip().splitlines()[-8:]:
            print("    " + ln)
        if p.returncode != 0:
            print(f"    !! exit {p.returncode}: {(p.stderr or '').strip()[-400:]}")
    except subprocess.TimeoutExpired:
        print("    !! timed out after 300s")
    except Exception as e:
        print(f"    !! error: {e}")


def run_reconcile():
    stamp = datetime.now(ET).strftime("%H:%M:%S")
    print(f"\n[{stamp} ET] ▶ reconcile (broker vs ledger)")
    try:
        r = requests.post(f"{BASE}/reconcile", headers={"X-API-Key": KEY}, timeout=30)
        j = r.json()
        disc = j.get("discrepancies") or {}
        if j.get("matched"):
            print("    ledger matches broker ✓")
        else:
            print(f"    !! DRIFT corrected — {len(disc)} discrepancy(ies): {disc}")
    except Exception as e:
        print(f"    !! reconcile error: {e}")


def sleep_until(when):
    """Sleep in short chunks so Ctrl-C stays responsive."""
    while True:
        remaining = (when - datetime.now(ET)).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(15.0, remaining))


def main():
    ap = argparse.ArgumentParser(description="One-shot intraday strategy scheduler.")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit (no server needed)")
    args = ap.parse_args()

    o, c = session_today()
    if o is None:
        print(f"{datetime.now(ET):%Y-%m-%d} is not an NYSE trading day — nothing to schedule.")
        return
    events = build_events(o, c)
    now = datetime.now(ET)

    print(f"\n\033[1mDAY SCHEDULE\033[0m  {now:%Y-%m-%d}  (now {now:%H:%M:%S} ET)")
    print(f"session: open {o:%H:%M} ET  close {c:%H:%M} ET")
    print("-" * 62)
    for when, label, _kind, _payload in events:
        delta = (when - now).total_seconds()
        tag = "past→skip" if delta < -GRACE_MIN * 60 else ("late→run" if delta < 0 else "")
        print(f"  {when:%H:%M} ET  {label}   {tag}")
    print("-" * 62)

    if args.dry_run:
        print("(dry run — not executing)")
        return

    preflight()
    print("\nrunning… (Ctrl-C to stop)\n")
    try:
        for when, label, kind, payload in events:
            delta = (when - datetime.now(ET)).total_seconds()
            if delta < -GRACE_MIN * 60:
                print(f"[skip] {label} — was {(-delta/60):.0f} min ago")
                continue
            if delta > 0:
                print(f"[wait] next: {label} at {when:%H:%M} ET "
                      f"(in {delta/60:.0f} min)")
                sleep_until(when)
            if kind == "reconcile":
                run_reconcile()
            else:
                run_event(label, payload)
        print(f"\n[{datetime.now(ET):%H:%M:%S} ET] day complete — all scheduled events done.")
    except KeyboardInterrupt:
        print("\nstopped by user — no further events will fire.")


if __name__ == "__main__":
    main()
