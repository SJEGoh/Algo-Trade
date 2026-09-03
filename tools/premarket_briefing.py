#!/usr/bin/env python3
"""
tools/premarket_briefing.py — Telegram pre-market briefing.

Fires ~30 min before the open (via day_scheduler). Sends a summary of:
  * Today's session times (open/close, half-day flag)
  * Current positions (net + per-strategy)
  * Strategy status (active / halted)
  * Today's scheduled events
  * ATR execution layer status

Talks to the executor server via REST; sends the message via Telegram directly
(not through the Alerter class, since this is a structured briefing, not a
one-line alert).
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

ET = ZoneInfo("America/New_York")
BASE = os.environ.get("EXECUTOR_URL", "http://127.0.0.1:8000")
KEY = os.environ.get("EXECUTOR_API_KEY", "")
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID")


def _get(path):
    r = requests.get(f"{BASE}{path}", headers={"X-API-Key": KEY}, timeout=10)
    r.raise_for_status()
    return r.json()


def _session_times():
    try:
        import pandas_market_calendars as mcal
        now = datetime.now(ET)
        sched = mcal.get_calendar("NYSE").schedule(start_date=now.date(), end_date=now.date())
        if sched.empty:
            return None, None
        o = sched.iloc[0]["market_open"].tz_convert(ET).to_pydatetime()
        c = sched.iloc[0]["market_close"].tz_convert(ET).to_pydatetime()
        return o, c
    except Exception:
        return None, None


def build_briefing():
    now = datetime.now(ET)
    lines = [f"\U0001f305 PRE-MARKET BRIEFING — {now:%A, %b %d %Y}"]
    lines.append("")

    # Session times
    o, c = _session_times()
    if o and c:
        duration = (c - o).total_seconds() / 3600
        half = " (HALF DAY)" if duration < 6 else ""
        lines.append(f"\U0001f552 Session: {o:%H:%M} – {c:%H:%M} ET{half}")
    else:
        lines.append("⚠️ No NYSE session today")
        return "\n".join(lines)

    # Health
    try:
        health = _get("/health")
        ib_status = "✅ Connected" if health.get("connected") else "❌ Disconnected"
        if health.get("killed"):
            ib_status += " \U0001f6d1 KILL SWITCH ACTIVE"
        lines.append(f"\U0001f4e1 IB Gateway: {ib_status}")
    except Exception as e:
        lines.append(f"❌ Server unreachable: {e}")
        return "\n".join(lines)

    # Strategies
    lines.append("")
    lines.append("\U0001f3af STRATEGIES")
    try:
        strats = _get("/strategies").get("strategies", [])
        for s in strats:
            sid = s["strategy_id"]
            # skip test/halt-test strategies
            if sid.startswith("test_suite") or sid.startswith("halt_test"):
                continue
            status = "✅" if s.get("active") else "\U0001f6d1 HALTED"
            alloc = s.get("capital_allocation", 0)
            dd = s.get("max_drawdown", 0)
            lines.append(f"  {status} {sid}  (${alloc:,.0f} / {dd:.0%} max DD)")
    except Exception as e:
        lines.append(f"  ⚠️ Could not fetch strategies: {e}")

    # Positions
    lines.append("")
    lines.append("\U0001f4ca POSITIONS")
    try:
        pos = _get("/positions")
        net = pos.get("current_positions", {})
        strat_pos = pos.get("strategy_positions", {})
        if not net:
            lines.append("  (flat — no open positions)")
        else:
            for sym, qty in sorted(net.items()):
                if qty != 0:
                    lines.append(f"  {sym}: {qty:+}")
            # per-strategy breakdown
            for sid, positions in sorted(strat_pos.items()):
                if sid.startswith("test_suite") or sid.startswith("halt_test"):
                    continue
                held = {s: q for s, q in positions.items() if q != 0}
                if held:
                    parts = ", ".join(f"{s} {q:+}" for s, q in sorted(held.items()))
                    lines.append(f"    [{sid}] {parts}")
    except Exception as e:
        lines.append(f"  ⚠️ Could not fetch positions: {e}")

    # P&L
    lines.append("")
    lines.append("\U0001f4b0 REALIZED P&L")
    try:
        pnl = _get("/pnl").get("realized_pnl", {})
        total = 0.0
        for sid, val in sorted(pnl.items()):
            if sid.startswith("test_suite") or sid.startswith("halt_test"):
                continue
            if val != 0:
                lines.append(f"  {sid}: ${val:+,.2f}")
                total += val
        if total != 0:
            lines.append(f"  ── Total: ${total:+,.2f}")
        else:
            lines.append("  (no realized P&L)")
    except Exception as e:
        lines.append(f"  ⚠️ Could not fetch P&L: {e}")

    # ATR layer
    try:
        atr = _get("/atr/status")
        if atr.get("enabled"):
            lines.append("")
            lines.append(f"\U0001f4c9 ATR Layer: ON (period={atr['atr_period']}, "
                         f"fraction={atr['atr_fraction']}, "
                         f"pending={atr.get('pending_orders', 0)})")
    except Exception:
        pass

    # Schedule preview
    lines.append("")
    lines.append(f"\U0001f4c5 TODAY'S SCHEDULE")
    dow = now.strftime("%A")
    lines.append(f"  {o:%H:%M}        Market open")
    lines.append(f"  {o + timedelta(minutes=30):%H:%M}  ORB first fire (then every 30 min)")
    lines.append(f"  {o + timedelta(minutes=60):%H:%M}  ovn_volsurge EXIT + momentum rebalance")
    if dow == "Thursday":
        lines.append(f"  {c - timedelta(minutes=10):%H:%M}  RRG rotation (Thursday)")
    lines.append(f"  {c - timedelta(minutes=5):%H:%M}   ATR cancel sweep")
    lines.append(f"  {c - timedelta(minutes=2):%H:%M}   ovn_volsurge ENTER")
    lines.append(f"  {c:%H:%M}        Market close")
    lines.append(f"  {c + timedelta(minutes=5):%H:%M}   VECM EOD run")
    lines.append(f"  {c + timedelta(minutes=10):%H:%M}  Post-market summary")

    lines.append("")
    lines.append("Good trading \U0001f44a")
    return "\n".join(lines)


def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        print("WARN: Telegram not configured — printing to stdout only")
        print(text)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text},
            timeout=10,
        )
        print("Telegram briefing sent OK")
    except Exception as e:
        print(f"Telegram send failed: {e}")
        print(text)


if __name__ == "__main__":
    msg = build_briefing()
    print(msg)
    print()
    send_telegram(msg)
