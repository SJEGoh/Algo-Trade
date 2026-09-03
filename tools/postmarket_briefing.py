#!/usr/bin/env python3
"""
tools/postmarket_briefing.py — Telegram post-market summary.

Fires ~10 min after the close (via day_scheduler). Sends a summary of:
  * Today's fills (what traded, at what price)
  * Realized P&L per strategy + total
  * End-of-day positions (the overnight book)
  * Strategy status (any halts during the day)
  * Reconciliation status

Talks to the executor server via REST; sends the message via Telegram directly.
"""
import os
import sys
from datetime import datetime
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


def build_summary():
    now = datetime.now(ET)
    lines = [f"\U0001f319 POST-MARKET SUMMARY — {now:%A, %b %d %Y}"]
    lines.append("")

    # Fills
    lines.append("\U0001f4dd TODAY'S FILLS")
    try:
        fills = _get("/fills?limit=200").get("fills", [])
        today_str = now.strftime("%Y-%m-%d")
        today_fills = [f for f in fills if f.get("timestamp", "").startswith(today_str)]
        if not today_fills:
            lines.append("  (no fills today)")
        else:
            lines.append(f"  {len(today_fills)} fill(s):")
            for f in today_fills:
                sym = f.get("symbol", "?")
                side = f.get("side", "?")
                qty = f.get("quantity", 0)
                price = f.get("fill_price", f.get("price", 0))
                sid = f.get("strategy_id", "")
                ts = f.get("timestamp", "")
                # extract time portion
                t_part = ts.split("T")[1][:8] if "T" in ts else ts
                lines.append(f"  {t_part}  {side.upper()} {abs(qty)} {sym} @ ${price:,.2f}  [{sid}]")
    except Exception as e:
        lines.append(f"  ⚠️ Could not fetch fills: {e}")

    # P&L
    lines.append("")
    lines.append("\U0001f4b0 REALIZED P&L")
    try:
        pnl = _get("/pnl").get("realized_pnl", {})
        total = 0.0
        any_pnl = False
        for sid, val in sorted(pnl.items()):
            if sid.startswith("test_suite") or sid.startswith("halt_test"):
                continue
            if val != 0:
                lines.append(f"  {sid}: ${val:+,.2f}")
                total += val
                any_pnl = True
        if any_pnl:
            emoji = "\U0001f7e2" if total >= 0 else "\U0001f534"
            lines.append(f"  ── {emoji} Total: ${total:+,.2f}")
        else:
            lines.append("  (no realized P&L)")
    except Exception as e:
        lines.append(f"  ⚠️ Could not fetch P&L: {e}")

    # Overnight book
    lines.append("")
    lines.append("\U0001f30d OVERNIGHT BOOK")
    try:
        pos = _get("/positions")
        net = pos.get("current_positions", {})
        strat_pos = pos.get("strategy_positions", {})
        held = {s: q for s, q in net.items() if q != 0}
        if not held:
            lines.append("  (flat — no overnight exposure)")
        else:
            for sym, qty in sorted(held.items()):
                lines.append(f"  {sym}: {qty:+}")
            # per-strategy breakdown
            for sid, positions in sorted(strat_pos.items()):
                if sid.startswith("test_suite") or sid.startswith("halt_test"):
                    continue
                strat_held = {s: q for s, q in positions.items() if q != 0}
                if strat_held:
                    parts = ", ".join(f"{s} {q:+}" for s, q in sorted(strat_held.items()))
                    lines.append(f"    [{sid}] {parts}")
    except Exception as e:
        lines.append(f"  ⚠️ Could not fetch positions: {e}")

    # Strategy status
    lines.append("")
    lines.append("\U0001f3af STRATEGY STATUS")
    try:
        strats = _get("/strategies").get("strategies", [])
        halted = []
        for s in strats:
            sid = s["strategy_id"]
            if sid.startswith("test_suite") or sid.startswith("halt_test"):
                continue
            if not s.get("active"):
                halted.append(sid)
        if halted:
            lines.append(f"  \U0001f6d1 HALTED: {', '.join(halted)}")
        else:
            lines.append("  ✅ All strategies active")
    except Exception as e:
        lines.append(f"  ⚠️ Could not fetch strategies: {e}")

    # Reconciliation
    lines.append("")
    try:
        recon = _get("/reconcile/status")
        if recon.get("ts"):
            matched = recon.get("matched", False)
            disc = recon.get("discrepancies", {})
            if matched:
                lines.append("✅ Last reconcile: ledger matches broker")
            else:
                lines.append(f"⚠️ Last reconcile: {len(disc)} discrepancy(ies)")
                for sym, detail in disc.items():
                    lines.append(f"    {sym}: {detail}")
        else:
            lines.append("ℹ️ No reconciliation ran today")
    except Exception:
        pass

    lines.append("")
    lines.append("Session complete \U0001f44b")
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
        print("Telegram summary sent OK")
    except Exception as e:
        print(f"Telegram send failed: {e}")
        print(text)


if __name__ == "__main__":
    msg = build_summary()
    print(msg)
    print()
    send_telegram(msg)
