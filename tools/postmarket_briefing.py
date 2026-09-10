#!/usr/bin/env python3
"""
tools/postmarket_briefing.py — Telegram post-market summary.

Fires ~10 min after the close (via day_scheduler):
  * Today's fills — what traded, when, at what price, and slippage vs expectation
  * Realized P&L per strategy, with fixtures and unattributed buckets kept separate
  * The overnight book, including anything no strategy claims
  * Halts, and the last reconciliation
"""
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))
load_dotenv(_ROOT / ".env")

from briefing_common import (ET, HEDGE_SID, INTERNAL_SIDS, et, get,  # noqa: E402
                             is_strategy, is_today_et, money, pnl_sections,
                             send_telegram)


def build_summary():
    now = datetime.now(ET)
    L = [f"\U0001f319 POST-MARKET SUMMARY — {now:%A, %b %d %Y}", ""]

    # ---------------------------------------------------------------- fills
    L.append("\U0001f4dd TODAY'S FILLS")
    try:
        fills = get("/fills?limit=500").get("fills", [])
        # The fill record's timestamp field is `filled_at`, and it is UTC. This used to read
        # `timestamp` — a key that does not exist — so the filter matched nothing and the
        # summary reported "(no fills today)" every single day regardless of what traded.
        today = [f for f in fills if is_today_et(f.get("filled_at"), now)]
        if not today:
            L.append("  (no fills today)")
        else:
            L.append(f"  {len(today)} fill(s):")
            for f in sorted(today, key=lambda x: x.get("filled_at") or ""):
                px, exp = f.get("price"), f.get("expected_price")
                slip = ""
                if px and exp:
                    # signed so a positive number always means "worse than expected"
                    sign = 1 if str(f.get("side", "")).upper().startswith("B") else -1
                    slip = f"  slip {sign * (px - exp) / exp * 100:+.2f}%"
                L.append(f"  {et(f.get('filled_at'))}  {str(f.get('side','?')).upper()} "
                         f"{abs(f.get('quantity', 0)):g} {f.get('symbol','?')} "
                         f"@ ${px or 0:,.2f}  [{f.get('strategy_id','?')}]{slip}")
            L.append("  note: pooled fills booked to __net__ are filtered out by /fills")
    except Exception as e:
        L.append(f"  ⚠️ Could not fetch fills: {e}")

    # ---------------------------------------------------------------- P&L
    L.append("")
    L.append("\U0001f4b0 REALIZED P&L (cumulative)")
    try:
        strat, fixture, internal = pnl_sections(get("/pnl").get("realized_pnl", {}))
        for sid, val in strat:
            L.append(f"  {sid}: {money(val)}")
        if not strat:
            L.append("  (none)")
        total = sum(v for _s, v in strat)
        emoji = "\U0001f7e2" if total >= 0 else "\U0001f534"
        L.append(f"  ── {emoji} Strategies: {money(total)}")
        if fixture:
            L.append("  fixtures: " + ", ".join(f"{s} {money(v)}" for s, v in fixture))
        if internal:
            # Real money, but booked to a ledger bucket rather than a strategy — listing
            # these as strategies is what made the old total misleading.
            L.append("  unattributed: "
                     + ", ".join(f"{s} {money(v)}" for s, v in internal))
    except Exception as e:
        L.append(f"  ⚠️ Could not fetch P&L: {e}")

    # ---------------------------------------------------------------- overnight book
    L.append("")
    L.append("\U0001f30d OVERNIGHT BOOK")
    try:
        pos = get("/positions")
        held = {s: q for s, q in pos.get("current_positions", {}).items() if q}
        if not held:
            L.append("  (flat — no overnight exposure)")
        else:
            for sym, qty in sorted(held.items()):
                L.append(f"  {sym}: {qty:+g}")
            for sid, positions in sorted(pos.get("strategy_positions", {}).items()):
                names = {s: q for s, q in positions.items() if q}
                if not names:
                    continue
                # Keep the bucket name. Collapsing __net__ and flatten_all to a shared
                # "unattributed" label put two lines of apparently contradictory quantities
                # next to each other (ANET -10 on one, +10 on the other) when they are
                # simply two books that net to zero.
                tag = f"unattributed:{sid}" if sid in INTERNAL_SIDS else sid
                L.append(f"    [{tag}] "
                         + ", ".join(f"{s} {q:+g}" for s, q in sorted(names.items())))
    except Exception as e:
        L.append(f"  ⚠️ Could not fetch positions: {e}")

    # ---------------------------------------------------------------- orphans
    try:
        orphans = get("/positions/orphans")
        if orphans.get("count"):
            L.append("")
            L.append("\U0001f6a8 POSITIONS NO STRATEGY CLAIMS")
            for row in orphans["orphans"]:
                notional = (f"  (${row['notional']:,.0f})" if row.get("notional") else "")
                L.append(f"  {row['symbol']}: {row['quantity']:+g}{notional}")
            L.append("  these sit outside NAV and outside /flatten — see /exposure")
    except Exception:
        pass          # endpoint is newer than some deployments; absence is not an error

    # ---------------------------------------------------------------- halts
    L.append("")
    L.append("\U0001f3af STRATEGY STATUS")
    try:
        strats = get("/strategies").get("strategies", [])
        halted = [s["strategy_id"] for s in strats
                  if is_strategy(s["strategy_id"]) and not s.get("active")]
        halted_fixtures = [s["strategy_id"] for s in strats
                           if not is_strategy(s["strategy_id"]) and not s.get("active")]
        L.append(f"  \U0001f6d1 HALTED: {', '.join(halted)}" if halted
                 else "  ✅ All strategies active")
        if halted_fixtures:
            L.append(f"  (fixtures halted: {', '.join(halted_fixtures)})")
    except Exception as e:
        L.append(f"  ⚠️ Could not fetch strategies: {e}")

    # ---------------------------------------------------------------- reconcile
    L.append("")
    try:
        recon = get("/reconcile/status")
        ts = recon.get("ts")
        if not ts:
            L.append("⚠️ No reconciliation has ever run")
        elif not is_today_et(ts, now):
            # The old text said "no reconciliation ran today" only when the field was
            # absent, so a week-old reconcile read as a fresh pass.
            L.append(f"⚠️ Last reconcile was {et(ts)} on {datetime.fromisoformat(ts).astimezone(ET):%b %d} — NOT today")
        elif recon.get("matched"):
            L.append(f"✅ Reconciled {et(ts)} ET: ledger matches broker")
        else:
            disc = recon.get("discrepancies", {})
            L.append(f"⚠️ Reconcile {et(ts)} ET: {len(disc)} discrepancy(ies)")
            for sym, detail in disc.items():
                L.append(f"    {sym}: {detail}")
    except Exception as e:
        L.append(f"⚠️ Could not fetch reconcile status: {e}")

    L.append("")
    L.append("Session complete \U0001f44b")
    return "\n".join(L)


if __name__ == "__main__":
    msg = build_summary()
    print(msg)
    print()
    send_telegram(msg, "summary")
