#!/usr/bin/env python3
"""
tools/premarket_briefing.py — Telegram pre-market briefing.

Fires ~30 min before the open (via day_scheduler):
  * Session times (open/close, half-day flag)
  * IB connection and kill-switch state
  * Strategies, with fixtures and the hedge overlay kept out of the strategy list
  * Positions carried into the day, and anything no strategy claims
  * Today's schedule, READ FROM THE SCHEDULER ITSELF

That last point is the one that mattered. The schedule used to be a hand-written list of
times, so every change to day_scheduler.py silently made the briefing wrong — by the time
the hedge, the rebalance and the plumbing strategies had been added it was describing a day
that no longer happened. It now calls build_events(), so the briefing cannot drift from what
will actually run: if it is in the briefing it is scheduled, and if it is scheduled it is in
the briefing.
"""
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))
sys.path.insert(0, str(_ROOT / "src"))
load_dotenv(_ROOT / ".env")

from briefing_common import (ET, HEDGE_SID, INTERNAL_SIDS, get,  # noqa: E402
                             is_strategy, money, send_telegram)


def _schedule_lines(o, c):
    """Today's plan, collapsed. A raw dump is ~45 lines once the ORB resyncs, the reconciles
    and the plumbing runs are in — so anything that repeats is shown as a range and a
    cadence rather than one line per firing."""
    from day_scheduler import build_events

    groups = OrderedDict()
    for when, label, kind, _payload in build_events(o, c):
        groups.setdefault(label, []).append(when)

    rows = []
    for label, times in groups.items():
        times.sort()
        if len(times) <= 2:
            rows.extend((t, label) for t in times)
        else:
            gap = round((times[1] - times[0]).total_seconds() / 60)
            rows.append((times[0],
                         f"{label} — every {gap} min until {times[-1]:%H:%M} "
                         f"({len(times)} runs)"))
    return [f"  {when:%H:%M}  {label}" for when, label in sorted(rows)]


def build_briefing():
    now = datetime.now(ET)
    L = [f"\U0001f305 PRE-MARKET BRIEFING — {now:%A, %b %d %Y}", ""]

    try:
        from day_scheduler import session_today
        o, c = session_today()
    except Exception as e:
        L.append(f"⚠️ Could not read the session calendar: {e}")
        return "\n".join(L)

    if o is None:
        L.append("⚠️ No NYSE session today")
        return "\n".join(L)

    hours = (c - o).total_seconds() / 3600
    L.append(f"\U0001f552 Session: {o:%H:%M} – {c:%H:%M} ET"
             + (" (HALF DAY)" if hours < 6 else ""))

    # ---------------------------------------------------------------- health
    try:
        h = get("/health")
        status = "✅ Connected" if h.get("connected") else "❌ Disconnected"
        if h.get("killed"):
            status += " \U0001f6d1 KILL SWITCH ACTIVE"
        if h.get("startup_degraded"):
            status += " ⚠️ STARTED DEGRADED (no broker reconciliation)"
        L.append(f"\U0001f4e1 IB Gateway: {status}")
        # `connected` only means the socket is up. The gateway can accept reads and refuse
        # every order, which is exactly how a read-only session went unnoticed for a day.
        L.append("   note: 'connected' does not prove orders are accepted")
    except Exception as e:
        L.append(f"❌ Server unreachable: {e}")
        return "\n".join(L)

    # ---------------------------------------------------------------- strategies
    L.append("")
    L.append("\U0001f3af STRATEGIES")
    try:
        strats = get("/strategies").get("strategies", [])
        shown = [s for s in strats if is_strategy(s["strategy_id"])]
        for s in shown:
            mark = "✅" if s.get("active") else "\U0001f6d1 HALTED"
            L.append(f"  {mark} {s['strategy_id']}  "
                     f"(${s.get('capital_allocation', 0):,.0f} / "
                     f"{s.get('max_drawdown', 0):.0%} max DD)")
        if not shown:
            L.append("  (none configured)")
        # The hedge overlay is not a strategy but it IS operationally live, so it gets its
        # own line rather than being buried in a list of fixtures.
        hedge = next((s for s in strats if s["strategy_id"] == HEDGE_SID), None)
        if hedge:
            mark = "✅" if hedge.get("active") else "\U0001f6d1 HALTED"
            L.append(f"  {mark} {HEDGE_SID} (overlay, "
                     f"${hedge.get('capital_allocation', 0):,.0f} notional ceiling)")
        # Fixtures are counted, not listed: naming them put a $600k demo_meanrev allocation
        # in the briefing every morning as though it were capital at work.
        fixtures = [s["strategy_id"] for s in strats
                    if not is_strategy(s["strategy_id"]) and s["strategy_id"] != HEDGE_SID]
        if fixtures:
            L.append(f"  ({len(fixtures)} fixtures not shown)")
    except Exception as e:
        L.append(f"  ⚠️ Could not fetch strategies: {e}")

    # ---------------------------------------------------------------- positions
    L.append("")
    L.append("\U0001f4ca POSITIONS CARRIED IN")
    try:
        pos = get("/positions")
        # current_positions keeps zero entries for every symbol ever traded, so the old
        # `if not net:` was truthy on a flat book and printed an empty section instead of
        # saying "flat".
        held = {s: q for s, q in pos.get("current_positions", {}).items() if q}
        if not held:
            L.append("  (flat — no open positions)")
        else:
            for sym, qty in sorted(held.items()):
                L.append(f"  {sym}: {qty:+g}")
            for sid, positions in sorted(pos.get("strategy_positions", {}).items()):
                names = {s: q for s, q in positions.items() if q}
                if names:
                    tag = f"unattributed:{sid}" if sid in INTERNAL_SIDS else sid
                    L.append(f"    [{tag}] "
                             + ", ".join(f"{s} {q:+g}" for s, q in sorted(names.items())))
    except Exception as e:
        L.append(f"  ⚠️ Could not fetch positions: {e}")

    try:
        orphans = get("/positions/orphans")
        if orphans.get("count"):
            L.append(f"  \U0001f6a8 {orphans['count']} position(s) no strategy claims: "
                     + ", ".join(f"{r['symbol']} {r['quantity']:+g}"
                                 for r in orphans["orphans"]))
    except Exception:
        pass

    # ---------------------------------------------------------------- schedule
    L.append("")
    L.append("\U0001f4c5 TODAY'S SCHEDULE")
    try:
        L.extend(_schedule_lines(o, c))
    except Exception as e:
        L.append(f"  ⚠️ Could not read the schedule: {e}")

    L.append("")
    L.append("Good trading \U0001f44a")
    return "\n".join(L)


if __name__ == "__main__":
    msg = build_briefing()
    print(msg)
    print()
    send_telegram(msg, "briefing")
