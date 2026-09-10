"""
tools/briefing_common.py — shared plumbing for the pre/post-market Telegram briefings.

The two briefings had their own copies of the REST helper, the Telegram sender, and the
"which strategies count" filter. The filter in particular was duplicated at six call sites
and had drifted: both scripts skipped `test_suite*` and `halt_test*` but neither skipped the
`demo_*` fixtures or the ledger's internal buckets, so `__net__` and `flatten_all` were
reported as strategies with P&L of their own. One definition here, used everywhere.
"""
from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

ET = ZoneInfo("America/New_York")
BASE = os.environ.get("EXECUTOR_URL", "http://127.0.0.1:8000")
KEY = os.environ.get("EXECUTOR_API_KEY", "")
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID")

#: Ledger bookkeeping buckets. They carry real P&L and real positions, but they are not
#: strategies — reporting them in a list of strategies invites the reader to look for a
#: signal that does not exist. Shown separately as "unattributed" instead.
INTERNAL_SIDS = frozenset({"__net__", "flatten_all", "kill_switch"})

#: Fixtures. `demo_*` are config placeholders that never trade; `test_suite*` back the test
#: suite; `halt_test*` are the disposable plumbing strategies.
FIXTURE_PREFIXES = ("test_suite", "halt_test", "demo_")

#: Not a strategy either — sized by exposure, holds no view. Reported in its own section.
HEDGE_SID = "hedge_overlay"


def is_strategy(sid: str) -> bool:
    """Is this something a human should read as a trading strategy?"""
    return (sid not in INTERNAL_SIDS
            and sid != HEDGE_SID
            and not sid.startswith(FIXTURE_PREFIXES))


def get(path: str, timeout: float = 15.0):
    r = requests.get(f"{BASE}{path}", headers={"X-API-Key": KEY}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def et(iso_ts: str) -> str:
    """UTC timestamp from the database -> HH:MM:SS in ET.

    The executor stores UTC; the briefings are headed and scheduled in ET. Printing the raw
    time portion put a 19:58 next to a session that closed at 16:00.
    """
    if not iso_ts:
        return "--:--:--"
    try:
        return datetime.fromisoformat(iso_ts).astimezone(ET).strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return str(iso_ts)[:8]


def is_today_et(iso_ts: str, now: datetime = None) -> bool:
    """Did this UTC timestamp fall on today's ET date? A fill at 20:30 UTC is still today
    in ET, and one at 01:00 UTC is yesterday's session — a naive string prefix gets both
    of those wrong."""
    if not iso_ts:
        return False
    try:
        stamp = datetime.fromisoformat(iso_ts).astimezone(ET)
    except (ValueError, TypeError):
        return False
    return stamp.date() == (now or datetime.now(ET)).date()


def money(v) -> str:
    return f"${v:+,.2f}"


def send_telegram(text: str, label: str = "briefing") -> None:
    if not TG_TOKEN or not TG_CHAT:
        print(f"WARN: Telegram not configured — {label} printed to stdout only")
        return
    try:
        r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          json={"chat_id": TG_CHAT, "text": text}, timeout=10)
        # Telegram answers 200 with ok:false for a bad chat id or a too-long message, so a
        # bare "sent OK" on any 2xx would hide the most common way this silently fails.
        body = r.json() if r.content else {}
        if r.status_code == 200 and body.get("ok"):
            print(f"Telegram {label} sent OK")
        else:
            print(f"Telegram {label} REJECTED: {r.status_code} "
                  f"{body.get('description') or r.text[:200]}")
    except Exception as e:
        print(f"Telegram {label} send failed: {e}")


def pnl_sections(pnl: dict) -> tuple:
    """Split realized P&L into (strategies, fixtures, unattributed) — each a sorted list of
    (sid, value) with zeros dropped."""
    strat, fixture, internal = [], [], []
    for sid, val in sorted(pnl.items()):
        if not val:
            continue
        if sid in INTERNAL_SIDS:
            internal.append((sid, val))
        elif sid.startswith(FIXTURE_PREFIXES):
            fixture.append((sid, val))
        else:
            strat.append((sid, val))
    return strat, fixture, internal
