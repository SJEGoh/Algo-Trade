#!/usr/bin/env python3
"""
tools/telegram_control.py — control the executor from Telegram.

Runs as its OWN container (see the `telegram-control` service in docker-compose.yml) and
drives the executor through its authenticated REST API, exactly as you would with curl.
Deliberately outside the trading process: a bug in message parsing can't touch the loop
that talks to IB, and the bot survives the executor to tell you when it's down — it also
serves as the watchdog, alerting when /health stops answering.

    /status /pnl /positions /orders /fills /strategies /journal   read-only, anyone in the chat
    /halt /resume /flatten /kill /unkill /allocate ...            allow-listed users only
    /kill /unkill /flatten /allocate                              also need /confirm <token>
                                                                  (/allocate shows the plan
                                                                   in the prompt first)

Where replies go
----------------
TELEGRAM_CONTROL_THREAD picks the topic replies land in. Default: the orders topic
(TELEGRAM_THREAD_ORDERS). To change it later, set that one variable:

    TELEGRAM_CONTROL_THREAD=<thread id>   a specific topic
    TELEGRAM_CONTROL_THREAD=here          reply in whichever topic the command came from
    TELEGRAM_CONTROL_THREAD=general       the group's main thread

Environment
-----------
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID         required (same bot as the alerter)
    TELEGRAM_ALLOWED_USER_IDS                    comma-separated user ids; REQUIRED for
                                                 any command that changes state
    TELEGRAM_CONTROL_THREAD                      see above (default: TELEGRAM_THREAD_ORDERS)
    TELEGRAM_THREAD_ERRORS                       watchdog alerts go here when set
    EXECUTOR_URL, EXECUTOR_API_KEY               how to reach the executor
    TELEGRAM_STATE_PATH                          update-offset file (default /app/state/...)
    TELEGRAM_WATCHDOG_SEC, TELEGRAM_WATCHDOG_FAILS, TELEGRAM_CONFIRM_TTL
"""
from dotenv import load_dotenv

load_dotenv()  # before anything reads env vars

import json
import logging
import os
import secrets
import threading
import time
from typing import NamedTuple
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] telegram-control: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("telegram-control")

API = "https://api.telegram.org/bot{token}/{method}"

TOKEN = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
CHAT_ID = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
EXECUTOR_URL = (os.environ.get("EXECUTOR_URL") or "http://127.0.0.1:8000").rstrip("/")
API_KEY = (os.environ.get("EXECUTOR_API_KEY") or "").strip()

# Reply target. Defaults to the orders topic; one variable moves it anywhere.
CONTROL_THREAD = (os.environ.get("TELEGRAM_CONTROL_THREAD")
                  or os.environ.get("TELEGRAM_THREAD_ORDERS") or "").strip()
ERRORS_THREAD = (os.environ.get("TELEGRAM_THREAD_ERRORS") or "").strip()

STATE_PATH = Path(os.environ.get("TELEGRAM_STATE_PATH")
                  or (Path(__file__).resolve().parents[1] / "db" / "telegram_offset.json"))
CONFIRM_TTL = float(os.environ.get("TELEGRAM_CONFIRM_TTL", "60"))
WATCHDOG_SEC = float(os.environ.get("TELEGRAM_WATCHDOG_SEC", "60"))
WATCHDOG_FAILS = int(os.environ.get("TELEGRAM_WATCHDOG_FAILS", "3"))
STALE_COMMAND_SEC = 300.0   # ignore anything Telegram redelivers from before a restart


def _allowed_users() -> set:
    raw = os.environ.get("TELEGRAM_ALLOWED_USER_IDS", "")
    out = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            log.warning("ignoring non-numeric TELEGRAM_ALLOWED_USER_IDS entry %r", part)
    return out


ALLOWED = _allowed_users()


# ---------------------------------------------------------------------------
# Telegram plumbing
# ---------------------------------------------------------------------------
def tg(method: str, **payload):
    """One Telegram API call. Returns the `result` field, or None on failure."""
    try:
        r = requests.post(API.format(token=TOKEN, method=method), json=payload, timeout=40)
        body = r.json()
        if not body.get("ok"):
            log.warning("telegram %s failed: %s", method, body.get("description"))
            return None
        return body.get("result")
    except Exception as e:
        log.warning("telegram %s error: %s", method, e)
        return None


def say(text: str, thread_id=None) -> None:
    """Send a reply. `thread_id` overrides the configured topic (used for watchdog alerts
    and for CONTROL_THREAD=here)."""
    payload = {"chat_id": CHAT_ID, "text": text, "disable_web_page_preview": True}
    target = thread_id if thread_id is not None else _configured_thread()
    if target:
        payload["message_thread_id"] = int(target)
    tg("sendMessage", **payload)


def _configured_thread():
    """The topic replies go to. 'general' (or empty) means the main thread; 'here' is
    resolved per-message by the caller."""
    if CONTROL_THREAD.lower() in ("", "general", "none"):
        return None
    if CONTROL_THREAD.lower() == "here":
        return None            # resolved by reply_thread_for()
    return CONTROL_THREAD


def reply_thread_for(message: dict):
    """Where THIS message's reply goes: the originating topic when CONTROL_THREAD=here,
    otherwise the configured one."""
    if CONTROL_THREAD.lower() == "here":
        return message.get("message_thread_id")
    return _configured_thread()


# ---------------------------------------------------------------------------
# Executor API
# ---------------------------------------------------------------------------
def api_get(path: str, **params):
    r = requests.get(f"{EXECUTOR_URL}{path}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def api_post(path: str, body: dict = None):
    r = requests.post(f"{EXECUTOR_URL}{path}", json=body or {},
                      headers={"X-API-Key": API_KEY}, timeout=30)
    r.raise_for_status()
    return r.json()


def journal(user: str, command: str, result: str) -> None:
    """Every command lands in the decision journal — you'll want this when reconstructing
    a bad day. Never let journalling failure break the command."""
    try:
        api_post("/journal", {"strategy_id": "telegram", "event_type": "control",
                              "summary": f"{user}: {command}", "detail": result[:2000]})
    except Exception as e:
        log.warning("could not journal %r: %s", command, e)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def money(v) -> str:
    """$1,234 / -$1,234 — the sign goes in front of the currency, not after it."""
    try:
        n = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{'-' if n < 0 else ''}${abs(n):,.0f}"


def _hidden(sid: str) -> bool:
    return sid in {"__net__", "flatten_all", "kill_switch"} or sid.startswith(
        ("test_suite", "halt_test"))


# ---------------------------------------------------------------------------
# Command handlers — each returns the text to reply with
# ---------------------------------------------------------------------------
def cmd_help(args, ctx) -> str:
    return (
        "Executor control\n"
        "\nRead-only:\n"
        "  /status         health, NAV, exposure\n"
        "  /pnl            per-strategy P&L and NAV\n"
        "  /positions      broker positions\n"
        "  /orders         working orders\n"
        "  /fills [n]      recent fills\n"
        "  /strategies     allocations and halt state\n"
        "  /journal [n]    recent decisions\n"
        "\nControl (allow-listed users):\n"
        "  /halt <strategy> [noflatten]   stop a strategy, close its book\n"
        "  /resume <strategy>             clear a halt\n"
        "  /flatten [strategy]            close a book, keep trading enabled *\n"
        "  /kill                          stop everything and flatten *\n"
        "  /unkill                        re-enable order flow *\n"
        "  /allocate <strategy> <amount>  re-allocate capital *\n"
        "        150k sets it, -40k takes it out; add 'equal' to split a\n"
        "        sell-down evenly instead of pro-rata\n"
        "  /reconcile                     resync with the broker\n"
        "  /reset_daily                   reset the daily loss baseline\n"
        "\n* needs /confirm <token> within "
        f"{CONFIRM_TTL:.0f}s")


def cmd_status(args, ctx) -> str:
    health = api_get("/health")
    lines = [
        f"IB: {'connected' if health.get('connected') else 'DISCONNECTED'}"
        f" | market {'open' if health.get('market_open') else 'closed'}"
        f"{' | KILL SWITCH ACTIVE' if health.get('killed') else ''}",
    ]
    if health.get("startup_degraded"):
        lines.append("DEGRADED: startup could not reconcile with the broker, so position "
                     "state is untrusted and trading is killed. Run /reconcile, then /unkill.")
    try:
        eq = api_get("/equity")
        totals = eq.get("totals") or {}
        if totals:
            pnl = totals.get("equity", 0.0)
            lines.append(f"NAV {money(totals.get('nav'))}  "
                         f"(cash {money(totals.get('cash'))}, "
                         f"positions {money(totals.get('position_value'))})")
            lines.append(f"P&L {'+' if pnl >= 0 else ''}{money(pnl)} "
                         f"on {money(totals.get('starting_cash'))} of capital")
        if eq.get("ts"):
            lines.append(f"last sample {eq['ts'][11:19]} UTC")
    except Exception as e:
        lines.append(f"(equity unavailable: {e})")
    try:
        working = [o for o in api_get("/orders").get("orders", [])
                   if o.get("status") in ("Submitted", "PreSubmitted")]
        halted = [s["strategy_id"] for s in api_get("/strategies").get("strategies", [])
                  if not s.get("active") and not _hidden(s["strategy_id"])]
        lines.append(f"working orders: {len(working)}")
        if halted:
            lines.append("HALTED: " + ", ".join(halted))
    except Exception:
        pass
    return "\n".join(lines)


def cmd_pnl(args, ctx) -> str:
    eq = api_get("/equity")
    strategies = eq.get("strategies") or {}
    if not strategies:
        return "no equity sample yet (the sampler runs every 60s)"
    rows = []
    for sid, v in sorted(strategies.items()):
        if _hidden(sid):
            continue
        pnl = v.get("equity", 0.0)
        rows.append(f"{sid}\n   NAV {money(v.get('nav'))} | cash {money(v.get('cash'))} | "
                    f"P&L {'+' if pnl >= 0 else ''}{money(pnl)}")
    totals = eq.get("totals") or {}
    rows.append(f"\nTOTAL NAV {money(totals.get('nav'))} | "
                f"P&L {'+' if totals.get('equity', 0) >= 0 else ''}{money(totals.get('equity'))}")
    return "\n".join(rows) or "nothing to report"


def cmd_positions(args, ctx) -> str:
    pos = api_get("/positions")
    current = {s: q for s, q in (pos.get("current_positions") or {}).items() if q}
    if not current:
        return "flat at the broker"
    marks = (api_get("/equity").get("marks") or {})
    lines = []
    for sym, qty in sorted(current.items()):
        mark = marks.get(sym)
        value = f" = {money(mark * qty)}" if mark else ""
        lines.append(f"{sym:>6} {qty:>+10,.0f}{value}")
    # positions no strategy owns are in nobody's P&L or drawdown limit — worth surfacing
    owned = {}
    for book in (pos.get("strategy_positions") or {}).values():
        for sym, q in book.items():
            owned[sym] = owned.get(sym, 0.0) + q
    orphans = {s: q - owned.get(s, 0.0) for s, q in current.items()
               if abs(q - owned.get(s, 0.0)) > 1e-9}
    if orphans:
        lines.append("\nunattributed (no strategy owns these, so no drawdown limit applies):")
        lines += [f"{s:>6} {q:>+10,.0f}" for s, q in sorted(orphans.items())]
    return "\n".join(lines)


def cmd_orders(args, ctx) -> str:
    orders = [o for o in api_get("/orders").get("orders", [])
              if o.get("status") in ("Submitted", "PreSubmitted")]
    if not orders:
        return "no working orders"
    return "\n".join(
        f"#{o['order_id']} {o.get('symbol')} {o.get('pending_qty', 0):+g} "
        f"filled {o.get('filled', 0):g}/{o.get('filled', 0) + o.get('remaining', 0):g} "
        f"[{o.get('strategy_id')}]"
        for o in sorted(orders, key=lambda o: -o["order_id"]))


def cmd_fills(args, ctx) -> str:
    n = _int_arg(args, 10)
    fills = api_get("/fills", limit=n).get("fills", [])
    if not fills:
        return "no fills recorded"
    return "\n".join(
        f"{(f.get('filled_at') or '')[11:19]} {f.get('symbol')} "
        f"{'BUY' if f.get('side') == 'BOT' else 'SELL'} {f.get('quantity'):g} "
        f"@ {f.get('price'):.2f} [{f.get('strategy_id')}]"
        for f in fills[:n])


def cmd_strategies(args, ctx) -> str:
    strategies = [s for s in api_get("/strategies").get("strategies", [])
                  if not _hidden(s["strategy_id"])]
    return "\n".join(
        f"{'OK ' if s['active'] else 'HALT'} {s['strategy_id']}  "
        f"alloc {money(s['capital_allocation'])}  maxDD {s['max_drawdown'] * 100:.0f}%"
        for s in sorted(strategies, key=lambda s: s["strategy_id"])) or "none configured"


def cmd_journal(args, ctx) -> str:
    n = _int_arg(args, 10)
    entries = api_get("/journal", limit=n).get("journal", [])
    if not entries:
        return "journal is empty"
    return "\n".join(f"{e['ts'][11:19]} {e['strategy_id']} {e['event_type']}: {e['summary']}"
                     for e in entries[:n])


def cmd_halt(args, ctx) -> str:
    if not args:
        return "usage: /halt <strategy> [noflatten]"
    sid = args[0]
    flatten = "noflatten" not in args[1:]
    r = api_post(f"/strategies/{sid}/halt", {"flatten": flatten,
                                             "reason": f"telegram: {ctx['user']}"})
    return f"halted {sid}" + (" and closed its book" if flatten else " (book left open)") \
        + (f" — {r['note']}" if r.get("note") else "")


def cmd_resume(args, ctx) -> str:
    if not args:
        return "usage: /resume <strategy>"
    api_post(f"/strategies/{args[0]}/reactivate")
    return (f"{args[0]} is active again.\nIt re-halts immediately if its drawdown is still "
            "breached — check /pnl first.")


def cmd_flatten(args, ctx) -> str:
    if args:
        r = api_post(f"/strategies/{args[0]}/flatten")
        closed = r.get("flattened") or []
        if not closed:
            return f"{args[0]} was already flat"
        return f"closing {args[0]}: " + ", ".join(
            f"{p['symbol']} {p['quantity']:+g}" for p in closed)
    r = api_post("/flatten")
    note = f"\n{r['note']}" if r.get("note") else ""
    return (f"cancelled {r.get('cancelled_orders', 0)} orders, "
            f"closing {len(r.get('flattened_positions', []))} positions{note}")


def cmd_kill(args, ctx) -> str:
    r = api_post("/kill", {"flatten": True})
    return f"KILL SWITCH ACTIVE (flatten={r.get('flattened')}). /unkill to re-enable."


def cmd_unkill(args, ctx) -> str:
    r = api_post("/unkill")
    return ("order flow re-enabled.\nStrategies halted by the breaker stay halted — "
            "/resume each one after you've looked."
            if not r.get("killed") else "kill switch is still set")


def cmd_reconcile(args, ctx) -> str:
    r = api_post("/reconcile")
    disc = r.get("discrepancies") or {}
    if not disc:
        return "reconciled: internal state matches the broker"
    return "discrepancies:\n" + "\n".join(
        f"{s}: internal {v.get('internal')} vs broker {v.get('broker')}"
        for s, v in disc.items())


def cmd_reset_daily(args, ctx) -> str:
    api_post("/reset_daily")
    return "daily loss baseline reset; circuit breaker cleared"


def _parse_amount(text: str):
    """'150k' -> (150000, absolute); '-40k' -> (40000 withdrawn, delta). A leading + or -
    means "change it by this much", anything else means "set it to this"."""
    raw = text.replace(",", "").replace("$", "").strip().lower()
    is_delta = raw.startswith(("+", "-"))
    factor = 1.0
    if raw.endswith("k"):
        factor, raw = 1_000.0, raw[:-1]
    elif raw.endswith("m"):
        factor, raw = 1_000_000.0, raw[:-1]
    try:
        return float(raw) * factor, is_delta
    except ValueError:
        raise ValueError(f"could not read an amount from {text!r} — try 150k, 1.2m or -40000")


ALLOCATE_USAGE = ("usage: /allocate <strategy> <amount> [pro_rata|equal]\n"
                  "  /allocate ovn_volsurge 150k        set the allocation to $150k\n"
                  "  /allocate ovn_volsurge -40k        take $40k out\n"
                  "  /allocate ovn_volsurge -40k equal  split the sell-down evenly")


def _allocate_body(args, dry_run: bool) -> tuple:
    if len(args) < 2:
        raise ValueError(ALLOCATE_USAGE)
    sid, method = args[0], (args[2] if len(args) > 2 else "pro_rata")
    if method not in ("pro_rata", "equal"):
        raise ValueError(f"unknown method {method!r} — use pro_rata or equal")
    amount, is_delta = _parse_amount(args[1])
    body = {"delta": amount} if is_delta else {"capital_allocation": amount}
    body.update(method=method, dry_run=dry_run)
    return sid, body


def _format_allocation(r: dict) -> str:
    """Render the plan the executor came back with — the same shape for a dry run and for
    the real thing, so what you confirm is what you get."""
    delta = r["allocation_after"] - r["allocation_before"]
    lines = [f"{r['strategy_id']}",
             f"allocation {money(r['allocation_before'])} -> {money(r['allocation_after'])}"
             f"  ({'+' if delta >= 0 else '-'}{money(abs(delta))})"]
    if r.get("cash_after") is not None:
        lines.append(f"cash {money(r['cash_before'])} -> {money(r['cash_after'])}")
    else:
        lines.append(f"cash now {money(r['cash_before'])}")
    cuts = r.get("liquidations") or []
    if cuts:
        lines.append(f"raising {money(r.get('cash_shortfall'))} by selling "
                     f"({r.get('method')}):")
        lines += [f"   {c['symbol']} {c['from_quantity']:g} -> {c['to_quantity']:g}"
                  f"  ({money(abs(c['freed']))})" for c in cuts]
        lines.append("cash goes negative until those fills settle — that is expected")
    elif delta < 0:
        lines.append("cash covers it — nothing to sell")
    if r.get("valued_at_cost"):
        lines.append("no live mark for " + ", ".join(r["valued_at_cost"]) + " (valued at cost)")
    return "\n".join(lines)


def preview_allocate(args, ctx) -> str:
    """Dry-run the change so the confirmation prompt shows exactly what will be sold. A
    plan that the executor would reject never reaches a confirmation token."""
    sid, body = _allocate_body(args, dry_run=True)
    return "PLAN (nothing changed yet)\n" + _format_allocation(
        api_post(f"/strategies/{sid}/allocation", body))


def cmd_allocate(args, ctx) -> str:
    sid, body = _allocate_body(args, dry_run=False)
    r = api_post(f"/strategies/{sid}/allocation", body)
    if r.get("note") == "allocation unchanged":
        return f"{sid} is already at {money(r['allocation_after'])}"
    failed = [o for o in (r.get("orders") or []) if o.get("error")]
    text = "DONE\n" + _format_allocation(r)
    if failed:
        text += "\nsome orders did NOT go through: " + ", ".join(
            f"{o['symbol']} ({o['error']})" for o in failed)
    return text


def _int_arg(args, default):
    try:
        return max(1, min(50, int(args[0])))
    except (IndexError, ValueError):
        return default


class Command(NamedTuple):
    handler: object
    restricted: bool = False    # allow-listed users only
    confirm: bool = False       # needs /confirm <token>
    preview: object = None      # optional dry run, shown in the confirmation prompt


COMMANDS = {
    "help":        Command(cmd_help),
    "start":       Command(cmd_help),
    "status":      Command(cmd_status),
    "pnl":         Command(cmd_pnl),
    "positions":   Command(cmd_positions),
    "orders":      Command(cmd_orders),
    "fills":       Command(cmd_fills),
    "strategies":  Command(cmd_strategies),
    "journal":     Command(cmd_journal),
    # protective: no confirmation, because a speed bump in front of de-risking is a bug
    "halt":        Command(cmd_halt, restricted=True),
    "resume":      Command(cmd_resume, restricted=True),
    "reconcile":   Command(cmd_reconcile, restricted=True),
    "reset_daily": Command(cmd_reset_daily, restricted=True),
    # these move money or stop trading
    "flatten":     Command(cmd_flatten, restricted=True, confirm=True),
    "kill":        Command(cmd_kill, restricted=True, confirm=True),
    "unkill":      Command(cmd_unkill, restricted=True, confirm=True),
    "allocate":    Command(cmd_allocate, restricted=True, confirm=True,
                           preview=preview_allocate),
}

# token -> (command, args, user_id, expires_at)
_pending = {}
_pending_lock = threading.Lock()


def _stash_confirmation(name, args, user_id) -> str:
    token = secrets.token_hex(2)
    with _pending_lock:
        now = time.time()
        for t, v in list(_pending.items()):        # drop anything expired
            if v[3] < now:
                del _pending[t]
        _pending[token] = (name, args, user_id, now + CONFIRM_TTL)
    return token


def _take_confirmation(token, user_id):
    with _pending_lock:
        entry = _pending.pop(token, None)
    if entry is None:
        return None, "nothing pending with that token (or it expired)"
    name, args, owner, expires = entry
    if owner != user_id:
        return None, "that confirmation belongs to someone else"
    if expires < time.time():
        return None, "that confirmation expired — run the command again"
    return (name, args), None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def handle(message: dict) -> None:
    text = (message.get("text") or "").strip()
    if not text.startswith("/"):
        return
    user = message.get("from") or {}
    user_id = user.get("id")
    user_name = user.get("username") or user.get("first_name") or str(user_id)
    thread = reply_thread_for(message)

    parts = text.split()
    name = parts[0][1:].split("@")[0].lower()     # strip a /cmd@botname suffix
    args = parts[1:]

    if name == "confirm":
        if not args:
            say("usage: /confirm <token>", thread)
            return
        taken, err = _take_confirmation(args[0], user_id)
        if err:
            say(err, thread)
            return
        name, args = taken
        _run(name, args, user_id, user_name, thread, confirmed=True)
        return

    entry = COMMANDS.get(name)
    if entry is None:
        say(f"unknown command /{name} — /help lists them", thread)
        return

    if entry.restricted:
        if not ALLOWED:
            say("no TELEGRAM_ALLOWED_USER_IDS configured, so control commands are "
                f"disabled. Add your id ({user_id}) to that variable and restart the "
                "telegram-control container.", thread)
            return
        if user_id not in ALLOWED:
            log.warning("rejected /%s from %s (%s) — not allow-listed", name, user_name, user_id)
            say(f"{user_name}, you are not allow-listed for control commands.", thread)
            return

    if entry.confirm:
        preview = ""
        if entry.preview is not None:
            try:
                preview = entry.preview(args, {"user": user_name, "user_id": user_id}) + "\n\n"
            except Exception as e:
                # A plan the executor would reject must never become a confirmation token
                say(_describe_failure(e, f"/{name} {' '.join(args)}"), thread)
                return
        token = _stash_confirmation(name, args, user_id)
        say(f"{preview}/{name} {' '.join(args)}\nThis changes live trading. Reply with:\n"
            f"/confirm {token}\n(expires in {CONFIRM_TTL:.0f}s)", thread)
        return

    _run(name, args, user_id, user_name, thread)


def _describe_failure(e: Exception, label: str) -> str:
    """Turn an exception into something readable in a chat window."""
    if isinstance(e, requests.HTTPError):
        detail = ""
        if e.response is not None:
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                detail = (e.response.text or "")[:300]
            return f"executor rejected {label}: {e.response.status_code} {detail}"
        return f"executor rejected {label}"
    if isinstance(e, requests.RequestException):
        return f"cannot reach the executor at {EXECUTOR_URL}: {e}"
    return f"{label} failed: {e}"


def _run(name, args, user_id, user_name, thread, confirmed=False) -> None:
    entry = COMMANDS[name]
    label = f"/{name} {' '.join(args)}".strip()
    try:
        reply = entry.handler(args, {"user": user_name, "user_id": user_id})
    except Exception as e:
        if not isinstance(e, (requests.RequestException, ValueError)):
            log.exception("handler %s failed", name)
        reply = _describe_failure(e, label)

    say(reply, thread)
    if entry.restricted:                       # journal state-changing commands only
        journal(user_name, label + (" (confirmed)" if confirmed else ""), reply)
    log.info("%s ran %s", user_name, label)


# ---------------------------------------------------------------------------
# Update loop
# ---------------------------------------------------------------------------
def load_offset():
    try:
        return int(json.loads(STATE_PATH.read_text())["offset"])
    except Exception:
        return None


def save_offset(offset: int) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps({"offset": offset}))
    except Exception as e:
        log.warning("could not persist update offset: %s", e)


def skip_backlog() -> int:
    """Acknowledge everything Telegram is holding without acting on it.

    Telegram redelivers un-acknowledged updates forever. Without this, a bot starting with
    no stored offset would replay whatever was sent while it was down — including a stale
    /kill or /flatten from hours ago."""
    result = tg("getUpdates", offset=-1, timeout=0) or []
    return (result[-1]["update_id"] + 1) if result else 0


def poll_loop() -> None:
    offset = load_offset()
    if offset is None:
        offset = skip_backlog()
        save_offset(offset)
        log.info("no stored offset — skipped the backlog, starting at %s", offset)

    backoff = 1.0
    while True:
        try:
            updates = tg("getUpdates", offset=offset, timeout=30,
                         allowed_updates=["message"])
            if updates is None:                       # transient API failure
                time.sleep(min(backoff, 30)); backoff *= 2
                continue
            backoff = 1.0
            for update in updates:
                offset = update["update_id"] + 1
                save_offset(offset)                   # ack BEFORE acting: a crash mid-command
                                                      # must not replay it on restart
                message = update.get("message") or {}
                if str(message.get("chat", {}).get("id")) != CHAT_ID:
                    continue                          # not our chat
                age = time.time() - float(message.get("date", 0))
                if age > STALE_COMMAND_SEC:
                    log.warning("ignoring stale command (%.0fs old): %r",
                                age, (message.get("text") or "")[:40])
                    continue
                handle(message)
        except Exception as e:
            log.exception("poll loop error: %s", e)
            time.sleep(min(backoff, 30)); backoff *= 2


# ---------------------------------------------------------------------------
# Watchdog — the reason this lives outside the executor
# ---------------------------------------------------------------------------
def watchdog_loop() -> None:
    """Alert when the executor stops answering. The in-process alerter cannot report its
    own death; this can."""
    fails, alerted = 0, False
    thread = ERRORS_THREAD or None
    while True:
        time.sleep(WATCHDOG_SEC)
        try:
            health = api_get("/health")
            problem = None if health.get("connected") else "executor up but IB DISCONNECTED"
        except Exception as e:
            problem = f"executor unreachable at {EXECUTOR_URL}: {e}"

        if problem:
            fails += 1
            if fails >= WATCHDOG_FAILS and not alerted:
                alerted = True
                say(f"\U0001f6a8 WATCHDOG: {problem}\n"
                    f"({fails} consecutive checks, {WATCHDOG_SEC:.0f}s apart)", thread)
        else:
            if alerted:
                say("✅ WATCHDOG: executor is answering again", thread)
            fails, alerted = 0, False


def main() -> None:
    if not TOKEN or not CHAT_ID:
        raise SystemExit("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")
    if not API_KEY:
        log.warning("EXECUTOR_API_KEY not set — control commands will be rejected (401)")
    if not ALLOWED:
        log.warning("TELEGRAM_ALLOWED_USER_IDS not set — read-only mode; /halt, /kill, "
                    "/flatten etc. will refuse to run")

    where = ("the topic each command came from" if CONTROL_THREAD.lower() == "here"
             else f"thread {CONTROL_THREAD}" if _configured_thread() else "the general thread")
    log.info("replying in %s; executor at %s; %d allow-listed user(s)",
             where, EXECUTOR_URL, len(ALLOWED))

    threading.Thread(target=watchdog_loop, daemon=True, name="watchdog").start()
    say("\U0001f916 Executor control online — /help for commands")
    poll_loop()


if __name__ == "__main__":
    main()
