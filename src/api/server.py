import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional, Union

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from execution.central_execution import CentralExecutor, is_market_open, moc_cutoff_at
from execution.netting import NettingCoordinator
from monitoring.alerter import Alerter, AlertingHandler
from monitoring.logging_config import setup_logging
from config import CONFIG, GLOBAL, is_funded, validate_config
from portfolio import hedger
from risk.exit_rules import ExitManager, ExitSpecError, parse_exits

import threading
import time
from datetime import datetime, timezone
import logging

# .env lives at repo root (one level above src/) — same pattern as main.py
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

STATIC_DIR = Path(__file__).resolve().parent / "static"
DB_DIR = Path(__file__).resolve().parent.parent.parent / "db"

EXECUTOR_API_KEY = os.environ.get("EXECUTOR_API_KEY")
#: Module logger. /atr/cancel has called logger.info and logger.warning since it was written,
#: but nothing ever defined `logger` here — so the daily sweep raised NameError the moment it
#: cancelled its first order (the except branch raised it again), returned a 500, and left
#: every remaining ATR limit working into the close.
logger = logging.getLogger("executor")

#: how long an order may sit unacknowledged by IB before /orders says something is wrong
UNACKED_WARN_SEC = float(os.environ.get("UNACKED_WARN_SEC", "20"))
SERVER_CLIENT_ID = int(os.environ.get("IB_CLIENT_ID", "8"))  # distinct from main.py / run_strat.py (both use 6)
IB_HOST = os.environ.get("IB_HOST", "127.0.0.1")               # 'ib-gateway' in Docker compose
IB_PORT = int(os.environ.get("IB_PORT", "4002"))              # 4002 paper / 4001 live (Gateway); 7497 TWS paper

executor: Optional[CentralExecutor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with _startup(app):
            yield
    except Exception:
        # A failed lifespan stops uvicorn serving but does NOT end the process, so Docker's
        # restart policy never fires and the container sits there answering nothing. Make
        # the failure real: log it, then exit so the container restarts (or crash-loops
        # visibly, which beats a zombie).
        logging.getLogger("executor").critical("STARTUP FAILED — exiting so the container "
                                               "restarts instead of running dead",
                                               exc_info=True)
        logging.shutdown()
        os._exit(1)


@asynccontextmanager
async def _startup(app: FastAPI):
    global executor
    setup_logging()
    if not EXECUTOR_API_KEY:
        raise RuntimeError(
            "EXECUTOR_API_KEY not set in .env — generate with secrets.token_hex(32)"
        )

    # Route any logger.critical(...) anywhere in the app to Telegram (drawdown halts,
    # kill switch, unexpected disconnects). No-op if TELEGRAM_* env vars are unset.
    alerter = Alerter()
    logging.getLogger().addHandler(AlertingHandler(alerter))
    app.state.alerter = alerter

    problems = validate_config()
    if problems:
        raise RuntimeError(
            "invalid strategy config — every strategy needs a positive capital_allocation "
            "and a max_drawdown in (0, 1], or its allocation cap and drawdown halt are "
            "silently disabled:\n  " + "\n  ".join(problems))

    executor = CentralExecutor()
    executor._alerter = alerter  # give executor access for read-only probe Telegram alerts
    recon = executor.start(host=IB_HOST, port=IB_PORT, client_id=SERVER_CLIENT_ID)
    executor.coordinator = NettingCoordinator(executor, CONFIG, state_path=str(DB_DIR / "netting.json"))
    executor.exit_manager = ExitManager(executor, alert=_alert,
                                        max_mark_age=GLOBAL.get("exit_mark_max_age_sec"))
    threading.Thread(target=_equity_sampler, args=(60.0,), daemon=True).start()
    threading.Thread(target=_exit_sampler, args=(float(GLOBAL.get("exit_check_sec", 30.0)),),
                     daemon=True, name="exit-checks").start()
    app.state.startup_reconciliation = recon
    _last_reconcile.update({
        "matched": recon.get("matched") if isinstance(recon, dict) else recon,
        "discrepancies": recon.get("discrepancies", []) if isinstance(recon, dict) else [],
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    alerter.send(f"\u2705 Executor server up \u2014 IB connected, reconciled "
                 f"(matched={recon.get('matched') if isinstance(recon, dict) else recon})")

    try:
        yield
    finally:
        _sampler_stop.set()
        try:
            app.state.alerter.send("\U0001F6D1 Executor server shutting down")
        except Exception:
            pass
        executor.shutdown()

app = FastAPI(title="Algo Trade Executor", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _alert(message: str, topic: str = None) -> None:
    """Send a Telegram alert if the alerter is initialised (no-op in tests / before lifespan)."""
    alerter = getattr(app.state, "alerter", None)
    if alerter:
        alerter.send(message, topic=topic)

def require_api_key(x_api_key: str = Header(default = "")) -> None:
    if not secrets.compare_digest(x_api_key, EXECUTOR_API_KEY or ""):
        raise HTTPException(status_code=401, detail="invalid or missing API key")

class KillRequest(BaseModel):
    flatten: bool = True

def _plan_exits(sid: str, intents: list):
    """Validate every intent's `exits` and apply same-session re-entry lockouts, BEFORE
    anything is submitted. Returns (plan, blocked, refusal):

      plan     {symbol: (target, spec, sec_type)} — what to arm once the submission is taken
      blocked  {symbol: lockout} — intents rewritten IN PLACE to target 0
      refusal  a rejection to return instead of submitting, or None

    Fails closed on the whole submission: arming some of a book's stops and not others would
    leave the strategy believing names are protected that are not."""
    em = getattr(executor, "exit_manager", None)
    plan, blocked, problems = {}, {}, []
    for it in intents:
        inst = it.get("instrument") or {}
        sym = inst.get("symbol")
        raw = it.get("exits")
        try:
            target = float(it.get("target_quantity"))
        except (TypeError, ValueError):
            target = None                     # the schema / coordinator refuses this one
        if raw is not None and em is None:
            problems.append(f"{sym}: exits sent but the exit manager is not running — "
                            "they would not be enforced")
            continue
        try:
            price = it.get("expected_price")
            spec = parse_exits(raw, target, price if price is not None else it.get("limit_price"))
        except ExitSpecError as e:
            problems.append(f"{sym}: {e}")
            continue
        if em is not None and target:
            lock = em.blocked(sid, sym, target)
            if lock is not None:
                blocked[sym] = lock
                it["target_quantity"] = 0
                target, spec = 0.0, None
        plan[sym] = (target or 0.0, spec, inst.get("sec_type", "STK"))
    if problems:
        return plan, blocked, {"accepted": False,
                               "reason": "invalid exits, nothing submitted: " + "; ".join(problems)}
    return plan, blocked, None


#: How a /targets intent may ask to be traded. The pool sends market orders, or MOC orders
#: for the close; anything else (a limit, say) it cannot honour.
BOOK_ORDER_TYPES = ("market", "moc")


def _check_book_order_types(intents: list):
    """Refuse the WHOLE book, before anything is placed, when it asks for an execution the
    pool cannot give: an unknown order_type would otherwise quietly trade at market, and an
    MOC past the cutoff would be refused by the exchange after the rest of the book traded."""
    bad = sorted({f"{(it.get('instrument') or {}).get('symbol')}: {it.get('order_type')!r}"
                  for it in intents
                  if it.get("order_type") not in (None,) + BOOK_ORDER_TYPES})
    if bad:
        return {"accepted": False,
                "reason": f"unsupported order_type in book (allowed: {', '.join(BOOK_ORDER_TYPES)}), "
                          f"nothing submitted: " + "; ".join(bad)}
    if any(it.get("order_type") == "moc" for it in intents) and executor.moc_closed():
        return {"accepted": False, "reason": executor.MOC_CLOSED_REASON}
    return None


def _is_noop(result) -> bool:
    return isinstance(result, dict) and str(result.get("reason", "")).startswith("no-op")


@app.post("/orders", dependencies = [Depends(require_api_key)])
def submit_order(intent: dict):
    em = getattr(executor, "exit_manager", None)
    exits = intent.pop("exits", None)
    sid = intent.get("strategy_id")
    sym = (intent.get("instrument") or {}).get("symbol")
    plan, blocked = None, {}
    if intent.get("intent_type") == "target_position":
        probe = {"instrument": intent.get("instrument"), "exits": exits,
                 "target_quantity": intent.get("target_quantity"),
                 "expected_price": intent.get("expected_price"),
                 "limit_price": intent.get("limit_price")}
        plan, blocked, refusal = _plan_exits(sid, [probe])
        if refusal:
            return refusal
        if sym in blocked:
            intent["target_quantity"] = 0
    elif exits is not None:
        return {"accepted": False, "reason": "exits need an absolute target — send "
                "intent_type target_position with target_quantity"}
    elif em is not None and intent.get("side") in ("buy", "sell"):
        lock = em.blocked(sid, sym, 1 if intent["side"] == "buy" else -1)
        if lock is not None:
            return {"accepted": False, "exits_blocked": {sym: lock},
                    "reason": f"exit lockout: {sym} hit its {lock['kind']} this session — "
                              f"a {intent['side']} would re-enter it"}

    result = executor.process_intent(intent)
    if plan is not None and em is not None and (result.get("accepted") or _is_noop(result)):
        em.set(sid, sym, *plan[sym])
    if blocked:
        result["exits_blocked"] = blocked
    # Alert order submission to orders topic
    if result.get("accepted"):
        symbol = intent.get("instrument", {}).get("symbol", "?")
        oid = result.get("order_id", "?")
        _alert(
            f"\U0001f4e8 Order submitted — {symbol} id={oid} "
            f"({intent.get('intent_type', '?')})",
            topic="orders",
        )
    return result


class TargetRequest(BaseModel):
    strategy_id: str
    symbol: str
    quantity: float
    instrument: Optional[dict] = None
    price: Optional[float] = None
    exits: Optional[dict] = None


def _pool_preflight(is_future_only: bool) -> None:
    """Same protections /orders enforces, for the pooled endpoints: no coordinator -> 503;
    kill switch -> 423; equities while the market is closed -> 409 (futures-only books pass,
    matching the futures bypass in process_intent)."""
    if executor.coordinator is None:
        raise HTTPException(status_code=503, detail="netting coordinator not initialised")
    if executor._killed:
        raise HTTPException(status_code=423, detail="kill switch active — pooled orders rejected")
    if not is_future_only and executor._enforce_market_hours and not is_market_open():
        raise HTTPException(status_code=409, detail="market closed — pooled equity orders rejected")


@app.post("/target", dependencies=[Depends(require_api_key)])
def set_target(req: TargetRequest):
    """Net-pooling: incremental. Set ONE symbol's absolute target for a strategy; the
    coordinator re-nets and trades the account to the pooled net. Exit = quantity 0."""
    _fut = (req.instrument or {}).get("sec_type", "STK") == "FUT"
    _pool_preflight(_fut)
    probe = {"instrument": {**(req.instrument or {}), "symbol": req.symbol},
             "target_quantity": req.quantity, "expected_price": req.price, "exits": req.exits}
    plan, blocked, refusal = _plan_exits(req.strategy_id, [probe])
    if refusal:
        return refusal
    result = executor.coordinator.set_target(
        req.strategy_id, req.symbol, probe["target_quantity"],
        instrument=req.instrument, price=req.price,
    )
    em = getattr(executor, "exit_manager", None)
    if em is not None and isinstance(result, dict) and result.get("accepted"):
        em.set(req.strategy_id, req.symbol, *plan[req.symbol])
    if isinstance(result, dict) and blocked:
        result["exits_blocked"] = blocked
    crosses = result.get("internal_crosses", []) if isinstance(result, dict) else []
    cross_note = ""
    if crosses:
        cross_note = " [\U0001f504 internally crossed]"
    _alert(
        f"\U0001f3af Target set — {req.strategy_id} {req.symbol} qty={req.quantity}{cross_note}",
        topic="orders",
    )
    return result


@app.post("/targets", dependencies=[Depends(require_api_key)])
def submit_book(body: dict):
    """Net-pooling: full-book resync. Authoritative snapshot of a strategy's whole book;
    any name dropped from the book is closed. body = {strategy_id, intents:[{instrument,
    target_quantity, expected_price, order_type?}]}. Run periodically to self-heal drift.
    order_type "moc" trades that name's change in the closing auction (default: market)."""
    sid = body.get("strategy_id")
    if not sid:
        raise HTTPException(status_code=422, detail="strategy_id required")
    intents = body.get("intents", [])
    fut_only = bool(intents) and all(
        (it.get("instrument") or {}).get("sec_type", "STK") == "FUT" for it in intents)
    _pool_preflight(fut_only)
    refusal = _check_book_order_types(intents)
    if refusal:
        return refusal
    plan, blocked, refusal = _plan_exits(sid, intents)
    if refusal:
        return refusal
    result = executor.coordinator.submit_book(sid, intents)
    em = getattr(executor, "exit_manager", None)
    if isinstance(result, dict):
        accepted = bool(result.get("accepted"))
        if em is not None and accepted:
            em.replace_book(sid, plan)        # authoritative, exits included
        result["exits"] = {"armed": sorted(s for s, (_q, spec, _t) in plan.items()
                                           if spec and accepted),
                           "blocked": blocked}
    orders = result.get("orders", []) if isinstance(result, dict) else []
    crosses = result.get("internal_crosses", []) if isinstance(result, dict) else []
    msg_parts = []
    if crosses:
        cross_summary = ", ".join(
            f"{c['symbol']} {c['side']} {c['quantity']:.0f}@{c['price']:.2f} ({c['strategy_id']})"
            for c in crosses
        )
        msg_parts.append(f"\U0001f504 Internal: {cross_summary}")
    if orders:
        order_summary = ", ".join(
            f"{o.get('symbol')} delta={o.get('delta')} id={o.get('order_id')}"
            for o in orders
        )
        msg_parts.append(f"\U0001f4e6 IB: {order_summary}")
    if msg_parts:
        _alert(f"Book resync — {sid}:\n" + "\n".join(msg_parts), topic="orders")
    # Journal: log every book resync with the order/cross summary
    n_intents = len(intents)
    n_orders = len(orders)
    n_crosses = len(crosses)
    summary = f"Book resync: {n_intents} intents, {n_orders} orders, {n_crosses} internal crosses"
    if blocked:
        summary += f", {len(blocked)} held flat by exit lockout ({', '.join(sorted(blocked))})"
    syms = list({it.get("instrument", {}).get("symbol", "?") for it in intents})
    import json as _json
    executor.logger_db.log_decision(
        sid, "rebalance", summary,
        detail=_json.dumps({"orders": orders, "internal_crosses": crosses}, default=str),
        symbols=syms,
    )
    return result


class ChainRequest(BaseModel):
    """Two legs, each an absolute target for the strategy: {instrument, target_quantity,
    expected_price}. The executor prices both as ATR limits (buy below, sell above); the
    first to fill leads and the other is hedged in proportion, through the pool."""
    strategy_id: str
    legs: list
    atr_fraction: Optional[float] = Field(None, gt=0)
    ttl_sec: Optional[float] = Field(None, gt=0)


@app.post("/chains", dependencies=[Depends(require_api_key)])
def submit_chain(body: ChainRequest):
    """Place a chained two-leg order. Expires at now + ttl_sec, and never later than today's
    MOC cutoff; on expiry what still rests is cancelled and the pair is left balanced."""
    sid = body.strategy_id
    _pool_preflight(False)
    em = getattr(executor, "exit_manager", None)
    problems, legs = [], []
    for i, leg in enumerate(body.legs):
        inst = (leg or {}).get("instrument") or {}
        sym = inst.get("symbol")
        where = sym or f"leg[{i}]"
        try:
            qty, px = float(leg["target_quantity"]), float(leg["expected_price"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"{where}: needs a numeric target_quantity and expected_price")
            continue
        if not sym or inst.get("sec_type", "STK") != "STK":
            problems.append(f"{where}: chain legs are equities with an instrument.symbol")
            continue
        if not (px > 0 and px == px and px != float("inf")):
            problems.append(f"{where}: expected_price must be positive — the ATR limit is "
                            "priced from it")
            continue
        if leg.get("exits"):
            problems.append(f"{where}: chain legs take no exits — arm them with a book once "
                            "the pair is on")
            continue
        held = executor.ledger.strategy_positions.get(sid, {}).get(sym, 0.0)
        lock = em.blocked(sid, sym, qty) if em is not None else None   # as /targets does
        if lock is not None:
            problems.append(f"{where}: re-entry blocked — hit its {lock['kind']} this session")
            continue
        legs.append({"instrument": {"asset_class": "equity", "exchange": "SMART",
                                    "sec_type": "STK", **inst},
                     "target_quantity": qty, "expected_price": px, "held": held})
    if problems:
        return {"accepted": False, "reason": "chain refused, nothing placed: " + "; ".join(problems)}

    # Price both legs BEFORE touching the book: an unavailable ATR refuses the chain rather
    # than quietly sending a leg at market.
    layer = executor.atr_layer
    for leg in legs:
        sym = leg["instrument"]["symbol"]
        buy = leg["target_quantity"] > leg["held"]
        lp = layer.compute_limit_price(sym, leg["expected_price"], buy, fraction=body.atr_fraction)
        if lp is None:
            return {"accepted": False,
                    "reason": f"chain refused, nothing placed: ATR unavailable for {sym}"}
        leg["limit_price"] = lp
        leg["atr"] = layer._get_cached(sym)

    now = time.time()
    cutoff = moc_cutoff_at()
    expires = cutoff.timestamp() if cutoff is not None else now
    if body.ttl_sec:
        expires = min(expires, now + body.ttl_sec)
    if expires <= now:
        return {"accepted": False, "reason": "chain refused, nothing placed: past today's "
                "MOC cutoff — too close to the close for resting legs"}

    result = executor.coordinator.chains.submit(sid, legs, expires, fraction=body.atr_fraction)
    if result.get("accepted"):
        c = result["chain"]
        summary = f"Chain {c['id']}: " + ", ".join(
            f"{l['symbol']} {l['delta']:+g} lmt {l['limit_price']:g}" for l in c["legs"])
        _alert(f"\U0001f517 {sid} — {summary}", topic="orders")
        import json as _json
        executor.logger_db.log_decision(sid, "chain", summary,
                                        detail=_json.dumps(c, default=str),
                                        symbols=[l["symbol"] for l in c["legs"]])
    return result


@app.get("/chains")
def list_chains(strategy_id: Optional[str] = None):
    """Chained orders with each leg's limit, fills and progress; `lead_symbol` is the leg
    that filled first."""
    return {"chains": executor.coordinator.chains.snapshot(strategy_id)}


@app.delete("/chains/{chain_id}", dependencies=[Depends(require_api_key)])
def cancel_chain(chain_id: str):
    """Cancel as on expiry: pull what rests, leave the pair balanced, send nothing at market."""
    c = executor.coordinator.chains.cancel(chain_id)
    if c is None:
        raise HTTPException(status_code=404, detail=f"no chain {chain_id}")
    return c


@app.get("/exits")
def get_exits(strategy_id: Optional[str] = None):
    """Armed exits with their current trigger levels, and today's re-entry lockouts."""
    em = getattr(executor, "exit_manager", None)
    if em is None:
        return {"enabled": False, "rules": {}, "lockouts": {}}
    marks = {**_last_equity.get("marks", {}), **em.last_marks}   # the exit loop's are fresher
    return {"enabled": True, **em.snapshot(marks, strategy_id)}


@app.delete("/exits/{strategy_id}/{symbol}", dependencies=[Depends(require_api_key)])
def clear_exit(strategy_id: str, symbol: str):
    """Remove a name's exit rules AND lift its lockout — the manual override for a lockout
    that should not stand (a stop hit on a bad print, say)."""
    em = getattr(executor, "exit_manager", None)
    if em is None:
        raise HTTPException(status_code=503, detail="exit manager not initialised")
    out = em.clear(strategy_id, symbol)
    if not any(out.values()):
        raise HTTPException(status_code=404,
                            detail=f"no exit rule or lockout for {strategy_id} {symbol}")
    executor.logger_db.log_decision(strategy_id, "exit",
                                    f"exit rules/lockout for {symbol} cleared by hand: {out}",
                                    symbols=[symbol])
    return out


_ACCOUNT_TTL = 30.0
_account_cache: dict = {"ts": 0.0, "accounts": None}
_account_lock = threading.Lock()


@app.get("/account")
def account_summary():
    """IB's own numbers for the account — net liquidation (its NAV), cash, gross positions,
    margin — next to the strategy ledger's NAV from /equity.

    The two differ by design: the ledger only counts capital allocated to strategies, IB
    counts the whole account. `unallocated` is the gap. Cached for 30s (the dashboard polls
    every few seconds and IB allows only two account-summary requests at once). Never an
    HTTP error for "IB didn't answer": `available: false` with the last good figures, so a
    slow gateway doesn't mark the whole dashboard stale."""
    with _account_lock:
        now = time.time()
        if now - _account_cache["ts"] >= _ACCOUNT_TTL and executor.isConnected():
            try:
                fresh = executor.fetch_account_summary()
            except Exception as e:
                logger.warning("account summary failed: %s", e)
                fresh = None
            if fresh:
                _account_cache.update(ts=now, accounts=fresh)
        accounts, ts = _account_cache["accounts"], _account_cache["ts"]
    if not accounts:
        return {"available": False, "accounts": [], "net_liquidation": None}

    def total(tag):
        vals = [v.get(tag) for v in accounts.values() if isinstance(v.get(tag), float)]
        return sum(vals) if vals else None

    net_liq = total("NetLiquidation")
    strategies_nav = (_last_equity.get("totals") or {}).get("nav")
    return {
        "available": True,
        "as_of": datetime.fromtimestamp(ts, timezone.utc).isoformat(),
        "age_sec": round(time.time() - ts, 1),
        "accounts": [{"account": a, **v} for a, v in sorted(accounts.items())],
        "net_liquidation": net_liq,
        "total_cash": total("TotalCashValue"),
        "gross_position_value": total("GrossPositionValue"),
        "unrealized_pnl": total("UnrealizedPnL"),
        "realized_pnl": total("RealizedPnL"),
        "available_funds": total("AvailableFunds"),
        "buying_power": total("BuyingPower"),
        "maint_margin": total("MaintMarginReq"),
        "strategies_nav": strategies_nav,
        "unallocated": (net_liq - strategies_nav
                        if net_liq is not None and strategies_nav is not None else None),
    }


@app.get("/price/{symbol}")
def get_price(symbol: str):
    """Last price for an equity: a market-data snapshot from IB, else the equity sampler's
    last mark. For sizing and the risk check's notional — not a quote to trade against."""
    sym = symbol.strip().upper()
    price, source = None, None
    if executor.isConnected():
        try:
            price, source = executor.fetch_price(sym), "ib"
        except Exception as e:
            logger.warning("price fetch for %s failed: %s", sym, e)
    if not price or price <= 0:
        price, source = (_last_equity.get("marks") or {}).get(sym), "last_mark"
    if not price or price <= 0:
        raise HTTPException(status_code=503, detail=f"no price available for {sym}")
    return {"symbol": sym, "price": float(price), "source": source}


@app.get("/net")
def get_net():
    """Inspect the pooled net book and each strategy's desired book (read-only)."""
    if executor.coordinator is None:
        return {"net": {}, "desired": {}}
    return {"net": executor.coordinator.net(), "desired": executor.coordinator.desired}

# NOTE: declared BEFORE /orders/{order_id} on purpose — FastAPI matches routes in
# declaration order, and the parameterised route would otherwise capture "acks"
# as an order_id and fail it as a non-integer.
@app.get("/orders/acks")
def order_acks(ids: str = ""):
    """Acknowledgement state for specific orders — what a strategy needs after submitting.

    A submission returning `accepted: true` only means the executor handed the order to the
    socket. It says nothing about whether IB took it, so a strategy that stops there records
    a position it may not have. Poll this with the ids from the submission instead.

    `ids` is a comma-separated list; omit it for every order this session.
    """
    wanted = None
    if ids.strip():
        try:
            wanted = {int(i) for i in ids.replace(" ", "").split(",") if i}
        except ValueError:
            raise HTTPException(status_code=422, detail="ids must be comma-separated integers")

    now = time.time()
    out = {}
    for oid, st in executor.order_status.items():
        if wanted is not None and oid not in wanted:
            continue
        out[str(oid)] = {
            "ack": st.get("ack", "live"),
            "status": st.get("status"),
            "symbol": st.get("symbol"),
            "filled": st.get("filled"),
            "remaining": st.get("remaining"),
            "avg_fill_price": st.get("avg_fill_price"),
            "last_error": st.get("last_error"),
            "age_sec": round(now - st["sent_at"], 1) if st.get("sent_at") else None,
        }
    missing = sorted(wanted - {int(k) for k in out}) if wanted else []
    return {"acks": out, "unknown_order_ids": missing,
            "pending": sum(1 for v in out.values() if v["ack"] == "pending"),
            "rejected": sum(1 for v in out.values() if v["ack"] == "rejected")}


@app.get("/orders/{order_id}")
def get_order(order_id: int):
    live = executor.order_status.get(order_id)
    if live is not None:
        return {"order_id": order_id, **live}
    persisted = executor.logger_db.get_order(order_id)
    if persisted is not None:
        return persisted
    raise HTTPException(status_code = 404, detail = f"unknown order_id {order_id}")

@app.get("/positions")
def get_positions():
    return {
        "current_positions": dict(executor.ledger.current_positions),
        "strategy_positions": {
            sid: dict(pos) for sid, pos in executor.ledger.strategy_positions.items()
        },
        "strategy_avg_cost": {
            sid: dict(costs) for sid, costs in executor.ledger.strategy_avg_cost.items()
        },
        # cash is a position too — one balance per strategy, plus its capital basis
        "strategy_cash": executor.ledger.cash_book(),
        "starting_cash": dict(executor.ledger.starting_cash),
        "multipliers": dict(executor.ledger.multipliers),
    }

@app.get("/exposure")
def get_exposure(trigger: float = 0.30, target: float = 0.25, release: float = 0.20):
    """Bucketed exposure, and the hedge that WOULD be placed. Read-only — places nothing.

    Run this alongside the live book before letting anything trade on it: the thresholds are
    the whole design, and the only way to know whether they fire sensibly is to watch them
    against real positions for a while.

    The `unpriced` and `unhedgeable` sections are the honest part. Exposure that could not
    be measured, or that has no hedge instrument, is reported rather than dropped — a
    coverage report that quietly omits what it could not handle is worse than no report.
    """
    marks = dict((_last_equity.get("marks") or {}))
    nav = float((_last_equity.get("totals") or {}).get("nav") or 0.0)
    if nav <= 0:
        raise HTTPException(status_code=503,
                            detail="no NAV sample yet — the equity sampler has not run")

    try:
        policy = hedger.BucketPolicy(trigger=trigger, target=target, release=release)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    exposures, unpriced = hedger.bucket_exposures(
        {sid: dict(pos) for sid, pos in executor.ledger.strategy_positions.items()},
        marks=marks, nav=nav,
        multipliers=dict(executor.ledger.multipliers),
        instruments=dict(getattr(executor, "_instruments", {}) or {}),
    )
    current = {sym: qty for sym, qty in
               (executor.ledger.strategy_positions.get(hedger.HEDGE_STRATEGY_ID) or {}).items()
               if abs(qty) > 1e-9}

    plan = hedger.plan(exposures, nav, prices=marks, default_policy=policy,
                       current_hedge=current, unpriced=unpriced)
    return {
        "nav": nav,
        "exposures": [{"bucket": b, "notional": e.notional, "fraction": e.fraction,
                       "symbols": e.symbols}
                      for b, e in sorted(exposures.items(), key=lambda kv: -abs(kv[1].fraction))],
        "hedge": [{"bucket": d.bucket, "symbol": d.hedge_symbol, "quantity": d.quantity,
                   "notional": d.hedge_notional, "reason": d.reason}
                  for d in plan.decisions],
        "book": plan.book,
        "coverage": plan.coverage,
        "unhedgeable": plan.unhedgeable,
        "policy": {"trigger": trigger, "target": target, "release": release},
    }


@app.get("/pending")
def get_pending():
    """What the ledger believes is still on its way, and whether working orders explain it.

    Pending feeds effective_position, which the netting rebalance trades against. Pending
    held for an order that no longer exists makes a symbol look already on target, and the
    strategy quietly stops getting fills in it — with nothing in the logs. Nothing exposed
    this before. `unexplained` is pending with no working order behind it; it should be
    empty, and a non-empty value is a leak to investigate, not a rounding error."""
    explained, orders = {}, []
    for oid, st in list(executor.order_status.items()):
        if st.get("pending_released") or st.get("status") in (
                "Filled", "Reconciled_Stale", *executor.TERMINAL_UNFILLED):
            continue
        original = float(st.get("pending_qty") or 0.0)
        remainder = original - float(st.get("exec_filled") or 0.0)
        if original == 0.0 or remainder * original <= 0.0:
            continue
        sym = st.get("symbol")
        explained[sym] = explained.get(sym, 0.0) + remainder
        orders.append({"order_id": oid, "symbol": sym, "strategy_id": st.get("strategy_id"),
                       "status": st.get("status"), "net": bool(st.get("net")),
                       "remainder": remainder})
    pending = {sym: q for sym, q in dict(executor.ledger.pending_deltas).items() if abs(q) > 1e-9}
    unexplained = {}
    for sym in set(pending) | set(explained):
        gap = pending.get(sym, 0.0) - explained.get(sym, 0.0)
        if abs(gap) > 1e-6:
            unexplained[sym] = gap
    return {"pending": pending, "explained_by_orders": explained, "orders": orders,
            "unexplained": unexplained, "consistent": not unexplained}


@app.get("/positions/orphans")
def get_orphans():
    """Broker positions no strategy claims — real risk the per-strategy views do not show.

    The equity sampler values NAV from strategy attribution, so an orphan contributes
    nothing to `position_value` while sitting at the broker. Worth checking after a flatten,
    a kill, or any reconcile that adopted positions."""
    orphans = executor.orphaned_positions()
    marks = (_last_equity.get("marks") or {})
    return {
        "orphans": [
            {"symbol": sym, "quantity": qty,
             "mark": marks.get(sym),
             "notional": (abs(qty) * marks[sym] * executor.ledger.multipliers.get(sym, 1.0))
                         if marks.get(sym) else None}
            for sym, qty in sorted(orphans.items())
        ],
        "count": len(orphans),
    }


_WORKING_STATUSES = ("PendingSubmit", "PreSubmitted", "Submitted")


def _bucket_snapshot(sid: str, sym: str) -> dict:
    led = executor.ledger
    return {"strategy_id": sid,
            "quantity": led.strategy_positions.get(sid, {}).get(sym, 0.0),
            "avg_cost": led.strategy_avg_cost.get(sid, {}).get(sym, 0.0),
            "realized": led.strategy_realized_pnl.get(sid, 0.0)}


@app.post("/positions/{symbol}/close_unowned", dependencies=[Depends(require_api_key)])
def close_unowned(symbol: str):
    """Close a position that only an internal bucket (__net__, flatten_all, kill_switch)
    holds — shares at the broker that no strategy owns.

    The market order is placed under that bucket's id, so its fill lands back in the same
    bucket and nets it to zero; /flatten's orphan sweep books the close to flatten_all
    instead, which zeroes the account but leaves __net__ and flatten_all holding equal and
    opposite phantom lots. Refuses whenever the close could touch shares a real strategy
    owns, or the records for the symbol don't add up — those want /positions/transfer."""
    sym = symbol.strip().upper()
    led = executor.ledger
    internal = {sid: led.strategy_positions.get(sid, {}).get(sym, 0.0)
                for sid in executor.INTERNAL_SIDS}
    internal = {sid: q for sid, q in internal.items() if abs(q) > 1e-9}
    if not internal:
        raise HTTPException(status_code=404, detail=f"no internal bucket holds {sym}")
    if len(internal) > 1:
        raise HTTPException(
            status_code=409,
            detail=f"{sym} is split across {internal} — /positions/transfer it into one "
                   "bucket first")
    (owner, qty), = internal.items()
    real = {sid: pos.get(sym) for sid, pos in led.strategy_positions.items()
            if sid not in executor.INTERNAL_SIDS and abs(pos.get(sym, 0.0)) > 1e-9}
    if real:
        raise HTTPException(
            status_code=409,
            detail=f"strategies also hold {sym} ({real}) — a close at the account would trade "
                   "their shares too. Correct the records with /positions/transfer instead.")
    account = led.current_positions.get(sym, 0.0)
    if abs(account - qty) > 1e-6:
        raise HTTPException(
            status_code=409,
            detail=f"records for {sym} don't add up: account {account:g}, {owner} {qty:g} — "
                   "run /reconcile first")
    working = sorted(oid for oid, st in executor.order_status.items()
                     if st.get("symbol") == sym and st.get("status") in _WORKING_STATUSES)
    if working:
        raise HTTPException(status_code=409,
                            detail=f"{sym} has working order(s) {working} — cancel them first")
    if not executor.isConnected():
        raise HTTPException(status_code=503, detail="not connected to IB")
    if executor._enforce_market_hours and not is_market_open():
        raise HTTPException(status_code=409, detail="market closed")
    inst = executor._instruments.get(sym) or {"symbol": sym, "asset_class": "equity",
                                              "sec_type": "STK", "exchange": "SMART"}
    if inst.get("sec_type", "STK") != "STK" or led.multipliers.get(sym, 1.0) != 1.0:
        raise HTTPException(status_code=422,
                            detail=f"{sym} is not a plain equity — close it manually")

    intent = {
        "client_order_id": f"unowned-{sym}-{int(time.time() * 1000)}",
        "strategy_id": owner,                  # the fill nets this bucket back to zero
        "instrument": inst,
        "side": "sell" if qty > 0 else "buy",
        "quantity": abs(qty),
        "order_type": "market",
        "time_in_force": "day",
        "expected_price": (executor._ref_value.get(sym)
                           or led.strategy_avg_cost.get(owner, {}).get(sym) or None),
    }
    order_id = executor.place_order(intent)
    summary = f"closing unowned {qty:+g} {sym} held by {owner}: {intent['side']} {abs(qty):g}"
    executor.logger_db.log_decision(owner, "close_unowned", summary, detail=f"order {order_id}")
    _alert(f"\U0001f9f9 {summary} (order {order_id})", topic="orders")
    return {"symbol": sym, "bucket": owner, "closing": qty, "order_id": order_id}


class TransferRequest(BaseModel):
    """Move part of one bucket's position into another — a records-only correction. No
    order is sent: the account's position does not change, only who it is booked to.
    `quantity` is signed as `from_strategy` holds it: +7 moves a long 7, -7 a short 7."""
    symbol: str
    from_strategy: str
    to_strategy: str
    quantity: float
    price: Optional[float] = Field(default=None, gt=0)
    reason: str = ""


@app.post("/positions/transfer", dependencies=[Depends(require_api_key)])
def transfer_position(req: TransferRequest):
    """Re-book shares between strategies (or internal buckets) without trading.

    Booked as an internal cross at `price` (default: the source's average cost), so it is
    zero-sum: the account position, total cash and the sum of strategy positions are all
    unchanged, and each side's cash, average cost and realized P&L move exactly as for a
    real cross at that price. It only moves shares the source actually holds — it can never
    increase or flip the source's position, so it cannot invent exposure."""
    sym = req.symbol.strip().upper()
    src, dst = req.from_strategy.strip(), req.to_strategy.strip()
    known = set(CONFIG) | set(executor.INTERNAL_SIDS)
    for sid in (src, dst):
        if sid not in known:
            raise HTTPException(status_code=404, detail=f"unknown strategy {sid}")
    if src == dst:
        raise HTTPException(status_code=422, detail="from_strategy and to_strategy are the same")
    q = float(req.quantity)
    led = executor.ledger
    held = led.strategy_positions.get(src, {}).get(sym, 0.0)
    if abs(q) < 1e-9 or held * q <= 0 or abs(q) > abs(held) + 1e-9:
        raise HTTPException(
            status_code=409,
            detail=f"{src} holds {held:g} {sym} — quantity must be part of that, same sign")
    price = req.price or led.strategy_avg_cost.get(src, {}).get(sym)
    if not price or price <= 0:
        raise HTTPException(status_code=422, detail=f"no price for {sym} — pass one")

    before = {"from": _bucket_snapshot(src, sym), "to": _bucket_snapshot(dst, sym)}
    led.apply_internal_cross(sym, -q, price, src)
    led.apply_internal_cross(sym, q, price, dst)
    after = {"from": _bucket_snapshot(src, sym), "to": _bucket_snapshot(dst, sym)}

    summary = f"transferred {q:+g} {sym} from {src} to {dst} at {price:g}"
    detail = f"{req.reason or 'records correction'}; before={before}; after={after}"
    for sid in (src, dst):
        executor.logger_db.log_decision(sid, "position_transfer", summary, detail=detail)
    try:
        led.save_state(executor.logger_db)
    except Exception as e:
        logger.critical("position transfer applied but NOT saved — it reverts on restart: %s", e)
        raise HTTPException(status_code=500,
                            detail=f"applied in memory but NOT saved (reverts on restart): {e}")
    _alert(f"\U0001f501 {summary}", topic="orders")
    return {"symbol": sym, "price": price, "before": before, "after": after,
            "account_position": led.current_positions.get(sym, 0.0)}


@app.get("/pnl")
def get_pnl():
    # realized_pnl is NET of commissions — apply_fee deducts them as IB reports them — and
    # `fees` is the running total deducted, so gross trading P&L is realized + fees.
    return {"realized_pnl": dict(executor.ledger.strategy_realized_pnl),
            "fees": dict(getattr(executor.ledger, "strategy_fees", {}) or {})}


@app.get("/equity")
def get_equity():
    """Latest sampled balance sheet: per-strategy cash + position value + NAV, and the
    portfolio totals. Served from the sampler's cache so a dashboard poll never triggers
    a market-data round trip."""
    return _last_equity


@app.get("/strategies/{strategy_id}/book")
def strategy_book(strategy_id: str):
    """One strategy's holdings with CASH as the first line, marked at the last sampled
    prices."""
    return {"strategy_id": strategy_id,
            "book": executor.ledger.strategy_book(strategy_id, _last_equity.get("marks", {}))}

@app.get("/health")
def health():
    return {
        "connected": executor.isConnected(),
        "killed": executor._killed,
        "market_open": is_market_open(),
        # True when startup could not reconcile against the broker: the executor is up and
        # inspectable but killed, because its position state is untrusted.
        "startup_degraded": getattr(executor, "_startup_degraded", False),
    }


@app.post("/kill", dependencies=[Depends(require_api_key)])
def kill(req: KillRequest):
    executor.kill_switch(flatten=req.flatten)
    _alert(
        f"\U0001f6d1 KILL SWITCH activated (flatten={req.flatten})",
        topic="errors",
    )
    return {"killed": True, "flattened": req.flatten}

@app.post("/unkill", dependencies=[Depends(require_api_key)])
def unkill():
    """Clear the kill switch and a tripped circuit breaker so orders flow again.
    Strategies halted by the breaker stay halted — reactivate them individually."""
    executor.clear_kill_switch()
    _alert("\u2705 Kill switch CLEARED — new orders accepted again", topic="errors")
    return {"killed": executor._killed, "circuit_broken": executor._circuit_broken}


class HaltRequest(BaseModel):
    flatten: bool = True
    reason: str = ""


@app.post("/strategies/{strategy_id}/halt", dependencies=[Depends(require_api_key)])
def halt_strategy(strategy_id: str, req: HaltRequest = HaltRequest()):
    """Stop ONE strategy (and by default close its book). The manual twin of the
    automatic drawdown halt — same path, so the unwind and its retry behave identically."""
    if strategy_id not in CONFIG:
        raise HTTPException(status_code=404, detail=f"unknown strategy {strategy_id}")
    if not executor.risk_manager.is_active(strategy_id):
        return {"strategy_id": strategy_id, "status": "halted", "note": "already halted"}
    reason = req.reason or "manual halt"
    if req.flatten:
        executor.halt_and_flatten(strategy_id, f"MANUAL HALT: {reason}")
    else:
        executor.risk_manager.halt_strategy(strategy_id, reason)
        executor.logger_db.save_halted_strategies(
            set(), executor.risk_manager._active_strategies, set(CONFIG.keys()), reason)
        executor.logger_db.log_decision(strategy_id, "halt", f"HALTED (no flatten): {reason}")
    return {"strategy_id": strategy_id, "status": "halted", "flattened": req.flatten}


@app.post("/strategies/{strategy_id}/flatten", dependencies=[Depends(require_api_key)])
def flatten_strategy(strategy_id: str):
    """Close one strategy's book WITHOUT halting it — it resumes on its next signal."""
    if strategy_id not in CONFIG:
        raise HTTPException(status_code=404, detail=f"unknown strategy {strategy_id}")
    result = executor.flatten_strategy(strategy_id)
    if result.get("flattened"):
        _alert(f"\U0001f4a8 Flatten {strategy_id}: "
               + ", ".join(f"{p['symbol']} {p['quantity']:+.0f}" for p in result["flattened"]),
               topic="orders")
    return result


@app.post("/flatten", dependencies=[Depends(require_api_key)])
def flatten_all():
    """Cancel open orders and flatten every position WITHOUT setting the kill switch.
    Strategies remain active and can resume trading on the next signal.
    If the market is CLOSED, only cancels open orders (no new closing orders placed)."""
    market_open = is_market_open()

    # 1. cancel open orders (always, regardless of market hours)
    cancelled = []
    for oid, status in list(executor.order_status.items()):
        if status.get("status") in ("PreSubmitted", "Submitted"):
            # cancel_order releases the order's unfilled pending and marks it PendingCancel.
            # A bare cancelOrder did neither: the closing orders below were sized against
            # shares still counted as on the way, and the rebalance then cancelled the same
            # order a second time.
            if executor.cancel_order(oid, "flatten"):
                cancelled.append(oid)

    # 2. flatten per-strategy — ONLY when market is open
    _INTERNAL = {"__net__", "flatten_all", "kill_switch"}
    flattened = []

    if market_open:
        # Collect what we're about to flatten (for the response)
        for sid, positions in list(executor.ledger.strategy_positions.items()):
            if sid in _INTERNAL:
                continue
            for symbol, qty in list(positions.items()):
                if abs(qty) < 1e-9:
                    continue
                flattened.append({"strategy_id": sid, "symbol": symbol, "qty": qty})

        if getattr(executor, "coordinator", None) is not None:
            # Coordinator path: zero out desired books and rebalance — the coordinator
            # internally crosses offsetting legs and sends net residual to IB.  Fills
            # flow back through attribute_fill, correctly updating each strategy.
            all_syms = set()
            for sid in list(executor.coordinator.desired):
                all_syms |= set(executor.coordinator.desired[sid])
                executor.coordinator.desired[sid] = {}
            # Also include symbols from strategy_positions (desired may already be empty
            # from a previous flatten, but positions still need closing)
            for sid, positions in executor.ledger.strategy_positions.items():
                if sid in _INTERNAL:
                    continue
                all_syms |= {s for s, q in positions.items() if abs(q) > 1e-9}
            # ...and anything the BROKER holds that no strategy claims. Every set above is
            # built from strategy attribution, so without this a position owned by nobody
            # survives a flatten that reports success.
            orphans = executor.orphaned_positions()
            for sym, qty in orphans.items():
                flattened.append({"strategy_id": None, "symbol": sym, "qty": qty})
            all_syms |= set(orphans)
            executor.coordinator._save()
            if all_syms:
                executor.coordinator._rebalance(all_syms, urgent=True)
        else:
            # No coordinator: flatten each strategy directly
            for sid, positions in list(executor.ledger.strategy_positions.items()):
                if sid in _INTERNAL:
                    continue
                if any(abs(q) > 1e-9 for q in positions.values()):
                    executor._flatten_direct(sid)
            for entry in executor._flatten_orphans():
                flattened.append({"strategy_id": None, **entry})

        # Clean up stale strategy positions for symbols already flat at the broker.
        # This handles leftover state from before per-strategy attribution was added.
        broker_flat = {s for s, q in executor.ledger.current_positions.items() if abs(q) < 1e-9}
        cleaned = []
        for sid, positions in list(executor.ledger.strategy_positions.items()):
            if sid in _INTERNAL:
                continue
            for sym in list(positions):
                if sym in broker_flat and abs(positions.get(sym, 0.0)) > 1e-9:
                    # write_off_position returns the cost basis to cash, so the strategy's
                    # NAV stays consistent instead of losing the phantom leg's value
                    was = executor.ledger.write_off_position(sid, sym)
                    if was:
                        cleaned.append({"strategy_id": sid, "symbol": sym, "was": was})
        if cleaned:
            executor.ledger.save_state(executor.logger_db)
            logging.getLogger("executor").info("Flatten cleanup: zeroed stale strategy positions: %s", cleaned)

    action = "cancelled orders + flattened positions" if market_open else "cancelled orders only (market closed)"
    _alert(
        f"💨 FLATTEN ALL: cancelled {len(cancelled)} orders, flattening {len(flattened)} positions — {action} (kill switch NOT set)",
        topic="orders",
    )
    return {
        "cancelled_orders": len(cancelled),
        "flattened_positions": flattened,
        "market_open": market_open,
        "note": "cancel-only mode, market is closed" if not market_open else None,
        "kill_switch": False,
    }


@app.get("/strategies/{strategy_id}/status")
def strategy_status(strategy_id: str):
    if strategy_id not in CONFIG:
        raise HTTPException(status_code=404, detail=f"unknown strategy {strategy_id}")
    # is_active() is a small getter added to RiskManager (see note below)
    active = executor.risk_manager.is_active(strategy_id)
    return {"strategy_id": strategy_id, "status": "active" if active else "halted"}


@app.get("/strategies/{strategy_id}/allocation")
def strategy_allocation(strategy_id: str):
    cfg = CONFIG.get(strategy_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"unknown strategy {strategy_id}")
    return {
        "strategy_id": strategy_id,
        "capital_allocation": cfg["capital_allocation"],
        "max_drawdown": cfg["max_drawdown"],
    }

class AllocationRequest(BaseModel):
    """Either an absolute `capital_allocation` or a `delta` to apply to the current one."""
    capital_allocation: Optional[float] = None
    delta: Optional[float] = None
    method: str = Field("pro_rata", pattern="^(pro_rata|equal)$")
    dry_run: bool = False


@app.post("/strategies/{strategy_id}/allocation", dependencies=[Depends(require_api_key)])
def set_allocation(strategy_id: str, req: AllocationRequest):
    """Re-allocate capital to a strategy.

    An increase goes straight to cash. A decrease comes out of cash first; anything cash
    can't cover is raised by selling positions — `pro_rata` (default) shrinks every position
    by the same fraction so the book keeps its shape, `equal` splits the amount evenly in
    dollars and redistributes whatever a small position can't cover. `dry_run` returns the
    plan without moving anything.
    """
    cfg = CONFIG.get(strategy_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"unknown strategy {strategy_id}")
    if (req.capital_allocation is None) == (req.delta is None):
        raise HTTPException(status_code=422,
                            detail="pass exactly one of capital_allocation or delta")
    target = (req.capital_allocation if req.capital_allocation is not None
              else cfg["capital_allocation"] + req.delta)
    try:
        result = executor.rebalance_allocation(strategy_id, target,
                                               method=req.method, dry_run=req.dry_run)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not req.dry_run:
        sells = result.get("liquidations") or []
        _alert(
            f"\U0001f4b0 Allocation {strategy_id}: {result['allocation_before']:,.0f} -> "
            f"{result['allocation_after']:,.0f}"
            + (f"\nSelling to raise {result.get('cash_shortfall', 0):,.0f}: "
               + ", ".join(f"{c['symbol']} {c['from_quantity']:g}->{c['to_quantity']:g}"
                           for c in sells) if sells else " (cash only)"),
            topic="orders")
    return result


@app.post("/strategies/{strategy_id}/reactivate", dependencies=[Depends(require_api_key)])
def reactivate_strategy(strategy_id: str):
    """Clear a halt: re-add the strategy to the active set (after reviewing a drawdown halt,
    or to reset a halt-test strategy without restarting the server)."""
    if strategy_id not in CONFIG:
        raise HTTPException(status_code=404, detail=f"unknown strategy {strategy_id}")
    executor.risk_manager.reactivate_strategy(strategy_id)
    # persist the reactivation so it survives a restart
    executor.logger_db.save_halted_strategies(
        set(), executor.risk_manager._active_strategies, set(CONFIG.keys()))
    executor.logger_db.log_decision(strategy_id, "reactivate",
                                    f"Strategy {strategy_id} reactivated via API")
    return {"strategy_id": strategy_id, "status": "active"}


@app.post("/reset_daily", dependencies=[Depends(require_api_key)])
def reset_daily():
    """Reset the portfolio circuit-breaker daily baseline (call at the open) and clear a
    tripped breaker. The next sampler cycle re-captures the baseline from current equity."""
    executor.reset_daily_baseline()
    return {"reset": True, "circuit_broken": executor._circuit_broken}


# last reconcile result for dashboard polling
_last_reconcile = {"matched": None, "discrepancies": [], "ts": None}

@app.post("/reconcile", dependencies=[Depends(require_api_key)])
def reconcile():
    try:
        result = executor.reconcile_and_log()   # pulls reqPositions, corrects ledger to broker + open orders
    except TimeoutError as e:
        raise HTTPException(status_code=503, detail=f"reconcile timed out talking to IB: {e}")

    order_recon = result.get("order_reconcile", {})
    _last_reconcile.update({
        "matched": result["matched"] and order_recon.get("matched", True),
        "discrepancies": result["discrepancies"],
        "order_reconcile": order_recon,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    return {
        "matched": result["matched"],
        "discrepancies": result["discrepancies"],
        "positions": dict(executor.ledger.current_positions),
        "order_reconcile": order_recon,
    }


@app.get("/reconcile/status")
def reconcile_status():
    return _last_reconcile


# ------------------------------------------------------------------
# Dashboard (read-only monitor) + the list endpoints it needs
# ------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


_INTERNAL_STRATS = {"__net__", "flatten_all", "kill_switch"}


@app.get("/orders")
def list_orders():
    # live, session-scoped view from in-memory order_status (richest: filled/remaining)
    # Enrich __net__ orders with the contributing strategy names so the dashboard
    # shows something meaningful instead of "__net__"
    orders = []
    for oid, st in executor.order_status.items():
        entry = {"order_id": oid, **st}
        if st.get("strategy_id") == "__net__" and executor.coordinator:
            sym = st.get("symbol")
            contributors = [sid for sid, book in executor.coordinator.desired.items()
                            if sym and sym in book and abs(book[sym]) > 1e-9]
            if not contributors:
                # desired already zeroed (post-flatten); check strategy_positions
                contributors = [sid for sid, pos in executor.ledger.strategy_positions.items()
                                if sid not in _INTERNAL_STRATS and sym
                                and abs(pos.get(sym, 0)) > 1e-9]
            entry["strategy_id"] = ", ".join(contributors) if contributors else "__net__"
        orders.append(entry)

    # Split on whether IB has actually acknowledged the order. Everything above is our own
    # record of what we SENT; only `ack != "pending"` means the broker has it.
    #
    # Deliberately split rather than filtered: hiding unacknowledged orders would have made
    # the read-only-gateway incident invisible — an empty dashboard and no explanation for
    # why nothing traded. The useful signal is "8 sent, 0 acknowledged", plus IB's reason.
    now = time.time()
    live, pending = [], []
    for entry in orders:
        (pending if entry.get("ack", "live") == "pending" else live).append(entry)
        if entry.get("sent_at"):
            entry["age_sec"] = round(now - entry["sent_at"], 1)

    stuck = [e for e in pending if (e.get("age_sec") or 0) > UNACKED_WARN_SEC]
    return {
        "orders": live,
        "unacknowledged": pending,
        "unacknowledged_count": len(pending),
        # A single order awaiting acknowledgement is normal for a moment. Several, or one
        # that has waited, means the gateway is not accepting orders — say so plainly.
        "warning": (f"{len(pending)} order(s) not acknowledged by IB"
                    + (f", oldest {max(e['age_sec'] for e in stuck):.0f}s" if stuck else "")
                    + " — the broker may be refusing orders (check read-only mode)")
                   if stuck else None,
    }


@app.get("/fills")
def list_fills(limit: int = 50):
    # Filter out internal pseudo-strategy fills server-side so the dashboard
    # never sees them regardless of browser cache
    all_fills = executor.logger_db.get_recent_fills(limit * 3)  # fetch extra to compensate for filtering
    filtered = [f for f in all_fills if f.get("strategy_id") not in _INTERNAL_STRATS]
    return {"fills": filtered[:limit]}


class NewStrategyRequest(BaseModel):
    """A strategy to add to the allowlist at runtime.

    `max_drawdown` is a FRACTION of the allocation (0.15 = halt at a 15% loss), and it is
    required rather than defaulted because a strategy without one has its drawdown halt
    silently disabled — see validate_config in config.py."""
    strategy_id: str = Field(..., min_length=1, max_length=64)
    capital_allocation: float = Field(..., gt=0)
    max_drawdown: float = Field(..., gt=0, le=1)
    starting_cash: Optional[float] = Field(None, ge=0)
    created_by: str = ""


@app.post("/strategies", dependencies=[Depends(require_api_key)])
def add_strategy(req: NewStrategyRequest):
    """Register a new strategy on the allowlist without a restart.

    CONFIG is fail-closed — an intent from an unknown strategy_id is rejected — and the
    ledger and risk manager both hold a reference to that same dict, so adding the entry
    here is what makes the strategy tradeable. Two things make this safe to expose:

      * it PERSISTS BEFORE it takes effect. A strategy live in memory but missing from the
        database vanishes at the next restart, and its orders start being rejected in the
        middle of a session with nothing to explain why;
      * it refuses to touch an id that already exists. Overwriting a live strategy's
        allocation is what /strategies/{id}/allocation is for — that path knows how to
        raise cash by selling, this one would just move the cap out from under an open book.
    """
    sid = req.strategy_id.strip()
    if not sid:
        raise HTTPException(status_code=422, detail="strategy_id required")
    if not all(c.isalnum() or c in "_-" for c in sid):
        raise HTTPException(status_code=422,
                            detail="strategy_id may only contain letters, digits, _ and -")
    if sid in CONFIG:
        raise HTTPException(
            status_code=409,
            detail=f"{sid} already exists — use POST /strategies/{sid}/allocation to change "
                   "its capital")

    entry = {"capital_allocation": float(req.capital_allocation),
             "max_drawdown": float(req.max_drawdown)}
    if req.starting_cash is not None:
        entry["starting_cash"] = float(req.starting_cash)

    problems = validate_config({sid: entry})
    if problems:
        raise HTTPException(status_code=422, detail="; ".join(problems))

    # Persist FIRST. If this raises, nothing has been mutated and the caller gets a clean
    # failure, rather than a strategy that trades today and disappears tomorrow.
    try:
        executor.logger_db.save_runtime_strategy(
            sid, entry["capital_allocation"], entry["max_drawdown"],
            entry.get("starting_cash"), req.created_by)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"could not persist {sid}, so it was NOT added: {e}")

    CONFIG[sid] = entry                       # shared with the ledger and the risk manager
    executor.risk_manager._active_strategies.add(sid)
    basis = executor.ledger._basis(sid)       # seeds cash from the entry we just installed

    executor.logger_db.log_decision(
        sid, "add_strategy",
        f"Strategy {sid} added: allocation {entry['capital_allocation']:,.0f}, "
        f"max_drawdown {entry['max_drawdown']:.0%}",
        detail=f"created_by={req.created_by or 'api'}, starting_cash={basis:,.2f}")
    _alert(f"\U0001f195 Strategy added: {sid} — allocation {entry['capital_allocation']:,.0f}, "
           f"max drawdown {entry['max_drawdown']:.0%}", topic="orders")

    return {"strategy_id": sid, "active": True, "starting_cash": basis, **entry}


@app.delete("/strategies/{strategy_id}", dependencies=[Depends(require_api_key)])
def remove_strategy(strategy_id: str):
    """Remove a runtime-added strategy. Refuses while it still holds anything — removing a
    strategy with an open book would orphan those positions: the executor would keep them at
    the broker with no allowlist entry to reconcile, halt, or flatten them through."""
    if strategy_id not in CONFIG:
        raise HTTPException(status_code=404, detail=f"unknown strategy {strategy_id}")
    if strategy_id not in executor.logger_db.load_runtime_strategies():
        raise HTTPException(status_code=409,
                            detail=f"{strategy_id} is defined in config.py, not at runtime — "
                                   "remove it there and restart")
    open_names = {sym: qty for sym, qty
                  in (executor.ledger.strategy_positions.get(strategy_id) or {}).items()
                  if qty}
    if open_names:
        raise HTTPException(
            status_code=409,
            detail=f"{strategy_id} still holds {', '.join(open_names)} — flatten it first")

    executor.logger_db.delete_runtime_strategy(strategy_id)
    CONFIG.pop(strategy_id, None)
    executor.risk_manager._active_strategies.discard(strategy_id)
    executor.logger_db.log_decision(strategy_id, "remove_strategy",
                                    f"Strategy {strategy_id} removed")
    _alert(f"\U0001f5d1 Strategy removed: {strategy_id}", topic="orders")
    return {"strategy_id": strategy_id, "removed": True}


@app.get("/strategies")
def list_strategies():
    return {"strategies": [
        {
            "strategy_id": sid,
            "capital_allocation": cfg["capital_allocation"],
            "max_drawdown": cfg["max_drawdown"],
            "active": executor.risk_manager.is_active(sid),
            "funded": is_funded(sid, cfg),
        }
        for sid, cfg in CONFIG.items()
    ]}

_sampler_stop = threading.Event()

# Latest sampled balance sheet, refreshed by _equity_sampler and served by GET /equity.
_last_equity: dict = {"ts": None, "strategies": {}, "totals": {}, "marks": {}}

# Consecutive sampler cycles a strategy's unrealized-drawdown check was skipped for want of
# a fresh mark. Escalates to a CRITICAL (-> Telegram) so the gap can't stay silent.
_stale_skips: dict = {}
_STALE_SKIP_ALERT_AFTER = 5

# Consecutive whole-cycle sampler failures. The sampler is the only path that halts on
# UNREALIZED losses, so it failing repeatedly has to page rather than log quietly.
_sampler_failures = [0]
_SAMPLER_FAIL_ALERT_AFTER = 3


def _portfolio_totals(strategies: dict) -> dict:
    """Portfolio balance sheet: the sum of every strategy's positions and cash. Cached for
    the dashboard and Telegram (GET /equity).

    An UNFUNDED strategy (test fixture, demo, hedge overlay — see config.is_funded) has no
    real capital behind its allocation, so only what it has actually done counts: its
    positions and P&L, never its nominal starting cash. Its cash enters as cash - basis,
    which keeps  nav == cash + position_value  and  equity == nav - starting_cash  for the
    totals exactly as for each strategy."""
    keys = ("cash", "position_value", "nav", "starting_cash", "realized", "unrealized", "equity")
    totals = dict.fromkeys(keys, 0.0)
    for sid, v in strategies.items():
        basis = 0.0 if is_funded(sid) else v.get("starting_cash", 0.0)
        for k in keys:
            totals[k] += v.get(k, 0.0)
        totals["cash"] -= basis
        totals["nav"] -= basis
        totals["starting_cash"] -= basis
    return totals


def _equity_sampler(interval: float = 60.0):
    log = logging.getLogger("executor")
    while not _sampler_stop.is_set():
        try:
            symbols = set()
            for pos in executor.ledger.strategy_positions.values():
                symbols |= {s for s, q in pos.items() if q != 0}
            marks = executor.get_marks(symbols) if symbols else {}
            ts = datetime.now(timezone.utc).isoformat()
            snap = executor.ledger.equity_snapshot(marks)
            for sid in CONFIG:  # ensure every configured strategy has a point, even flat
                basis = executor.ledger.starting_cash.get(sid, 0.0)
                snap.setdefault(sid, {"realized": 0.0, "unrealized": 0.0, "equity": 0.0,
                                      "cash": basis, "position_value": 0.0, "nav": basis,
                                      "starting_cash": basis, "unmarked": []})
            _INTERNAL = {"__net__", "flatten_all", "kill_switch"}
            visible = {}
            for strat, v in snap.items():
                if strat in _INTERNAL:
                    continue
                visible[strat] = v
                # Recording history must never cost us a risk check — see the enforcement
                # block below.
                try:
                    executor.logger_db.log_equity(
                        ts, strat, v["realized"], v["unrealized"], v["equity"],
                        cash=v.get("cash"), position_value=v.get("position_value"),
                        nav=v.get("nav"),
                    )
                except Exception as e:
                    log.error("equity snapshot not recorded for %s: %s", strat, e)
            totals = _portfolio_totals(visible)
            _last_equity.update({"ts": ts, "strategies": visible, "totals": totals,
                                 "marks": {k: v for k, v in marks.items() if v is not None}})
            # ---- risk enforcement -------------------------------------------------
            # Isolated from everything above. This loop is the ONLY thing that halts a
            # strategy on unrealized losses, and a single throw anywhere earlier in the
            # cycle used to skip it entirely — silently, every cycle, logged at ERROR.
            _enforce(log, snap)
            _sampler_failures[0] = 0
        except Exception as e:
            _sampler_failures[0] += 1
            if _sampler_failures[0] in (1, _SAMPLER_FAIL_ALERT_AFTER):
                level = log.critical if _sampler_failures[0] >= _SAMPLER_FAIL_ALERT_AFTER else log.error
                level("equity sampler error (%d in a row)%s: %s", _sampler_failures[0],
                      " — UNREALIZED DRAWDOWN CHECKS ARE NOT RUNNING"
                      if _sampler_failures[0] >= _SAMPLER_FAIL_ALERT_AFTER else "", e)
            else:
                log.error("equity sampler error (%d in a row): %s", _sampler_failures[0], e)
        _sampler_stop.wait(interval)   # sleep, wakes early on stop


_exit_failures = [0]
_EXIT_FAIL_ALERT_AFTER = 3
_EXIT_FAIL_REALERT_EVERY = 60       # ~30 minutes at the default cadence


def _check_exits() -> list:
    """One exit pass: re-price every name with an armed exit straight from IB, then check.

    Its own loop rather than a step in the equity sampler: exits run at their own, faster
    cadence without doubling the equity history, and a failure in one cannot skip the other.
    Runs even with nothing to price — lockout expiry and retries of closes that did not take
    need no marks."""
    em = getattr(executor, "exit_manager", None)
    if em is None:
        return []
    symbols = em.watched_symbols()
    marks = executor.get_marks(symbols) if symbols else {}
    return em.check(marks, market_open=is_market_open())


def _expire_chains() -> list:
    """Cancel chained orders past their expiry (on the exit loop's cadence)."""
    co = getattr(executor, "coordinator", None)
    if co is None:
        return []
    expired = co.chains.expire()
    for c in expired:
        _alert(f"\u23f1 Chain expired — {c['strategy_id']} {c['id']}: " + ", ".join(
            f"{l['symbol']} {l['filled']:+g}/{l['delta']:+g}" for l in c["legs"]),
            topic="orders")
    return expired


def _exit_sampler(interval: float = 30.0):
    log = logging.getLogger("executor")
    while not _sampler_stop.is_set():
        started = time.monotonic()
        try:
            _expire_chains()
        except Exception as e:
            log.critical("chain expiry failed — resting chain legs may outlive their "
                         "expiry: %s", e)
        try:
            _check_exits()
            _exit_failures[0] = 0
        except Exception as e:
            _exit_failures[0] += 1
            n = _exit_failures[0]
            if n == _EXIT_FAIL_ALERT_AFTER or (
                    n > _EXIT_FAIL_ALERT_AFTER and n % _EXIT_FAIL_REALERT_EVERY == 0):
                log.critical("EXIT CHECKS FAILING (%d in a row) — stop-losses, take-profits and "
                             "trailing stops are NOT being enforced: %s", n, e)
            else:
                log.error("exit check failed (%d in a row): %s", n, e)
        # a fixed cadence: a slow price fetch shortens the wait rather than stretching the cycle
        _sampler_stop.wait(max(1.0, interval - (time.monotonic() - started)))


def _enforce(log, snap: dict) -> None:
    """Portfolio breaker + per-strategy drawdown halts + retry of halted-but-holding
    unwinds.

    Every step is guarded on its own: this is the only path that halts a strategy on
    UNREALIZED losses, so neither a bad strategy nor a failed write may stop the rest of
    the book from being checked. Failures here page (CRITICAL -> Telegram) — an unguarded
    strategy must never be a quiet log line."""
    try:
        # portfolio circuit breaker on total equity (cash + positions across all strategies;
        # the baseline is a same-basis level, so the loss it measures is unchanged)
        executor.enforce_daily_loss(sum(v.get("nav", 0.0) for v in snap.values()))
    except Exception as e:
        log.critical("portfolio circuit breaker did not run: %s", e)

    for sid in CONFIG:
        try:
            # A halted strategy that is still holding gets its unwind retried — the
            # one-shot halt+flatten can fail (rejection, market closed, partial fill)
            # and would otherwise never try again.
            if not executor.risk_manager.is_active(sid):
                executor.ensure_flat(sid)
                continue

            # Per-strategy total-equity drawdown. SKIP a strategy if any held symbol's mark
            # is stale/missing: halting+flattening on incomplete unrealized data is worse
            # than waiting, and the realized fast path still guards it on every fill.
            held = [s for s, q in executor.ledger.strategy_positions.get(sid, {}).items()
                    if q != 0]
            stale = [s for s in held if not executor.mark_is_fresh(s)]
            if stale:
                # Skipping is deliberate, but an UNBOUNDED skip is an unguarded strategy —
                # escalate so the gap can't stay invisible.
                n = _stale_skips.get(sid, 0) + 1
                _stale_skips[sid] = n
                if n == _STALE_SKIP_ALERT_AFTER:
                    log.critical(
                        "UNREALIZED DRAWDOWN CHECK UNGUARDED — %s skipped %d cycles: no "
                        "fresh mark for %s. Only the realized-P&L halt is protecting it.",
                        sid, n, ", ".join(stale))
                else:
                    log.warning("total-drawdown check skipped for %s (stale/missing mark: %s)",
                                sid, ", ".join(stale))
                continue
            _stale_skips.pop(sid, None)

            # "equity" here is P&L (realized + unrealized) — the drawdown limit is a
            # fraction of allocation, not of NAV.
            executor.enforce_drawdown(sid, snap.get(sid, {}).get("equity", 0.0), "total")
        except Exception as e:
            log.critical("DRAWDOWN CHECK FAILED for %s — strategy unguarded this cycle: %s",
                         sid, e)


class JournalEntry(BaseModel):
    strategy_id: str
    event_type: str
    summary: str
    detail: str = ""
    symbols: list[str] = Field(default_factory=list)


@app.post("/journal", dependencies=[Depends(require_api_key)])
def post_journal(entry: JournalEntry):
    """External journal entry — strategies log their signal/rebalance decisions here."""
    executor.logger_db.log_decision(
        entry.strategy_id, entry.event_type, entry.summary,
        detail=entry.detail, symbols=entry.symbols,
    )
    return {"logged": True}


@app.get("/journal")
def get_journal(strategy: Optional[str] = None, event_type: Optional[str] = None,
                since: Optional[str] = None, limit: int = 100):
    return {"journal": executor.logger_db.get_journal(
        strategy_id=strategy, event_type=event_type, since=since, limit=limit)}


@app.get("/pnl/history")
def pnl_history(strategy: Optional[str] = None, since: Optional[str] = None):
    return {"history": executor.logger_db.get_equity_history(strategy_id=strategy, since=since)}


@app.get("/resolve_front/{symbol}")
def resolve_front(symbol: str, exchange: str = "NYMEX"):
    r = executor.resolve_front_month(symbol, exchange=exchange)
    if r is None:
        raise HTTPException(status_code=404, detail=f"no front-month contract for {symbol}")
    return r


# ---------------------------------------------------------------------------
# ATR execution layer — EOD cancel sweep
# ---------------------------------------------------------------------------
@app.post("/atr/cancel", dependencies=[Depends(require_api_key)])
def atr_cancel_unfilled():
    """Cancel all unfilled limit orders placed by the ATR pullback layer.
    Called by day_scheduler ~5 min before close."""
    cancelled = []
    for oid in executor.atr_layer.pending_order_ids():
        status = executor.order_status.get(oid, {})
        if status.get("status") in ("PreSubmitted", "Submitted"):
            # releases the unfilled pending too. This sweep runs every day before the close,
            # and a bare cancelOrder left each cancelled limit's shares in pending — so the
            # next day's rebalance treated those names as already on target.
            if executor.cancel_order(oid, "ATR end-of-day sweep"):
                cancelled.append(oid)
                logger.info("ATR cancel: cancelled unfilled order %s (%s)", oid, status.get("symbol"))
    executor.atr_layer.clear_tracked()
    return {"cancelled": cancelled, "count": len(cancelled)}


@app.get("/atr/status")
def atr_status():
    """Current ATR execution layer state. `strategies` empty means every strategy."""
    layer = executor.atr_layer
    return {
        "enabled": layer.enabled,
        "strategies": list(layer.strategies),
        "skip_exits": layer.skip_exits,
        "atr_period": layer.atr_period,
        "atr_fraction": layer.atr_fraction,
        "bar_size": layer.bar_size,
        "duration": layer.duration,
        "pending_orders": len(layer.pending_order_ids()),
        "cached_symbols": list(layer._cache.keys()),
        "cached_atrs": {s: round(v[0], 4) for s, v in layer._cache.items()},
    }


def _layer_status(name: str, layer) -> dict:
    return {"layer": name, "enabled": layer.enabled, "strategies": list(layer.strategies),
            "skip_exits": layer.skip_exits,
            "pending_orders": len(layer.pending_order_ids())}


@app.get("/execution")
def execution_layers():
    """Every execution layer (ATR, and any added later): on/off and the strategies it
    applies to. `strategies` empty means every strategy."""
    return {"layers": [_layer_status(n, l) for n, l in executor.execution_layers.items()]}


class ExecutionLayerRequest(BaseModel):
    """Change which orders an execution layer reworks. Omitted fields keep their current
    value. `strategies` is a list of ids or "all"; an empty list is refused, because a layer
    reads [] as "every strategy" — the opposite of what it looks like."""
    enabled: Optional[bool] = None
    strategies: Optional[Union[List[str], Literal["all"]]] = None
    changed_by: str = ""


@app.post("/execution/{layer_name}", dependencies=[Depends(require_api_key)])
def set_execution_layer(layer_name: str, req: ExecutionLayerRequest):
    """Turn an execution layer on or off and choose its strategies, without a restart.
    PERSISTS BEFORE it takes effect and is restored at startup, where it wins over the
    layer's block in config.py. Affects orders placed from now on; orders already working
    stay as they are (ATR limits until they fill or the end-of-day sweep cancels them)."""
    layer = executor.execution_layers.get(layer_name)
    if layer is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown execution layer {layer_name!r} — have: "
                   f"{', '.join(executor.execution_layers)}")
    enabled = layer.enabled if req.enabled is None else bool(req.enabled)
    if req.strategies is None:
        strategies = list(layer.strategies)
    elif req.strategies == "all":
        strategies = []
    else:
        strategies = list(dict.fromkeys(s.strip() for s in req.strategies if s.strip()))
        if not strategies:
            raise HTTPException(
                status_code=422,
                detail=f"an empty list would apply {layer_name} to EVERY strategy — send "
                       "\"all\" for that, or enabled=false to turn it off")
        unknown = [s for s in strategies if s not in CONFIG]
        if unknown:
            raise HTTPException(status_code=422,
                                detail=f"unknown strategy: {', '.join(unknown)}")

    try:
        executor.logger_db.save_execution_settings(layer_name, enabled, strategies)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"could not persist the {layer_name} setting, so it was "
                                   f"NOT changed: {e}")

    before = ("on" if layer.enabled else "off", list(layer.strategies))
    layer.enabled = enabled
    layer.strategies = strategies

    applies = ", ".join(strategies) if strategies else "all strategies"
    summary = f"{layer_name} execution {'ON' if enabled else 'OFF'} — {applies}"
    executor.logger_db.log_decision(
        f"{layer_name}_execution", "execution_config", summary,
        detail=f"was {before[0]} for {before[1] or 'all'}; "
               f"changed_by={req.changed_by or 'api'}")
    _alert(f"\u2699\ufe0f {summary}", topic="orders")
    return _layer_status(layer_name, layer)
