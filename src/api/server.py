import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from execution.central_execution import CentralExecutor, is_market_open
from execution.netting import NettingCoordinator
from monitoring.alerter import Alerter, AlertingHandler
from monitoring.logging_config import setup_logging
from config import CONFIG, validate_config
from portfolio import hedger

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
    threading.Thread(target=_equity_sampler, args=(60.0,), daemon=True).start()
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

@app.post("/orders", dependencies = [Depends(require_api_key)])
def submit_order(intent: dict):
    result = executor.process_intent(intent)
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
    result = executor.coordinator.set_target(
        req.strategy_id, req.symbol, req.quantity,
        instrument=req.instrument, price=req.price,
    )
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
    target_quantity, expected_price}]}. Run periodically to self-heal drift."""
    sid = body.get("strategy_id")
    if not sid:
        raise HTTPException(status_code=422, detail="strategy_id required")
    intents = body.get("intents", [])
    fut_only = bool(intents) and all(
        (it.get("instrument") or {}).get("sec_type", "STK") == "FUT" for it in intents)
    _pool_preflight(fut_only)
    result = executor.coordinator.submit_book(sid, intents)
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
    syms = list({it.get("instrument", {}).get("symbol", "?") for it in intents})
    import json as _json
    executor.logger_db.log_decision(
        sid, "rebalance", summary,
        detail=_json.dumps({"orders": orders, "internal_crosses": crosses}, default=str),
        symbols=syms,
    )
    return result


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


@app.get("/pnl")
def get_pnl():
    return {"realized_pnl": dict(executor.ledger.strategy_realized_pnl)}


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
            # Portfolio balance sheet: NAV is the sum of every position of every strategy,
            # cash included. Cached for the dashboard (GET /equity).
            totals = {
                "cash": sum(v.get("cash", 0.0) for v in visible.values()),
                "position_value": sum(v.get("position_value", 0.0) for v in visible.values()),
                "nav": sum(v.get("nav", 0.0) for v in visible.values()),
                "starting_cash": sum(v.get("starting_cash", 0.0) for v in visible.values()),
                "realized": sum(v.get("realized", 0.0) for v in visible.values()),
                "unrealized": sum(v.get("unrealized", 0.0) for v in visible.values()),
                "equity": sum(v.get("equity", 0.0) for v in visible.values()),
            }
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
    """Current ATR execution layer state."""
    layer = executor.atr_layer
    return {
        "enabled": layer.enabled,
        "atr_period": layer.atr_period,
        "atr_fraction": layer.atr_fraction,
        "bar_size": layer.bar_size,
        "duration": layer.duration,
        "pending_orders": len(layer.pending_order_ids()),
        "cached_symbols": list(layer._cache.keys()),
        "cached_atrs": {s: round(v[0], 4) for s, v in layer._cache.items()},
    }
