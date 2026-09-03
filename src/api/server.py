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
from config import CONFIG

import threading
from datetime import datetime, timezone
import logging

# .env lives at repo root (one level above src/) — same pattern as main.py
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

STATIC_DIR = Path(__file__).resolve().parent / "static"
DB_DIR = Path(__file__).resolve().parent.parent.parent / "db"

EXECUTOR_API_KEY = os.environ.get("EXECUTOR_API_KEY")
SERVER_CLIENT_ID = int(os.environ.get("IB_CLIENT_ID", "8"))  # distinct from main.py / run_strat.py (both use 6)
IB_HOST = os.environ.get("IB_HOST", "127.0.0.1")               # 'ib-gateway' in Docker compose
IB_PORT = int(os.environ.get("IB_PORT", "4002"))              # 4002 paper / 4001 live (Gateway); 7497 TWS paper

executor: Optional[CentralExecutor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
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

    executor = CentralExecutor()
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
    }

@app.get("/pnl")
def get_pnl():
    return {"realized_pnl": dict(executor.ledger.strategy_realized_pnl)}

@app.get("/health")
def health():
    return {
        "connected": executor.isConnected(),
        "killed": executor._killed,
        "market_open": is_market_open(),
    }


@app.post("/kill", dependencies=[Depends(require_api_key)])
def kill(req: KillRequest):
    executor.kill_switch(flatten=req.flatten)
    _alert(
        f"\U0001f6d1 KILL SWITCH activated (flatten={req.flatten})",
        topic="errors",
    )
    return {"killed": True, "flattened": req.flatten}

@app.post("/flatten", dependencies=[Depends(require_api_key)])
def flatten_all():
    """Cancel open orders and flatten every position WITHOUT setting the kill switch.
    Strategies remain active and can resume trading on the next signal."""
    # 1. cancel open orders
    cancelled = []
    for oid, status in list(executor.order_status.items()):
        if status.get("status") in ("PreSubmitted", "Submitted"):
            try:
                executor.cancelOrder(oid)
                cancelled.append(oid)
            except Exception:
                pass

    # 2. flatten per-strategy so fills attribute to the correct strategy
    _INTERNAL = {"__net__", "flatten_all", "kill_switch"}
    flattened = []

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

    # Clean up stale strategy positions for symbols already flat at the broker.
    # This handles leftover state from before per-strategy attribution was added.
    broker_flat = {s for s, q in executor.ledger.current_positions.items() if abs(q) < 1e-9}
    cleaned = []
    for sid, positions in list(executor.ledger.strategy_positions.items()):
        if sid in _INTERNAL:
            continue
        for sym in list(positions):
            if sym in broker_flat and abs(positions.get(sym, 0.0)) > 1e-9:
                cleaned.append({"strategy_id": sid, "symbol": sym, "was": positions[sym]})
                positions[sym] = 0.0
                executor.ledger.strategy_avg_cost.get(sid, {})[sym] = 0.0
    if cleaned:
        executor.ledger.save_state(executor.logger_db)
        logging.getLogger("executor").info("Flatten cleanup: zeroed stale strategy positions: %s", cleaned)

    _alert(
        f"💨 FLATTEN ALL: cancelled {len(cancelled)} orders, flattening {len(flattened)} positions (kill switch NOT set)",
        topic="orders",
    )
    return {"cancelled_orders": len(cancelled), "flattened_positions": flattened, "kill_switch": False}


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
        result = executor.reconcile_and_log()   # pulls reqPositions, corrects ledger to broker
    except TimeoutError as e:
        raise HTTPException(status_code=503, detail=f"reconcile timed out talking to IB: {e}")
    _last_reconcile.update({
        "matched": result["matched"],
        "discrepancies": result["discrepancies"],
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    return {
        "matched": result["matched"],
        "discrepancies": result["discrepancies"],
        "positions": dict(executor.ledger.current_positions),
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


@app.get("/orders")
def list_orders():
    # live, session-scoped view from in-memory order_status (richest: filled/remaining)
    return {"orders": [{"order_id": oid, **st} for oid, st in executor.order_status.items()]}


@app.get("/fills")
def list_fills(limit: int = 50):
    return {"fills": executor.logger_db.get_recent_fills(limit)}


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
                snap.setdefault(sid, {"realized": 0.0, "unrealized": 0.0, "equity": 0.0})
            _INTERNAL = {"__net__", "flatten_all", "kill_switch"}
            for strat, v in snap.items():
                if strat in _INTERNAL:
                    continue
                executor.logger_db.log_equity(ts, strat, v["realized"], v["unrealized"], v["equity"])
            # portfolio circuit breaker on total equity (realized + unrealized across all strategies)
            total_equity = sum(v.get("equity", 0.0) for v in snap.values())
            executor.enforce_daily_loss(total_equity)
            # per-strategy total-equity drawdown; SKIP a strategy if any held symbol's mark is
            # stale/missing (don't halt+flatten on incomplete unrealized data — realized fast
            # path still guards it).
            for sid in CONFIG:
                held = [s for s, q in executor.ledger.strategy_positions.get(sid, {}).items() if q != 0]
                if held and not all(executor.mark_is_fresh(s) for s in held):
                    log.warning("total-drawdown check skipped for %s (stale/missing mark)", sid)
                    continue
                executor.enforce_drawdown(sid, snap.get(sid, {}).get("equity", 0.0), "total")
        except Exception as e:
            log.error("equity sampler error: %s", e)
        _sampler_stop.wait(interval)   # sleep, wakes early on stop


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
            try:
                executor.cancelOrder(oid)
                cancelled.append(oid)
                logger.info("ATR cancel: cancelled unfilled order %s (%s)", oid, status.get("symbol"))
            except Exception as e:
                logger.warning("ATR cancel: failed to cancel order %s: %s", oid, e)
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
