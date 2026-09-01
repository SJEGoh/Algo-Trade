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
SERVER_CLIENT_ID = 8  # distinct from main.py / run_strat.py (both use 6)

executor: Optional[CentralExecutor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor
    setup_logging()
    if not EXECUTOR_API_KEY:
        raise RuntimeError(
            "EXECUTOR_API_KEY not set in .env — generate with secrets.token_hex(32)"
        )

    executor = CentralExecutor()
    recon = executor.start(client_id = SERVER_CLIENT_ID)
    executor.coordinator = NettingCoordinator(executor, CONFIG, state_path=str(DB_DIR / "netting.json"))
    threading.Thread(target=_equity_sampler, args=(60.0,), daemon=True).start()
    app.state.startup_reconciliation = recon

    try:
        yield
    finally:
        _sampler_stop.set()
        executor.shutdown()

app = FastAPI(title="Algo Trade Executor", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def require_api_key(x_api_key: str = Header(default = "")) -> None:
    if not secrets.compare_digest(x_api_key, EXECUTOR_API_KEY or ""):
        raise HTTPException(status_code=401, detail="invalid or missing API key")

class KillRequest(BaseModel):
    flatten: bool = True

@app.post("/orders", dependencies = [Depends(require_api_key)])
def submit_order(intent: dict):
    result = executor.process_intent(intent)
    return result


class TargetRequest(BaseModel):
    strategy_id: str
    symbol: str
    quantity: float
    instrument: Optional[dict] = None
    price: Optional[float] = None


@app.post("/target", dependencies=[Depends(require_api_key)])
def set_target(req: TargetRequest):
    """Net-pooling: incremental. Set ONE symbol's absolute target for a strategy; the
    coordinator re-nets and trades the account to the pooled net. Exit = quantity 0."""
    if executor.coordinator is None:
        raise HTTPException(status_code=503, detail="netting coordinator not initialised")
    return executor.coordinator.set_target(
        req.strategy_id, req.symbol, req.quantity,
        instrument=req.instrument, price=req.price,
    )


@app.post("/targets", dependencies=[Depends(require_api_key)])
def submit_book(body: dict):
    """Net-pooling: full-book resync. Authoritative snapshot of a strategy's whole book;
    any name dropped from the book is closed. body = {strategy_id, intents:[{instrument,
    target_quantity, expected_price}]}. Run periodically to self-heal drift."""
    if executor.coordinator is None:
        raise HTTPException(status_code=503, detail="netting coordinator not initialised")
    sid = body.get("strategy_id")
    if not sid:
        raise HTTPException(status_code=422, detail="strategy_id required")
    return executor.coordinator.submit_book(sid, body.get("intents", []))


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
        }
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
    return {"killed": True, "flattened": req.flatten}

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

@app.post("/reconcile", dependencies=[Depends(require_api_key)])
def reconcile():
    try:
        result = executor.reconcile_and_log()   # pulls reqPositions, corrects ledger to broker
    except TimeoutError as e:
        raise HTTPException(status_code=503, detail=f"reconcile timed out talking to IB: {e}")
    return {
        "matched": result["matched"],
        "discrepancies": result["discrepancies"],
        "positions": dict(executor.ledger.current_positions),
    }


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
            for strat, v in snap.items():
                executor.logger_db.log_equity(ts, strat, v["realized"], v["unrealized"], v["equity"])
        except Exception as e:
            log.error("equity sampler error: %s", e)
        _sampler_stop.wait(interval)   # sleep, wakes early on stop


@app.get("/pnl/history")
def pnl_history(strategy: Optional[str] = None, since: Optional[str] = None):
    return {"history": executor.logger_db.get_equity_history(strategy_id=strategy, since=since)}


@app.get("/resolve_front/{symbol}")
def resolve_front(symbol: str, exchange: str = "NYMEX"):
    r = executor.resolve_front_month(symbol, exchange=exchange)
    if r is None:
        raise HTTPException(status_code=404, detail=f"no front-month contract for {symbol}")
    return r
