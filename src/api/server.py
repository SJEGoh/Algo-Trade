import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from execution.central_execution import CentralExecutor
from monitoring.logging_config import setup_logging
from config import CONFIG

# .env lives at repo root (one level above src/) — same pattern as main.py
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

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
    app.state.startup_reconciliation = recon

    try:
        yield
    finally:
        executor.shutdown()

app = FastAPI(title="Algo Trade Executor", version="0.1.0", lifespan=lifespan)

def require_api_key(x_api_key: str = Header(default = "")) -> None:
    if not secrets.compare_digest(x_api_key, EXECUTOR_API_KEY or ""):
        raise HTTPException(status_code=401, detail="invalid or missing API key")

class KillRequest(BaseModel):
    flatten: bool = True

@app.post("/orders", dependencies = [Depends(require_api_key)])
def submit_order(intent: dict):
    result = executor.process_intent(intent)
    return result

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
