#!/usr/bin/env python3
"""run_rrg.py — drive the combined Kalman/RRG rotation strategy against the running server.

    python3 run_rrg.py          # run one daily rebalance cycle

Periodic rebalancing strategy — uses /targets (full-book resync) so any name
the strategy drops is automatically closed.
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "src"))

import requests
from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from data.alpaca_data_provider import AlpacaDataProvider
from models.rotation import CombinedRotationStrategy

BASE = os.environ.get("EXECUTOR_URL", "http://127.0.0.1:8000")
KEY = os.environ.get("EXECUTOR_API_KEY")
STRATEGY_ID = "kalman_rrg_combined"

UNIVERSE = ["NVDA", "AMD", "AVGO", "MU", "AMZN", "META", "GOOGL", "MSFT", "ORCL", "TSLA",
            "PLTR", "CRWD", "PANW", "FTNT", "SNOW", "ARM", "ANET", "MRVL", "TSM", "ASML",
            "APP", "UBER", "ABNB", "SHOP", "MELI", "NFLX", "COIN", "HOOD", "RDDT", "DDOG",
            "NET", "ZS", "OKTA", "HUBS", "NOW", "CRM", "ADBE", "INTU", "KLAC", "LRCX",
            "AMAT", "QCOM", "TXN", "ON", "MCHP", "NXPI", "CDNS", "SNPS", "VRT", "SMCI"]


if __name__ == "__main__":
    if not KEY:
        raise SystemExit("EXECUTOR_API_KEY not set in .env")
    try:
        h = requests.get(f"{BASE}/health", timeout=10).json()
    except requests.exceptions.ConnectionError:
        raise SystemExit(f"executor not reachable at {BASE} — is uvicorn running?")
    if not h.get("connected"):
        raise SystemExit(f"IB not connected: {h}")
    if h.get("killed"):
        raise SystemExit("kill switch active — refusing to trade")
    if not h.get("market_open"):
        raise SystemExit("market closed — skipping cycle (equities)")

    st = requests.get(f"{BASE}/strategies/{STRATEGY_ID}/status", timeout=10).json()
    if st.get("status") != "active":
        raise SystemExit(f"strategy {STRATEGY_ID} is '{st.get('status')}' — skipping cycle")
    alloc = requests.get(f"{BASE}/strategies/{STRATEGY_ID}/allocation",
                         timeout=10).json()["capital_allocation"]

    data_provider = AlpacaDataProvider(
        api_key=os.environ.get("ALPACA_KEY"),
        secret_key=os.environ.get("ALPACA_SECRET"),
    )
    DB_DIR = _ROOT / "db"
    strat = CombinedRotationStrategy(
        data_provider=data_provider,
        universe=UNIVERSE,
        capital_allocation=alloc,
        state_path=str(DB_DIR / "rrg_state.json"),
    )

    print(f"=== generating {STRATEGY_ID} intents ===")
    intents = strat.generate_intents()
    active = [i for i in intents if i["target_quantity"] != 0]
    print(f"{len(intents)} names evaluated, {len(active)} with a non-zero target")

    # Journal: log signal snapshot before trading
    state = strat.load_state() or {}
    signal_summary = {sym: {"score": round(v.get("score", 0), 4), "quadrant": v.get("quadrant")}
                      for sym, v in state.get("signal", {}).items()}
    active_syms = [i["instrument"]["symbol"] for i in active]
    import json as _json
    journal_entry = {
        "strategy_id": STRATEGY_ID,
        "event_type": "signal",
        "summary": f"Signal computed: {len(active)} active positions out of {len(intents)} universe",
        "detail": _json.dumps({
            "signal": signal_summary,
            "combined_weights": state.get("combined_weights", {}),
            "sleeve_a_weights": state.get("sleeve_a_weights", {}),
            "sleeve_b_weights": state.get("sleeve_b_weights", {}),
        }, default=str),
        "symbols": active_syms,
    }
    try:
        requests.post(f"{BASE}/journal", json=journal_entry,
                      headers={"X-API-Key": KEY}, timeout=10)
    except Exception:
        pass  # best-effort

    # Full-book resync via /targets — dropped names are automatically closed
    book = {"strategy_id": STRATEGY_ID, "intents": [
        {"instrument": i["instrument"], "target_quantity": i["target_quantity"],
         "expected_price": i["expected_price"]} for i in intents]}
    print("\n=== submitting full-book resync ===")
    try:
        r = requests.post(f"{BASE}/targets", json=book,
                          headers={"X-API-Key": KEY}, timeout=30)
        r.raise_for_status()
        result = r.json()
        orders = result.get("orders", [])
        crosses = result.get("internal_crosses", [])
        if crosses:
            print(f"  internal crosses: {len(crosses)}")
        if orders:
            for o in orders:
                print(f"  {o.get('symbol')} delta={o.get('delta')} id={o.get('order_id')}")
        if not orders and not crosses:
            print("  already at target — no orders needed")
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"resync request failed: {e}")
