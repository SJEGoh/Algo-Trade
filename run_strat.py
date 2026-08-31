"""
run_strat.py

Drives the cross-sectional momentum strategy against the RUNNING executor
server (api/server.py) over HTTP.

This replaces the old in-process mode that created its own CentralExecutor and
called process_intent directly.

IMPORTANT: the server owns the one and only CentralExecutor. Do NOT instantiate
one here — two executors don't share ledger / dedup / pending-exposure state and
will double-submit. This script is now a pure HTTP client.

Prereq — server running (paper account behind it):
    uvicorn api.server:app --host 127.0.0.1 --port 8000

Run modes (set MODE below):
  "inspect" - generate intents and print them, no network calls at all
  "live"    - health + halt check, then POST each intent to the executor
"""

import os
from pathlib import Path

import time
import requests
from dotenv import load_dotenv

from data.alpaca_data_provider import AlpacaDataProvider
from models.xs_momentum import MomentumStrategy

load_dotenv(Path(__file__).resolve().parent / ".env")

MODE = "live"   # "inspect" or "live"

BASE_URL = os.environ.get("EXECUTOR_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("EXECUTOR_API_KEY")
STRATEGY_ID = "cross_sectional_momentum"

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
            "JPM", "XOM", "JNJ", "PG", "KO", "WMT"]


def build_strategy(allocation: float) -> MomentumStrategy:
    data_provider = AlpacaDataProvider(
        api_key=os.environ.get("ALPACA_KEY"),
        secret_key=os.environ.get("ALPACA_SECRET"),
    )
    # allocation comes from the server (single source of truth), not a hardcoded default
    return MomentumStrategy(data_provider, universe=UNIVERSE, capital_allocation=allocation)


def submit(intent: dict) -> dict:
    """POST one intent. A domain rejection is 200 with accepted=false — returned
    as-is. An HTTP-level failure (401 bad key, 5xx) is caught and turned into a
    reject-shaped dict so one bad call doesn't abort the whole batch."""
    try:
        resp = requests.post(
            f"{BASE_URL}/orders",
            json=intent,
            headers={"X-API-Key": API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        return {"accepted": False, "reason": f"HTTP {resp.status_code}: {resp.text[:120]}"}
    except requests.exceptions.RequestException as e:
        return {"accepted": False, "reason": f"request failed: {e}"}


def summarize(intents: list[dict], results: list[dict]) -> None:
    print(f"\n{'SYMBOL':<8}{'TARGET':>8}  {'RESULT':<10}{'DETAIL'}")
    print("-" * 60)
    accepted = rejected = 0
    for intent, result in zip(intents, results):
        symbol = intent["instrument"]["symbol"]
        target = intent["target_quantity"]
        if result.get("accepted"):
            accepted += 1
            detail = result.get("note") or f"order_id={result.get('order_id')}"
            verdict = "ACCEPTED"
        else:
            rejected += 1
            verdict = "REJECTED"
            detail = result.get("reason", "")
        print(f"{symbol:<8}{target:>8}  {verdict:<10}{detail}")
    print("-" * 60)
    print(f"accepted: {accepted}  rejected: {rejected}  total: {len(intents)}")


if __name__ == "__main__":
    # --- inspect mode: no server needed, just print what we'd send ---
    if MODE == "inspect":
        # allocation for inspect can't come from the server (it may be down),
        # so use the strategy's own default purely for a dry-run preview.
        strategy = MomentumStrategy(
            AlpacaDataProvider(
                api_key=os.environ.get("ALPACA_KEY"),
                secret_key=os.environ.get("ALPACA_SECRET"),
            ),
            universe=UNIVERSE,
        )
        for intent in strategy.generate_intents():
            print(intent)
        raise SystemExit(0)

    # --- live mode ---
    if not API_KEY:
        raise SystemExit("EXECUTOR_API_KEY not set in .env")

    # 1. health gate — fail fast with a clear message
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5).json()
    except requests.exceptions.ConnectionError:
        raise SystemExit(f"executor not reachable at {BASE_URL} — is uvicorn running?")
    if not health.get("connected"):
        raise SystemExit(f"server up but IB not connected: {health}")
    if health.get("killed"):
        raise SystemExit("server is in kill-switch state — refusing to trade")
    if not health.get("market_open"):
        raise SystemExit("market is closed — skipping cycle (executor would reject anyway)")

    # 2. pull-before-act — skip the cycle entirely if halted
    status = requests.get(f"{BASE_URL}/strategies/{STRATEGY_ID}/status", timeout=5).json()
    if status.get("status") != "active":
        raise SystemExit(f"strategy {STRATEGY_ID} is '{status.get('status')}' — skipping cycle")
    # 2.5 sync the server's ledger to broker truth BEFORE computing targets
    rec = requests.post(f"{BASE_URL}/reconcile",
                        headers={"X-API-Key": API_KEY}, timeout=15).json()
    if rec["matched"]:
        print(f"reconciled — ledger matches broker: {rec['positions']}")
    else:
        print(f"reconciled — corrected drift: {rec['discrepancies']}")
    time.sleep(15)
    # 3. single source of truth for allocation
    alloc = requests.get(f"{BASE_URL}/strategies/{STRATEGY_ID}/allocation", timeout=5).json()
    allocation = alloc["capital_allocation"]

    # 4. generate + submit
    strategy = build_strategy(allocation)
    print("=== generating intents ===")
    intents = strategy.generate_intents()
    print(f"generated {len(intents)} intents (allocation={allocation:,.0f})")

    print("\n=== POSTing intents to executor ===")
    results = [submit(intent) for intent in intents]
    summarize(intents, results)
