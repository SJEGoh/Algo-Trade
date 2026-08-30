"""
run_strat.py
 
Wires the cross-sectional momentum strategy into the CentralExecutor.
 
Run modes (set MODE below):
  "inspect"  - generate intents and print them, no executor at all
  "isolated" - start executor, push intents through process_intent, report each
               result — validates the full pipeline (schema, dedup, risk, resolution)
               without depending on fills. Safe to run market-closed.
 
Order-placement itself still goes to whatever TWS/Gateway you point at, so use a
PAPER account. This script does NOT wait for or assert on fills — it's testing the
intent -> validation -> risk -> resolution path, which is what actually needs proving
first.
"""
 
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent / ".env")
 
import time
import signal
 
from monitoring.logging_config import setup_logging
from data.alpaca_data_provider import AlpacaDataProvider
from models.xs_momentum import MomentumStrategy
from execution.central_execution import CentralExecutor
 
MODE = "isolated"   # "inspect" or "isolated"
 
UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
            "JPM", "XOM", "JNJ", "PG", "KO", "WMT"]
 
 
def build_strategy() -> MomentumStrategy:
    data_provider = AlpacaDataProvider(
        api_key=os.environ.get("ALPACA_KEY"),
        secret_key=os.environ.get("ALPACA_SECRET"),
    )
    return MomentumStrategy(data_provider, universe=UNIVERSE)
 
 
def summarize(intents: list[dict], results: list[dict]) -> None:
    """Compact table of what happened to each intent."""
    print(f"\n{'SYMBOL':<8}{'TARGET':>8}  {'RESULT':<10}{'DETAIL'}")
    print("-" * 60)
    accepted = rejected = 0
    for intent, result in zip(intents, results):
        symbol = intent["instrument"]["symbol"]
        target = intent["target_quantity"]
        if result.get("accepted"):
            accepted += 1
            detail = f"order_id={result.get('order_id')}"
            if result.get("note"):
                detail = result["note"]
            verdict = "ACCEPTED"
        else:
            rejected += 1
            verdict = "REJECTED"
            detail = result.get("reason", "")
        print(f"{symbol:<8}{target:>8}  {verdict:<10}{detail}")
    print("-" * 60)
    print(f"accepted: {accepted}  rejected: {rejected}  total: {len(intents)}")
 
 
if __name__ == "__main__":
    setup_logging()
    strategy = build_strategy()
 
    print("=== generating intents ===")
    intents = strategy.generate_intents()
    print(f"generated {len(intents)} intents")
 
    if MODE == "inspect":
        for intent in intents:
            print(intent)
        raise SystemExit(0)
 
    # --- isolated mode: push through the executor ---
    app = CentralExecutor()
 
    def handle_sigint(sig, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, handle_sigint)
 
    try:
        app.start(client_id=6)
        time.sleep(2)
 
        print("\n=== pushing intents through process_intent ===")
        results = [app.process_intent(intent) for intent in intents]
        summarize(intents, results)
 
        # give fills/status callbacks a moment to arrive (for any that would fill)
        time.sleep(5)
 
        print("\n=== post-run state ===")
        print(f"positions: {dict(app.ledger.current_positions)}")
        print(f"open orders: "
              f"{[(oid, s['status']) for oid, s in app.order_status.items() if s['status'] in ('PreSubmitted', 'Submitted')]}")
        print(f"strategy P&L: {dict(app.ledger.strategy_realized_pnl)}")
 
    finally:
        app.shutdown()
 
