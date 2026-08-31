#!/usr/bin/env python3
"""
tools/first_fill.py — place ONE small market order, wait for the fill, print
fill price vs expected price and the slippage in bps. Validates Phase 6's last
unconfirmed piece (execDetails -> fills table -> slippage) with trivial risk.

Requires the market to be OPEN (a market order needs to actually fill).

Usage:
    python3 tools/first_fill.py                      # 1 share AAPL, buy
    python3 tools/first_fill.py --symbol MSFT --qty 1 --side sell
    python3 tools/first_fill.py --expected-price 231.50   # skip the price fetch
"""
import argparse, os, sys, time
from pathlib import Path
from datetime import datetime, timezone

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))            # data/ and models/ live at the repo root
sys.path.insert(0, str(_ROOT / "src"))    # execution/, ledger/, etc. live under src/
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
BASE = os.environ.get("EXECUTOR_URL", "http://127.0.0.1:8000")
KEY = os.environ.get("EXECUTOR_API_KEY")


def reference_price(symbol):
    """Last daily close via Alpaca — the same reference the strategy uses."""
    from data.alpaca_data_provider import AlpacaDataProvider
    dp = AlpacaDataProvider(api_key=os.environ.get("ALPACA_KEY"),
                            secret_key=os.environ.get("ALPACA_SECRET"))
    return float(dp.get_daily_bars(symbol, 5).iloc[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--qty", type=float, default=1)
    ap.add_argument("--side", choices=["buy", "sell"], default="buy")
    ap.add_argument("--expected-price", type=float, default=None)
    ap.add_argument("--strategy", default="cross_sectional_momentum")
    ap.add_argument("--timeout", type=float, default=30.0)
    a = ap.parse_args()

    if not KEY:
        sys.exit("EXECUTOR_API_KEY not set in .env")

    # must be open — a market order needs to fill, not queue
    try:
        h = requests.get(f"{BASE}/health", timeout=10).json()
    except requests.exceptions.ConnectionError:
        sys.exit(f"server not reachable at {BASE} — is uvicorn running?")
    if not h.get("connected"):
        sys.exit(f"IB not connected: {h}")
    if not h.get("market_open"):
        sys.exit("market is closed — a market order would be rejected; run this at the open")

    expected = a.expected_price if a.expected_price is not None else reference_price(a.symbol)
    print(f"placing {a.side} {a.qty:g} {a.symbol} (market), expected_price={expected:.2f}")

    intent = {
        "strategy_id": a.strategy,
        "client_order_id": f"first-fill-{int(time.time())}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
        "instrument": {"symbol": a.symbol, "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "delta", "side": a.side, "quantity": a.qty,
        "order_type": "market", "expected_price": expected, "time_in_force": "day",
    }
    r = requests.post(f"{BASE}/orders", json=intent, headers={"X-API-Key": KEY}, timeout=20)
    rj = r.json()
    if not (r.status_code == 200 and rj.get("accepted")):
        sys.exit(f"order not accepted ({r.status_code}): {rj}")
    oid = rj["order_id"]
    print(f"accepted, order_id={oid} — waiting for fill...")

    # poll order status until filled
    deadline = time.time() + a.timeout
    status = None
    while time.time() < deadline:
        o = requests.get(f"{BASE}/orders/{oid}", timeout=10).json()
        if o.get("status") != status:
            status = o.get("status")
            print(f"  status: {status}  filled={o.get('filled')}  remaining={o.get('remaining')}")
        if status == "Filled":
            break
        time.sleep(1)
    if status != "Filled":
        sys.exit(f"not filled within {a.timeout:.0f}s (status={status}); check TWS")

    # pull the authoritative fill row (price + expected) from /fills
    fills = requests.get(f"{BASE}/fills", timeout=10).json().get("fills", [])
    row = next((f for f in fills if f["order_id"] == oid), None)
    if row is None:
        sys.exit("filled, but no row in the fills table — the slippage-logging path did NOT fire")

    fill, exp = row["price"], row.get("expected_price")
    print(f"\nFILLED: {row['side']} {row['quantity']:g} {row['symbol']} @ {fill:.4f}")
    if exp:
        sign = 1 if row["side"] == "BOT" else -1        # +ve = worse than expected
        slip = (fill - exp) * sign
        bps = slip / exp * 1e4
        print(f"expected {exp:.4f} | slippage {slip:+.4f} ({bps:+.1f} bps)  [+ = worse]")
    else:
        print("no expected_price recorded on the fill")
    print("\nPhase 6 confirmed: fill -> fills table -> slippage all working.")


if __name__ == "__main__":
    main()
