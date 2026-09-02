#!/usr/bin/env python3
"""
tools/halt_test.py — exercise the drawdown-HALT path LIVE against the running server, using a
halt_test_* strategy (max_drawdown = 1%). It does small losing round-trips on a micro future
(works while equities are closed) until realized P&L breaches 1% of the strategy's allocation,
then confirms the strategy is HALTED and that a further order is REJECTED.

Because each round-trip only loses the spread + commission (~a few $ on MCL) and 1% of the
$10k allocation is ~$100, expect on the order of a few dozen round-trips — this is deliberate
turnover to accumulate a real realized loss. For a fast, deterministic proof of the same path
without trading, see tests/test_halt_config.py.

    uvicorn src.api.server:app ...        # server up, paper TWS on 7497
    python3 tools/halt_test.py                        # halt_test_1 on MCL, loop until halted
    python3 tools/halt_test.py --strategy halt_test_2 --max-trips 80
    python3 tools/halt_test.py --reactivate-first     # clear a prior halt before starting
    python3 tools/halt_test.py --price 68.5 -y        # skip the TWS price probe + confirmation

Trades real (paper) contracts. Leaves the strategy HALTED (that's the point); clear it with
--reactivate-first next run, or POST /strategies/<id>/reactivate.
"""
import argparse
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "src"))

import requests
from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

BASE = os.environ.get("EXECUTOR_URL", "http://127.0.0.1:8000")
KEY = os.environ.get("EXECUTOR_API_KEY")


def get(path, **kw):
    r = requests.get(f"{BASE}{path}", timeout=kw.pop("timeout", 15), **kw); r.raise_for_status(); return r.json()


def post(path, body):
    r = requests.post(f"{BASE}{path}", json=body, headers={"X-API-Key": KEY}, timeout=20)
    r.raise_for_status(); return r.json()


def die(msg, code=2):
    print(f"\n\033[31mABORT:\033[0m {msg}"); sys.exit(code)


def fetch_price(symbol, exchange, ltd, mult, host, port, cid):
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    TICKS = {1, 2, 4, 9, 66, 67, 68, 75, 76}

    class _P(EClient, EWrapper):
        def __init__(s): EClient.__init__(s, s); s.ready = threading.Event(); s.px = None; s.nid = 0
        def nextValidId(s, oid): s.nid = oid; s.ready.set()
        def error(s, *a): pass
        def tickPrice(s, rid, tt, price, attrib):
            if s.px is None and tt in TICKS and price and price > 0: s.px = price
    app = _P()
    try: app.connect(host, port, clientId=cid)
    except Exception: return None
    threading.Thread(target=app.run, daemon=True).start()
    if not app.ready.wait(timeout=8): app.disconnect(); return None
    try:
        app.reqMarketDataType(3)
        c = Contract(); c.symbol = symbol; c.secType = "FUT"; c.exchange = exchange
        c.currency = "USD"; c.lastTradeDateOrContractMonth = ltd
        if mult: c.multiplier = str(int(mult))
        app.reqMktData(app.nid + 1, c, "", True, False, [])
        t = time.time() + 6
        while time.time() < t and app.px is None: time.sleep(0.2)
        return app.px
    finally:
        app.disconnect()


def target_intent(strategy, inst, target_qty, price):
    return {
        "strategy_id": strategy,
        "client_order_id": f"halttest-{inst['symbol']}-{target_qty}-{int(time.time()*1000)}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
        "instrument": inst,
        "intent_type": "target_position",
        "target_quantity": int(target_qty),
        "order_type": "market",
        "expected_price": float(price),
    }


def wait_fill(oid, timeout):
    if oid is None: return None
    end = time.time() + timeout
    last = {}
    while time.time() < end:
        try: last = get(f"/orders/{oid}")
        except requests.HTTPError: last = {}
        st = (last.get("status") or "").lower()
        if st in ("filled",) or (last.get("filled") and abs(last["filled"]) >= 1):
            return last
        if st in ("cancelled", "inactive", "apicancelled"): return last
        time.sleep(0.8)
    return last


def realized(sid):
    return (get("/pnl").get("realized_pnl") or {}).get(sid, 0.0)


def status(sid):
    return get(f"/strategies/{sid}/status").get("status")


def main():
    ap = argparse.ArgumentParser(description="Live drawdown-halt test using a halt_test_* strategy.")
    ap.add_argument("--strategy", default="halt_test_1")
    ap.add_argument("--symbol", default="MCL", help="micro future (default MCL, Micro WTI)")
    ap.add_argument("--exchange", default="NYMEX")
    ap.add_argument("--qty", type=int, default=1)
    ap.add_argument("--max-trips", type=int, default=60, help="safety cap on round-trips")
    ap.add_argument("--price", type=float, default=None)
    ap.add_argument("--tws-host", default="127.0.0.1")
    ap.add_argument("--tws-port", type=int, default=7497)
    ap.add_argument("--price-client-id", type=int, default=13)
    ap.add_argument("--timeout", type=int, default=40)
    ap.add_argument("--reactivate-first", action="store_true", help="clear a prior halt before starting")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()
    if not KEY: die("EXECUTOR_API_KEY not set in .env")

    print(f"\033[1mLIVE HALT TEST\033[0m  strategy={args.strategy}  symbol={args.symbol}  server={BASE}")

    try: h = get("/health")
    except requests.exceptions.ConnectionError: die(f"server not reachable at {BASE}")
    if not h.get("connected"): die(f"IB not connected: {h}")
    if h.get("killed"): die("kill switch active")

    try: alloc = get(f"/strategies/{args.strategy}/allocation")
    except requests.HTTPError: die(f"strategy '{args.strategy}' not in config")
    cap, dd = alloc["capital_allocation"], alloc["max_drawdown"]
    print(f"allocation ${cap:,.0f}  max_drawdown {dd:.1%}  -> halt at realized <= -${cap*dd:,.0f}")

    if args.reactivate_first:
        print("reactivate:", post(f"/strategies/{args.strategy}/reactivate", {}))
    if status(args.strategy) != "active":
        die(f"strategy '{args.strategy}' is already halted — re-run with --reactivate-first")

    meta = get(f"/resolve_front/{args.symbol}", params={"exchange": args.exchange})
    ltd, mult = meta.get("last_trade_date"), meta.get("multiplier")
    inst = {"symbol": args.symbol, "asset_class": "future", "sec_type": "FUT",
            "exchange": args.exchange, "multiplier": mult, "last_trade_date": ltd}
    print(f"front month: {meta.get('local_symbol')}  expiry {ltd}  mult {mult}")

    price = args.price or fetch_price(args.symbol, args.exchange, ltd, mult,
                                      args.tws_host, args.tws_port, args.price_client_id)
    if not price: die("no reference price — re-run with --price <px>")
    print(f"reference price {price:.4f}  (1 {args.symbol} notional ~ ${price*(mult or 1):,.0f})")

    if not args.yes:
        if input(f"\nRun losing round-trips on {args.symbol} under '{args.strategy}' until it "
                 f"halts (~{int((cap*dd)/3)+1} trips)? [y/N] ").strip().lower() not in ("y", "yes"):
            die("cancelled", 1)

    start_pnl = realized(args.strategy)
    print(f"\nstarting realized P&L: ${start_pnl:,.2f}\n")

    trips = 0
    halted = False
    while trips < args.max_trips:
        trips += 1
        # open +qty, then close to 0 -> realizes the round-trip loss (spread + commission)
        r_open = post("/orders", target_intent(args.strategy, inst, args.qty, price))
        if not r_open.get("accepted"):
            print(f"trip {trips}: open rejected ({r_open.get('reason')}) — strategy may be halted")
            break
        wait_fill(r_open.get("order_id"), args.timeout)
        r_close = post("/orders", target_intent(args.strategy, inst, 0, price))
        wait_fill(r_close.get("order_id"), args.timeout)
        time.sleep(0.6)  # let execDetails -> check_drawdown settle
        pnl = realized(args.strategy)
        stt = status(args.strategy)
        dd_pct = (-pnl / cap) if pnl < 0 else 0.0
        print(f"trip {trips:>3}: realized ${pnl:,.2f}  drawdown {dd_pct:.2%}  status {stt}")
        if stt != "active":
            halted = True
            break

    print()
    if not halted:
        die(f"not halted after {trips} trips (realized ${realized(args.strategy):,.2f}). "
            f"Increase --max-trips.", code=1)

    # confirm the halt gates new orders
    r = post("/orders", target_intent(args.strategy, inst, args.qty, price))
    blocked = not r.get("accepted") and "not active" in (r.get("reason", ""))
    print(f"\033[1m{'PASS' if blocked else 'FAIL'}\033[0m — halted after {trips} round-trips; "
          f"post-halt order {'REJECTED' if blocked else 'NOT rejected: ' + str(r)}")
    print(f"realized P&L: ${realized(args.strategy):,.2f}  (started ${start_pnl:,.2f})")
    print(f"'{args.strategy}' is now halted. Clear it with --reactivate-first or "
          f"POST /strategies/{args.strategy}/reactivate. Position is flat.")
    sys.exit(0 if blocked else 1)


if __name__ == "__main__":
    main()
