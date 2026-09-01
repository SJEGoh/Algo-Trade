#!/usr/bin/env python3
"""
tools/live_futures_test.py — end-to-end LIVE validation of the running executor server,
using a FUTURES contract so it works while the equity market is closed (energy futures
trade ~23h/day; futures bypass the NYSE market-hours gate by design).

Unlike tools/test_futures.py (a whatIf permissions probe that places NO real orders), this
places a REAL order on your PAPER account through the server and checks the whole pipeline:

    health -> resolve front month -> reference price -> submit -> fill -> ledger/positions
    -> fills log -> flatten back to start

It talks to the RUNNING server over HTTP (the same path the competition will use), so it
also exercises whatever is wired into the live server — including the NettingCoordinator
when you run with --path pool (POST /target), which drives place_net_order + the net-fill
attribution in execDetails on a real fill.

PREREQS: paper TWS up on 7497, `uvicorn src.api.server:app` running, EXECUTOR_API_KEY in
.env. Run in the algo_trade venv:

    python3 tools/live_futures_test.py                 # CL, direct path (/orders), 1 contract, then flatten
    python3 tools/live_futures_test.py --path pool     # route through the netting coordinator (/target)
    python3 tools/live_futures_test.py --symbol MCL    # Micro WTI — ~1/10 the notional of CL
    python3 tools/live_futures_test.py --no-flatten     # leave the position on (default: flatten at end)
    python3 tools/live_futures_test.py --price 68.50 -y # skip the TWS price probe; don't prompt

This trades real (paper) contracts — you'll be asked to confirm unless you pass -y/--yes.
"""
import argparse
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import requests
from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

BASE = os.environ.get("EXECUTOR_URL", "http://127.0.0.1:8000")
KEY = os.environ.get("EXECUTOR_API_KEY")

# ---- pretty check accounting ------------------------------------------------
_RESULTS = []


def check(name, ok, detail=""):
    _RESULTS.append(ok)
    mark = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
    print(f"  {mark} {name}" + (f"  — {detail}" if detail else ""))
    return ok


def die(msg, code=2):
    print(f"\n\033[31mABORT:\033[0m {msg}")
    sys.exit(code)


def hdr(t):
    print(f"\n\033[1m{t}\033[0m")


# ---- HTTP helpers -----------------------------------------------------------
def get(path, **kw):
    r = requests.get(f"{BASE}{path}", timeout=kw.pop("timeout", 15), **kw)
    r.raise_for_status()
    return r.json()


def post(path, body):
    r = requests.post(f"{BASE}{path}", json=body, headers={"X-API-Key": KEY}, timeout=20)
    r.raise_for_status()
    return r.json()


# ---- transient TWS client, only to fetch a futures reference price ----------
def fetch_reference_price(symbol, exchange, last_trade_date, multiplier, host, port, client_id):
    """One-off delayed/live futures price straight from TWS (the server has no futures-price
    endpoint). Pins the exact resolved contract so the price matches what we trade."""
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract

    PRICE_TICKS = {1, 2, 4, 9, 66, 67, 68, 75, 76}  # bid/ask/last/close incl. delayed variants
    INFO = {2104, 2106, 2158, 2107, 2119, 2100, 2108, 10197, 10167}

    class _P(EClient, EWrapper):
        def __init__(self):
            EClient.__init__(self, self)
            self.ready = threading.Event()
            self.price = None
            self.nid = 0

        def nextValidId(self, oid):
            self.nid = oid
            self.ready.set()

        def error(self, reqId, code, msg, *a):
            if code not in INFO:
                print(f"    [tws price] {code}: {msg}")

        def tickPrice(self, reqId, tickType, price, attrib):
            if self.price is None and tickType in PRICE_TICKS and price and price > 0:
                self.price = price

    app = _P()
    try:
        app.connect(host, port, clientId=client_id)
    except Exception as e:
        print(f"    could not connect to TWS for price probe: {e}")
        return None
    threading.Thread(target=app.run, daemon=True).start()
    if not app.ready.wait(timeout=8):
        app.disconnect()
        return None
    try:
        app.reqMarketDataType(3)  # delayed is fine for a reference price
        c = Contract()
        c.symbol = symbol
        c.secType = "FUT"
        c.exchange = exchange
        c.currency = "USD"
        c.lastTradeDateOrContractMonth = last_trade_date
        if multiplier:
            c.multiplier = str(int(multiplier))
        app.reqMktData(app.nid + 1, c, "", True, False, [])
        deadline = time.time() + 6
        while time.time() < deadline and app.price is None:
            time.sleep(0.2)
        return app.price
    finally:
        app.disconnect()


# ---- intent / order helpers -------------------------------------------------
def make_instrument(symbol, exchange, multiplier, last_trade_date):
    return {
        "symbol": symbol,
        "asset_class": "future",
        "sec_type": "FUT",
        "exchange": exchange,
        "multiplier": multiplier,
        "last_trade_date": last_trade_date,
    }


def target_intent(strategy, symbol, target_qty, instrument, price):
    """A target_position futures intent (matches VECMStrategy's shape)."""
    stamp = int(time.time() * 1000)
    return {
        "strategy_id": strategy,
        "client_order_id": f"livetest-{symbol}-{target_qty}-{stamp}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
        "instrument": instrument,
        "intent_type": "target_position",
        "target_quantity": int(target_qty),
        "order_type": "market",
        "expected_price": float(price),
    }


def submit(path, strategy, symbol, target_qty, instrument, price):
    """Returns (order_id_or_None, raw_response). direct -> /orders ; pool -> /target."""
    if path == "pool":
        r = post("/target", {
            "strategy_id": strategy, "symbol": symbol, "quantity": int(target_qty),
            "instrument": instrument, "price": float(price),
        })
        if not r.get("accepted"):
            return None, r
        orders = r.get("orders", [])
        return (orders[0]["order_id"] if orders else None), r
    else:
        r = post("/orders", target_intent(strategy, symbol, target_qty, instrument, price))
        return r.get("order_id"), r


def wait_for_fill(order_id, qty_abs, timeout):
    """Poll /orders/{id} until Filled (or filled qty reached) or timeout. Returns final dict."""
    if order_id is None:
        return None
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        try:
            last = get(f"/orders/{order_id}")
        except requests.HTTPError:
            last = {}
        status = (last.get("status") or "").lower()
        filled = last.get("filled") or 0
        if status in ("filled",) or (qty_abs and filled and abs(filled) >= qty_abs):
            return last
        if status in ("cancelled", "inactive", "apicancelled"):
            return last
        time.sleep(1.0)
    return last


def pos_of(positions, symbol):
    return (positions.get("current_positions") or {}).get(symbol, 0.0)


# ---- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Live futures end-to-end test against the running server.")
    ap.add_argument("--symbol", default="CL", help="futures root: CL (WTI, ~$70k/contract), MCL (Micro WTI, ~$7k), BZ, RB (default CL)")
    ap.add_argument("--exchange", default="NYMEX")
    ap.add_argument("--qty", type=int, default=1, help="signed contracts to open (default 1; negative = short)")
    ap.add_argument("--path", choices=["direct", "pool"], default="direct",
                    help="direct -> POST /orders (unpooled); pool -> POST /target (through the NettingCoordinator)")
    ap.add_argument("--strategy", default="test_suite", help="strategy_id to trade under (must be in config, default test_suite)")
    ap.add_argument("--price", type=float, default=None, help="reference price; skips the TWS price probe if given")
    ap.add_argument("--tws-host", default="127.0.0.1")
    ap.add_argument("--tws-port", type=int, default=7497)
    ap.add_argument("--price-client-id", type=int, default=12, help="distinct from server(8)/run_strat(6)/test_futures(11)")
    ap.add_argument("--timeout", type=int, default=40, help="seconds to wait for each fill")
    ap.add_argument("--no-flatten", action="store_true", help="leave the position on at the end (default: flatten)")
    ap.add_argument("-y", "--yes", action="store_true", help="skip the 'this places real paper orders' confirmation")
    args = ap.parse_args()

    if not KEY:
        die("EXECUTOR_API_KEY not set in .env")
    if args.qty == 0:
        die("--qty 0 is a no-op")

    print(f"\033[1mLIVE FUTURES TEST\033[0m  server={BASE}  symbol={args.symbol}  qty={args.qty}  path={args.path}  strategy={args.strategy}")

    # 1) preflight -----------------------------------------------------------
    hdr("1. Preflight")
    try:
        h = get("/health")
    except requests.exceptions.ConnectionError:
        die(f"server not reachable at {BASE} — is uvicorn running?")
    check("server reachable", True, BASE)
    if not check("IB connected", bool(h.get("connected")), str(h)):
        die("IB not connected — start/log in to paper TWS on 7497")
    if h.get("killed"):
        die("kill switch is active — refusing to trade")
    check("equity market closed (expected — that's why we use futures)", not h.get("market_open"),
          f"market_open={h.get('market_open')}")

    try:
        st = get(f"/strategies/{args.strategy}/status")
    except requests.HTTPError:
        die(f"strategy '{args.strategy}' not in config")
    if not check(f"strategy '{args.strategy}' active", st.get("status") == "active", st.get("status")):
        die("strategy is halted — pick another with --strategy or reactivate it")

    # 2) resolve front month -------------------------------------------------
    hdr("2. Resolve front-month contract (validates the roll-buffer logic live)")
    try:
        meta = get(f"/resolve_front/{args.symbol}", params={"exchange": args.exchange})
    except requests.HTTPError as e:
        die(f"no front-month contract for {args.symbol} on {args.exchange} ({e}) — "
            f"try --symbol CL, or check the account sees this product")
    ltd, mult, local = meta.get("last_trade_date"), meta.get("multiplier"), meta.get("local_symbol")
    check("front month resolved", bool(ltd), f"{local}  expiry={ltd}  multiplier={mult}")
    today = datetime.now().strftime("%Y%m%d")
    expf = ltd if ltd and len(ltd) == 8 else (ltd + "01" if ltd else "")
    check("resolved contract is not past/expiring (roll buffer skipped near-expiry)",
          bool(expf) and expf >= today, f"expiry {ltd} vs today {today}")
    instrument = make_instrument(args.symbol, args.exchange, mult, ltd)

    # 3) reference price -----------------------------------------------------
    hdr("3. Reference price")
    price = args.price
    if price is None:
        price = fetch_reference_price(args.symbol, args.exchange, ltd, mult,
                                      args.tws_host, args.tws_port, args.price_client_id)
    if not price:
        die("no reference price (no market-data permission off-hours?) — re-run with --price <px>")
    check("have a reference price", True, f"{price:.4f}  (notional/contract ≈ ${price*(mult or 1):,.0f})")

    # confirmation ------------------------------------------------------------
    if not args.yes:
        ans = input(f"\nPlace a REAL paper order: {args.path} {args.qty:+d} {local or args.symbol} "
                    f"under '{args.strategy}'? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            die("cancelled by user", code=1)

    start_pos = pos_of(get("/positions"), args.symbol)

    # 4) open ----------------------------------------------------------------
    hdr(f"4. Open {args.qty:+d} {args.symbol} via {args.path}")
    oid, raw = submit(args.path, args.strategy, args.symbol, args.qty, instrument, price)
    if not check("order accepted", raw.get("accepted", False), str(raw)):
        die(f"submission rejected: {raw}")
    if oid is None and raw.get("note"):
        # pool path can no-op if already at target (shouldn't happen from flat, but be graceful)
        check("no order needed (already at net target)", True, raw.get("note"))
    else:
        check("order id issued", oid is not None, f"order_id={oid}")
        final = wait_for_fill(oid, abs(args.qty), args.timeout)
        filled = (final or {}).get("filled")
        avg = (final or {}).get("avg_fill_price")
        check("order filled", bool(final) and (final.get("status", "").lower() == "filled"
              or (filled and abs(filled) >= abs(args.qty))),
              f"status={(final or {}).get('status')} filled={filled} avg={avg}")

    # 5) verify state --------------------------------------------------------
    hdr("5. Verify ledger / positions / fills")
    time.sleep(1.0)  # let execDetails settle
    positions = get("/positions")
    now_pos = pos_of(positions, args.symbol)
    check("net position moved by the traded qty",
          abs((now_pos - start_pos) - args.qty) < 1e-6,
          f"{start_pos:g} -> {now_pos:g} (Δ {now_pos-start_pos:+g}, expected {args.qty:+d})")
    strat_pos = (positions.get("strategy_positions") or {}).get(args.strategy, {}).get(args.symbol, 0.0)
    check(f"position attributed to '{args.strategy}'", abs(strat_pos - args.qty) < 1e-6 or args.path == "direct" and abs(strat_pos - args.qty) < 1e-6,
          f"{args.strategy}[{args.symbol}]={strat_pos:g}")
    fills = get("/fills", params={"limit": 5}).get("fills", [])
    check("fill recorded in the fills log", any(f.get("symbol") == args.symbol for f in fills),
          f"{len(fills)} recent fills")
    if args.path == "pool":
        net = get("/net")
        check("coordinator net book reflects the target",
              abs((net.get("net") or {}).get(args.symbol, 0.0) - now_pos) < 1e-6,
              f"net[{args.symbol}]={ (net.get('net') or {}).get(args.symbol) }")

    # 6) flatten -------------------------------------------------------------
    if args.no_flatten:
        hdr("6. Flatten — SKIPPED (--no-flatten). You are holding the position.")
    else:
        hdr("6. Flatten back to start")
        oid2, raw2 = submit(args.path, args.strategy, args.symbol, start_pos, instrument, price)
        if check("flatten accepted", raw2.get("accepted", False), str(raw2)):
            if oid2 is not None:
                wait_for_fill(oid2, abs(args.qty), args.timeout)
            time.sleep(1.0)
            end_pos = pos_of(get("/positions"), args.symbol)
            check("position back to start", abs(end_pos - start_pos) < 1e-6,
                  f"{args.symbol} now {end_pos:g} (start {start_pos:g})")

    # summary ----------------------------------------------------------------
    passed = sum(1 for r in _RESULTS if r)
    total = len(_RESULTS)
    ok = passed == total
    print(f"\n\033[1m{'PASS' if ok else 'FAIL'}\033[0m — {passed}/{total} checks")
    if not ok:
        print("Some checks failed. If this was off-hours, a non-fill can mean the futures "
              "session is in its daily maintenance break — retry, or use --no-flatten to inspect.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
