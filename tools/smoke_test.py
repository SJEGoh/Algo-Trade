#!/usr/bin/env python3
"""
tools/smoke_test.py — live smoke test of the executor server against paper TWS.

Validates the whole stack short of a real fill: server boot + IB connection,
every read endpoint, dashboard assets, auth on the write endpoints, reconcile,
the POST /orders processing path (via a deliberately-rejected intent — no order
is placed), and that the equity sampler writes a real snapshot.

Run (server must be up):   python3 tools/smoke_test.py
Optional real-order test:  python3 tools/smoke_test.py --place
    places a far, non-filling limit order. There is NO cancel endpoint, so it
    stays open until you cancel it in TWS (or POST /kill cancels ALL open orders).
"""
import os, sys, time
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
BASE = os.environ.get("EXECUTOR_URL", "http://127.0.0.1:8000")
KEY = os.environ.get("EXECUTOR_API_KEY")

_passed = _failed = 0
def check(name, ok, detail=""):
    global _passed, _failed
    _passed += bool(ok); _failed += (not ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return ok

def get(path, **kw):  return requests.get(f"{BASE}{path}", timeout=15, **kw)
def post(path, **kw): return requests.post(f"{BASE}{path}", timeout=20, **kw)

print(f"== smoke test against {BASE} ==\n")

# 0. reachable + auth key present
if not KEY:
    print("[FAIL] EXECUTOR_API_KEY not set in .env"); sys.exit(1)
try:
    h = get("/health")
except requests.exceptions.ConnectionError:
    print(f"[FAIL] server not reachable at {BASE} — is uvicorn running?"); sys.exit(1)

# 1. health / IB connection
hj = h.json()
check("GET /health 200", h.status_code == 200, str(hj))
check("IB connected", hj.get("connected") is True, "start/confirm paper TWS (port 7497)")
check("not in kill state", hj.get("killed") is False)
check("GET /health reports market_open", "market_open" in hj, f"market_open={hj.get('market_open')}")

# 2. read endpoints
for path in ["/strategies", "/positions", "/pnl", "/orders", "/fills", "/pnl/history"]:
    try:
        r = get(path)
        check(f"GET {path} 200", r.status_code == 200)
    except Exception as e:
        check(f"GET {path} 200", False, str(e))
strats = get("/strategies").json().get("strategies", [])
check("cross_sectional_momentum configured",
      any(s["strategy_id"] == "cross_sectional_momentum" for s in strats))

# 3. dashboard + self-hosted asset
check("GET / (dashboard) 200", get("/").status_code == 200)
check("GET /static/uPlot.min.js 200", get("/static/uPlot.min.js").status_code == 200)

# 4. auth is enforced on write endpoints
check("POST /orders no key -> 401", post("/orders", json={}).status_code == 401)
check("POST /orders wrong key -> 401",
      post("/orders", json={}, headers={"X-API-Key": "nope"}).status_code == 401)
check("POST /kill no key -> 401", post("/kill", json={}).status_code == 401)

# 5. reconcile (auth)
r = post("/reconcile", headers={"X-API-Key": KEY})
check("POST /reconcile 200", r.status_code == 200,
      str(r.json())[:120] if r.status_code == 200 else r.text[:120])

# 6. POST /orders processing — deliberately-rejected intent, NO order placed
reject = {
    "strategy_id": "smoke_test_unconfigured",           # not on allowlist -> risk rejects
    "client_order_id": f"smoke-{int(time.time())}",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "schema_version": "1.0",
    "instrument": {"symbol": "SPY", "asset_class": "equity", "exchange": "SMART"},
    "intent_type": "delta", "side": "buy", "quantity": 1,
    "order_type": "market", "expected_price": 100.0, "time_in_force": "day",
}
r = post("/orders", json=reject, headers={"X-API-Key": KEY})
rj = r.json() if r.status_code == 200 else {}
check("POST /orders (auth) processes -> 200", r.status_code == 200, str(rj)[:120])
check("POST /orders rejected cleanly, no order placed (risk or market gate)",
      rj.get("accepted") is False, rj.get("reason", ""))

# 7. optional real placement (far limit that won't fill)
if "--place" in sys.argv:
    place = {
        "strategy_id": "cross_sectional_momentum",
        "client_order_id": f"smoke-place-{int(time.time())}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
        "instrument": {"symbol": "AAPL", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "delta", "side": "buy", "quantity": 1,
        "order_type": "limit", "limit_price": 1.00,     # far below market -> won't fill
        "metadata": {"allow_when_closed": True},        # opt past the market-hours gate
    }
    r = post("/orders", json=place, headers={"X-API-Key": KEY}); rj = r.json()
    oid = rj.get("order_id")
    check("POST /orders places real order",
          r.status_code == 200 and rj.get("accepted") and oid is not None, str(rj)[:120])
    if oid is not None:
        gr = get(f"/orders/{oid}").json()
        check(f"GET /orders/{oid} reflects it", gr.get("order_id") == oid, gr.get("status", ""))
        check("order appears in GET /orders list",
              any(o.get("order_id") == oid for o in get("/orders").json().get("orders", [])))
        print(f"    NOTE: order {oid} is a far limit that won't fill; no cancel endpoint — "
              "cancel it in TWS, or POST /kill cancels ALL open orders.")

# 8. equity sampler wrote a real (non-demo) snapshot
print("\n-- waiting up to 75s for the equity sampler's first real snapshot --")
seen = False
deadline = time.time() + 75
while time.time() < deadline:
    real = [x for x in get("/pnl/history").json().get("history", [])
            if not x["strategy_id"].startswith("demo_")]
    if real:
        seen = True
        last = real[-1]
        print(f"    snapshot: {last['strategy_id']} equity={last['equity']} @ {last['ts']}")
        break
    time.sleep(5)
check("equity sampler produced a real snapshot (sampler + marks working)", seen,
      "" if seen else "none in 75s — check the sampler thread / get_marks")

print(f"\n== {_passed} passed, {_failed} failed ==")
sys.exit(1 if _failed else 0)
