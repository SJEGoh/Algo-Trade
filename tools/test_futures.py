#!/usr/bin/env python3
"""
tools/test_futures.py — does the connected IBKR PAPER account see & trade energy futures?

For WTI (CL), Brent (BZ), RBOB (RB) on NYMEX it:
  1. resolves the front-month contract      (reqContractDetails)  -> can the account see it?
  2. pulls a delayed price                   (reqMktData type 3)   -> market-data permission?
  3. runs a whatIf order (margin only, NO real order placed)       -> TRADING permission?

Run in the algo_trade venv with paper TWS up on 7497:
    python3 tools/test_futures.py
"""
import threading, time
from datetime import datetime

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order

SYMBOLS = [("CL", "WTI crude"), ("BZ", "Brent crude"), ("RB", "RBOB gasoline")]
EXCHANGE = "NYMEX"
INFORMATIONAL = {2104, 2106, 2158, 2107, 2119, 2100, 2108}
PRICE_TICKS = {1, 2, 4, 9, 66, 67, 68, 75}   # bid/ask/last/close incl. delayed variants


class FuturesTest(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self._ready = threading.Event()
        self.next_id = 0
        self.details = {}
        self.details_end = {}
        self.prices = {}
        self.whatif = {}
        self.whatif_evt = {}
        self.errors = []

    def nextValidId(self, orderId):
        self.next_id = orderId
        self._ready.set()

    def error(self, reqId, code, msg, *a):
        if code in INFORMATIONAL:
            return
        self.errors.append((reqId, code, msg))

    def contractDetails(self, reqId, cd):
        self.details.setdefault(reqId, []).append(cd.contract)

    def contractDetailsEnd(self, reqId):
        self.details_end.get(reqId, threading.Event()).set()

    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType in PRICE_TICKS and price and price > 0 and reqId not in self.prices:
            self.prices[reqId] = price

    def openOrder(self, orderId, contract, order, orderState):
        self.whatif[orderId] = {
            "init_margin": orderState.initMarginChange,
            "maint_margin": orderState.maintMarginChange,
            "commission": orderState.commission,
        }
        self.whatif_evt.get(orderId, threading.Event()).set()

    def _rid(self):
        self.next_id += 1
        return self.next_id

    def resolve_front(self, symbol):
        rid = self._rid()
        self.details_end[rid] = threading.Event()
        c = Contract(); c.symbol = symbol; c.secType = "FUT"; c.exchange = EXCHANGE; c.currency = "USD"
        self.reqContractDetails(rid, c)
        self.details_end[rid].wait(timeout=8)
        cands = self.details.get(rid, [])
        today = datetime.now().strftime("%Y%m%d")
        dated = []
        for k in cands:
            exp = k.lastTradeDateOrContractMonth
            expf = exp if len(exp) == 8 else exp + "01"
            if expf >= today:
                dated.append((expf, k))
        dated.sort()
        return dated[0][1] if dated else None

    def get_price(self, contract):
        rid = self._rid()
        self.reqMktData(rid, contract, "", True, False, [])
        t = time.time() + 5
        while time.time() < t and rid not in self.prices:
            time.sleep(0.2)
        self.cancelMktData(rid)
        return self.prices.get(rid)

    def what_if(self, contract):
        oid = self.next_id; self.next_id += 1
        self.whatif_evt[oid] = threading.Event()
        o = Order()
        o.action = "BUY"; o.orderType = "MKT"; o.totalQuantity = 1
        o.whatIf = True; o.eTradeOnly = False; o.firmQuoteOnly = False
        self.placeOrder(oid, contract, o)
        self.whatif_evt[oid].wait(timeout=8)
        return self.whatif.get(oid)


def main():
    app = FuturesTest()
    app.connect("127.0.0.1", 7497, clientId=11)
    threading.Thread(target=app.run, daemon=True).start()
    if not app._ready.wait(timeout=8):
        print("FAIL: could not connect / no nextValidId — is paper TWS up on 7497?")
        return
    app.reqMarketDataType(3)  # delayed OK for a permissions probe

    print(f"{'SYM':<5}{'RESOLVED':<22}{'PRICE':>10}   TRADING PERMISSION")
    print("-" * 70)
    for sym, label in SYMBOLS:
        front = app.resolve_front(sym)
        if front is None:
            print(f"{sym:<5}{'NOT FOUND':<22}{'-':>10}   (no contract — check exchange/permissions)")
            continue
        desc = f"{front.localSymbol or sym} {front.lastTradeDateOrContractMonth}"
        price = app.get_price(front)
        wi = app.what_if(front)
        if wi and wi.get("init_margin") not in ("", None):
            perm = f"OK (init margin ~${float(wi['init_margin']):,.0f})"
        else:
            perm = "NO / unknown — see errors below"
        print(f"{sym:<5}{desc:<22}{(f'{price:.2f}' if price else 'no data'):>10}   {perm}")

    if app.errors:
        print("\n-- IB messages (non-informational) --")
        for rid, code, msg in app.errors:
            print(f"  [{code}] {msg}")
        print("\nkey codes: 200=contract not found  10089/10091/354=market-data not subscribed"
              "  201/463=no trading permission")
    else:
        print("\nno error messages — clean.")

    app.disconnect()
    time.sleep(1)


if __name__ == "__main__":
    main()
