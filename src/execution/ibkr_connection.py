from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import BarData
from ibapi.order_state import OrderState
from ibapi.execution import Execution
import threading

from typing import Dict, Optional
import pandas as pd
import time


class TradingApp(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self.data: Dict[int, pd.DataFrame] = {}

        # --- order management state ---
        self._next_order_id: Optional[int] = None
        self._order_id_ready = threading.Event()
        self.order_status: Dict[int, dict] = {}   # orderId -> latest status info

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = "") -> None:
        if advancedOrderRejectJson:
            print(f"Advanced order reject: {advancedOrderRejectJson}")
        print(f"Error: {reqId}, {errorCode}, {errorString}")

    # --- required for order IDs: IB tells you the next valid ID on connect ---
    def nextValidId(self, orderId: int) -> None:
        self._next_order_id = orderId
        self._order_id_ready.set()
        print(f"Next valid order ID: {orderId}")

    def get_next_order_id(self, timeout: float = 5.0) -> int:
        if not self._order_id_ready.wait(timeout=timeout):
            raise TimeoutError("Timed out waiting for nextValidId from IB")
        order_id = self._next_order_id
        self._next_order_id += 1  # reserve it locally so concurrent calls don't collide
        return order_id

    # --- order lifecycle callbacks ---
    def orderStatus(self, orderId: int, status: str, filled: float, remaining: float,
                     avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float,
                     clientId: int, whyHeld: str, mktCapPrice: float) -> None:
        self.order_status[orderId] = {
            "status": status, "filled": filled, "remaining": remaining,
            "avg_fill_price": avgFillPrice, "last_fill_price": lastFillPrice,
        }
        print(f"OrderStatus - id:{orderId} status:{status} filled:{filled} "
              f"remaining:{remaining} avgFillPrice:{avgFillPrice}")

    def openOrder(self, orderId: int, contract: Contract, order: Order, orderState: OrderState) -> None:
        print(f"OpenOrder - id:{orderId} {contract.symbol} {order.action} {order.totalQuantity} "
              f"@ {order.orderType} status:{orderState.status}")

    def execDetails(self, reqId: int, contract: Contract, execution: Execution) -> None:
        print(f"ExecDetails - {contract.symbol} {execution.side} {execution.shares} "
              f"@ {execution.price} orderId:{execution.orderId}")

    # --- order construction helpers ---
    @staticmethod
    def get_contract(symbol: str) -> Contract:
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    @staticmethod
    def market_order(action: str, quantity: float) -> Order:
        order = Order()
        order.action = action          # "BUY" or "SELL"
        order.orderType = "MKT"
        order.totalQuantity = quantity
        order.eTradeOnly = False       # required on newer ibapi versions or orders get rejected
        order.firmQuoteOnly = False
        return order

    @staticmethod
    def limit_order(action: str, quantity: float, limit_price: float) -> Order:
        order = Order()
        order.action = action
        order.orderType = "LMT"
        order.totalQuantity = quantity
        order.lmtPrice = limit_price
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        return order

    def submit_order(self, contract: Contract, order: Order) -> int:
        order_id = self.get_next_order_id()
        self.placeOrder(order_id, contract, order)
        return order_id

    def cancel_order_by_id(self, order_id: int) -> None:
        self.cancelOrder(order_id, "")  # second arg is manual cancel reason, "" is fine

    # --- historical data (unchanged) ---
    def get_historical_data(self, req_id: int, contract: Contract) -> pd.DataFrame:
        self.data[req_id] = pd.DataFrame(columns=["time", "high", "low", "close"]).set_index("time")
        self.reqHistoricalData(
            reqId=req_id, contract=contract, endDateTime="", durationStr="1 D",
            barSizeSetting="1 min", whatToShow="MIDPOINT", useRTH=0,
            formatDate=2, keepUpToDate=False, chartOptions=[]
        )
        time.sleep(3)
        return self.data[req_id]

    def historicalData(self, req_id: int, bar: BarData) -> None:
        df = self.data[req_id]
        df.loc[pd.to_datetime(int(bar.date), unit="s"), ["high", "low", "close"]] = \
            [bar.high, bar.low, bar.close]
        self.data[req_id] = df.astype(float)


if __name__ == "__main__":
    app = TradingApp()
    app.connect("127.0.0.1", 7497, clientId=5)
    threading.Thread(target=app.run, daemon=True).start()

    nvda = TradingApp.get_contract("NVDA")

    data = app.get_historical_data(0, nvda)
    print(data)

    # example: submit a limit buy
    order = TradingApp.limit_order("BUY", quantity=10, limit_price=180.00)
    order_id = app.submit_order(nvda, order)
    print(f"Submitted order id {order_id}")

    time.sleep(5)  # let status/exec callbacks arrive — replace with proper wait/poll logic
    print(app.order_status.get(order_id))
