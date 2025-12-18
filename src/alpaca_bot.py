from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data import StockHistoricalDataClient, StockTradesRequest
from datetime import datetime

trading_client = TradingClient(KEY, SECRET)


data_client = StockHistoricalDataClient(KEY, SECRET)
request_params = StockTradesRequest(
    symbol_or_symbols = "AAPL"
)

market_order_request = MarketOrderRequest(
    symbol = "SPY",
    qty = 1,
    side = OrderSide.BUY,
    time_in_force = TimeInForce.DAY
)

market_order = trading_client(market_order_request)

limit_order_request = LimitOrderRequest(
    symbol = "SPY",
    qty = 1,
    side = OrderSide.SELL,
    time_in_force = TimeInForce.DAY,
    limit_price = 486.0
)

request_params = GetOrdersRequest(
    status = QueryOrderStatus.OPEN
)

orders = trading_client.get_orders(request_params)

trading_client.cancel_order_by_id(order.id)

positions = trading_client.get_all_positions()

trading_client.close_all_positions(True)

from alpaca.data.live import StockDataStream

stream = StockDataStream(KEY, SECRET)

async def handle_trade(data):
    print(data)

stream.subscribe_trade(handle_trade, "AAPL")
