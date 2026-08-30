# data/alpaca_data_provider.py
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
from datetime import datetime, timedelta

from data.data_provider import DataProvider

class AlpacaDataProvider(DataProvider):
    def __init__(self, api_key: str, secret_key: str):
        self._client = StockHistoricalDataClient(api_key, secret_key)

    def get_daily_bars(self, symbol: str, lookback_days: int) -> pd.Series:
        start = datetime.now() - timedelta(days=lookback_days * 2)  # *2 to account for weekends/holidays
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
        )
        bars = self._client.get_stock_bars(request)
        df = bars.df
        if df.empty:
            raise ValueError(f"Alpaca returned no bars for {symbol} (start={start})")
        return df["close"]
    
    def get_current_price(self, symbol: str) -> float:
        request = StockLatestTradeRequest(symbol_or_symbols=symbol)
        latest = self._client.get_stock_latest_trade(request)
        return latest[symbol].price
