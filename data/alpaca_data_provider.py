# data/alpaca_data_provider.py
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
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

    def get_daily_bars_many(self, symbols: list[str], lookback_days: int) -> dict[str, pd.Series]:
        """One paginated request for the whole list — per-symbol calls on a 500-name universe
        run into Alpaca's rate limit and the scheduler's 300s runner timeout.

        Split- and dividend-adjusted: Alpaca's default is raw prices, where a 10:1 split reads
        as a 90% crash and a reverse split as a 400% rally. The latest close is unchanged by
        adjustment, so it is still the right price to size and value orders with."""
        start = datetime.now() - timedelta(days=lookback_days * 2)
        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=TimeFrame.Day,
            start=start,
            adjustment=Adjustment.ALL,
        )
        df = self._client.get_stock_bars(request).df
        if df.empty:
            return {}
        closes = df["close"]
        return {sym: closes.xs(sym, level="symbol")
                for sym in closes.index.get_level_values("symbol").unique()}

    def get_current_price(self, symbol: str) -> float:
        request = StockLatestTradeRequest(symbol_or_symbols=symbol)
        latest = self._client.get_stock_latest_trade(request)
        return latest[symbol].price
