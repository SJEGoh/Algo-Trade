from abc import ABC, abstractmethod
import pandas as pd

class DataProvider(ABC):
    @abstractmethod
    def get_daily_bars(self, symbol: str, lookback_days: int) -> pd.Series:
        """Return a series of daily closing prices, most recent last."""
        ...

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        ...

    def get_daily_bars_many(self, symbols: list[str], lookback_days: int) -> dict[str, pd.Series]:
        """Closes for many symbols at once. Symbols with no data are left out rather than
        raising, so one bad name can't sink a whole universe. Override to batch requests."""
        out = {}
        for symbol in symbols:
            try:
                out[symbol] = self.get_daily_bars(symbol, lookback_days)
            except Exception:
                pass
        return out
