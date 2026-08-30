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
