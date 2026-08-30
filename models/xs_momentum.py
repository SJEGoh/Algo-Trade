# strategies/momentum.py
import time
from datetime import datetime, timezone
import pandas as pd

from data.data_provider import DataProvider   # the interface, not a concrete impl


class MomentumStrategy:
    def __init__(self, data_provider: DataProvider, universe: list[str],
                 lookback_days: int = 252, skip_days: int = 21,
                 capital_allocation: float = 100_000):
        self._data = data_provider          # injected — strategy doesn't know or care if it's Alpaca/IB/cached
        self.universe = universe
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.capital_allocation = capital_allocation
        self.strategy_id = "cross_sectional_momentum"

    def _fetch_all_bars(self) -> dict[str, pd.Series]:
        """Pull daily bars for every symbol via the data provider."""
        bars = {}
        for symbol in self.universe:
            bars[symbol] = self._data.get_daily_bars(symbol, self.lookback_days)
        return bars

    def compute_signal(self, price_history: dict[str, pd.Series]) -> dict[str, float]:
        scores = {}
        for symbol, prices in price_history.items():
            # 12-1 momentum: return from lookback_days ago to skip_days ago
            past = prices.iloc[-self.lookback_days]
            recent = prices.iloc[-self.skip_days - 1]
            scores[symbol] = recent / past - 1
        return scores

    def construct_weights(self, scores: dict[str, float]) -> dict[str, float]:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        n = len(ranked) // 3
        longs = [s for s, _ in ranked[:n]]
        shorts = [s for s, _ in ranked[-n:]]
        weights = {s: 1.0 / len(longs) for s in longs}
        weights.update({s: -1.0 / len(shorts) for s in shorts})
        return weights

    def generate_intents(self) -> list[dict]:
        """Full rebalance cycle: fetch → signal → weights → intents."""
        bars = self._fetch_all_bars()
        scores = self.compute_signal(bars)
        weights = self.construct_weights(scores)

        intents = []
        for symbol in self.universe:
            weight = weights.get(symbol, 0.0)
            current_price = bars[symbol].iloc[-1]   # most recent close, for sizing + expected_price
            target_dollars = weight * self.capital_allocation
            target_shares = round(target_dollars / current_price)
            intents.append({
                "strategy_id": self.strategy_id,
                "client_order_id": f"{self.strategy_id}-{symbol}-{int(time.time())}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "schema_version": "1.0",
                "instrument": {"symbol": symbol, "asset_class": "equity", "exchange": "SMART"},
                "intent_type": "target_position",
                "target_quantity": target_shares,
                "order_type": "market",
                "expected_price": float(current_price),
                "time_in_force": "day",
                "metadata": {"signal_score": float(scores[symbol])},
            })
        return intents
