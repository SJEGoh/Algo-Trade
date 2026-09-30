# strategies/momentum.py
import time
from datetime import datetime, timezone
import pandas as pd

from data.data_provider import DataProvider   # the interface, not a concrete impl


class MomentumStrategy:
    def __init__(self, data_provider: DataProvider, universe: list[str],
                 lookback_days: int = 252, skip_days: int = 21,
                 capital_allocation: float = 100_000, max_positions: int = 20,
                 gross_utilization: float = 0.95):
        self._data = data_provider          # injected — strategy doesn't know or care if it's Alpaca/IB/cached
        self.universe = universe
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.capital_allocation = capital_allocation
        self.max_positions = max_positions            # names held at once, half long / half short
        self.gross_utilization = gross_utilization    # share of the allocation the book's gross uses
        self.strategy_id = "cross_sectional_momentum"

    def compute_signal(self, price_history: dict[str, pd.Series]) -> dict[str, float]:
        scores = {}
        for symbol, prices in price_history.items():
            # 12-1 momentum: return from lookback_days ago to skip_days ago
            past = prices.iloc[-self.lookback_days]
            recent = prices.iloc[-self.skip_days - 1]
            scores[symbol] = recent / past - 1
        return scores

    def construct_weights(self, scores: dict[str, float],
                          prices: dict[str, float] = None) -> dict[str, float]:
        """Long the strongest names, short the weakest: max_positions // 2 a side, and never
        more than a third of the ranked names a side, so a small universe keeps its long and
        short books apart.

        Each side gets half of gross_utilization, so the whole book's gross notional fits
        inside capital_allocation — the executor refuses any order that would take gross past
        it. With `prices`, a name whose slice of capital is less than one share is passed
        over for the next one in line (longs from the top half, shorts from the bottom half),
        instead of holding a side that is a name short."""
        ranked = sorted(scores, key=scores.get, reverse=True)
        n = min(len(ranked) // 3, self.max_positions // 2)
        if n == 0:
            return {}
        side = 0.5 * self.gross_utilization
        per_name = side / n * self.capital_allocation

        def affordable(s):
            return prices is None or prices[s] <= per_name

        half = len(ranked) // 2
        longs = [s for s in ranked[:half] if affordable(s)][:n]
        shorts = [s for s in reversed(ranked[half:]) if affordable(s)][:n]
        weights = {s: side / n for s in longs}
        weights.update({s: -side / n for s in shorts})
        return weights

    def generate_intents(self, held: dict[str, float] = None) -> list[dict]:
        """Full rebalance cycle: fetch → signal → weights → intents.

        Only the selected names get a target, so pass `held` (symbol -> quantity this
        strategy owns now) to close the rest: every held name that was not selected gets a
        target of 0, including one since dropped from the universe. Without it, names that
        fall out of the selection are never sold. Reductions are ordered first so they free
        allocation before the buys are checked against it."""
        held = {s: q for s, q in (held or {}).items() if q}
        bars = self._data.get_daily_bars_many(sorted(set(self.universe) | set(held)),
                                              self.lookback_days)

        # A name without a full lookback (recent IPO / spin-off, or no data at all) can't be
        # scored; skip it rather than let one short series raise and sink the whole universe.
        rankable = {s: bars[s] for s in self.universe
                    if s in bars and len(bars[s]) >= self.lookback_days}
        skipped = [s for s in self.universe if s not in rankable]
        if skipped:
            print(f"skipping {len(skipped)} symbol(s) without {self.lookback_days} days of "
                  f"history: {', '.join(skipped)}")

        prices = {s: float(b.iloc[-1]) for s, b in bars.items()}   # most recent close
        scores = self.compute_signal(rankable)
        weights = self.construct_weights(scores, prices)

        # int() truncates toward zero, so rounding can never push gross over the allocation.
        targets = {s: int(w * self.capital_allocation / prices[s]) for s, w in weights.items()}
        for s in held:
            if s not in targets:
                if s in prices:
                    targets[s] = 0
                else:
                    print(f"WARNING: holding {held[s]:g} {s} but have no price for it — "
                          f"left as is")

        order = sorted(targets, key=lambda s: abs(targets[s]) >= abs(held.get(s, 0)))
        intents = []
        for symbol in order:
            score = scores.get(symbol)
            intents.append({
                "strategy_id": self.strategy_id,
                "client_order_id": f"{self.strategy_id}-{symbol}-{int(time.time())}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "schema_version": "1.0",
                "instrument": {"symbol": symbol, "asset_class": "equity", "exchange": "SMART"},
                "intent_type": "target_position",
                "target_quantity": targets[symbol],
                "order_type": "market",
                "expected_price": prices[symbol],
                "time_in_force": "day",
                "metadata": {"signal_score": None if score is None else float(score)},
            })
        return intents
