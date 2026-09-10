"""models/test_strategies.py — throwaway strategies for exercising the plumbing.

NOT trading candidates. These exist to put real orders through the whole path — signal ->
intent -> risk check -> netting -> IB -> fill -> attribution -> drawdown halt — using
textbook signals nobody is claiming an edge for. `models/equity_strategies.py` is where a
strategy with a backtest behind it belongs; anything here is disposable.

  * MacdCrossStrategy      ("halt_test_macd")      — long while MACD is above its signal.
  * BollingerReversionStrategy ("halt_test_bollinger") — long a close below the lower band,
                                                     out when it recovers the mid.

Both run on INTRADAY (30-minute) bars and are fired repeatedly through the session, for the
same reason the drawdown limits are tight: a strategy that trades once a day would take a
week to reach its halt threshold, where one re-evaluating every half hour reaches it in an
afternoon. On daily bars the Bollinger signal is flat most of the time — historically 13-35%
of days held per name — which exercises nothing.

Both are deliberately sized to trip a halt
------------------------------------------
Their config entries carry `max_drawdown: 0.01` against a $10,000 allocation, so $100 of
loss halts and flattens them. That is the point: the halt path is the hardest thing in the
system to test, because a strategy you actually want to keep should never breach. A pair of
disposable strategies that WILL breach within a day or two exercises it for real — halt,
flatten, persistence across a restart, and the Telegram alert — without risking anything
that matters.

The universe is small and the lot size low for the same reason: 5 names x $1,000 is $5,000
of gross against a $10,000 cap, so orders are not rejected by the allocation check before
they can reach the thing being tested.

NAMING: the `halt_test` prefix is load-bearing. run_rebalance.py excludes those ids from
capital allocation, so the rebalancer never hands real money to a test fixture.
"""
from __future__ import annotations

from typing import Optional

import indicators as ind
import indicators_ext as ext
from models.equity_strategies import _EquityBase
from equity_signals import held_state

#: Small, liquid and cheap enough per share that $1,000 buys a sane round lot.
TEST_UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMD", "META"]


class _IntradayBarStrategy(_EquityBase):
    """30-minute OHLC for a small universe, matching OrbBreakoutStrategy's download shape.

    10 days of 30-minute bars is ~130 observations — enough for MACD(26) and Bollinger(20)
    to be defined, and short enough that yfinance will serve it (intraday history is capped
    at 60 days).
    """

    def __init__(self, strategy_id, universe=None, lot_dollars: float = 1_000.0,
                 intraday_period: str = "10d", interval: str = "30m", **kw):
        super().__init__(strategy_id, universe=universe or TEST_UNIVERSE,
                         lot_dollars=lot_dollars, **kw)
        self.intraday_period = intraday_period
        self.interval = interval

    def _yf_ohlc(self):
        import yfinance as yf
        raw = yf.download(self.universe, period=self.intraday_period, interval=self.interval,
                          auto_adjust=True, progress=False, group_by="ticker")
        data = {}
        for t in self.universe:
            try:
                df = raw[t].dropna().rename(columns=str.lower)
            except Exception:
                continue
            if len(df) > 60:
                data[t] = df[["open", "high", "low", "close", "volume"]]
        return data



class MacdCrossStrategy(_IntradayBarStrategy):
    """Long while the MACD line sits above its signal line, flat otherwise.

    Deliberately the textbook version — no filters, no confirmation. On 30-minute bars it
    will whipsaw badly, which is useful here: a strategy that trades often reaches its
    drawdown limit sooner, and reaching it is the whole purpose.
    """

    def __init__(self, strategy_id: str = "halt_test_macd", fast: int = 12, slow: int = 26,
                 signal: int = 9, **kw):
        super().__init__(strategy_id, **kw)
        self.fast, self.slow, self.signal = int(fast), int(slow), int(signal)

    def _selected(self, data):
        held = set()
        for t, df in data.items():
            line, sig, _hist = ind.macd(df["close"], self.fast, self.slow, self.signal)
            if int(held_state(ind.cross_up(line, sig), ind.cross_down(line, sig)).iloc[-1]):
                held.add(t)
        return held


class BollingerReversionStrategy(_IntradayBarStrategy):
    """Buy a close below the lower Bollinger band, exit when it recovers the middle band.

    Mean reversion, so it complements the trend-following MACD test — between them the two
    exercise both a strategy that holds through a drift and one that trades against it,
    which is what makes the netting and attribution paths interesting. On intraday bars the
    bands are crossed often enough to actually trade, where the daily version sits flat for
    weeks at a time.
    """

    def __init__(self, strategy_id: str = "halt_test_bollinger", n: int = 20,
                 k: float = 2.0, **kw):
        super().__init__(strategy_id, **kw)
        self.n, self.k = int(n), float(k)

    def _selected(self, data):
        held = set()
        for t, df in data.items():
            mid, _upper, lower = ext.bollinger(df["close"], self.n, self.k)
            buy = ind.cross_down(df["close"], lower)     # crossing BELOW the lower band
            sell = ind.cross_up(df["close"], mid)        # back through the middle
            if int(held_state(buy, sell).iloc[-1]):
                held.add(t)
        return held
