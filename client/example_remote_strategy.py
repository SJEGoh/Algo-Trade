#!/usr/bin/env python3
"""
client/example_remote_strategy.py — a strategy running off the executor's host.

A working template, not a strategy worth trading: equal-weight the N strongest names of a
watchlist by 20-day return. It exists to show how little a RemoteStrategy has to be — the
preflight, capital lookup, validation, submission, journalling and exit codes all come from
the base class, so what is left below is the signal and nothing else.

    python3 example_remote_strategy.py --dry-run   # generate and validate, submit nothing
    python3 example_remote_strategy.py             # submit it

Deploying elsewhere: copy this `client/` directory, `pip install requests yfinance`, set
EXECUTOR_URL and EXECUTOR_API_KEY (see README.md), and run it from cron.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.remote_strategy import RemoteStrategy


class MomentumStrategy(RemoteStrategy):
    strategy_id = os.environ.get("STRATEGY_ID", "cross_sectional_momentum")
    require_market_open = True

    WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JNJ", "XOM", "PG", "WMT"]
    TOP_N = 4
    LOOKBACK_DAYS = 20

    def generate_book(self, capital):
        import yfinance as yf

        bars = yf.download(self.WATCHLIST, period=f"{self.LOOKBACK_DAYS * 2}d",
                           progress=False, auto_adjust=True)["Close"].dropna()
        self.returns = (bars.iloc[-1] / bars.iloc[-self.LOOKBACK_DAYS] - 1.0) \
            .sort_values(ascending=False)
        winners = list(self.returns.head(self.TOP_N).index)
        per_name = capital / max(len(winners), 1)

        # Every watchlist name is listed, with 0 for the ones we don't want. The book is
        # authoritative either way, but saying it explicitly puts "evaluated and rejected"
        # in the journal instead of silence.
        return [self.intent(symbol, int(per_name // float(bars[symbol].iloc[-1]))
                            if symbol in winners else 0,
                            round(float(bars[symbol].iloc[-1]), 2))
                for symbol in self.WATCHLIST]

    def describe(self, book):
        held = [i["instrument"]["symbol"] for i in book if i["target_quantity"]]
        return (f"top {self.TOP_N} of {len(self.WATCHLIST)} by {self.LOOKBACK_DAYS}d return: "
                + (", ".join(held) or "none passed the filter"))

    def journal_detail(self, book):
        return str({s: round(float(r), 4) for s, r in self.returns.items()})


if __name__ == "__main__":
    raise SystemExit(MomentumStrategy.cli())
