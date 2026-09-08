#!/usr/bin/env python3
"""
client/local_strategy.py — a single-name strategy you can actually run, from this laptop.

The point of this file is to prove the whole path works end to end with real money at a size
that cannot hurt: laptop -> Tailscale -> executor -> IB. It fetches a live price, sizes one
position against a notional you choose, and submits it as an absolute book.

    python3 client/local_strategy.py --dry-run              # build + validate, send nothing
    python3 client/local_strategy.py --symbol AAPL --notional 500
    python3 client/local_strategy.py --symbol AAPL --flat    # close it again

Why `test_suite_small_alloc` is the default strategy_id
------------------------------------------------------
Two reasons, and both matter before you point this at a real strategy:

  1. `/targets` is AUTHORITATIVE for the strategy_id it names — any symbol missing from the
     book gets CLOSED. Sending a one-name book as `cross_sectional_momentum` would flatten
     every other name that strategy holds. A dedicated id can't collide with a live book.
  2. That id is allocated $1,000 in config.CONFIG, so the executor's own allocation cap
     refuses anything larger. The guard rail is server-side, not a promise made here.

Once you trust the path, set STRATEGY_ID (or --strategy) to the real strategy and let a
proper signal generate the book.

Environment (see client/README.md):
    EXECUTOR_URL=https://executor.tail82fc30.ts.net
    EXECUTOR_API_KEY=...        # same value as the executor's .env
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.remote_strategy import RemoteStrategy, StrategyError

log = logging.getLogger("local-strategy")


class LocalStrategy(RemoteStrategy):
    strategy_id = os.environ.get("STRATEGY_ID", "test_suite_small_alloc")
    require_market_open = True

    def __init__(self, symbol="AAPL", notional=500.0, flat=False, **kwargs):
        super().__init__(**kwargs)
        self.symbol = symbol.upper()
        self.notional = float(notional)
        self.flat = flat

    def last_price(self) -> float:
        """Most recent close. This becomes `expected_price`, which the executor values the
        leg with when it applies the allocation cap — so it has to be a real number, not a
        placeholder, or the order AND the limit meant to contain it are both wrong."""
        import yfinance as yf

        close = yf.download(self.symbol, period="5d", progress=False,
                            auto_adjust=True)["Close"]
        if close.empty:
            raise StrategyError(f"no price data for {self.symbol} — check the ticker")
        return round(float(close.squeeze().iloc[-1]), 2)

    def generate_book(self, capital):
        price = self.last_price()

        if self.flat:
            log.info("closing %s", self.symbol)
            return [self.intent(self.symbol, 0, price)]

        budget = min(self.notional, capital)
        if budget < self.notional:
            log.warning("sizing against the executor's $%s allocation, not the $%s asked "
                        "for", f"{capital:,.0f}", f"{self.notional:,.0f}")
        quantity = int(budget // price)
        if quantity < 1:
            raise StrategyError(
                f"{self.symbol} is ${price:,.2f} but the budget is ${budget:,.2f} — "
                f"raise --notional above one share, or pick a cheaper name")
        log.info("%s @ $%.2f -> %d shares (~$%.2f of $%.0f)",
                 self.symbol, price, quantity, quantity * price, budget)
        return [self.intent(self.symbol, quantity, price)]

    def describe(self, book):
        entry = book[0]
        qty = entry["target_quantity"]
        if not qty:
            return f"local smoke test: closing {self.symbol}"
        return (f"local smoke test: {self.symbol} target {qty:g} @ "
                f"{entry['expected_price']:.2f} (~${qty * entry['expected_price']:,.0f})")

    def journal_detail(self, book):
        return f"submitted from {os.uname().nodename} via {self.client.base_url}"

    @classmethod
    def cli(cls, argv=None, **kwargs):
        """Parse this strategy's own flags, then hand the rest to the base class so the
        exit codes, --dry-run and logging setup stay identical to every other strategy."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--symbol", default="AAPL")
        parser.add_argument("--notional", type=float, default=500.0,
                            help="dollars to put to work (default 500)")
        parser.add_argument("--flat", action="store_true",
                            help="target zero — close the position instead of opening it")
        mine, rest = parser.parse_known_args(argv)
        return super().cli(rest, symbol=mine.symbol, notional=mine.notional,
                           flat=mine.flat, **kwargs)


if __name__ == "__main__":
    raise SystemExit(LocalStrategy.cli())
