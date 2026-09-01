"""src/equity_signals.py — equities signals ported from the NUSSIF claude research.

orb_breakout (intraday opening-range breakout, copied verbatim from strategies_new.py)
+ held_state (event->position). Runs on INTRADAY (30-min) bars — it uses the day's
opening range and bar-of-day. pairs_ratio / keltner_squeeze were removed (biased-universe
picks; did not survive the rigorous unbiased evaluation — STRATEGY_LOG.md).
"""
from __future__ import annotations
import pandas as pd
import indicators as ind
import indicators_ext as ext


def orb_breakout_signal(df, or_bars: int = 2, vol_n: int = 20):
    """Opening-range breakout: buy the first close above the day's opening-range high
    (first hour) on above-average volume, early in the session; exit on a failed
    breakout (close back below the OR high or the range mid). INTRADAY bars only."""
    or_hi, or_lo, nday = ext.opening_range(df, or_bars)
    vol_ok = df["volume"] > ind.sma(df["volume"], vol_n)
    early = nday <= 8                                   # act before ~13:30 on 30-min bars
    buy = ind.cross_up(df["close"], or_hi) & vol_ok & early
    mid = (or_hi + or_lo) / 2
    sell = ind.cross_down(df["close"], or_hi) | ind.cross_down(df["close"], mid)
    return buy.fillna(False), sell.fillna(False)


def held_state(buy, sell) -> pd.Series:
    """Event buy/sell booleans -> held-position state (1 after a buy until a sell)."""
    held, out = 0, []
    for b, s in zip(buy.fillna(False).values, sell.fillna(False).values):
        if held == 0 and b:
            held = 1
        elif held == 1 and s:
            held = 0
        out.append(held)
    return pd.Series(out, index=buy.index)
