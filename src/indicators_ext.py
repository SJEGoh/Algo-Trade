"""Extra indicators for the new strategy set (30-min bars).

Kept separate from indicators.py so the original swing-dip library is
untouched. Per-ticker indicators take/return Series or DataFrames; the
cross-sectional helpers operate on a wide panel of closes (columns = tickers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ind


# ---------- Per-ticker volatility bands ----------

def bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    mid = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return mid, mid + k * sd, mid - k * sd


def keltner(df: pd.DataFrame, n: int = 20, k: float = 1.5):
    mid = ind.ema(df["close"], n)
    rng = k * ind.atr(df, n)
    return mid, mid + rng, mid - rng


def squeeze_on(df: pd.DataFrame, n: int = 20, bb_k: float = 2.0,
               kc_k: float = 1.5) -> pd.Series:
    """TTM-style squeeze: Bollinger band sits INSIDE the Keltner channel
    (volatility compressed). Returns a boolean Series."""
    _, bb_up, bb_lo = bollinger(df["close"], n, bb_k)
    _, kc_up, kc_lo = keltner(df, n, kc_k)
    return (bb_up < kc_up) & (bb_lo > kc_lo)


# ---------- Session structure helpers (intraday) ----------

def _day(df: pd.DataFrame):
    return pd.Series(df.index.date, index=df.index)


def opening_range(df: pd.DataFrame, bars: int = 2):
    """Rolling opening-range high/low from the first `bars` of each session.
    Value is only defined (non-NaN) from the `bars`-th bar of the day onward,
    so there is no lookahead when used on bar close."""
    day = _day(df)
    n_in_day = day.groupby(day).cumcount()  # 0,1,2,... within each day
    hi = df["high"].where(n_in_day < bars)
    lo = df["low"].where(n_in_day < bars)
    or_hi = hi.groupby(day).transform(lambda x: x.expanding().max())
    or_lo = lo.groupby(day).transform(lambda x: x.expanding().min())
    # freeze the range after the opening window (ffill within day)
    or_hi = or_hi.groupby(day).ffill()
    or_lo = or_lo.groupby(day).ffill()
    valid = n_in_day >= bars
    return or_hi.where(valid), or_lo.where(valid), n_in_day


def prev_close(df: pd.DataFrame) -> pd.Series:
    """Previous session's last close, broadcast across the current day."""
    day = _day(df)
    last = df["close"].groupby(day).last()
    prev = last.shift(1)
    return day.map(prev)


def bar_of_day(df: pd.DataFrame) -> pd.Series:
    day = _day(df)
    return day.groupby(day).cumcount()


# ---------- Cross-sectional panel helpers ----------

def close_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide frame of closes (index = union of timestamps, cols = tickers)."""
    panel = pd.DataFrame({t: df["close"] for t, df in data.items()})
    return panel.sort_index()


def xs_return_z(panel: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Cross-sectional z-score of each name's trailing `lookback`-bar return,
    measured against the basket at each timestamp. z>0 => outperforming peers."""
    ret = panel.pct_change(lookback, fill_method=None)
    mu = ret.mean(axis=1)
    sd = ret.std(axis=1).replace(0, np.nan)
    return ret.sub(mu, axis=0).div(sd, axis=0)


def basket_index(panel: pd.DataFrame) -> pd.Series:
    """Equal-weight normalized basket level (each name rebased to its first
    valid observation, then averaged)."""
    norm = panel / panel.apply(lambda c: c.loc[c.first_valid_index()])
    return norm.mean(axis=1)


def cross_up_df(a: pd.DataFrame, thr) -> pd.DataFrame:
    prev = a.shift()
    return (a > thr) & (prev <= thr)


def cross_down_df(a: pd.DataFrame, thr) -> pd.DataFrame:
    prev = a.shift()
    return (a < thr) & (prev >= thr)
