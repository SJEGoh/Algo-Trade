"""Indicator library for the swing-dip strategy (30-min bars).

All functions take/return pandas Series/DataFrames indexed by bar timestamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- Trend / lag-reduction ----------

def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def hma(s: pd.Series, n: int = 21) -> pd.Series:
    """Hull Moving Average."""
    return wma(2 * wma(s, n // 2) - wma(s, n), int(np.sqrt(n)))


def zlema(s: pd.Series, n: int = 21) -> pd.Series:
    """Zero-Lag EMA."""
    lag = (n - 1) // 2
    return (2 * s - s.shift(lag)).ewm(span=n, adjust=False).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig  # macd, signal, histogram


# ---------- Momentum / timing ----------

def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3):
    lo = df["low"].rolling(k).min()
    hi = df["high"].rolling(k).max()
    raw_k = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    k_line = raw_k.rolling(smooth).mean()
    d_line = k_line.rolling(d).mean()
    return k_line, d_line


def kdj(df: pd.DataFrame, k: int = 9, d: int = 3):
    """KDJ: J = 3K - 2D (RSV-based, EMA smoothing)."""
    lo = df["low"].rolling(k).min()
    hi = df["high"].rolling(k).max()
    rsv = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    k_line = rsv.ewm(alpha=1 / d, adjust=False).mean()
    d_line = k_line.ewm(alpha=1 / d, adjust=False).mean()
    return k_line, d_line, 3 * k_line - 2 * d_line


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def rsi_bull_divergence(df: pd.DataFrame, n: int = 14, lookback: int = 40,
                        pivot: int = 3) -> pd.Series:
    """True where price makes a lower low but RSI makes a higher low.

    A pivot low is a bar whose low is the minimum of +/- `pivot` bars.
    Compares the two most recent confirmed pivot lows within `lookback` bars.
    Signal is emitted `pivot` bars after the pivot (when it is confirmed) —
    no lookahead.
    """
    r = rsi(df["close"], n)
    low = df["low"]
    is_pivot = (low == low.rolling(2 * pivot + 1, center=True).min())
    out = pd.Series(False, index=df.index)
    pivots: list[tuple[int, float, float]] = []  # (bar_idx, price_low, rsi)
    lows = low.values
    rsis = r.values
    piv = is_pivot.values
    for i in range(len(df)):
        j = i - pivot  # pivot confirmed `pivot` bars later
        if j >= 0 and piv[j] and not np.isnan(rsis[j]):
            pivots.append((j, lows[j], rsis[j]))
            if len(pivots) >= 2:
                (j0, p0, r0), (j1, p1, r1) = pivots[-2], pivots[-1]
                if j1 - j0 <= lookback and p1 < p0 and r1 > r0:
                    out.iloc[i] = True
    return out


# ---------- Volume ----------

def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff()).fillna(0)
    return (sign * df["volume"]).cumsum()


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP anchored to each trading day."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.date
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum()
    return pv / vv.replace(0, np.nan)


# ---------- Volatility ----------

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ---------- Helpers ----------

def cross_up(a: pd.Series, b) -> pd.Series:
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a > b) & (a.shift() <= b.shift())


def cross_down(a: pd.Series, b) -> pd.Series:
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a < b) & (a.shift() >= b.shift())
