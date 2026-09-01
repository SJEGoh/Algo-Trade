"""tests/test_equity_strategies.py — equities adapters (ovn_volsurge, orb_breakout).
Injects synthetic data so it runs with numpy/pandas (no yfinance/IBKR)."""
import numpy as np
import pandas as pd

from equity_signals import orb_breakout_signal, held_state
from models.equity_strategies import OvernightVolSurgeStrategy, OrbBreakoutStrategy


# ---------- daily OHLC (for ovn_volsurge) ----------
def _daily(n=120, seed=1):
    rng = np.random.default_rng(seed)
    close = np.abs(100 + np.cumsum(rng.normal(0, 1, n))) + 10
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": rng.integers(1e6, 5e6, n).astype(float)}, index=idx)


def _set_last_vol(df, mult):
    df.iloc[-1, df.columns.get_loc("volume")] = df["volume"].iloc[-20:-1].mean() * mult
    return df


# ---------- intraday 30-min bars (for orb_breakout) ----------
def _intraday(days=4, bpd=13, breakout=False):
    rows, start, price = [], pd.Timestamp("2024-06-03 09:30"), 100.0
    for d in range(days):
        day0 = start + pd.Timedelta(days=d)
        for b in range(bpd):
            ts = day0 + pd.Timedelta(minutes=30 * b)
            rows.append((ts, price, price + 0.3, price - 0.3, price, 2e5))
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).set_index("ts")
    if breakout:
        ls = len(df) - bpd
        ci, hi, li, vi = (df.columns.get_loc(c) for c in ("close", "high", "low", "volume"))
        for j in (ls, ls + 1):                       # opening range: high 100.5
            df.iloc[j, hi] = 100.5; df.iloc[j, ci] = 100.0; df.iloc[j, li] = 99.5
        df.iloc[ls + 2, ci] = 99.8; df.iloc[ls + 2, hi] = 100.0   # bar 2: below OR-hi
        for j in range(ls + 3, ls + bpd):            # bar 3+: break above & hold
            df.iloc[j, ci] = 106.0; df.iloc[j, hi] = 106.5; df.iloc[j, li] = 105.5
        df.iloc[ls + 3, vi] = df["volume"].iloc[:ls + 3].mean() * 6   # breakout volume
    return df


def test_held_state_sequence():
    buy = pd.Series([False, True, False, False, False])
    sell = pd.Series([False, False, False, True, False])
    assert list(held_state(buy, sell)) == [0, 1, 1, 0, 0]


# ---- ovn_volsurge ----
def test_volsurge_enter_selects_only_surge_names():
    d = {"AAA": _set_last_vol(_daily(seed=1), 3.0), "BBB": _set_last_vol(_daily(seed=2), 0.3)}
    strat = OvernightVolSurgeStrategy(phase="enter", universe=["AAA", "BBB"],
                                      lot_dollars=10_000, ohlc_fn=lambda: d)
    tgt = {i["instrument"]["symbol"]: i["target_quantity"] for i in strat.generate_intents()}
    assert tgt["AAA"] > 0 and tgt["BBB"] == 0


def test_volsurge_exit_flattens_everything():
    d = {"AAA": _set_last_vol(_daily(seed=1), 3.0)}
    strat = OvernightVolSurgeStrategy(phase="exit", universe=["AAA"], ohlc_fn=lambda: d)
    assert all(i["target_quantity"] == 0 for i in strat.generate_intents())


# ---- orb_breakout (intraday) ----
def test_orb_signal_shape():
    buy, sell = orb_breakout_signal(_intraday())
    assert buy.dtype == bool and sell.dtype == bool


def test_orb_breakout_goes_long():
    df = _intraday(breakout=True)
    strat = OrbBreakoutStrategy(universe=["AAA"], lot_dollars=10_000, ohlc_fn=lambda: {"AAA": df})
    i = strat.generate_intents()[0]
    assert i["target_quantity"] > 0
    assert i["instrument"]["asset_class"] == "equity"


def test_orb_no_breakout_flat():
    df = _intraday(breakout=False)
    strat = OrbBreakoutStrategy(universe=["AAA"], ohlc_fn=lambda: {"AAA": df})
    assert strat.generate_intents()[0]["target_quantity"] == 0
