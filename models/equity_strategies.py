"""models/equity_strategies.py — equities adapters emitting target_position intents.

  * OvernightVolSurgeStrategy ("ovn_volsurge") — best unbiased strategy (Sharpe 2.62).
    Overnight hold of volume-surge names. DAILY bars. Two fires/day:
      phase="enter" near CLOSE -> long the surge names;  phase="exit" at OPEN -> flat.
  * OrbBreakoutStrategy ("orb_breakout") — INTRADAY (30-min) opening-range breakout.
    Fires through the session; long on breakout, flat on failure.

Injectable data fns for testing. Equity path (STK, SMART) — existing executor.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Callable, Optional

from equity_signals import orb_breakout_signal, held_state

DEFAULT_UNIVERSE = ["NVDA","AMD","AVGO","MU","AMZN","META","GOOGL","MSFT","ORCL","TSLA",
                    "PLTR","CRWD","PANW","FTNT","SNOW","ARM","ANET","MRVL","TSM","ASML",
                    "APP","UBER","ABNB","SHOP","MELI","NFLX","COIN","HOOD","RDDT","DDOG",
                    "NET","ZS","OKTA","HUBS","NOW","CRM","ADBE","INTU","KLAC","LRCX",
                    "AMAT","QCOM","TXN","ON","MCHP","NXPI","CDNS","SNPS","VRT","SMCI"]


class _EquityBase:
    def __init__(self, strategy_id, universe=None, lot_dollars: float = 2000.0,
                 ohlc_fn: Optional[Callable] = None, lookback_days: int = 400):
        self.strategy_id = strategy_id
        self.universe = list(universe or DEFAULT_UNIVERSE)
        self.lot_dollars = float(lot_dollars)
        self._ohlc_fn = ohlc_fn or self._yf_ohlc
        self.lookback_days = lookback_days

    def _yf_ohlc(self):
        raise NotImplementedError

    def _targets(self, data) -> dict:
        raise NotImplementedError

    def generate_intents(self) -> list:
        data = self._ohlc_fn()
        targets = self._targets(data)
        now = datetime.now(timezone.utc).isoformat()
        stamp = int(datetime.now().timestamp())
        intents = []
        for t, shares in targets.items():
            price = float(data[t]["close"].iloc[-1])
            intents.append({
                "strategy_id": self.strategy_id,
                "client_order_id": f"{self.strategy_id}-{t}-{stamp}",
                "timestamp": now, "schema_version": "1.0",
                "instrument": {"symbol": t, "asset_class": "equity", "exchange": "SMART"},
                "intent_type": "target_position",
                "target_quantity": int(shares),
                "order_type": "market",
                "expected_price": price,
                "time_in_force": "day",
                "metadata": {"phase": getattr(self, "phase", None)},
            })
        return intents


class OvernightVolSurgeStrategy(_EquityBase):
    """Best unbiased strategy (Sharpe 2.62). Overnight hold of volume-surge names."""
    def __init__(self, strategy_id: str = "ovn_volsurge", phase: str = "enter",
                 surge_mult: float = 1.5, **kw):
        super().__init__(strategy_id, **kw)
        self.phase = phase
        self.surge_mult = float(surge_mult)

    def _yf_ohlc(self):
        import yfinance as yf
        raw = yf.download(self.universe, period=f"{self.lookback_days}d", interval="1d",
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

    def _targets(self, data):
        if self.phase == "exit":
            return {t: 0 for t in data}
        out = {}
        for t, df in data.items():
            v = df["volume"]
            price = float(df["close"].iloc[-1])
            surge = len(v) >= 20 and v.iloc[-1] > self.surge_mult * v.iloc[-20:].mean()
            out[t] = round(self.lot_dollars / price) if surge else 0
        return out


class OrbBreakoutStrategy(_EquityBase):
    """Intraday opening-range breakout (30-min bars). Fire through the session."""
    def __init__(self, strategy_id: str = "orb_breakout", intraday_period: str = "10d",
                 interval: str = "30m", **kw):
        super().__init__(strategy_id, **kw)
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
            if len(df) > 20:
                data[t] = df[["open", "high", "low", "close", "volume"]]
        return data

    def _targets(self, data):
        out = {}
        for t, df in data.items():
            buy, sell = orb_breakout_signal(df)
            held = int(held_state(buy, sell).iloc[-1])
            out[t] = round(self.lot_dollars / float(df["close"].iloc[-1])) if held else 0
        return out
