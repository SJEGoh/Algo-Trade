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


def inverse_vol_sizing(data, selected, deployable: float, max_weight: float = 0.25,
                       vol_lookback: int = 20) -> dict:
    """Split `deployable` dollars across `selected` names, inversely to their volatility.

    Returns symbol -> dollars. A calmer name carries more capital than a jumpy one, so each
    position contributes roughly the same risk instead of the same notional — which is what
    equal dollar lots quietly got wrong: $2,000 of a 60%-vol name is several times the risk
    of $2,000 of a 15%-vol one.

    The weights are capped at `max_weight` and DELIBERATELY NOT renormalised afterwards.
    Portfolio weights must sum to 1, so capping there redistributes the excess; here the cap
    exists to stop one name taking the book, and redistributing would defeat it — with two
    names and a 25% cap, renormalising hands each 50%. Leaving the sum short instead means a
    thin book simply deploys less than its full allocation, which is the safe direction.
    """
    if not selected or deployable <= 0:
        return {}

    vols = {}
    for t in selected:
        df = data.get(t)
        if df is None or len(df) < 3:
            continue
        r = df["close"].pct_change().dropna().tail(vol_lookback)
        v = float(r.std()) if len(r) >= 2 else 0.0
        if v > 0 and v == v:                     # positive and not NaN
            vols[t] = v
    if not vols:
        # No usable volatility for anything selected — fall back to equal weight rather
        # than silently sizing nothing, but cap it the same way.
        w = min(max_weight, 1.0 / len(selected))
        return {t: deployable * w for t in selected}

    inv_total = sum(1.0 / v for v in vols.values())
    return {t: deployable * min(max_weight, (1.0 / v) / inv_total)
            for t, v in vols.items()}


class _EquityBase:
    def __init__(self, strategy_id, universe=None, lot_dollars: float = 2000.0,
                 ohlc_fn: Optional[Callable] = None, lookback_days: int = 400,
                 capital_allocation: Optional[float] = None,
                 deploy_fraction: float = 1.0, max_weight: float = 0.25,
                 vol_lookback: int = 20):
        self.strategy_id = strategy_id
        self.universe = list(universe or DEFAULT_UNIVERSE)
        self.lot_dollars = float(lot_dollars)
        self._ohlc_fn = ohlc_fn or self._yf_ohlc
        self.lookback_days = lookback_days
        #: The strategy's cap, read from the executor at run time. When it is None the
        #: strategy falls back to fixed `lot_dollars` lots — the old behaviour — so a caller
        #: that has not been taught to fetch it keeps working unchanged.
        self.capital_allocation = (None if capital_allocation is None
                                   else float(capital_allocation))
        self.deploy_fraction = float(deploy_fraction)
        self.max_weight = float(max_weight)
        self.vol_lookback = int(vol_lookback)

    def _yf_ohlc(self):
        raise NotImplementedError

    def _selected(self, data) -> set:
        """Which names the signal wants to hold. Sizing is not the signal's business."""
        raise NotImplementedError

    def _targets(self, data) -> dict:
        """Selection -> share counts, sized against the live allocation."""
        held = set(self._selected(data))
        if self.capital_allocation is None:
            dollars = {t: self.lot_dollars for t in held}
        else:
            dollars = inverse_vol_sizing(
                data, held, self.capital_allocation * self.deploy_fraction,
                max_weight=self.max_weight, vol_lookback=self.vol_lookback)

        out, rounded_out = {}, []
        for t, df in data.items():
            price = float(df["close"].iloc[-1])
            d = dollars.get(t, 0.0)
            shares = int(d / price) if d > 0 and price > 0 else 0
            if d > 0 and shares == 0:
                # Selected, but its risk-weighted share of the allocation does not buy one
                # share. Reported rather than left to look like the signal skipped it.
                rounded_out.append(f"{t} (${d:,.0f} < 1 share @ ${price:,.2f})")
            out[t] = shares
        if rounded_out:
            print(f"  note: selected but sized to zero — {', '.join(rounded_out)}")
        return out

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

    def _selected(self, data):
        if self.phase == "exit":
            return set()
        return {t for t, df in data.items()
                if len(df["volume"]) >= 20
                and df["volume"].iloc[-1] > self.surge_mult * df["volume"].iloc[-20:].mean()}


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

    def _selected(self, data):
        held = set()
        for t, df in data.items():
            buy, sell = orb_breakout_signal(df)
            if int(held_state(buy, sell).iloc[-1]):
                held.add(t)
        return held
