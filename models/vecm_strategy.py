"""
models/vecm_strategy.py — VECM strategy adapter.

Wraps the pure Kalman VECM core (src/vecm/) into a strategy that emits target_position
OrderIntents in FUTURES CONTRACTS for the executor. Runs once daily (EOD, after
settlement). Persists Kalman filter state between runs so each day advances one bar.

WTI (CL) vs a Brent (BZ) + RBOB (RB) hedge basket. Pure/injectable: pass `price_fn` and
`contract_resolver` for testing; defaults use yfinance + the server's front-month resolver.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from vecm.kalman_vecm import KalmanVECM, KalmanState
from vecm.vol_scale import RollingVolScaler
from vecm.sizing import target_contracts
from vecm.risk_limits import RiskLimits, apply_limits

LEG_SYMBOL = {"y": "CL", "Brent": "BZ", "RBOB": "RB"}      # filter leg -> IB symbol
LEG_PRICE_COL = {"y": "WTI", "Brent": "Brent", "RBOB": "RBOB"}
DEFAULT_MULT = {"CL": 1000, "BZ": 1000, "RB": 42000}
DEFAULT_EXCH = {"CL": "NYMEX", "BZ": "NYMEX", "RB": "NYMEX"}


class VECMStrategy:
    def __init__(self, strategy_id: str = "kalman_vecm", allocation: float = 2_000_000.0,
                 delta: float = 1e-4, R: float = 1e-3, zcap: float = 2.0,
                 vol_window: int = 20, vol_target: float = 0.15, vol_cap: float = 5.0,
                 state_path: Optional[str] = None,
                 price_fn: Optional[Callable[[], pd.DataFrame]] = None,
                 contract_resolver: Optional[Callable[[str], dict]] = None,
                 risk_limits: Optional[RiskLimits] = None, lookback_days: int = 1500):
        self.strategy_id = strategy_id
        self.allocation = float(allocation)
        self.delta, self.R, self.zcap = delta, R, zcap
        self.vol_window, self.vol_target, self.vol_cap = vol_window, vol_target, vol_cap
        self.state_path = Path(state_path) if state_path else None
        self._price_fn = price_fn or self._yfinance_prices
        self._resolver = contract_resolver
        self.risk_limits = risk_limits or RiskLimits(
            max_contracts_per_leg=10, max_leg_notional=250_000,
            max_gross_notional=500_000, max_abs_fractional=50)
        self.lookback_days = lookback_days

    # ---- data (default: yfinance continuous front-month) ----
    def _yfinance_prices(self) -> pd.DataFrame:
        import yfinance as yf
        tick = {"WTI": "CL=F", "Brent": "BZ=F", "RBOB": "RB=F"}
        px = yf.download(list(tick.values()), period=f"{self.lookback_days}d",
                         interval="1d", auto_adjust=True, progress=False)["Close"]
        px = px.rename(columns={v: k for k, v in tick.items()})
        return px[["WTI", "Brent", "RBOB"]].dropna()

    # ---- persisted Kalman state ----
    def _load_state(self) -> Optional[KalmanState]:
        if self.state_path and self.state_path.exists():
            return KalmanState.from_dict(json.loads(self.state_path.read_text()))
        return None

    def _save_state(self, state: KalmanState) -> None:
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(state.to_dict()))

    def _resolve(self, symbol: str) -> dict:
        r = (self._resolver(symbol) or {}) if self._resolver else {}
        return {"last_trade_date": r.get("last_trade_date"),
                "multiplier": r.get("multiplier") or DEFAULT_MULT[symbol],
                "exchange": r.get("exchange") or DEFAULT_EXCH[symbol]}

    # ---- main daily cycle ----
    def generate_intents(self) -> list[dict]:
        prices = self._price_fn()
        if len(prices) < self.vol_window + 5:
            raise ValueError("not enough price history for the VECM")

        logy = np.log(prices["WTI"].values)
        logx = np.log(prices[["Brent", "RBOB"]].values)
        idx_dates = [str(d.date()) for d in prices.index]

        state = self._load_state()
        filt = KalmanVECM(n_hedges=2, delta=self.delta, R=self.R, zcap=self.zcap,
                          hedge_names=["Brent", "RBOB"], state=state)
        if state is None:
            step = filt.warm_start(logy, logx, dates=idx_dates)      # cold start: full history
        else:
            start = idx_dates.index(state.last_date) + 1 if state.last_date in idx_dates else 0
            step = None
            for t in range(start, len(logy)):
                step = filt.step(logy[t], logx[t], date=idx_dates[t])
            if step is None:                                          # already up to date
                step = filt.evaluate(logy[-1], logx[-1])
        self._save_state(filt.snapshot())

        # normalized positions -> vol-scaled effective exposure per leg
        eff_pos = {}
        for leg in ("y", "Brent", "RBOB"):
            rets = prices[LEG_PRICE_COL[leg]].pct_change().dropna().values
            m = RollingVolScaler.latest_from_series(rets, window=self.vol_window,
                                                    target=self.vol_target, cap=self.vol_cap) or 0.0
            eff_pos[LEG_SYMBOL[leg]] = step.positions[leg] * m

        last_px = {LEG_SYMBOL[leg]: float(prices[LEG_PRICE_COL[leg]].iloc[-1])
                   for leg in ("y", "Brent", "RBOB")}
        meta = {sym: self._resolve(sym) for sym in eff_pos}
        mults = {sym: float(meta[sym]["multiplier"]) for sym in eff_pos}

        sizings = target_contracts(eff_pos, last_px, mults, self.allocation)
        decision = apply_limits(sizings, self.risk_limits)
        if decision.aborted:
            raise ValueError(f"VECM risk abort: {decision.abort_reason}")

        now = datetime.now(timezone.utc).isoformat()
        stamp = int(datetime.now().timestamp())
        intents = []
        for sym, contracts in decision.approved_contracts.items():
            intents.append({
                "strategy_id": self.strategy_id,
                "client_order_id": f"{self.strategy_id}-{sym}-{stamp}",
                "timestamp": now,
                "schema_version": "1.0",
                "instrument": {"symbol": sym, "asset_class": "future", "sec_type": "FUT",
                               "exchange": meta[sym]["exchange"], "multiplier": mults[sym],
                               "last_trade_date": meta[sym]["last_trade_date"]},
                "intent_type": "target_position",
                "target_quantity": int(contracts),
                "order_type": "market",
                "expected_price": last_px[sym],
                "time_in_force": "day",
                "metadata": {"z": step.z, "breaches": decision.breaches},
            })
        return intents
