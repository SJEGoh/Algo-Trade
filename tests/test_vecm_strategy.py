"""tests/test_vecm_strategy.py — VECM adapter (models/vecm_strategy.py).

Injects synthetic prices + a stub contract resolver so it runs with just numpy/pandas
(no yfinance, no IBKR). Covers intent shape, state persistence/resume, and risk clamping.
"""
import os
import tempfile

import numpy as np
import pandas as pd

from models.vecm_strategy import VECMStrategy
from vecm.risk_limits import RiskLimits


def _prices_df(n=800, seed=7):
    rng = np.random.default_rng(seed)
    brent = 60 + np.cumsum(rng.normal(0, 0.5, n))
    rbob = 1.8 + np.cumsum(rng.normal(0, 0.02, n))
    wti = 0.9 * brent + 3.0 * rbob + rng.normal(0, 0.4, n)
    idx = pd.bdate_range("2016-01-01", periods=n)
    return pd.DataFrame({"WTI": np.abs(wti) + 5, "Brent": np.abs(brent) + 5,
                         "RBOB": np.abs(rbob) + 1}, index=idx)


_RESOLVED = {
    "CL": {"last_trade_date": "20261022", "multiplier": 1000, "exchange": "NYMEX"},
    "BZ": {"last_trade_date": "20261030", "multiplier": 1000, "exchange": "NYMEX"},
    "RB": {"last_trade_date": "20261031", "multiplier": 42000, "exchange": "NYMEX"},
}
def _resolver(sym): return _RESOLVED[sym]
def _tmp(): return os.path.join(tempfile.mkdtemp(), "state.json")


def test_generates_three_futures_intents():
    strat = VECMStrategy(allocation=5_000_000, state_path=_tmp(),
                         price_fn=lambda: _prices_df(), contract_resolver=_resolver)
    intents = strat.generate_intents()
    assert {i["instrument"]["symbol"] for i in intents} == {"CL", "BZ", "RB"}
    for i in intents:
        ins = i["instrument"]
        assert ins["sec_type"] == "FUT"
        assert ins["multiplier"] in (1000, 42000)
        assert ins["last_trade_date"]
        assert i["intent_type"] == "target_position"
        assert isinstance(i["target_quantity"], int)
        assert i["expected_price"] > 0


def test_state_persists_and_resumes():
    sp = _tmp()
    df = _prices_df(n=600)
    VECMStrategy(state_path=sp, price_fn=lambda: df, contract_resolver=_resolver).generate_intents()
    import json
    st = json.loads(open(sp).read())
    assert st["n_obs"] > 0 and st["last_date"] is not None
    # second run over the same data resumes (no new bars) without crashing
    intents = VECMStrategy(state_path=sp, price_fn=lambda: df,
                           contract_resolver=_resolver).generate_intents()
    assert len(intents) == 3


def test_cold_start_matches_warm_start_signal():
    """A cold-start run's z equals a direct warm_start of the core on the same data."""
    from vecm.kalman_vecm import KalmanVECM
    df = _prices_df(n=500)
    logy = np.log(df["WTI"].values); logx = np.log(df[["Brent", "RBOB"]].values)
    ref = KalmanVECM(n_hedges=2, hedge_names=["Brent", "RBOB"]).warm_start(logy, logx)
    strat = VECMStrategy(state_path=_tmp(), price_fn=lambda: df, contract_resolver=_resolver)
    intents = strat.generate_intents()
    assert np.isclose(intents[0]["metadata"]["z"], ref.z, atol=1e-10)


def test_risk_limits_clamp_contracts():
    strat = VECMStrategy(allocation=50_000_000, state_path=_tmp(),
                         price_fn=lambda: _prices_df(), contract_resolver=_resolver,
                         risk_limits=RiskLimits(max_contracts_per_leg=2, max_leg_notional=1e12,
                                                max_gross_notional=1e12, max_abs_fractional=1e9))
    for i in strat.generate_intents():
        assert abs(i["target_quantity"]) <= 2
