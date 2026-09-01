"""
tests/test_vecm.py — full unit tests for the ported Kalman VECM strategy core
(src/vecm/). Includes the notebook-regression test: the incremental filter must
reproduce the research batch filter to floating-point precision, or the live
signal has diverged from the validated backtest.

Run:  PYTHONPATH=src pytest tests/test_vecm.py -v
"""
import numpy as np
import pandas as pd

from vecm.kalman_vecm import KalmanVECM, KalmanState
from vecm.vol_scale import RollingVolScaler
from vecm.sizing import target_contracts
from vecm.risk_limits import RiskLimits, apply_limits, validate_inputs


# ------------------------------------------------------------------
# verbatim copy of the notebook's batch filter (research Section 4)
# ------------------------------------------------------------------
def kalman_multi_signal(y_px, x_pxs, delta=1e-4, R=1e-3, zcap=2.0):
    names = list(x_pxs.keys())
    k = len(names)
    logy = np.log(y_px).values
    logx = np.column_stack([np.log(x_pxs[nm]).values for nm in names])
    n = len(logy)
    d = k + 1
    theta = np.zeros(d)
    P = np.eye(d) * 1.0
    Q = np.eye(d) * delta
    pos = {nm: np.zeros(n) for nm in ['__y__'] + names}
    for t in range(1, n):
        P_pred = P + Q
        xt = np.concatenate([[1.0], logx[t]])
        e = logy[t] - xt @ theta
        S = xt @ P_pred @ xt + R
        K = P_pred @ xt / S
        theta = theta + K * e
        P = P_pred - np.outer(K, xt) @ P_pred
        z = e / np.sqrt(S) if S > 0 else 0.0
        z_clipped = np.clip(z, -zcap, zcap) / zcap
        pos['__y__'][t] = -z_clipped
        for j, nm in enumerate(names):
            pos[nm][t] = z_clipped * theta[1 + j]
    idx = y_px.index
    return {kk: pd.Series(v, index=idx) for kk, v in pos.items()}


def _synthetic_prices(n=800, seed=7):
    rng = np.random.default_rng(seed)
    brent = 60 + np.cumsum(rng.normal(0, 0.5, n))
    rbob = 1.8 + np.cumsum(rng.normal(0, 0.02, n))
    wti = 0.9 * brent + 3.0 * rbob + rng.normal(0, 0.4, n)
    idx = pd.bdate_range("2016-01-01", periods=n)
    return (pd.Series(np.abs(wti) + 5, idx),
            pd.Series(np.abs(brent) + 5, idx),
            pd.Series(np.abs(rbob) + 1, idx))


# ------------------------------------------------------------------
# Kalman VECM core
# ------------------------------------------------------------------
def test_warm_start_matches_batch():
    """The incremental filter reproduces the batch filter to 1e-12."""
    wti, brent, rbob = _synthetic_prices()
    d, R, zcap = 1e-4, 1e-3, 2.0
    ref = kalman_multi_signal(wti, {"Brent": brent, "RBOB": rbob}, delta=d, R=R, zcap=zcap)

    filt = KalmanVECM(n_hedges=2, delta=d, R=R, zcap=zcap, hedge_names=["Brent", "RBOB"])
    logy = np.log(wti.values)
    logx = np.log(np.column_stack([brent.values, rbob.values]))
    y_pos = np.zeros(len(wti)); b_pos = np.zeros(len(wti)); r_pos = np.zeros(len(wti))
    for t in range(1, len(wti)):
        res = filt.step(logy[t], logx[t])
        y_pos[t] = res.positions["y"]; b_pos[t] = res.positions["Brent"]; r_pos[t] = res.positions["RBOB"]

    assert np.allclose(y_pos, ref["__y__"].values, atol=1e-12)
    assert np.allclose(b_pos, ref["Brent"].values, atol=1e-12)
    assert np.allclose(r_pos, ref["RBOB"].values, atol=1e-12)


def test_warm_start_method_equals_manual_loop():
    """The warm_start() convenience equals a manual bar-by-bar step loop."""
    wti, brent, rbob = _synthetic_prices(n=400)
    logy = np.log(wti.values); logx = np.log(np.column_stack([brent.values, rbob.values]))

    manual = KalmanVECM(n_hedges=2, hedge_names=["Brent", "RBOB"])
    last_manual = None
    for t in range(1, len(wti)):
        last_manual = manual.step(logy[t], logx[t])

    ws = KalmanVECM(n_hedges=2, hedge_names=["Brent", "RBOB"])
    last_ws = ws.warm_start(logy, logx)
    assert np.allclose(ws.state.theta, manual.state.theta, atol=1e-12)
    assert np.isclose(last_ws.z, last_manual.z, atol=1e-12)


def test_incremental_equals_full_replay():
    """Resume from a snapshot + one step == replaying the whole history."""
    wti, brent, rbob = _synthetic_prices(n=600)
    logy = np.log(wti.values); logx = np.log(np.column_stack([brent.values, rbob.values]))

    full = KalmanVECM(n_hedges=2, hedge_names=["Brent", "RBOB"])
    last_full = None
    for t in range(1, len(wti)):
        last_full = full.step(logy[t], logx[t])

    part = KalmanVECM(n_hedges=2, hedge_names=["Brent", "RBOB"])
    for t in range(1, len(wti) - 1):
        part.step(logy[t], logx[t])
    resumed = KalmanVECM(n_hedges=2, hedge_names=["Brent", "RBOB"], state=part.snapshot())
    last_inc = resumed.step(logy[-1], logx[-1])

    assert np.allclose(resumed.state.theta, full.state.theta, atol=1e-12)
    assert np.isclose(last_inc.z, last_full.z, atol=1e-12)
    assert np.allclose(last_inc.betas, last_full.betas, atol=1e-12)


def test_evaluate_does_not_mutate_state():
    """Intraday evaluate() computes a signal without advancing the filter."""
    wti, brent, rbob = _synthetic_prices(n=300)
    logy = np.log(wti.values); logx = np.log(np.column_stack([brent.values, rbob.values]))
    filt = KalmanVECM(n_hedges=2, hedge_names=["Brent", "RBOB"])
    filt.warm_start(logy, logx)
    theta_before = filt.state.theta.copy(); n_before = filt.state.n_obs
    r = filt.evaluate(logy[-1], logx[-1])
    assert np.allclose(filt.state.theta, theta_before)      # unchanged
    assert filt.state.n_obs == n_before
    assert np.isfinite(r.z)


def test_state_dict_roundtrip():
    wti, brent, rbob = _synthetic_prices(n=200)
    logy = np.log(wti.values); logx = np.log(np.column_stack([brent.values, rbob.values]))
    filt = KalmanVECM(n_hedges=2, hedge_names=["Brent", "RBOB"])
    filt.warm_start(logy, logx)
    snap = filt.snapshot()
    restored = KalmanState.from_dict(snap.to_dict())
    assert np.allclose(restored.theta, snap.theta)
    assert np.allclose(restored.P, snap.P)
    assert restored.n_obs == snap.n_obs


def test_bad_dimensions_raise():
    import pytest
    with pytest.raises(ValueError):
        KalmanVECM(n_hedges=0)
    filt = KalmanVECM(n_hedges=2, hedge_names=["a", "b"])
    with pytest.raises(ValueError):
        filt.step(1.0, [1.0])            # wrong hedge vector length


# ------------------------------------------------------------------
# Vol scaler
# ------------------------------------------------------------------
def test_vol_scaler_matches_pandas():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0, 0.01, 300))
    ref = (0.15 / (rets.rolling(20).std() * np.sqrt(252))).clip(upper=5).iloc[-1]
    got = RollingVolScaler.latest_from_series(rets, window=20, target=0.15, cap=5.0)
    assert np.isclose(ref, got, atol=1e-12)


def test_vol_scaler_none_until_window_full():
    vs = RollingVolScaler(window=20)
    for i in range(19):
        assert vs.update(0.01) is None
    assert vs.update(0.01) is not None


# ------------------------------------------------------------------
# Sizing + hard risk limits
# ------------------------------------------------------------------
PRICES = {"WTI": 75.0, "Brent": 80.0, "RBOB": 2.3}
MULTS = {"WTI": 1000, "Brent": 1000, "RBOB": 42000}


def test_sizing_rounds_and_signs():
    eff = {"WTI": -0.8, "Brent": 0.5, "RBOB": 0.5}
    s = target_contracts(eff, PRICES, MULTS, 2_000_000)
    assert s["WTI"].contracts == -21          # 2e6*-0.8/(75*1000) = -21.33
    assert s["WTI"].fractional < 0
    assert abs(s["RBOB"].contracts) <= 11      # huge multiplier -> small count


def test_per_leg_contract_cap():
    eff = {"WTI": -5.0, "Brent": 5.0, "RBOB": 5.0}
    s = target_contracts(eff, PRICES, MULTS, 5_000_000)
    dec = apply_limits(s, RiskLimits(max_contracts_per_leg=3, max_leg_notional=1e12,
                                     max_gross_notional=1e12, max_abs_fractional=1e9))
    assert all(abs(v) <= 3 for v in dec.approved_contracts.values())
    assert dec.breaches


def test_gross_notional_cap_scales_down():
    eff = {"WTI": -1.0, "Brent": 1.0, "RBOB": 1.0}
    s = target_contracts(eff, PRICES, MULTS, 3_000_000)
    dec = apply_limits(s, RiskLimits(max_contracts_per_leg=1000, max_leg_notional=1e12,
                                     max_gross_notional=100_000, max_abs_fractional=1e9))
    gross = sum(abs(dec.approved_contracts[l]) * PRICES[l] * MULTS[l] for l in dec.approved_contracts)
    assert gross <= 100_000 * 1.05
    assert any("gross notional" in b for b in dec.breaches)


def test_abort_on_absurd_fractional():
    eff = {"WTI": -1000.0, "Brent": 0.0, "RBOB": 0.0}
    s = target_contracts(eff, PRICES, MULTS, 1_000_000_000)
    dec = validate_inputs(s, RiskLimits(max_abs_fractional=50))
    assert dec.aborted


def test_sizing_rejects_bad_price():
    import pytest
    with pytest.raises(ValueError):
        target_contracts({"WTI": 1.0}, {"WTI": 0.0}, {"WTI": 1000}, 1_000_000)
