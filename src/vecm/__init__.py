"""Kalman VECM strategy core (WTI vs Brent+RBOB), ported from the tested
NUSSIF trading_comp ibkr_kalman_vecm reference. Pure — no IBKR / algo_trade deps."""
from vecm.kalman_vecm import KalmanVECM, KalmanState, StepResult
from vecm.vol_scale import RollingVolScaler
from vecm.sizing import target_contracts, LegSizing
from vecm.risk_limits import RiskLimits, RiskDecision, apply_limits, validate_inputs
