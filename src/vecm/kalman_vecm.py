"""Stateful, incremental multivariate Kalman-filter VECM.

Models a time-varying cointegrating relationship:

    log(y_t) = alpha_t + sum_j beta_{j,t} * log(x_{j,t}) + e_t

with the coefficient vector theta_t = [alpha, beta_1, ..., beta_k] following a random
walk (state noise `delta`), observed through log(y_t) with observation noise `R`.

This is a *causal, one-observation-at-a-time* filter. Feeding the full price history
through `.step()` bar by bar reproduces, to floating-point precision, the batch
`kalman_multi_signal` in the research notebook (there is a regression test that asserts
this). The difference is that this object keeps (theta, P) as mutable state, so the live
system can persist it between days and advance a single step per new close instead of
refitting from scratch.

The trading signal is the standardized innovation z_t = e_t / sqrt(S_t): how many
standard deviations the observed log(y) is from the model's prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np


@dataclass
class KalmanState:
    """Serializable filter state. `theta` is [alpha, beta_1, ..., beta_k]."""

    theta: np.ndarray
    P: np.ndarray
    n_obs: int = 0                       # number of bars processed
    last_date: Optional[str] = None      # ISO date of the most recent bar processed

    def to_dict(self) -> dict:
        return {
            "theta": np.asarray(self.theta, dtype=float).tolist(),
            "P": np.asarray(self.P, dtype=float).tolist(),
            "n_obs": int(self.n_obs),
            "last_date": self.last_date,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KalmanState":
        return cls(
            theta=np.asarray(d["theta"], dtype=float),
            P=np.asarray(d["P"], dtype=float),
            n_obs=int(d.get("n_obs", 0)),
            last_date=d.get("last_date"),
        )


@dataclass
class StepResult:
    """Everything produced by advancing the filter one bar."""

    z: float                 # standardized innovation (the raw signal)
    z_clipped: float         # clip(z, -zcap, zcap) / zcap, in [-1, 1]
    innovation: float        # e_t = log(y) - prediction
    S: float                 # innovation variance
    betas: np.ndarray        # current hedge betas [beta_1, ..., beta_k]
    alpha: float             # current intercept
    positions: dict          # normalized target exposure per leg (pre vol-scaling)


class KalmanVECM:
    """Online multivariate Kalman VECM.

    Parameters
    ----------
    n_hedges : int
        Number of hedge legs (k). For WTI vs Brent+RBOB this is 2.
    delta : float
        State-transition (process) noise; Q = delta * I. Selected annually.
    R : float
        Observation noise. Selected annually.
    zcap : float
        Symmetric clip applied to the standardized innovation before normalizing.
    hedge_names : sequence of str, optional
        Names for the hedge legs (defaults to x0, x1, ...). Used only for the
        `positions` dict keys. The dependent leg is always keyed `y`.
    state : KalmanState, optional
        Resume from a persisted state. If omitted, a cold-start state is created
        (theta = 0, P = I) identical to the notebook's initialization.
    """

    Y_KEY = "y"

    def __init__(
        self,
        n_hedges: int,
        delta: float = 1e-4,
        R: float = 1e-3,
        zcap: float = 2.0,
        hedge_names: Optional[Sequence[str]] = None,
        state: Optional[KalmanState] = None,
    ):
        if n_hedges < 1:
            raise ValueError("n_hedges must be >= 1")
        self.k = int(n_hedges)
        self.d = self.k + 1  # dimension of theta (intercept + k betas)
        self.delta = float(delta)
        self.R = float(R)
        self.zcap = float(zcap)
        if hedge_names is None:
            hedge_names = [f"x{j}" for j in range(self.k)]
        if len(hedge_names) != self.k:
            raise ValueError("len(hedge_names) must equal n_hedges")
        self.hedge_names = list(hedge_names)

        if state is None:
            state = KalmanState(theta=np.zeros(self.d), P=np.eye(self.d) * 1.0)
        if state.theta.shape != (self.d,) or state.P.shape != (self.d, self.d):
            raise ValueError("state dimensions do not match n_hedges")
        self.state = state

    # -- core recursion -----------------------------------------------------
    def step(self, log_y: float, log_x: Sequence[float], date: Optional[str] = None) -> StepResult:
        """Advance the filter by one observation and return the trading signal.

        `log_x` is the vector [log(x_1), ..., log(x_k)] for this bar. Mutates state.
        """
        log_x = np.asarray(log_x, dtype=float)
        if log_x.shape != (self.k,):
            raise ValueError(f"log_x must have shape ({self.k},)")

        theta = self.state.theta
        P = self.state.P
        Q = np.eye(self.d) * self.delta

        # predict
        P_pred = P + Q
        xt = np.concatenate([[1.0], log_x])          # design row [1, log_x...]
        y_pred = xt @ theta
        e = log_y - y_pred                           # innovation
        S = float(xt @ P_pred @ xt + self.R)         # innovation variance

        # update
        K = P_pred @ xt / S                          # Kalman gain
        theta = theta + K * e
        P = P_pred - np.outer(K, xt) @ P_pred

        # commit state
        self.state.theta = theta
        self.state.P = P
        self.state.n_obs += 1
        if date is not None:
            self.state.last_date = date

        # signal
        z = e / np.sqrt(S) if S > 0 else 0.0
        z_clipped = float(np.clip(z, -self.zcap, self.zcap) / self.zcap)

        betas = theta[1:].copy()
        positions = {self.Y_KEY: -z_clipped}         # short y when it's rich vs the model
        for j, nm in enumerate(self.hedge_names):
            positions[nm] = z_clipped * betas[j]

        return StepResult(
            z=float(z),
            z_clipped=z_clipped,
            innovation=float(e),
            S=S,
            betas=betas,
            alpha=float(theta[0]),
            positions=positions,
        )

    # -- non-mutating evaluation (intraday) ---------------------------------
    def evaluate(self, log_y: float, log_x: Sequence[float]) -> StepResult:
        """Compute the trading signal for a (log_y, log_x) observation using the CURRENT
        state, WITHOUT advancing it. This is the intraday path: the cointegrating vector
        (theta, P) is frozen at the last daily close, and we only re-price the mispricing
        z against live intraday prices. Calling this many times per day does not move the
        filter -- only `step()` (run once per finished daily bar) does.

        Uses the same predict step as `step()` (P_pred = P + Q, innovation variance
        S = xt' P_pred xt + R) so the standardized z is on the identical scale as the
        daily signal; it simply omits the state update.
        """
        log_x = np.asarray(log_x, dtype=float)
        if log_x.shape != (self.k,):
            raise ValueError(f"log_x must have shape ({self.k},)")
        theta = self.state.theta
        P_pred = self.state.P + np.eye(self.d) * self.delta
        xt = np.concatenate([[1.0], log_x])
        e = float(log_y - xt @ theta)
        S = float(xt @ P_pred @ xt + self.R)
        z = e / np.sqrt(S) if S > 0 else 0.0
        z_clipped = float(np.clip(z, -self.zcap, self.zcap) / self.zcap)
        betas = theta[1:].copy()
        positions = {self.Y_KEY: -z_clipped}
        for j, nm in enumerate(self.hedge_names):
            positions[nm] = z_clipped * betas[j]
        return StepResult(z=float(z), z_clipped=z_clipped, innovation=e, S=S,
                          betas=betas, alpha=float(theta[0]), positions=positions)

    # -- convenience --------------------------------------------------------
    def warm_start(self, log_y: Sequence[float], log_x: np.ndarray, dates: Optional[Sequence] = None) -> StepResult:
        """Replay a full history bar by bar to build up (theta, P).

        `log_x` has shape (n, k). Returns the final StepResult (i.e. the signal for
        the most recent bar). This exactly reproduces the notebook's batch filter when
        called on a cold-start instance.
        """
        log_y = np.asarray(log_y, dtype=float)
        log_x = np.asarray(log_x, dtype=float)
        n = len(log_y)
        if log_x.shape != (n, self.k):
            raise ValueError("log_x must have shape (len(log_y), n_hedges)")
        last: Optional[StepResult] = None
        # Mirror the notebook: the first bar (t=0) only seeds; updates start at t=1.
        # We achieve identical numbers by processing bars 1..n-1 (bar 0 leaves theta=0,
        # P=I, position 0), which is what the batch loop `for t in range(1, n)` does.
        for t in range(1, n):
            d = None
            if dates is not None:
                d = str(dates[t])
            last = self.step(log_y[t], log_x[t], date=d)
        if last is None:  # only one bar supplied
            last = StepResult(
                z=0.0, z_clipped=0.0, innovation=0.0, S=float("nan"),
                betas=self.state.theta[1:].copy(), alpha=float(self.state.theta[0]),
                positions={self.Y_KEY: 0.0, **{nm: 0.0 for nm in self.hedge_names}},
            )
            if dates is not None:
                self.state.last_date = str(dates[0])
            self.state.n_obs = max(self.state.n_obs, 1)
        return last

    def snapshot(self) -> KalmanState:
        """Deep copy of current state for persistence."""
        return KalmanState(
            theta=self.state.theta.copy(),
            P=self.state.P.copy(),
            n_obs=self.state.n_obs,
            last_date=self.state.last_date,
        )
