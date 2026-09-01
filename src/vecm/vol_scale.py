"""Rolling realized-volatility targeting, matching the notebook's `vol_scale`.

Each leg's normalized exposure is multiplied by (target_vol / trailing_realized_vol),
capped at `cap` to avoid extreme leverage in unusually quiet periods. The notebook uses
a 20-day window, 15% annualized target, and a 5x cap.

For the live bot we only ever need the *latest* multiplier, but the class keeps a small
trailing buffer of returns so it can compute the same rolling std the backtest used.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np


class RollingVolScaler:
    def __init__(self, window: int = 20, target: float = 0.15, periods: int = 252, cap: float = 5.0):
        self.window = int(window)
        self.target = float(target)
        self.periods = int(periods)
        self.cap = float(cap)
        self._buf: Deque[float] = deque(maxlen=self.window)

    def update(self, ret: float) -> Optional[float]:
        """Push a new simple return; return the current multiplier (or None until the
        window is full, matching pandas `rolling(window).std()` which needs `window`
        observations)."""
        self._buf.append(float(ret))
        return self.multiplier()

    def multiplier(self) -> Optional[float]:
        if len(self._buf) < self.window:
            return None
        # pandas rolling std uses ddof=1
        realized = float(np.std(self._buf, ddof=1)) * np.sqrt(self.periods)
        if realized <= 0:
            return self.cap
        return float(min(self.target / realized, self.cap))

    @staticmethod
    def latest_from_series(returns, window: int = 20, target: float = 0.15,
                           periods: int = 252, cap: float = 5.0) -> Optional[float]:
        """One-shot: compute the latest multiplier from a full return series
        (pandas Series or 1-D array), identical to `vol_scale(returns).iloc[-1]`."""
        arr = np.asarray(returns, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < window:
            return None
        realized = float(np.std(arr[-window:], ddof=1)) * np.sqrt(periods)
        if realized <= 0:
            return cap
        return float(min(target / realized, cap))
