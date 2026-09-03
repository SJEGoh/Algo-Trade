"""execution/atr_execution.py — pluggable execution-layer framework.

Base class ``ExecutionLayer`` provides the common wiring (enable/disable, strategy
filter, exit passthrough, order tracking for EOD cancel, ATR cache).  Concrete
subclasses override ``compute_limit_price()`` to implement different entry tactics:

  * AtrPullbackLayer  — limit at  price - fraction * ATR  (buy cheaper on pullback)
  * (future) VwapLayer, TwapLayer, IcebergLayer, …

Usage:
  1. Set the desired layer in config.py → ATR_EXECUTION  (or a future config block).
  2. CentralExecutor calls  layer.transform(intent)  inside process_intent.
  3. day_scheduler fires POST /atr/cancel at EOD to sweep unfilled limit orders.

Only market-order intents are transformed; limit/stop orders pass through unchanged.
Exit intents (target_quantity == 0) are NEVER transformed — exits always fire at
market to guarantee the position is closed.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from execution.central_execution import CentralExecutor

logger = logging.getLogger("executor.atr")


# ======================================================================
# Base class — common infrastructure for any execution layer
# ======================================================================

class ExecutionLayer:
    """Abstract base for execution layers that sit between strategy intents
    and order placement.

    Subclasses must override ``compute_limit_price(symbol, price, is_buy)``
    to return a limit price (or None to fall back to market).

    Config keys consumed here (subclasses may add their own):
      enabled        – global toggle (default False)
      cache_ttl_sec  – how long to cache per-symbol data (default 300)
      strategies     – list of strategy_ids to apply; [] = all
      skip_exits     – never transform exit intents (default True)
    """

    # subclasses set this to tag metadata (e.g. "atr_execution", "vwap_execution")
    METADATA_KEY: str = "execution_layer"

    def __init__(self, cfg: dict, executor: "CentralExecutor" = None):
        self.enabled: bool = cfg.get("enabled", False)
        self.cache_ttl: float = float(cfg.get("cache_ttl_sec", 300))
        self.strategies: list = list(cfg.get("strategies", []))
        self.skip_exits: bool = cfg.get("skip_exits", True)
        self._executor = executor

        # symbol -> (cached_value, fetch_time)  — subclasses use via _get_cached / _set_cached
        self._cache: Dict[str, tuple] = {}
        # track order_ids placed by this layer so we can cancel them at EOD
        self._tracked_order_ids: list = []

    # ------------------------------------------------------------------
    # Public API (common to all layers)
    # ------------------------------------------------------------------

    def transform(self, intent: dict) -> dict:
        """Possibly convert a market intent into a limit intent.

        Returns a (possibly modified) intent dict.  Non-transforming cases
        (returns intent unchanged): layer disabled, order_type != "market",
        strategy not in allow-list, exit intent, compute_limit_price returns None.
        """
        if not self.enabled:
            return intent

        if intent.get("order_type") != "market":
            return intent

        sid = intent.get("strategy_id", "")
        if self.strategies and sid not in self.strategies:
            return intent

        # never delay an exit — close at market
        if self.skip_exits and intent.get("intent_type") == "target_position":
            tgt = intent.get("target_quantity")
            if tgt is not None and tgt == 0:
                return intent

        symbol = intent.get("instrument", {}).get("symbol")
        if not symbol:
            return intent

        price = intent.get("expected_price")
        if not price or price <= 0:
            return intent

        is_buy = self._is_buy(intent)
        limit_price = self.compute_limit_price(symbol, price, is_buy)
        if limit_price is None or limit_price <= 0:
            return intent

        # transform: market -> limit
        transformed = dict(intent)
        transformed["order_type"] = "limit"
        transformed["limit_price"] = limit_price
        transformed["metadata"] = dict(intent.get("metadata", {}))
        transformed["metadata"][self.METADATA_KEY] = self._build_metadata(
            symbol, price, limit_price, is_buy)
        logger.info("%s: %s %s market@%.2f -> limit@%.2f",
                    self.__class__.__name__, "BUY" if is_buy else "SELL",
                    symbol, price, limit_price)
        return transformed

    def record_order(self, order_id: int) -> None:
        """Track an order placed via this layer (for EOD cancel sweep)."""
        self._tracked_order_ids.append(order_id)

    def pending_order_ids(self) -> list:
        """Return order_ids placed by this layer that haven't been swept yet."""
        return list(self._tracked_order_ids)

    def clear_tracked(self) -> None:
        """Reset after an EOD cancel sweep."""
        self._tracked_order_ids.clear()

    # ------------------------------------------------------------------
    # Cache helpers (for subclasses)
    # ------------------------------------------------------------------

    def _get_cached(self, symbol: str) -> Optional[float]:
        """Return a cached value for ``symbol`` if still fresh, else None."""
        cached = self._cache.get(symbol)
        if cached and (time.time() - cached[1]) < self.cache_ttl:
            return cached[0]
        return None

    def _set_cached(self, symbol: str, value: float) -> None:
        self._cache[symbol] = (value, time.time())

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def compute_limit_price(self, symbol: str, price: float, is_buy: bool) -> Optional[float]:
        """Return the limit price for this symbol/direction, or None to fall back to market.
        Subclasses MUST override this."""
        raise NotImplementedError

    def _build_metadata(self, symbol: str, price: float,
                        limit_price: float, is_buy: bool) -> dict:
        """Return a dict to attach under metadata[METADATA_KEY]. Override for richer info."""
        return {
            "original_order_type": "market",
            "expected_price": price,
            "limit_price": limit_price,
            "layer": self.__class__.__name__,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _is_buy(intent: dict) -> bool:
        if intent.get("side") == "buy":
            return True
        if intent.get("side") == "sell":
            return False
        tgt = intent.get("target_quantity")
        if tgt is not None:
            return tgt > 0
        return True


# ======================================================================
# Concrete layer: ATR limit-at-pullback (IBKR intraday bars)
# ======================================================================

class AtrPullbackLayer(ExecutionLayer):
    """Converts market-order intents into limit orders at a pullback price.

    limit_price = price - atr_fraction * ATR(atr_period)   (for buys)
    limit_price = price + atr_fraction * ATR(atr_period)   (for sells)

    ATR is computed from intraday bars fetched via IBKR reqHistoricalData.

    Config (extends ExecutionLayer):
      atr_period     – ATR lookback in bars (default 14)
      atr_fraction   – fraction of ATR to offset (default 0.5)
      bar_size       – IBKR bar size string (default "5 mins")
      duration       – IBKR duration string (default "2 D")
    """

    METADATA_KEY = "atr_execution"

    def __init__(self, cfg: dict, executor: "CentralExecutor" = None):
        super().__init__(cfg, executor=executor)
        self.atr_period: int = int(cfg.get("atr_period", 14))
        self.atr_fraction: float = float(cfg.get("atr_fraction", 0.5))
        self.bar_size: str = cfg.get("bar_size", "5 mins")
        self.duration: str = cfg.get("duration", "2 D")

    def compute_limit_price(self, symbol: str, price: float, is_buy: bool) -> Optional[float]:
        atr = self._get_atr(symbol)
        if atr is None or atr <= 0:
            logger.warning("ATR unavailable for %s — falling through to market order", symbol)
            return None
        if is_buy:
            lp = round(price - self.atr_fraction * atr, 2)
        else:
            lp = round(price + self.atr_fraction * atr, 2)
        return lp if lp > 0 else None

    def _build_metadata(self, symbol: str, price: float,
                        limit_price: float, is_buy: bool) -> dict:
        atr = self._get_cached(symbol) or 0.0
        return {
            "original_order_type": "market",
            "atr": round(atr, 4),
            "atr_fraction": self.atr_fraction,
            "bar_size": self.bar_size,
            "expected_price": price,
            "limit_price": limit_price,
        }

    # ------------------------------------------------------------------
    # ATR computation with cache — fetches from IBKR via executor
    # ------------------------------------------------------------------

    def _get_atr(self, symbol: str) -> Optional[float]:
        cached = self._get_cached(symbol)
        if cached is not None:
            return cached
        atr = self._fetch_atr(symbol)
        if atr is not None:
            self._set_cached(symbol, atr)
        return atr

    def _fetch_atr(self, symbol: str) -> Optional[float]:
        """Fetch ATR via the executor's IBKR historical data connection."""
        if self._executor is None:
            logger.warning("ATR layer has no executor reference — cannot fetch bars")
            return None
        try:
            return self._executor.fetch_atr(
                symbol,
                period=self.atr_period,
                bar_size=self.bar_size,
                duration=self.duration,
            )
        except Exception as e:
            logger.warning("ATR fetch error for %s: %s", symbol, e)
            return None
