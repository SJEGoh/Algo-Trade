"""Tests for the ATR limit-at-pullback execution layer."""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import pytest
from execution.atr_execution import AtrPullbackLayer


def _cfg(**overrides):
    base = {"enabled": True, "atr_period": 14, "atr_fraction": 0.5,
            "cache_ttl_sec": 300, "strategies": [], "skip_exits": True}
    base.update(overrides)
    return base


def _intent(symbol="AAPL", price=150.0, order_type="market", target_quantity=10,
            strategy_id="test_suite", intent_type="target_position", **extra):
    d = {
        "strategy_id": strategy_id,
        "client_order_id": f"test-{symbol}-1",
        "timestamp": "2024-01-01T00:00:00Z",
        "schema_version": "1.0",
        "instrument": {"symbol": symbol, "asset_class": "equity", "exchange": "SMART"},
        "intent_type": intent_type,
        "target_quantity": target_quantity,
        "order_type": order_type,
        "expected_price": price,
        "time_in_force": "day",
        "metadata": {},
    }
    d.update(extra)
    return d


class TestAtrPullbackLayer:
    def test_disabled_passthrough(self):
        layer = AtrPullbackLayer(_cfg(enabled=False))
        intent = _intent()
        assert layer.transform(intent) is intent

    def test_limit_order_passthrough(self):
        layer = AtrPullbackLayer(_cfg())
        intent = _intent(order_type="limit")
        assert layer.transform(intent) is intent

    def test_exit_passthrough(self):
        """Exit intents (target_qty=0) should never be transformed."""
        layer = AtrPullbackLayer(_cfg())
        # mock ATR so we know it would transform if not an exit
        layer._cache["AAPL"] = (5.0, 1e18)
        intent = _intent(target_quantity=0)
        result = layer.transform(intent)
        assert result["order_type"] == "market"

    def test_strategy_filter(self):
        layer = AtrPullbackLayer(_cfg(strategies=["only_this"]))
        layer._cache["AAPL"] = (5.0, 1e18)
        intent = _intent(strategy_id="test_suite")
        result = layer.transform(intent)
        assert result["order_type"] == "market"  # not in allow-list

    def test_transform_buy(self):
        layer = AtrPullbackLayer(_cfg(atr_fraction=0.5))
        layer._cache["AAPL"] = (10.0, 1e18)  # ATR=10
        intent = _intent(price=150.0, target_quantity=10)
        result = layer.transform(intent)
        assert result["order_type"] == "limit"
        assert result["limit_price"] == 145.0  # 150 - 0.5*10
        assert result["metadata"]["atr_execution"]["atr"] == 10.0
        assert result["metadata"]["atr_execution"]["original_order_type"] == "market"

    def test_transform_sell(self):
        layer = AtrPullbackLayer(_cfg(atr_fraction=0.5))
        layer._cache["AAPL"] = (10.0, 1e18)
        intent = _intent(price=150.0, target_quantity=-10)  # selling
        result = layer.transform(intent)
        assert result["order_type"] == "limit"
        assert result["limit_price"] == 155.0  # 150 + 0.5*10

    def test_atr_zero_fallback(self):
        layer = AtrPullbackLayer(_cfg())
        layer._cache["AAPL"] = (0.0, 1e18)
        intent = _intent()
        result = layer.transform(intent)
        assert result["order_type"] == "market"

    def test_atr_cache(self):
        layer = AtrPullbackLayer(_cfg(cache_ttl_sec=300))
        import time
        layer._cache["AAPL"] = (5.0, time.time())
        assert layer._get_atr("AAPL") == 5.0

    def test_record_and_clear(self):
        layer = AtrPullbackLayer(_cfg())
        layer.record_order(100)
        layer.record_order(101)
        assert layer.pending_order_ids() == [100, 101]
        layer.clear_tracked()
        assert layer.pending_order_ids() == []

    def test_no_price_passthrough(self):
        layer = AtrPullbackLayer(_cfg())
        layer._cache["AAPL"] = (5.0, 1e18)
        intent = _intent(price=None)
        # expected_price=None should pass through
        result = layer.transform(intent)
        assert result["order_type"] == "market"

    def test_negative_limit_fallback(self):
        """If ATR is huge relative to price, limit would go negative — fall back to market."""
        layer = AtrPullbackLayer(_cfg(atr_fraction=2.0))
        layer._cache["AAPL"] = (100.0, 1e18)
        intent = _intent(price=50.0, target_quantity=10)
        result = layer.transform(intent)
        # 50 - 2.0*100 = -150, should fall back to market
        assert result["order_type"] == "market"
