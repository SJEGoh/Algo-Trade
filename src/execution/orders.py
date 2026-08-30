"""
test_orders.py

Standalone test intent definitions for the CentralExecutor test suite.
Import with:

    from test_orders import test_orders, get_test_order

Usage:
    result = app.process_intent(get_test_order("test-001-aapl-buy-mkt"))

Schema notes:
- "delta" intents use side + quantity (quantity always positive, side carries direction).
- "target_position" intents use a single signed target_quantity field instead of
  side + quantity — a target of 0 (flatten) is a fully valid, unambiguous value,
  and the resolved buy/sell direction is derived later from the computed delta,
  not from the intent itself.

NOTE: limit_price fields marked None are deliberately left for you to fill in
at test-run time based on the current quote for that symbol — hardcoding a
price here would go stale. A couple of entries use a placeholder low price
(e.g. 5) as a stand-in for "well below market" — replace with a real
below-market value for the symbol/session you're testing against.
"""

from typing import Optional


test_orders = [

    # --- Tier 1: Happy path (delta intents) ---
    {
        "strategy_id": "test_suite",
        "client_order_id": "test-001-aapl-buy-mkt",
        "timestamp": "2026-08-27T09:30:00Z",
        "schema_version": "1.0",
        "instrument": {"symbol": "AAPL", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "delta",
        "side": "buy",
        "quantity": 10,
        "order_type": "market",
        "time_in_force": "day",
        "metadata": {"test_case": "1 - market buy happy path"}
    },
    {
        "strategy_id": "test_suite",
        "client_order_id": "test-002-aapl-sell-mkt",
        "timestamp": "2026-08-27T09:31:00Z",
        "schema_version": "1.0",
        "instrument": {"symbol": "AAPL", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "delta",
        "side": "sell",
        "quantity": 10,
        "order_type": "market",
        "time_in_force": "day",
        "metadata": {"test_case": "2 - market sell, close position"}
    },
    {
        "strategy_id": "test_suite",
        "client_order_id": "test-003-nvda-buy-lmt",
        "timestamp": "2026-08-27T09:32:00Z",
        "schema_version": "1.0",
        "instrument": {"symbol": "NVDA", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "delta",
        "side": "buy",
        "quantity": 5,
        "order_type": "limit",
        "limit_price": None,  # fill in: ~1% below last traded price at test time
        "time_in_force": "day",
        "metadata": {"test_case": "3 - limit order below market, should not fill"}
    },
    {
        "strategy_id": "test_suite",
        "action": "cancel",
        "client_order_id": "test-003-nvda-buy-lmt",
        "timestamp": "2026-08-27T09:33:00Z",
        "schema_version": "1.0",
        "metadata": {"test_case": "4 - cancel order from test 3"}
    },

    # --- Tier 2: Idempotency & concurrency ---
    {
        "strategy_id": "test_suite",
        "client_order_id": "test-005-dedup",
        "timestamp": "2026-08-27T09:34:00Z",
        "schema_version": "1.0",
        "instrument": {"symbol": "MSFT", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "delta",
        "side": "buy",
        "quantity": 5,
        "order_type": "market",
        "time_in_force": "day",
        "metadata": {"test_case": "5 - submit this exact payload twice, second should be rejected/no-op"}
    },
    {
        "strategy_id": "test_suite",
        "client_order_id": "test-006-target-repeat",
        "timestamp": "2026-08-27T09:35:00Z",
        "schema_version": "1.0",
        "instrument": {"symbol": "MSFT", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "target_position",
        "target_quantity": 100,
        "order_type": "market",
        "time_in_force": "day",
        "metadata": {"test_case": "6 - submit twice (same client_order_id), second call should hit dedup"}
    },
    {
        "strategy_id": "test_suite",
        "client_order_id": "test-006b-target-repeat-new-id",
        "timestamp": "2026-08-27T09:35:10Z",
        "schema_version": "1.0",
        "instrument": {"symbol": "MSFT", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "target_position",
        "target_quantity": 100,
        "order_type": "market",
        "time_in_force": "day",
        "metadata": {"test_case": "6b - new client_order_id, same target as 6 - should resolve to zero delta and be rejected as no-op"}
    },
    {
        "strategy_id": "test_suite",
        "client_order_id": "test-007a-target-open",
        "timestamp": "2026-08-27T09:36:00Z",
        "schema_version": "1.0",
        "instrument": {"symbol": "TSLA", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "target_position",
        "target_quantity": 100,
        "order_type": "limit",
        "limit_price": 5,  # placeholder — set well below current market so it stays open
        "time_in_force": "day",
        "metadata": {"test_case": "7a - target +100, expect this to still be open when 7b arrives"}
    },
    {
        "strategy_id": "test_suite",
        "client_order_id": "test-007b-target-flatten",
        "timestamp": "2026-08-27T09:36:05Z",
        "schema_version": "1.0",
        "instrument": {"symbol": "TSLA", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "target_position",
        "target_quantity": 0,
        "order_type": "market",
        "time_in_force": "day",
        "metadata": {"test_case": "7b - flatten (target 0) while 7a still open, tests cancel-and-replace"}
    },

    # --- Tier 3: Risk layer ---
    {
        "strategy_id": "test_suite_small_alloc",
        "client_order_id": "test-009-exceeds-allocation",
        "timestamp": "2026-08-27T09:38:00Z",
        "schema_version": "1.0",
        "instrument": {"symbol": "NVDA", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "delta",
        "side": "buy",
        "quantity": 10000,  # deliberately far exceeds any reasonable test allocation
        "order_type": "market",
        "expected_price": 180.0,
        "time_in_force": "day",
        "risk": {"strategy_capital_allocation": 1000, "expected_notional": None, "max_slippage_bps": 15},
        "metadata": {"test_case": "9 - order size exceeds strategy's capital allocation, expect pre-trade rejection"}
    },
    {
        "strategy_id": "test_suite_drawdown_breached",
        "client_order_id": "test-010-post-drawdown-breach",
        "timestamp": "2026-08-27T09:39:00Z",
        "schema_version": "1.0",
        "instrument": {"symbol": "AAPL", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "delta",
        "side": "buy",
        "quantity": 5,
        "order_type": "market",
        "time_in_force": "day",
        "metadata": {"test_case": "10 - submit after manually setting this strategy's P&L below drawdown limit, expect hard block"}
    },
    {
        "strategy_id": "test_suite",
        "action": "kill_switch",
        "timestamp": "2026-08-27T09:40:00Z",
        "schema_version": "1.0",
        "metadata": {"test_case": "11 - trigger global kill switch while an order is open, confirm cancel/flatten"}
    },

    # --- Tier 4: Failure & edge cases ---
    {
        "strategy_id": "test_suite",
        "client_order_id": "test-012-invalid-symbol",
        "timestamp": "2026-08-27T09:41:00Z",
        "schema_version": "1.0",
        "instrument": {"symbol": "ZZZZZINVALID", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "delta",
        "side": "buy",
        "quantity": 1,
        "order_type": "market",
        "time_in_force": "day",
        "metadata": {"test_case": "12 - invalid symbol, expect error() callback caught and surfaced"}
    },
    {
        "strategy_id": "test_suite",
        "client_order_id": "test-013-zero-qty",
        "timestamp": "2026-08-27T09:42:00Z",
        "schema_version": "1.0",
        "instrument": {"symbol": "AAPL", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "delta",
        "side": "buy",
        "quantity": 0,
        "order_type": "market",
        "time_in_force": "day",
        "metadata": {"test_case": "13 - zero quantity, expect schema validation rejection before placeOrder"}
    },
    {
        "strategy_id": "test_suite",
        "client_order_id": "test-014-far-limit",
        "timestamp": "2026-08-27T09:43:00Z",
        "schema_version": "1.0",
        "instrument": {"symbol": "AAPL", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "delta",
        "side": "buy",
        "quantity": 1,
        "order_type": "limit",
        "limit_price": 0.01,
        "time_in_force": "day",
        "metadata": {"test_case": "14 - absurd limit price, should sit unfilled, tests staleness/timeout handling"}
    },
    {
        "strategy_id": "test_suite",
        "client_order_id": "test-015-crash-recovery",
        "timestamp": "2026-08-27T09:44:00Z",
        "schema_version": "1.0",
        "instrument": {"symbol": "AAPL", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "delta",
        "side": "buy",
        "quantity": 5,
        "order_type": "limit",
        "limit_price": 5,  # placeholder — set well below current market so it's still open when you kill the process
        "time_in_force": "day",
        "metadata": {"test_case": "15 - submit, then kill process before fill, restart, confirm reconciliation picks it up"}
    },
    {
        "strategy_id": "test_suite",
        "client_order_id": "test-016-disconnect",
        "timestamp": "2026-08-27T09:45:00Z",
        "schema_version": "1.0",
        "instrument": {"symbol": "AAPL", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "delta",
        "side": "buy",
        "quantity": 5,
        "order_type": "limit",
        "limit_price": None,  # fill in: well below market at test time
        "time_in_force": "day",
        "metadata": {"test_case": "16 - kill IB Gateway / network mid-order, confirm reconnect and no false-fill assumption"}
    },
    {
        "strategy_id": "test_suite",
        "client_order_id": "test-017-outside-hours",
        "timestamp": "2026-08-27T02:00:00Z",  # outside RTH
        "schema_version": "1.0",
        "instrument": {"symbol": "AAPL", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "delta",
        "side": "buy",
        "quantity": 1,
        "order_type": "market",
        "time_in_force": "day",
        "metadata": {"test_case": "17 - submitted outside market hours, confirm queued or rejected as intended"}
    },

    # --- Tier 5: Multi-strategy netting (target_position intents) ---
    {
        "strategy_id": "strategy_a",
        "client_order_id": "test-018a-vst-long",
        "timestamp": "2026-08-27T09:46:00Z",
        "schema_version": "1.0",
        "instrument": {"symbol": "VST", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "target_position",
        "target_quantity": 500,
        "order_type": "market",
        "time_in_force": "day",
        "metadata": {"test_case": "18 - strategy A leg, expect net order of +200 after netting with B"}
    },
    {
        "strategy_id": "strategy_b",
        "client_order_id": "test-018b-vst-short",
        "timestamp": "2026-08-27T09:46:00Z",
        "schema_version": "1.0",
        "instrument": {"symbol": "VST", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "target_position",
        "target_quantity": -300,
        "order_type": "market",
        "time_in_force": "day",
        "metadata": {"test_case": "18 - strategy B leg, same cycle as A, expect net order of +200"}
    },
    {
        "strategy_id": "strategy_a",
        "client_order_id": "test-019a-vst-long",
        "timestamp": "2026-08-27T09:47:00Z",
        "schema_version": "1.0",
        "instrument": {"symbol": "VST", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "target_position",
        "target_quantity": 500,
        "order_type": "market",
        "time_in_force": "day",
        "metadata": {"test_case": "19a - strategy A leg, arrives first"}
    },
    {
        "strategy_id": "strategy_b",
        "client_order_id": "test-019b-vst-short-delayed",
        "timestamp": "2026-08-27T09:47:45Z",  # 45s later
        "schema_version": "1.0",
        "instrument": {"symbol": "VST", "asset_class": "equity", "exchange": "SMART"},
        "intent_type": "target_position",
        "target_quantity": -300,
        "order_type": "market",
        "time_in_force": "day",
        "metadata": {"test_case": "19b - strategy B leg arrives 45s after A, tests synchronization window"}
    },
]


def get_test_order(client_order_id: str) -> dict:
    """
    Look up a test intent by its client_order_id rather than list position,
    so reordering/adding test cases can't silently point you at the wrong one.
    Returns a fresh copy so mutating the result doesn't affect the shared list.
    """
    for order in test_orders:
        if order.get("client_order_id") == client_order_id:
            return dict(order)
    raise KeyError(f"No test order found with client_order_id={client_order_id!r}")


def list_test_case_ids() -> list:
    """Convenience: print/inspect all available client_order_ids at a glance."""
    return [o["client_order_id"] for o in test_orders]
def get_burst_test_orders() -> list:
    """
    Test 8: rapid multi-symbol burst, simulating 5 strategies submitting
    concurrently. Returns a fresh list of 5 intents each time (not cached
    in test_orders, since these exist to be fired concurrently/repeatedly
    rather than looked up by a single client_order_id).
    """
    symbols = ["AAPL", "MSFT", "NVDA", "AMD", "GOOGL"]
    return [
        {
            "strategy_id": f"test_suite_sim_{i}",
            "client_order_id": f"test-008-burst-{i}",
            "timestamp": "2026-08-27T09:37:00Z",
            "schema_version": "1.0",
            "instrument": {"symbol": sym, "asset_class": "equity", "exchange": "SMART"},
            "intent_type": "delta",
            "side": "buy",
            "quantity": 1,
            "order_type": "market",
            "time_in_force": "day",
            "metadata": {"test_case": "8 - rapid multi-symbol burst, tests rate-limit queue and lock safety"}
        }
        for i, sym in enumerate(symbols)
    ]

if __name__ == "__main__":
    # quick sanity check when run directly: python3 test_orders.py
    print(f"{len(test_orders)} test orders loaded:")
    for cid in list_test_case_ids():
        print(f"  - {cid}")
