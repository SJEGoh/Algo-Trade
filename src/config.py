"""
config.py

Minimal risk configuration for testing the CentralExecutor.

Shape: strategy_id -> {
    "capital_allocation": float,   # max gross notional this strategy may hold
    "max_drawdown_pct":  float,    # halt strategy when realized loss / allocation >= this
}

Only strategy_ids listed here are on the allowlist — any intent from a
strategy_id NOT in this dict is rejected by RiskManager (fail-closed).

The strategy_ids below match the ones used across test_orders.py so the
test suite actually flows through the risk checks instead of being rejected
as "not active".
"""

CONFIG = {
    # generic test strategy used by most Tier 1/2 cases
    "test_suite": {
        "capital_allocation": 1_000_000.0,   # large, so normal happy-path tests aren't blocked
        "max_drawdown": 0.20,
    },

    # Tier 3 test 9 — deliberately tiny allocation so an oversized order is rejected
    "test_suite_small_alloc": {
        "capital_allocation": 1_000.0,
        "max_drawdown": 0.20,
    },

    # Tier 3 test 10 — used to exercise the drawdown-breach halt
    "test_suite_drawdown_breached": {
        "capital_allocation": 10_000.0,
        "max_drawdown": 0.10,
    },

    # Tier 5 netting strategies
    "demo_momentum": {
        "capital_allocation": 500_000.0,
        "max_drawdown": 0.15,
    },
    "demo_meanrev": {
        "capital_allocation": 500_000.0,
        "max_drawdown": 0.15,
    },
        "cross_sectional_momentum": {
        "capital_allocation": 100_000.0,   # match the capital_allocation in your MomentumStrategy
        "max_drawdown": 0.15,              # fraction of allocation, e.g. 0.15 = 15%
    },
}
