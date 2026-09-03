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

    # Kalman VECM (WTI vs Brent+RBOB futures). capital_allocation here is the sizing NAV +
    # the executor's loose multiplier-aware risk backstop; the strategy's own hard risk_limits
    # (per-leg / gross notional in models/vecm_strategy.py) are the BINDING caps — tune both
    # to your competition capital.
    "kalman_vecm": {
        "capital_allocation": 2_000_000.0,
        "max_drawdown": 0.15,
    },

    # Best-2 equities strategies from the claude research (daily-rebalanced port).
    "ovn_volsurge": {   # best unbiased strategy (Sharpe 2.62) — overnight vol-surge hold
        "capital_allocation": 200_000.0,
        "max_drawdown": 0.15,
    },
    "orb_breakout": {   # intraday opening-range breakout (30-min bars); unbiased Sharpe ~2.5
        "capital_allocation": 200_000.0,
        "max_drawdown": 0.15,
    },
    "kalman_rrg_combined": {   # combined Kalman/RRG rotation (gated long-leg + naive risk parity)
        "capital_allocation": 200_000.0,
        "max_drawdown": 0.15,
    },
    # --- halt-test strategies: tight 1% max_drawdown to exercise the drawdown -> halt path
    #     LIVE (RiskManager.check_drawdown -> halt_strategy -> is_active False -> subsequent
    #     orders rejected). Allocation fits one micro future (e.g. MCL ~$6.85k notional).
    #     Drive with tools/halt_test.py. 1% of 10k = $100 realized-loss halt threshold.
    "halt_test_1": {
        "capital_allocation": 10_000.0,
        "max_drawdown": 0.01,
    },
    "halt_test_2": {
        "capital_allocation": 10_000.0,
        "max_drawdown": 0.01,
    },
    "halt_test_3": {
        "capital_allocation": 10_000.0,
        "max_drawdown": 0.01,
    },
}


# ---------------------------------------------------------------------------
# GLOBAL (portfolio-level) risk + safety guards. See ROADMAP "Safety / survival".
# Breaker + margin check are DISABLED by default (set thresholds to enable) so they can't
# surprise-halt an unsupervised run; the protective guards default ON.
# ---------------------------------------------------------------------------
GLOBAL = {
    # Portfolio circuit breaker: if total P&L (realized + unrealized) since the daily baseline
    # falls by >= this many dollars, HALT + FLATTEN every strategy and set the kill switch.
    "max_daily_loss": None,          # e.g. 50_000.0 ; None = disabled
    # Cap on total gross notional across ALL strategies (pre-trade reject). None = disabled.
    "max_gross_exposure": None,      # e.g. 5_000_000.0 ; None = disabled

    # Fill-price sanity guard (post-fill; market orders can't be pre-rejected).
    "fill_slippage_alert_pct": 0.05, # alert (CRITICAL -> Telegram) when |fill-expected|/expected exceeds this
    "fill_slippage_halt_pct": None,  # also halt+flatten the strategy beyond this; None = alert only

    # Pre-trade margin probe (whatIf) before placing. Adds a blocking round-trip; OFF by default.
    "pretrade_margin_check": False,
    "max_order_init_margin": None,   # $ cap on an order's init-margin change (required if the check is on)

    # Unrealized-drawdown data-quality guard: a mark older than this (or missing) makes the
    # total-equity drawdown check SKIP that strategy for the cycle (realized fast path still runs).
    "mark_staleness_sec": 120.0,

    # Auto-reconnect to IB on an unexpected disconnect (then reconcile + recover open orders).
    "auto_reconnect": True,
    "reconnect_max_attempts": 30,
    "reconnect_backoff_sec": 10.0,
}


# ---------------------------------------------------------------------------
# ATR LIMIT-AT-PULLBACK execution layer.  When enabled, market-order intents
# are converted to limit orders at  price - fraction * ATR(period).  The idea:
# buy a little cheaper on a normal intraday pullback.  Unfilled orders are
# cancelled before the close (POST /atr/cancel from day_scheduler).
# ---------------------------------------------------------------------------
ATR_EXECUTION = {
    "enabled": True,                 # flip to True to activate
    "atr_period": 14,                 # ATR lookback in intraday bars
    "atr_fraction": 0.5,             # limit = price - fraction * ATR
    "bar_size": "5 mins",            # IBKR bar size (e.g. "1 min", "5 mins", "15 mins")
    "duration": "2 D",               # IBKR lookback duration (must cover atr_period+1 bars)
    "cache_ttl_sec": 300,            # ATR cache lifetime (seconds)
    "strategies": ["orb_breakout"],   # only orb_breakout uses ATR; rest go market
    "skip_exits": True,              # never transform exit intents (target_qty == 0)
    "cancel_before_close_min": 5,    # EOD cancel sweep (used by day_scheduler)
}
