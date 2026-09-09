#!/usr/bin/env python3
"""run_rebalance.py — recommend capital weights across strategies, once, from the scheduler.

    python3 run_rebalance.py              # report the recommendation, change nothing
    python3 run_rebalance.py --apply      # actually move the capital

REPORT-ONLY BY DEFAULT, and that is not timidity. Applying a reallocation SELLS positions to
raise cash (`/strategies/{id}/allocation` pro-rata), so a wrong answer does not sit in a log
— it trades. The allocator refuses to use expected returns until there is enough history and
falls back to inverse-volatility or equal weight, so for a young book the honest output is
"hold". Run it in report mode for a few weeks and watch whether the weights it suggests are
stable before letting it move anything.

Exit codes: 0 reported/applied, 1 refused, 2 executor unreachable.
"""
import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd
from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from client.executor_client import (ExecutorClient, ExecutorRejected,  # noqa: E402
                                    ExecutorUnreachable)
from portfolio.allocator import NON_ALLOCATABLE, recommend             # noqa: E402

log = logging.getLogger("run-rebalance")
EXIT_OK, EXIT_REFUSED, EXIT_UNREACHABLE = 0, 1, 2

#: fixtures, not strategies — they would otherwise be handed real capital
TEST_PREFIXES = ("test_suite", "halt_test", "demo_")


def returns_from_history(rows: list, freq: str = "1D") -> pd.DataFrame:
    """Daily returns per strategy from the equity sampler's snapshots.

    Resampled to `freq` deliberately. The sampler writes about once a minute, so a week is
    ~10,000 rows — a sample size that looks enormous and is not, because the extra ticks add
    no independent information about a mean. The allocator's guards count calendar days for
    the same reason; feeding it raw ticks would defeat them.
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    basis = "nav" if "nav" in df.columns and df["nav"].notna().any() else "equity"
    df = df[df[basis].notna()]
    if df.empty:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="mixed")
    wide = (df.pivot_table(index="ts", columns="strategy_id", values=basis, aggfunc="last")
              .sort_index().resample(freq).last().dropna(how="all"))
    return wide.pct_change().dropna(how="all")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually reallocate (SELLS positions to raise cash)")
    ap.add_argument("--min-history-days", type=float, default=14.0,
                    help="a strategy younger than this is exempt (held at its weight)")
    ap.add_argument("--freq", default="1D",
                    help="resample frequency for the return series, e.g. 1D or 15min. "
                         "Anything intraday needs --periods-per-year to match, or every "
                         "annualised figure downstream is wrong by that ratio.")
    ap.add_argument("--periods-per-year", type=float, default=252.0)
    ap.add_argument("--min-days-cov", type=float, default=20.0,
                    help="calendar days of history before a covariance is trusted")
    ap.add_argument("--min-days-means", type=float, default=90.0,
                    help="calendar days before EXPECTED RETURNS are trusted. Lowering this "
                         "is the one that lets a lucky run win capital — the guard exists "
                         "because a max-Sharpe fit on a short window loads on noise.")
    ap.add_argument("--max-weight", type=float, default=0.40)
    ap.add_argument("--l2", type=float, default=0.5)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    client = ExecutorClient(strategy_id="rebalancer")
    client.preflight()

    history = client._request("GET", "/pnl/history").get("history", [])
    returns = returns_from_history(history, freq=args.freq)
    if returns.empty:
        log.warning("no NAV history yet — nothing to allocate on")
        return EXIT_OK

    keep = [c for c in returns.columns if not c.startswith(TEST_PREFIXES)]
    returns = returns[keep]
    if returns.empty or returns.shape[1] == 0:
        log.warning("only test fixtures in the history — nothing to allocate on")
        return EXIT_OK

    strategies = client._request("GET", "/strategies").get("strategies", [])
    total = sum(s["capital_allocation"] for s in strategies
                if not s["strategy_id"].startswith(TEST_PREFIXES))
    current = {s["strategy_id"]: s["capital_allocation"] / total
               for s in strategies
               if total > 0 and not s["strategy_id"].startswith(TEST_PREFIXES)}

    alloc = recommend(returns, current_weights=current,
                      min_history_days=args.min_history_days,
                      max_weight=args.max_weight, l2_lambda=args.l2,
                      min_days_for_cov=args.min_days_cov,
                      min_days_for_means=args.min_days_means,
                      periods_per_year=args.periods_per_year)
    if args.min_days_means < 30.0 or args.freq != "1D":
        log.warning("SHAKEOUT SETTINGS — freq=%s, means trusted after %.2f days. These "
                    "defeat the guards that stop a short lucky run winning capital. Fine "
                    "for proving the pipeline runs; not a basis for moving money.",
                    args.freq, args.min_days_means)
    log.info("\n%s", alloc)

    targets = alloc.capital(total)
    for sid, dollars in sorted(targets.items(), key=lambda kv: -kv[1]):
        delta = dollars - current.get(sid, 0.0) * total
        log.info("   %-28s %12s  (%s)", sid, f"{dollars:,.0f}", f"{delta:+,.0f}")

    if alloc.method == "hold":
        log.info("nothing changes — %s", alloc.reason)
        return EXIT_OK
    if not args.apply:
        log.info("report only — pass --apply to move capital")
        return EXIT_OK

    for sid, dollars in targets.items():
        if sid in NON_ALLOCATABLE or sid in alloc.exempt:
            continue          # never sized here; leaving them alone is the whole point
        result = client._request("POST", f"/strategies/{sid}/allocation", auth=True,
                                 json={"capital_allocation": dollars})
        log.info("%s -> %s: %s", sid, f"{dollars:,.0f}",
                 {k: result.get(k) for k in ("allocation_before", "allocation_after")})
    return EXIT_OK


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExecutorUnreachable as e:
        log.critical("ABORTED — %s", e)
        raise SystemExit(EXIT_UNREACHABLE)
    except ExecutorRejected as e:
        log.error("%s", e)
        raise SystemExit(EXIT_REFUSED)
