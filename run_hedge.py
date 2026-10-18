#!/usr/bin/env python3
"""run_hedge.py — put the portfolio hedge on, once, from the scheduler.

    python3 run_hedge.py --dry-run     # print the plan, submit nothing
    python3 run_hedge.py --dry-run --any-time   # ...even with the market closed
    python3 run_hedge.py --model pca --dry-run  # statistical factor hedge, print only
    python3 run_hedge.py --model pca            # ...and submit it
    python3 run_hedge.py --model minvar --dry-run   # spend the allocation on variance
    python3 run_hedge.py               # submit it

Asks the executor for `GET /exposure`, which buckets every holding and decides which buckets
have run past their trigger, then submits the resulting book to `/targets` under
`hedge_overlay`. The book is absolute, so re-running is a no-op when nothing has moved and a
correction when it has.

The rule that matters
---------------------
An order the executor accepted is NOT a hedge. `/orders` returning `accepted: true` means the
order reached the socket; whether IB took it is a separate question with a separate answer.
A hedge you believe is on but isn't is worse than no hedge at all, because the book gets
sized as though it were protected. So this waits for the broker's acknowledgement and exits
NON-ZERO if any leg is unconfirmed — the scheduler's log and the Telegram alert then say the
book is UNHEDGED, rather than a green run implying cover that does not exist.

Exit codes: 0 hedged (or nothing to do), 1 refused, 2 executor unreachable,
3 submitted but NOT confirmed at the broker — treat as unhedged.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from client.executor_client import (ExecutorClient, ExecutorRejected,  # noqa: E402
                                    ExecutorUnreachable)
from portfolio.hedger import HEDGE_STRATEGY_ID                          # noqa: E402

log = logging.getLogger("run-hedge")

EXIT_OK, EXIT_REFUSED, EXIT_UNREACHABLE, EXIT_NOT_ACKED = 0, 1, 2, 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, submit nothing")
    ap.add_argument("--any-time", action="store_true",
                    help="with --dry-run: plan even when the market is closed (marks are "
                         "the last prices IB has, so treat sizes as approximate)")
    ap.add_argument("--model", choices=["buckets", "pca", "minvar"], default="buckets",
                    help="buckets: sector exposure share of NAV. pca: statistical factor "
                         "risk as daily vol share of NAV (pca_hedge.py). minvar: spend the "
                         "overlay's allocation on the ETF hedge that removes the most "
                         "variance (minvar_hedge.py) — sized by capital, not thresholds")
    ap.add_argument("--utilization", type=float, default=0.95,
                    help="minvar: share of the overlay's allocation to spend")
    ap.add_argument("--budget", type=float, default=None,
                    help="minvar: spend this instead (what-if; must fit the allocation)")
    ap.add_argument("--max-unmeasured", type=float, default=0.25,
                    help="pca: refuse to TRADE when more than this share of the book's gross "
                         "could not be measured (no history) — a hedge sized on a partial "
                         "book is a guess")
    ap.add_argument("--trigger", type=float, default=None,
                    help="default 0.30 for buckets (exposure), 0.006 for pca (daily vol)")
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--release", type=float, default=None)
    ap.add_argument("--exclude", nargs="+", default=[], metavar="SYMBOL",
                    help="pca: measure the book as if these symbols were not held (what-if)")
    ap.add_argument("--exclude-strategy", nargs="+", default=[], metavar="STRATEGY",
                    help="pca: measure the book without these strategies' holdings")
    ap.add_argument("--ack-timeout", type=float, default=60.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.any_time and not args.dry_run:
        ap.error("--any-time only works with --dry-run — a live hedge never trades a closed market")
    if args.model == "minvar" and any(v is not None for v in
                                      (args.trigger, args.target, args.release)):
        ap.error("--model minvar has no thresholds — it is sized by the overlay's allocation "
                 "(--utilization / --budget)")
    if (args.exclude or args.exclude_strategy) and args.model != "pca":
        ap.error("--exclude / --exclude-strategy are what-ifs for --model pca")
    if (args.exclude or args.exclude_strategy) and not args.dry_run:
        ap.error("--exclude / --exclude-strategy measure a book you do not hold — a what-if "
                 "can never be traded; add --dry-run")

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # the first PCA call of a day pulls a year of daily bars per holding from IB
    client = ExecutorClient(strategy_id=HEDGE_STRATEGY_ID,
                            **({"timeout": 300.0, "retries": 1}
                               if args.model in ("pca", "minvar") else {}))
    health = client.preflight()
    if not health.get("market_open"):
        if not args.any_time:
            log.info("market closed — no hedge cycle")
            return EXIT_OK
        log.warning("market closed — planning anyway (--any-time); prices are stale")

    if args.model in ("pca", "minvar"):
        view = report_pca(client, args) if args.model == "pca" else report_minvar(client, args)
        book = view.get("book") or []
        if not book:
            log.info("nothing to hedge and nothing held — nothing to do")
            return EXIT_OK
        if args.dry_run:
            log.info("dry run — %d leg(s) NOT submitted", len(book))
            return EXIT_OK
        cov = view.get("coverage") or {}
        gross = (cov.get("measured_notional") or 0.0) + (cov.get("unmeasured_notional") or 0.0)
        share = (cov.get("unmeasured_notional") or 0.0) / gross if gross > 0 else 0.0
        if share > args.max_unmeasured:
            log.critical("HEDGE NOT PLACED — %.0f%% of the book has no usable history "
                         "(limit %.0f%%): %s. Fix the history, or raise --max-unmeasured.",
                         share * 100, args.max_unmeasured * 100, cov.get("unmeasured_symbols"))
            return EXIT_REFUSED
        if args.model == "minvar":
            if view.get("held_kept"):
                log.info("held hedge kept — %s", view.get("reason"))
            return submit_and_confirm(
                client, book, args,
                summary=(f"minvar hedge cycle: {len(view['legs'])} leg(s), "
                         f"{view['gross']:,.0f} of {view['budget']:,.0f} budget"),
                detail=(f"nav={view['nav']:,.0f}; vol {view['vol_before']:.2%} -> "
                        f"{view['vol_after']:.2%}/day, {view['variance_removed']:.0%} of "
                        f"variance removed; {view['reason']}"))
        acting = [e for e in view["exposures"] if e.get("hedge_notional")]
        return submit_and_confirm(
            client, book, args,
            summary=f"pca hedge cycle: {len(acting)} factor(s) hedged, {len(book)} leg(s)",
            detail=(f"nav={view['nav']:,.0f}; factors="
                    + ", ".join(f"{e['factor']} {e['vol_contribution']:.2%}/day"
                                for e in view["exposures"])))

    defaults = {"trigger": 0.30, "target": 0.25, "release": 0.20}
    view = client._request("GET", "/exposure", params={
        k: v if v is not None else defaults[k]
        for k, v in (("trigger", args.trigger), ("target", args.target),
                     ("release", args.release))})

    log.info("NAV %s", f"{view['nav']:,.0f}")
    for e in view["exposures"]:
        log.info("   %-20s %+7.1f%%  %+14s", e["bucket"], e["fraction"] * 100,
                 f"{e['notional']:,.0f}")

    coverage = view.get("coverage") or {}
    if coverage.get("unpriced_symbols"):
        # Not fatal, but it must be said: this exposure was not measured, so the hedge below
        # is sized against an incomplete picture of the book.
        log.error("UNMEASURED exposure — no mark for %s", coverage["unpriced_symbols"])
    if view.get("unhedgeable"):
        log.error("UNHEDGEABLE buckets (carried naked): %s",
                  {k: f"{v:,.0f}" for k, v in view["unhedgeable"].items()})

    book = view.get("book") or []
    acting = [d for d in view["hedge"] if abs(d.get("quantity") or 0) > 0]
    for d in view["hedge"]:
        log.info("   %-20s %-5s %+10s — %s", d["bucket"], d["symbol"] or "-",
                 f"{d['quantity']:,.0f}", d["reason"])

    if not book:
        log.info("no bucket past its trigger and nothing held — nothing to do")
        return EXIT_OK
    if args.dry_run:
        log.info("dry run — %d leg(s) NOT submitted", len(book))
        return EXIT_OK

    return submit_and_confirm(
        client, book, args,
        summary=f"hedge cycle: {len(acting)} bucket(s) hedged, {len(book)} leg(s) submitted",
        detail=f"nav={view['nav']:,.0f}; unhedgeable={view.get('unhedgeable')}")


def submit_and_confirm(client, book, args, summary, detail) -> int:
    """Submit the overlay's absolute book, wait for the BROKER, journal it. Shared by both
    models: an accepted order is not a hedge until IB has taken it."""
    result = client.submit_book(book, strategy_id=HEDGE_STRATEGY_ID)
    log.info("submitted: %s", {k: v for k, v in result.items() if k != "internal_crosses"})

    acks = client.wait_for_acks(client.order_ids(result), timeout=args.ack_timeout)
    try:
        client.journal(
            "hedge", summary,
            detail=(f"{detail}; broker: {len(acks['live'])} live, "
                    f"{len(acks['rejected'])} rejected, {len(acks['pending'])} unacknowledged"),
            symbols=[i["instrument"]["symbol"] for i in book],
            strategy_id=HEDGE_STRATEGY_ID)
    except Exception as e:
        log.warning("hedged, but could not journal it: %s", e)

    if acks["rejected"] or acks["pending"]:
        log.critical("HEDGE NOT IN PLACE — %d rejected, %d unacknowledged. The book is "
                     "UNHEDGED; do not treat it as covered.",
                     len(acks["rejected"]), len(acks["pending"]))
        return EXIT_NOT_ACKED

    log.info("hedge confirmed at the broker: %d leg(s)", len(acks["live"]))
    return EXIT_OK


def report_minvar(client, args) -> dict:
    """Print the variance-minimising hedge for the overlay's capital, and return the view."""
    params = {"utilization": args.utilization}
    if args.budget is not None:
        params["budget"] = args.budget
    v = client._request("GET", "/exposure/minvar", params=params)
    log.info("NAV %s   overlay allocation %s, budget %s (%d days of history)",
             f"{v['nav']:,.0f}", f"{v['allocation']:,.0f}", f"{v['budget']:,.0f}", v["days"])
    log.info("book volatility %.2f%%/day -> %.2f%%/day hedged: %.0f%% of variance removed "
             "(%s)", v["vol_before"] * 100, v["vol_after"] * 100,
             v["variance_removed"] * 100,
             "budget-bound" if v["binding"] else "full min-variance hedge fits the budget")
    if v["held_kept"]:
        log.info("currently held hedge: %.2f%%/day — %s", v["vol_with_held"] * 100, v["reason"])
    for sym, d in sorted(v["dollars"].items(), key=lambda kv: -abs(kv[1])):
        log.info("   %-5s %+12s  (%+.0f sh)", sym, f"{d:,.0f}", v["legs"][sym])
    log.info("   gross %s of %s", f"{v['gross']:,.0f}", f"{v['budget']:,.0f}")
    cov = v["coverage"]
    if cov.get("unpriced_symbols"):
        log.error("UNPRICED (no mark, not measured): %s", cov["unpriced_symbols"])
    if cov.get("unmeasured_symbols"):
        log.error("NO USABLE HISTORY (not measured, %s): %s",
                  f"{cov['unmeasured_notional']:,.0f}", cov["unmeasured_symbols"])
    if cov.get("priced_from_close"):
        log.warning("valued at last daily close, no live mark: %s", cov["priced_from_close"])
    for leg in v["book"]:
        log.info("   leg  %-5s %+8.0f  @ %s", leg["instrument"]["symbol"],
                 leg["target_quantity"], leg["expected_price"])
    return v


def report_pca(client, args) -> dict:
    """Print the factor hedge the executor would place, and return the view. Submits nothing."""
    params = {k: v for k, v in (("trigger", args.trigger), ("target", args.target),
                                ("release", args.release)) if v is not None}
    if args.exclude:
        params["exclude"] = args.exclude
    if args.exclude_strategy:
        params["exclude_strategy"] = args.exclude_strategy
    view = client._request("GET", "/exposure/pca", params=params or None)
    pol = view["policy"]
    ex = view.get("excluded") or {}
    if ex.get("symbols") or ex.get("strategies"):
        log.warning("WHAT-IF — book measured without symbols %s, strategies %s",
                    ex.get("symbols") or "-", ex.get("strategies") or "-")
    log.info("NAV %s   (%d days of history; gate = daily vol share of NAV, trigger %.2f%% "
             "/ target %.2f%% / release %.2f%%)", f"{view['nav']:,.0f}", view["history_days"],
             pol["trigger"] * 100, pol["target"] * 100, pol["release"] * 100)
    for f in view["factors"]:
        top = ", ".join(f"{s} {w:+.2f}" for s, w in list(f["weights"].items())[:5])
        log.info("   %-4s explains %5.1f%%  vol %.2f%%/day   %s",
                 f["factor"], f["explained"] * 100, f["daily_vol"] * 100, top)
    log.info("   PC1 explains >= 60%% of the universe: %s (reported, not used to decide)",
             view["pc1_explains_60pct"])
    for e in view["exposures"]:
        top = ", ".join(f"{s} {v:+,.0f}" for s, v in e["top_contributors"].items())
        log.info("   %-4s exposure %+12s (%+6.1f%% NAV)  risk %.2f%%/day  hedge %+10s "
                 "(held %s) — %s", e["factor"], f"{e['exposure']:,.0f}", e["fraction"] * 100,
                 e["vol_contribution"] * 100, f"{e['hedge_notional']:,.0f}",
                 f"{e.get('held') or 0.0:+,.0f}", e["reason"])
        if top:
            log.info("          largest: %s", top)
    br = view.get("book_risk") or {}
    if br.get("daily_vol") is not None:
        log.info("whole book (historical replay, %d days): %.2f%%/day = %.1f%%/yr of NAV, "
                 "1-day 99%% loss %.2f%%, worst day %.2f%% — factors above are the hedgeable "
                 "part of this", br["days"], br["daily_vol"] * 100, br["annual_vol"] * 100,
                 br["var99"] * 100, br["worst_day"] * 100)
    cov = view["coverage"]
    if cov.get("priced_from_close"):
        log.warning("valued at last daily close, no live mark: %s", cov["priced_from_close"])
    if cov.get("factor_r2") is not None:
        log.info("factors explain %.0f%% of the measured book's daily moves (value-weighted R²)",
                 cov["factor_r2"] * 100)
    if cov.get("unpriced_symbols"):
        log.error("UNPRICED (no mark, not measured): %s", cov["unpriced_symbols"])
    if cov.get("unmeasured_symbols"):
        log.error("NO USABLE HISTORY (not measured, %s): %s",
                  f"{cov['unmeasured_notional']:,.0f}", cov["unmeasured_symbols"])
    for leg in view["book"]:
        log.info("   leg  %-5s %+8.0f  @ %s", leg["instrument"]["symbol"],
                 leg["target_quantity"], leg["expected_price"])
    return view


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExecutorUnreachable as e:
        log.critical("ABORTED — %s", e)
        raise SystemExit(EXIT_UNREACHABLE)
    except ExecutorRejected as e:
        log.error("%s", e)
        raise SystemExit(EXIT_REFUSED)
