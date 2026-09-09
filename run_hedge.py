#!/usr/bin/env python3
"""run_hedge.py — put the portfolio hedge on, once, from the scheduler.

    python3 run_hedge.py --dry-run     # print the plan, submit nothing
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
    ap.add_argument("--trigger", type=float, default=0.30)
    ap.add_argument("--target", type=float, default=0.25)
    ap.add_argument("--release", type=float, default=0.20)
    ap.add_argument("--ack-timeout", type=float, default=60.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    client = ExecutorClient(strategy_id=HEDGE_STRATEGY_ID)
    health = client.preflight()
    if not health.get("market_open"):
        log.info("market closed — no hedge cycle")
        return EXIT_OK

    view = client._request("GET", "/exposure", params={
        "trigger": args.trigger, "target": args.target, "release": args.release})

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

    result = client.submit_book(book, strategy_id=HEDGE_STRATEGY_ID)
    log.info("submitted: %s", {k: v for k, v in result.items() if k != "internal_crosses"})

    acks = client.wait_for_acks(client.order_ids(result), timeout=args.ack_timeout)
    try:
        client.journal(
            "hedge",
            f"hedge cycle: {len(acting)} bucket(s) hedged, {len(book)} leg(s) submitted",
            detail=(f"nav={view['nav']:,.0f}; "
                    f"broker: {len(acks['live'])} live, {len(acks['rejected'])} rejected, "
                    f"{len(acks['pending'])} unacknowledged; "
                    f"unhedgeable={view.get('unhedgeable')}"),
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


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExecutorUnreachable as e:
        log.critical("ABORTED — %s", e)
        raise SystemExit(EXIT_UNREACHABLE)
    except ExecutorRejected as e:
        log.error("%s", e)
        raise SystemExit(EXIT_REFUSED)
