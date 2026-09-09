"""
client/remote_strategy.py — the shape of a strategy that runs off the executor's host.

Subclass RemoteStrategy, implement `generate_book`, and the framework does the rest: the
preflight, the capital lookup, book validation, submission, the journal entry, and the exit
codes your scheduler reads.

    class MyStrategy(RemoteStrategy):
        strategy_id = "my_strategy"

        def generate_book(self, capital):
            return [self.intent("AAPL", 78, 319.97)]

    if __name__ == "__main__":
        raise SystemExit(MyStrategy.cli())

Everything the framework does is something that is easy to get wrong once and never notice:

  * `preflight()` before any work, so a broken executor fails before a half-book exists;
  * capital read from the executor, so the strategy follows /allocate instead of a constant
    someone has to remember to edit;
  * every book VALIDATED before it is sent — most importantly that `expected_price` is a
    real, positive, finite number, because the executor values your book with it when it
    applies the allocation cap. A stale or missing price mis-sizes the order AND the limit
    meant to contain it;
  * submission through `/targets` (absolute, authoritative, safe to repeat);
  * a journal entry on EVERY run, including the ones that decide to do nothing — those are
    the runs you cannot reconstruct later;
  * exit codes: 0 submitted AND confirmed by the broker, 1 refused,
    2 unreachable, 3 the executor took the orders but IB did not.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
from abc import ABC, abstractmethod

from client.executor_client import (ExecutorClient, ExecutorError, ExecutorRejected,
                                    ExecutorUnreachable)

log = logging.getLogger("remote-strategy")


class StrategyError(ExecutorError):
    """The strategy produced something that must not be sent."""


class RemoteStrategy(ABC):
    #: must match a strategy_id in the executor's config.CONFIG
    strategy_id: str = None
    #: "book" submits the whole book to /targets (recommended); "orders" posts each intent
    #: to /orders individually, for strategies that are not net-pooled.
    mode: str = "book"
    #: skip the run (exit 0) when the equity market is closed
    require_market_open: bool = False
    #: capital to assume for --dry-run, where the executor is not consulted
    dry_run_capital: float = float(os.environ.get("DRY_RUN_CAPITAL", 100_000.0))
    #: after submitting, wait for IB to confirm it took the orders before reporting success
    confirm_with_broker: bool = True
    #: how long to wait for that confirmation
    ack_timeout: float = 30.0

    EXIT_OK = 0
    EXIT_REFUSED = 1        # the executor answered and said no — config or risk
    EXIT_UNREACHABLE = 2    # orders did NOT go in
    EXIT_NOT_ACKED = 3      # the executor took them; the BROKER did not

    def __init__(self, strategy_id: str = None, client: ExecutorClient = None,
                 dry_run: bool = False):
        self.strategy_id = strategy_id or self.strategy_id
        if not self.strategy_id:
            raise ValueError("set strategy_id on the class or pass it to the constructor")
        self.dry_run = dry_run
        self.client = client or ExecutorClient(strategy_id=self.strategy_id)

    # ------------------------------------------------------------------ implement this
    @abstractmethod
    def generate_book(self, capital: float) -> list:
        """Return this strategy's ENTIRE desired book as a list of intents.

        `capital` is the allocation the executor currently holds for this strategy, so size
        against it rather than a constant. Use `self.intent(...)` to build each entry.

        The book is authoritative: any name you leave out gets closed. Including a name with
        `quantity=0` says the same thing, but says it out loud in the journal — worth doing
        for names you evaluated and rejected."""

    # ------------------------------------------------------------------ optional hooks
    def should_run(self, health: dict) -> bool:
        """Decide whether to trade at all this cycle. Return False to skip cleanly (exit 0).

        Override for a strategy that only acts at certain times; the default honours
        `require_market_open`."""
        if self.require_market_open and not health.get("market_open", True):
            log.info("market is closed — skipping")
            return False
        return True

    def describe(self, book: list) -> str:
        """One line for the journal. Override to record WHY, not just what."""
        held = [i for i in book if i["target_quantity"]]
        if not held:
            return f"{self.strategy_id}: flat — no names selected"
        return (f"{self.strategy_id}: {len(held)} names — "
                + ", ".join(f"{i['instrument']['symbol']}:{i['target_quantity']:g}"
                            for i in held))

    def journal_detail(self, book: list) -> str:
        """Extra context for the journal entry — scores, weights, anything you would want
        when reconstructing this decision in two months."""
        return ""

    def on_submitted(self, result: dict) -> None:
        """Called with the executor's response after a successful submission."""

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def intent(symbol: str, quantity: float, price: float, sec_type: str = "STK",
               exchange: str = "SMART", asset_class: str = "equity",
               multiplier: float = None) -> dict:
        """Build one intent in the shape the executor expects.

        `price` is not decoration: it becomes `expected_price`, which the executor uses to
        value this leg against the strategy's allocation cap (and to measure slippage on the
        fill). Pass a current price, not a placeholder."""
        instrument = {"symbol": symbol, "asset_class": asset_class,
                      "sec_type": sec_type, "exchange": exchange}
        if multiplier is not None:
            instrument["multiplier"] = multiplier
        return {"instrument": instrument, "target_quantity": quantity,
                "expected_price": price}

    def validate(self, book: list) -> None:
        """Refuse to send a book that cannot be valued — the same fail-closed rule the
        executor applies, enforced here so the mistake never leaves the strategy.

        Raises StrategyError listing every problem, rather than the first one."""
        problems = []
        seen = set()
        for i, entry in enumerate(book):
            where = f"intent[{i}]"
            symbol = ((entry.get("instrument") or {}).get("symbol") or "").strip()
            if not symbol:
                problems.append(f"{where}: missing instrument.symbol")
            elif symbol in seen:
                problems.append(f"{where}: {symbol} appears more than once")
            else:
                seen.add(symbol)
                where = f"{symbol}"

            qty = entry.get("target_quantity")
            if not isinstance(qty, (int, float)) or isinstance(qty, bool) or not math.isfinite(qty):
                problems.append(f"{where}: target_quantity must be a finite number, got {qty!r}")

            price = entry.get("expected_price")
            if not isinstance(price, (int, float)) or isinstance(price, bool):
                problems.append(f"{where}: expected_price must be a number, got {price!r}")
            elif not math.isfinite(price) or price <= 0:
                problems.append(f"{where}: expected_price must be positive and finite, "
                                f"got {price!r} — the allocation cap is computed from it")

            if (entry.get("instrument") or {}).get("sec_type") == "FUT" \
                    and not (entry.get("instrument") or {}).get("multiplier"):
                problems.append(f"{where}: a futures leg needs instrument.multiplier, or its "
                                "notional is understated by the multiplier")
        if problems:
            raise StrategyError(f"{self.strategy_id} produced an unsendable book:\n  "
                                + "\n  ".join(problems))

    # ------------------------------------------------------------------ the run loop
    def run(self) -> int:
        """Preflight, size, generate, validate, submit, journal. Returns an exit code."""
        log.info("%s -> %s%s", self.strategy_id, self.client.base_url,
                 " (dry run)" if self.dry_run else "")

        if self.dry_run:
            capital = self.dry_run_capital
        else:
            health = self.client.preflight()
            if not self.should_run(health):
                return self.EXIT_OK
            capital = float(self.client.allocation()["capital_allocation"])

        book = self.generate_book(capital)
        if book is None:
            raise StrategyError("generate_book returned None — return a list of intents")
        self.validate(book)

        held = [i for i in book if i["target_quantity"]]
        log.info("%d/%d names with a target, $%s of capital",
                 len(held), len(book), f"{capital:,.0f}")
        for entry in held:
            log.info("   %-6s %10.4g @ %10.4f", entry["instrument"]["symbol"],
                     entry["target_quantity"], entry["expected_price"])

        if self.dry_run:
            log.info("dry run — nothing submitted")
            return self.EXIT_OK

        result = (self.client.submit_book(book) if self.mode == "book"
                  else self.client.submit_orders(book))
        log.info("submitted: %s", {k: v for k, v in result.items() if k != "internal_crosses"})
        self.on_submitted(result)

        # A clean submission means the EXECUTOR took the orders — it handed them to the
        # socket. Whether the BROKER took them is a different question, and not asking it is
        # how a strategy ends a run reporting success while holding nothing: a read-only
        # gateway refuses every order and the submission still comes back accepted.
        acks = None
        if self.confirm_with_broker:
            if not hasattr(self.client, "wait_for_acks"):
                # A client predating broker confirmation. Skipping is the only option, but
                # it must be said out loud: the run is about to report success on the
                # executor's word alone, which is the very thing this step exists to stop.
                log.warning("%s cannot confirm orders with the broker — reporting success "
                            "on the executor's acceptance alone",
                            type(self.client).__name__)
            else:
                acks = self.client.wait_for_acks(self.client.order_ids(result),
                                                 timeout=self.ack_timeout)
            if acks["live"]:
                log.info("%d order(s) confirmed at the broker", len(acks["live"]))

        # journal AFTER submitting, so the record reflects what was actually sent, and even
        # when the book was empty — a decision to hold nothing is still a decision
        try:
            self.client.journal("signal", self.describe(book),
                                detail=self._with_ack_detail(self.journal_detail(book), acks),
                                symbols=[i["instrument"]["symbol"] for i in held])
        except ExecutorError as e:
            log.warning("submitted, but could not journal it: %s", e)

        if acks and (acks["rejected"] or acks["pending"]):
            log.critical("%d rejected, %d unacknowledged — the book is NOT in place at the "
                         "broker", len(acks["rejected"]), len(acks["pending"]))
            return self.EXIT_NOT_ACKED
        return self.EXIT_OK

    @staticmethod
    def _with_ack_detail(detail: str, acks: dict) -> str:
        """Put the broker's answer in the journal too — a run that submitted and was refused
        must not read the same as one that worked."""
        if not acks:
            return detail
        note = (f"broker: {len(acks['live'])} live, {len(acks['rejected'])} rejected, "
                f"{len(acks['pending'])} unacknowledged")
        for oid in acks["rejected"]:
            err = (acks["acks"].get(str(oid)) or {}).get("last_error") or {}
            note += f"\n  order {oid} rejected: {err.get('code', '')} {err.get('message', '')}"
        return f"{detail}\n{note}" if detail else note

    # ------------------------------------------------------------------ entry point
    @classmethod
    def cli(cls, argv: list = None, **kwargs) -> int:
        """Command line front end: `raise SystemExit(MyStrategy.cli())`.

        Turns every failure into the exit code a scheduler can act on — the whole point of
        which is that an unreachable executor (2) can never be mistaken for a quiet success."""
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--dry-run", action="store_true",
                            help="generate and validate the book, submit nothing")
        parser.add_argument("--strategy", default=cls.strategy_id,
                            help="override the strategy_id")
        parser.add_argument("-v", "--verbose", action="store_true")
        args = parser.parse_args(argv)

        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
        try:
            return cls(strategy_id=args.strategy, dry_run=args.dry_run, **kwargs).run()
        except ExecutorUnreachable as e:
            # The client has already alerted Telegram directly. Non-zero so the scheduler
            # sees it: the orders did NOT go in.
            log.critical("ABORTED — %s", e)
            return cls.EXIT_UNREACHABLE
        except (ExecutorRejected, StrategyError) as e:
            log.error("%s", e)
            return cls.EXIT_REFUSED
