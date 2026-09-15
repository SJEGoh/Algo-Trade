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
  * optional exits per name — `self.intent("AAPL", 78, 319.97, stop_pct=0.03,
    trail_pct=0.05)` — validated here, enforced by the executor, and a name held flat
    after one fires is expected flat rather than reported unfilled;
  * exit codes: 0 submitted, confirmed by the broker AND filled, 1 refused,
    2 unreachable, 3 the executor took the orders but IB did not,
    4 IB acknowledged them but the book did not reach target in time.
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
    #: then wait for the strategy's own holdings to reach its targets — i.e. for FILLS
    confirm_fills: bool = True
    #: how long to wait for fills. A strategy that works resting limit orders, which can
    #: legitimately sit unfilled for hours, should raise this or set confirm_fills = False.
    fill_timeout: float = 60.0

    EXIT_OK = 0
    EXIT_REFUSED = 1        # the executor answered and said no — config or risk
    EXIT_UNREACHABLE = 2    # orders did NOT go in
    EXIT_NOT_ACKED = 3      # the executor took them; the BROKER did not
    EXIT_NOT_FILLED = 4     # the broker took them; the book did NOT reach target in time

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
               multiplier: float = None, *, stop_price: float = None,
               stop_pct: float = None, take_profit_price: float = None,
               take_profit_pct: float = None, trail_pct: float = None,
               trail_amount: float = None) -> dict:
        """Build one intent in the shape the executor expects.

        `price` is not decoration: it becomes `expected_price`, which the executor uses to
        value this leg against the strategy's allocation cap (and to measure slippage on the
        fill). Pass a current price, not a placeholder.

        Exits are optional and independent — pass any combination, or none. `*_pct` values
        are fractions (0.02 = 2%) of the strategy's average cost, so they can go out with
        the entry. The executor enforces them; see client/README.md."""
        instrument = {"symbol": symbol, "asset_class": asset_class,
                      "sec_type": sec_type, "exchange": exchange}
        if multiplier is not None:
            instrument["multiplier"] = multiplier
        entry = {"instrument": instrument, "target_quantity": quantity,
                 "expected_price": price}
        exits = {k: v for k, v in (("stop_price", stop_price), ("stop_pct", stop_pct),
                                   ("take_profit_price", take_profit_price),
                                   ("take_profit_pct", take_profit_pct),
                                   ("trail_pct", trail_pct), ("trail_amount", trail_amount))
                 if v is not None}
        if exits:
            entry["exits"] = exits
        return entry

    #: exit field pairs — one of each at most
    EXIT_PAIRS = (("stop_price", "stop_pct"), ("take_profit_price", "take_profit_pct"),
                  ("trail_amount", "trail_pct"))

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
            problems += self._exit_problems(where, entry)
        if problems:
            raise StrategyError(f"{self.strategy_id} produced an unsendable book:\n  "
                                + "\n  ".join(problems))

    def _exit_problems(self, where: str, entry: dict) -> list:
        """The executor refuses the WHOLE submission over one unenforceable exit, so catch
        it here, where the message can name the strategy's own mistake."""
        exits = entry.get("exits")
        if exits is None:
            return []
        if not isinstance(exits, dict):
            return [f"{where}: exits must be a dict"]
        allowed = {f for pair in self.EXIT_PAIRS for f in pair}
        out = [f"{where}: unknown exit field {k!r}" for k in exits if k not in allowed]
        for k, v in exits.items():
            if k in allowed and (isinstance(v, bool) or not isinstance(v, (int, float))
                                 or not math.isfinite(v) or v <= 0):
                out.append(f"{where}: {k} must be a positive number, got {v!r}")
        for a, b in self.EXIT_PAIRS:
            if exits.get(a) is not None and exits.get(b) is not None:
                out.append(f"{where}: set {a} or {b}, not both")
        if out:
            return out
        qty, price = entry.get("target_quantity"), entry.get("expected_price")
        if not isinstance(qty, (int, float)) or not qty:
            return out
        for k in ("stop_pct", "trail_pct") + (("take_profit_pct",) if qty < 0 else ()):
            if exits.get(k, 0) >= 1:
                out.append(f"{where}: {k} is a fraction (0.05 = 5%), got {exits[k]!r}")
        if isinstance(price, (int, float)) and price > 0:
            long, stop, tp = qty > 0, exits.get("stop_price"), exits.get("take_profit_price")
            if stop is not None and (stop >= price if long else stop <= price):
                out.append(f"{where}: stop_price {stop} is on the wrong side of {price} — "
                           "it would trigger immediately")
            if tp is not None and (tp <= price if long else tp >= price):
                out.append(f"{where}: take_profit_price {tp} is on the wrong side of {price} "
                           "— it would trigger immediately")
        return out

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

        # A refusal comes back as HTTP 200 with accepted:false — per intent in orders mode,
        # for the whole book in book mode. Both used to be logged, or not even that, and then
        # ignored: the run went on to wait for fills that could never come.
        if self.mode == "orders":
            refused = list(result.get("rejected") or [])
        elif result.get("accepted") is False:
            refused = [{"symbol": "book", "reason": result.get("reason")}]
        else:
            refused = []
        for r in refused:
            log.error("refused: %s — %s", r.get("symbol"), r.get("reason"))
        book_refused = self.mode == "book" and bool(refused)

        # A name that hit a stop / take-profit / trailing stop earlier this session is held
        # flat by the executor, whatever this book asked for. Expect it flat.
        blocked = dict((result.get("exits") or {}).get("blocked") or {})
        for entry in (result.get("submitted") or []) + (result.get("noop") or []):
            blocked.update(entry.get("exits_blocked") or {})
        for sym, lock in blocked.items():
            log.warning("%s exited on its %s earlier this session — re-entry is blocked "
                        "until the next session, so it stays flat", sym, lock.get("kind"))

        # Three different things have to be true, and each was once assumed from the one
        # before it: the EXECUTOR accepted the orders (the submission above), the BROKER took
        # them (acknowledgement), and they TRADED (fills). A read-only gateway fails the second
        # while the first looks clean; a resting or cancelled order fails the third while the
        # second looks clean.
        acks = fills = None
        if self.confirm_with_broker and not book_refused:
            if not hasattr(self.client, "wait_for_acks"):
                # A client predating broker confirmation. Skipping is the only option, but it
                # must be said out loud: the run is about to report success on the executor's
                # word alone, which is the very thing this step exists to stop.
                log.warning("%s cannot confirm orders with the broker — reporting success "
                            "on the executor's acceptance alone",
                            type(self.client).__name__)
            else:
                order_ids = self.client.order_ids(result)
                acks = self.client.wait_for_acks(order_ids, timeout=self.ack_timeout)
                if acks["live"]:
                    log.info("%d order(s) confirmed at the broker", len(acks["live"]))
                acknowledged = not (acks["rejected"] or acks["pending"])
                if acknowledged and self.confirm_fills:
                    if not hasattr(self.client, "wait_for_fills"):
                        log.warning("%s cannot confirm fills — reporting success on the "
                                    "broker's acknowledgement alone",
                                    type(self.client).__name__)
                    else:
                        refused_names = {r.get("symbol") for r in refused}
                        targets = {i["instrument"]["symbol"]:
                                   0 if i["instrument"]["symbol"] in blocked
                                   else i["target_quantity"]
                                   for i in book
                                   if i["instrument"]["symbol"] not in refused_names}
                        fills = self.client.wait_for_fills(
                            targets, order_ids, authoritative=(self.mode == "book"),
                            timeout=self.fill_timeout, strategy_id=self.strategy_id)
                        if not fills["unfilled"]:
                            log.info("book in place: %d name(s) at target",
                                     len(fills["filled"]))

        self._log_pnl()

        # journal AFTER submitting, so the record reflects what was actually sent, and even
        # when the book was empty — a decision to hold nothing is still a decision
        try:
            detail = self._with_ack_detail(self.journal_detail(book), acks)
            detail = self._with_fill_detail(detail, fills)
            if blocked:
                detail += ("\n" if detail else "") + "held flat by exit lockout: " + ", ".join(
                    f"{sym} ({lock.get('kind')})" for sym, lock in sorted(blocked.items()))
            if refused:
                detail += ("\n" if detail else "") + "refused: " + ", ".join(
                    f"{r.get('symbol')} ({r.get('reason')})" for r in refused)
            self.client.journal("signal", self.describe(book), detail=detail,
                                symbols=[i["instrument"]["symbol"] for i in held])
        except ExecutorError as e:
            log.warning("submitted, but could not journal it: %s", e)

        if refused:
            log.error("%d intent(s) refused by the executor", len(refused))
            return self.EXIT_REFUSED
        if acks and (acks["rejected"] or acks["pending"]):
            log.critical("%d rejected, %d unacknowledged — the book is NOT in place at the "
                         "broker", len(acks["rejected"]), len(acks["pending"]))
            return self.EXIT_NOT_ACKED
        if fills and fills["unfilled"]:
            log.critical("%d name(s) not at target — %s. The book is NOT in place.",
                         len(fills["unfilled"]), fills["reason"])
            return self.EXIT_NOT_FILLED
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

    @staticmethod
    def _with_fill_detail(detail: str, fills: dict) -> str:
        """The fill outcome belongs in the journal for the same reason the broker's answer
        does: a run whose orders never traded must not read like one that did."""
        if not fills:
            return detail
        note = f"fills: {len(fills['filled'])} at target, {len(fills['unfilled'])} not"
        if fills.get("reason"):
            note += f" ({fills['reason']})"
        for sym, gap in fills["unfilled"].items():
            note += f"\n  {sym}: holding {gap['held']:g}, target {gap['target']:g}"
        return f"{detail}\n{note}" if detail else note

    def _log_pnl(self) -> None:
        """Realized P&L net of commissions, and the fees. Informational — a failed read must
        never change a run's outcome."""
        if self.dry_run or not hasattr(self.client, "strategy_pnl"):
            return
        try:
            p = self.client.strategy_pnl(self.strategy_id)
            log.info("realized P&L %s, net of %s in fees (gross %s)",
                     f"{p['realized']:+,.2f}", f"{p['fees']:,.2f}", f"{p['gross']:+,.2f}")
        except Exception as e:
            log.debug("could not read P&L: %s", e)

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
