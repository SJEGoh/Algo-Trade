# Net Pooling — the NettingCoordinator

`src/execution/netting.py`

## Why this exists

Several strategies will trade the same names during the 5-month competition. If each
strategy sends its own orders straight to the broker, two problems appear:

1. **Wasted trading.** Strategy A wants +100 MSFT, Strategy B wants −60 MSFT. Sent
   separately that's 160 shares of turnover (and commission, and slippage) to hold a net
   book of +40. Pooled, it's a single +40 order.
2. **Double exposure / risk blind spots.** The account's real position is the *sum* of
   what every strategy wants. No single strategy sees that total, so per-name risk and
   margin can't be reasoned about from any one book.

Net pooling fixes both: the coordinator keeps each strategy's **desired book**, sums them
into **one net position per symbol**, and trades the account to that net. Fills are then
**attributed back** to the strategies so per-strategy P&L and risk stay exactly correct —
including when two strategies hold opposing legs of the same name.

## The core idea in one line

> Each strategy owns a *desired book*. The broker holds *one net position* per symbol =
> the sum of the books. Trade the account to the net; decompose each fill back onto the
> strategies at the fill price.

**Invariant** (the thing the tests pin down):

```
sum over strategies of strategy_positions[strat][symbol]  ==  net position[symbol]
```

## The two ways a strategy updates its book

A strategy never sends "buy 10" / "sell 5" deltas to the coordinator. It declares **where
it wants to be** (absolute targets). That's what makes the whole thing self-correcting: a
dropped message or a missed fill is healed by the next target, because the target is the
truth, not the increment.

### 1. `set_target(sid, symbol, qty, instrument=, price=)` — incremental

Set **one** symbol's absolute target for a strategy, then re-net just that symbol. This is
the cheap, event-driven path: a strategy that reacts to a single name touches only that
name. `qty=0` is an **explicit exit** — it removes the name from the book.

```python
coord.set_target("ovn_volsurge", "MSFT", 100, instrument=inst, price=505.0)
# ovn_volsurge now wants +100 MSFT; the account is traded to the new net for MSFT only.
```

Why absolute-not-delta: the earlier design question was "why can't I just store each
strategy's desired position and receive new orders, rather than resend the whole book each
cycle?" — this is exactly that. You keep the store (`desired`) and push single-symbol
updates into it. The one rule that makes it safe is that an update is an *absolute target*,
so re-sending it is a no-op and a lost update is corrected by the next one.

### 2. `submit_book(sid, intents)` — full-book resync

The authoritative snapshot of a strategy's **entire** book. Any name in the strategy's old
book that is **absent** from the new snapshot is closed. This is the safety net for the one
thing incremental updates can't self-heal: the **stale-exit trap**.

> Stale-exit trap: an incremental strategy stops *mentioning* MSFT (it moved on) but never
> sent `MSFT=0`. Its desired book still says +100. Incremental updates will never close it,
> because nothing ever references MSFT again.

A periodic `submit_book` (say, once per bar or once a minute) fixes this: the coordinator
diffs the new book against the old, and closes MSFT because the snapshot doesn't contain
it. Run incremental for latency; run `submit_book` on a timer for correctness. You get both.

```python
coord.submit_book("orb_breakout", [
    {"instrument": aapl_inst, "target_quantity": 10, "expected_price": 200.0},
    # MSFT not present -> if orb_breakout used to hold MSFT, it is closed.
])
```

## How a rebalance works

Both entry points end in `_rebalance(symbols)`. For each affected symbol:

1. `target = net()[symbol]` — the new pooled target (sum of all desired books).
2. If `target` already equals the ledger's **effective** position (filled + pending), do
   nothing — no order. (This is why re-running the same target is a clean no-op and why the
   coordinator doesn't churn cancels.)
3. Otherwise cancel any stale in-flight order for the symbol, recompute
   `delta = target − effective_position`, and if it's non-trivial, call
   `executor.place_net_order(symbol, delta, instrument, ref_price)`.

`place_net_order` submits the pooled order under a synthetic strategy id `__net__` and
records the pending at the **net** level only (`record_net_pending`) — deliberately *not*
per strategy, because the account doesn't yet know how to split the fill. That split
happens on the fill.

## Fill attribution — the clever bit

When the net order fills, `execDetails` sees the order is flagged `net` and calls
`coordinator.attribute_fill(symbol, filled_signed, price)`. That method:

1. Computes, for every strategy, `want − have` for the symbol (`want` = desired book,
   `have` = what the strategy is currently booked at). These are the per-strategy changes
   the fill is *supposed* to deliver.
2. `total = sum(changes)`. On a **full** fill, `total == filled_signed`, so each strategy
   gets exactly its own change. On a **partial** fill, `scale = filled_signed / total`
   pro-rates every strategy's change by the same fraction.
3. For each strategy, `apply_attributed_fill(symbol, change*scale, price, strat)` books the
   sub-fill at the **fill price** — so realized P&L is correct even when strategies hold
   opposing legs (A closes into B's open at the true traded price), and reverses the net
   pending.

Edge case: if `total ≈ 0` (the desired books net out to no change but a fill still arrived,
e.g. a reconciliation artifact), the fill is booked to `__net__` rather than silently
dropped, keeping the invariant intact.

## Halting a strategy

`halt(sid)` sets the strategy's desired book to empty **but keeps the entry** in `desired`,
then rebalances the affected symbols. Keeping the (now-empty) entry matters: it means the
unwind trades still attribute back to *that* strategy's book (driving it to flat and booking
its realized P&L), instead of leaking onto `__net__`. The drawdown halt in the risk manager
composes with this — after any net fill, `execDetails` runs `check_drawdown` for each
strategy with a book, and a breach halts that strategy, whose next rebalance unwinds it.

## Risk / allocation check

Before accepting a target, the coordinator values the strategy's **whole desired book** at
gross notional (`|qty| * ref_price * multiplier`, so futures multipliers are respected) and
rejects it if it exceeds that strategy's `capital_allocation`. On rejection the book is
**reverted** to its prior state and **no order is placed** — the check is on the strategy's
own book, so pooling can never let one strategy quietly exceed its allocation by hiding
behind another's offsetting position. A strategy the risk manager has already halted
(`is_active` false) can't submit at all.

## Persistence

If constructed with `state_path`, the coordinator writes `desired`, `instrument`, and
`ref_price` to JSON on every accepted change and reloads them on start, so desired books
survive a server restart. The server wires this to `db/netting.json`.

## HTTP surface (FastAPI server)

- `POST /target` — incremental. Body: `{strategy_id, symbol, quantity, instrument?, price?}`.
- `POST /targets` — full-book resync. Body: `{strategy_id, intents:[{instrument,
  target_quantity, expected_price}]}`.
- `GET /net` — inspect the pooled net book and every strategy's desired book (read-only).

The coordinator is created in the server's lifespan and attached as `executor.coordinator`;
the executor's `place_net_order` / `execDetails` net-fill routing do the rest.

## What's tested (`tests/test_netting.py`)

- single strategy nets to its target;
- offsetting legs net to the difference **and** each strategy books its own side (invariant);
- re-running the same target is a no-op;
- a full-book resync closes a stale position (the stale-exit trap);
- halt unwinds a strategy and attributes the unwind to it, leaving others untouched;
- an over-allocation target is rejected and the book reverted (no order placed);
- a halted strategy cannot submit;
- a partial fill is split pro-rata across strategies.
