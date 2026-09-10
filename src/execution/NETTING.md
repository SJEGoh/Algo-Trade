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

Both entry points end in `_rebalance(symbols)`. **No strategy position moves here** — positions
move only when the broker reports a fill (next section). For each affected symbol:

1. `target = net()[symbol]` — the new pooled target (sum of all desired books).
2. If `target` already equals the ledger's **effective** position (filled + pending), place
   nothing — re-running the same target is a clean no-op. The one exception: strategies whose
   gaps cancel out *exactly* (s1 wants +60, s2 wants −60) with no working order to settle
   them. A single order for zero shares would never fill, so neither position could ever
   move; the buyers and the sellers are sent to IB as **two orders**, each owned by its side.
3. Otherwise cancel any stale in-flight order for the symbol, recompute
   `delta = target − effective_position`, snapshot each strategy's gap (`want − filled`), call
   `executor.place_net_order(symbol, delta, instrument, ref_price)`, and **record those gaps
   as the order's owners** against its order id.

`place_net_order` submits the pooled order under the synthetic id `__net__` and records the
pending at the **net** level only (`record_net_pending`). Opposing legs still net into one
order — s1 +100 and s2 −60 send +40 — and each strategy books its own side when that order
fills, at the real fill price.

**Internal crossing is off by default.** It booked offsetting legs against each other at the
reference price the moment an order was made, with no fill, which broke the rule that a
position moves only on a fill. It remains available as `NettingCoordinator(...,
internal_crossing=True)`, and `tests/test_internal_crossing.py` pins the algorithm with it on.

## Fill attribution — to the order's owners

When a net order fills, `execDetails` calls
`coordinator.attribute_fill(symbol, filled_signed, price, order_id=...)`. That method uses the
owners **frozen when the order was placed**:

1. `total = sum(owner gaps)`; `scale = filled_signed / total`. A full fill gives each owner
   exactly its gap; a partial fill pro-rates every owner by the same fraction.
2. For each owner, `apply_attributed_fill(symbol, gap*scale, price, strat)` books the
   sub-fill at the **fill price** and reverses the net pending.
3. Once the order's full quantity has filled, its owner record is dropped.

Why frozen rather than recomputed at fill time: attribution used to re-derive owners from the
live desired book when the fill arrived, so any strategy with an open gap in the symbol could
claim it. A fill from strategy B's order landed on strategy A's fresh target — A's position
moved before A's own order had filled — and one IB sale of 14 MSFT was booked 10 to
`kalman_vecm`, a futures strategy. A cancelled order that fills anyway (the cancel lost the
race) now also books to the strategies it was actually for.

The owner map is **in memory only**. IB reuses order ids across restarts, so a persisted entry
could attach to an unrelated order next session. An order with no record — placed before a
restart — falls back to the live desired book, and the log says so.

Edge case: an order nobody owned (e.g. closing a position no strategy holds), or a fill whose
owners net to zero, is booked to `__net__` rather than dropped, keeping the invariant intact.

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

Owner-based attribution — positions move only on the owning order's fill, late fills on
cancelled orders, exact offsets as two orders — is in `tests/test_fill_owned_attribution.py`.
