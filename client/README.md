# Running a strategy off-box

Everything a strategy needs to submit orders from somewhere other than the executor's host.
Copy this directory, `pip install requests`, set two environment variables.

```
client/
  remote_strategy.py         the base class you subclass — start here
  executor_client.py         the HTTP layer it sits on; requests is the only dependency
  example_remote_strategy.py a working template you can gut and replace
  local_strategy.py          one name, one small position — run this first
  env.sh                     `source client/env.sh` to set the variables below
```

## Reaching the executor

Keep the executor bound to `127.0.0.1:8000` and put a tunnel in front of it. Do **not**
open port 8000: 18 of its endpoints need no API key, so anything that can reach the port
can read your entire book — positions, P&L, fills, journal.

**Tailscale (what this is set up for).** Install on both boxes. Joining the tailnet is not
enough on its own — uvicorn is bound to loopback, so nothing on the tailnet can reach 8000
until you put `tailscale serve` in front of it. On the executor:

```bash
sudo tailscale serve --bg 8000      # -> https://<host>.<tailnet>.ts.net, real cert, 443
tailscale serve status              # note the hostname it prints
```

uvicorn stays on 127.0.0.1, no security group changes, and the API key travels under TLS
rather than in cleartext. On the strategy host:

```bash
EXECUTOR_URL=https://executor.<your-tailnet>.ts.net
```

**SSH tunnel (nothing new to install).** From the strategy host:

```bash
ssh -N -L 8000:127.0.0.1:8000 ubuntu@executor-box     # then EXECUTOR_URL=http://127.0.0.1:8000
```

Run it under `autossh` or a systemd unit so it survives a reconnect. Either way the traffic
is encrypted, which matters: `X-API-Key` crosses the wire in cleartext.

## Environment

```bash
EXECUTOR_URL=https://executor.tail82fc30.ts.net   # tailnet address, or your tunnel
EXECUTOR_API_KEY=…                     # same key the executor checks
STRATEGY_ID=cross_sectional_momentum   # must exist in the executor's config.CONFIG

# optional, and worth setting — this is how you hear about a partition
TELEGRAM_BOT_TOKEN=…
TELEGRAM_CHAT_ID=-100…
TELEGRAM_THREAD_ERRORS=2
```

`client/env.sh` sets all of these for you, reading the key out of the repo's gitignored
`.env` rather than keeping a second copy: `source client/env.sh`.

The API key currently grants **everything**, including `/kill`, `/flatten` and now
`/strategies`. Until keys are scoped per strategy, only put it on a host you trust as much
as the executor's.

## Writing the strategy

Subclass `RemoteStrategy` and implement one method:

```python
from client.remote_strategy import RemoteStrategy

class MyStrategy(RemoteStrategy):
    strategy_id = "my_strategy"
    require_market_open = True

    def generate_book(self, capital):
        """Return the ENTIRE desired book. `capital` is the allocation the executor
        currently holds, so this follows /allocate automatically."""
        return [self.intent("AAPL", 78, 319.97)]

if __name__ == "__main__":
    raise SystemExit(MyStrategy.cli())
```

That is the whole strategy. The base class does the rest, and each piece of it is something
that is easy to get wrong once and never notice:

1. **`preflight()` before any work** — refuses to trade into an executor that is
   disconnected from IB, killed, or started degraded, so you fail before a half-book exists.
2. **Capital read from the executor** rather than a constant someone has to remember to edit.
3. **Every book validated before it is sent.** Most importantly `expected_price`: the
   executor values your book with it when applying the allocation cap, so a missing, zero,
   negative or NaN price mis-sizes the order *and* the limit meant to contain it. Futures
   legs must carry a `multiplier` for the same reason. Validation reports every problem at
   once, not the first.
4. **`submit_book()`** — one request, absolute targets, and any name you stop mentioning
   gets closed, so it self-heals drift and is safe to re-run.
5. **A journal entry on every run**, including the ones that hold nothing — those are the
   runs you cannot reconstruct afterwards.
6. **Exit codes**, below.

Hooks worth overriding: `describe()` (the journal line — record *why*), `journal_detail()`
(scores, weights), `should_run(health)` (skip cleanly), `on_submitted(result)`.

Set `mode = "orders"` to post each intent to `/orders` individually instead of submitting a
pooled book. Use `ExecutorClient` directly if you want none of this.

Stops, take-profits and trailing stops are keyword arguments on the same call —
`self.intent("AAPL", 78, 319.97, stop_pct=0.03, trail_pct=0.05)` — see
[Stops, take-profits and trailing stops](#stops-take-profits-and-trailing-stops) below.

## Scheduling

Nothing here schedules anything — use cron or a systemd timer on the strategy host. The
exit codes are meant for that:

| code | meaning |
|---|---|
| `0` | submitted, confirmed by the broker **and filled** (or deliberately skipped) |
| `1` | the executor refused, or the book was invalid |
| `2` | executor unreachable — orders did **not** go in, Telegram already alerted |
| `3` | the executor took the orders, **IB did not** |
| `4` | IB acknowledged the orders, but the book did **not** reach its targets in time |

Codes `3` and `4` are the ones worth understanding: each is a run that looks fine from one
step earlier.

**`3` — the broker refused.** A submission returning `accepted: true` only means the executor
handed the order to the socket. A gateway in read-only mode refuses every order while
submissions still come back clean, so `run()` polls `/orders/acks` and will not report success
until IB has answered. `confirm_with_broker = False` opts out; `ack_timeout` (default 30s)
sets the wait.

**`4` — acknowledged, not filled.** IB taking an order is not the order trading. It can rest
unfilled, be cancelled by a later rebalance or the end-of-day sweep, or fill in part — and a
cancelled order still reads as acknowledged. So once the broker has answered, `run()` waits for
the strategy's **own book** to reach its targets. That is the right thing to watch: the
executor moves a strategy's position only when an order that strategy owns fills, whereas a
pooled order's fill count covers every strategy it was for. In book mode a name you dropped
must reach zero. The wait stops early once no order is left working. `fill_timeout` (default
60s) sets it; a strategy that works resting limit orders should raise it or set
`confirm_fills = False`.

Realized P&L is **net of commissions** — the executor deducts each fee as IB reports it.
`client.strategy_pnl()` returns `realized`, `fees` and `gross`, and `run()` logs them.

```cron
35 15 * * 1-5  cd /home/ubuntu/strategy && ./venv/bin/python client/example_remote_strategy.py
```

## Reading the executor's state

Everything the strategy host can see, on `ExecutorClient`:

| method | endpoint | what |
|---|---|---|
| `holdings()` | `/strategies/{id}/book` | this strategy's positions — they move only on its own fills |
| `strategy_pnl()` | `/pnl` | realized P&L net of fees, the fees, and gross |
| `acks(ids)`, `wait_for_acks()` | `/orders/acks` | did the broker take these orders |
| `wait_for_fills(targets, ids)` | book + acks | did the book actually reach its targets |
| `pending()` | `/pending` | shares the executor still expects, and whether working orders explain them |
| `fills()`, `orders()` | `/fills`, `/orders` | recent fills (with commission) and working orders |
| `exposure()`, `orphans()` | `/exposure`, `/positions/orphans` | bucketed exposure; positions no strategy claims |
| `strategies()`, `strategy_status()` | `/strategies` | allocations and halt state |
| `add_strategy()`, `remove_strategy()` | `POST` / `DELETE /strategies` | register a strategy id at runtime (needs the key) |
| `exits()` | `/exits` | armed stops / take-profits / trails, their trigger levels, today's lockouts |
| `clear_exit(symbol)` | `DELETE /exits/{id}/{symbol}` | drop a name's exits and lift its lockout (needs the key) |

`mode = "orders"` posts each intent to `/orders`, and the client fills in the `intent_type` and
`order_type` the executor requires. A target the book already meets comes back as a no-op and
is not counted as a refusal; a real refusal — of one intent, or of a whole book — makes the run
exit `1`.

## Stops, take-profits and trailing stops

Optional, per name, in any combination — or none:

```python
self.intent("AAPL", 78, 319.97, stop_pct=0.03)                        # stop only
self.intent("NVDA", 40, 181.20, trail_pct=0.05)                       # trailing only
self.intent("MSFT", 20, 505.10, stop_price=490, take_profit_pct=0.08) # both
```

| field | fires when, for a long (mirrored for a short) |
|---|---|
| `stop_price` / `stop_pct` | price falls to the level |
| `take_profit_price` / `take_profit_pct` | price rises to the level |
| `trail_amount` / `trail_pct` | price falls that far from its best mark since entry |

`*_pct` values are fractions (`0.03` = 3%) of this strategy's **average cost**, which the
executor records from the actual fills. So send exits **with the entry** — there is no need to
wait for the fill to know where "3% below my fill" is, and waiting leaves the position
unprotected for however long that takes.

**Without `self.intent()`** — `ExecutorClient` directly, or anything that speaks HTTP — it is
one more key on the intent:

```python
client.submit_book([
    {"instrument": {"symbol": "AAPL", "asset_class": "equity",
                    "sec_type": "STK", "exchange": "SMART"},
     "target_quantity": 78, "expected_price": 319.97,
     "exits": {"stop_pct": 0.03, "trail_pct": 0.05}},
])
```

In orders mode the same object goes on a `target_position` intent posted to `/orders`; it is
refused on a `side` + `quantity` delta, which has no target to protect.

**Moving an exit after the fill.** Exits are resent with every book, so a later run can tighten
them. The strategy's book carries its average cost:

```python
def generate_book(self, capital):
    price = latest_price("AAPL")
    held = {r["symbol"]: r for r in self.client.book()["book"] if not r.get("is_cash")}
    exits = {"trail_pct": 0.05}
    cost = (held.get("AAPL") or {}).get("avg_cost")
    if cost and price > cost * 1.05:
        exits["stop_price"] = cost            # up 5%: never let it turn into a loss
    return [self.intent("AAPL", 78, price, **exits)]
```

**Checking what is armed.** `client.exits()` returns each name's rule, the position and average
cost it is protecting, and `levels` — the price each exit fires at right now (a `*_pct` level
appears once the name has filled; a trail's once it has been priced). Names that already
exited today are under `lockouts`.

How it behaves:

- **Each submission replaces that name's exits.** Resend them to keep them, change them to move
  them (a stop to breakeven, a tighter trail), leave them out to remove them. A trail keeps its
  best mark when resent.
- **The executor enforces them; nothing rests at IB.** A resting stop would be cancelled by
  the next rebalance, and on a pooled position it would close other strategies' shares. Every
  name with an armed exit is **re-priced from IB every 30 seconds** (equities only while the
  market is open), and a price more than 65 seconds old is never acted on. A hit sets this
  strategy's target for the name to zero and closes it with a **market order that skips ATR
  and every other execution layer** — an exit left resting as a limit is not an exit. It is
  journalled and posted to Telegram. Not tick-by-tick, so a fast gap can fill past the level,
  and nothing is protected while the executor is down.
- **Re-entry in the same direction is blocked for the rest of the session** (ET date).
  Otherwise a strategy that still likes the name buys it straight back and the stop achieved
  nothing. The book may keep asking for it; the executor holds it flat, reports it under
  `exits.blocked`, and `run()` expects it flat rather than exiting `4`. The opposite direction
  is allowed. `client.clear_exit("AAPL")` lifts a lockout by hand.
- **One unenforceable exit refuses the whole submission** — an unknown field, both forms of
  one kind, a percentage of 1 or more, a price stop already on the wrong side of the price.
  `validate()` catches these before anything is sent.
- Exits need an absolute target: book mode, or `intent_type: target_position` in orders mode.

## Proving the path before you trust it

`local_strategy.py` is a single-name strategy sized in dollars, for the first run from a new
machine. It defaults to `test_suite_small_alloc`, which config.CONFIG caps at $1,000 — the
guard rail is server-side, not a promise the script makes:

```bash
source client/env.sh
python3 client/local_strategy.py --dry-run                    # build and validate only
python3 client/local_strategy.py --symbol AAPL --notional 500 # open it
python3 client/local_strategy.py --symbol AAPL --flat         # close it again
```

Watch a full open/close cycle before pointing anything at a real strategy_id. The reason for
a dedicated id is `/targets`: the book is AUTHORITATIVE for the strategy it names, so a
one-name test book sent as a live strategy would close every other name that strategy holds.

## Adding a strategy without a restart

`config.CONFIG` is a fail-closed allowlist, so a new `strategy_id` is rejected until it
exists there. Rather than editing config.py and restarting, register it from Telegram:

```
/addstrategy pairs_v2 150k 10%      -> /confirm <token>
```

or over HTTP:

```bash
curl -X POST $EXECUTOR_URL/strategies -H "X-API-Key: $EXECUTOR_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"strategy_id":"pairs_v2","capital_allocation":150000,"max_drawdown":0.10}'
```

It persists before it takes effect and is restored at boot, so the strategy does not quietly
stop existing at the next restart. `/delstrategy` (or `DELETE /strategies/<id>`) removes one,
and refuses while it still holds a position.

## Checking it works

```bash
python3 client/example_remote_strategy.py --dry-run   # prints the book, submits nothing
```

`--dry-run` skips preflight and the allocation lookup, so it works before the tunnel is up
(set `DRY_RUN_CAPITAL` to size it).
