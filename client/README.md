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

## Scheduling

Nothing here schedules anything — use cron or a systemd timer on the strategy host. The
exit codes are meant for that: `0` submitted (or deliberately skipped), `1` the executor
refused or the book was invalid, `2` unreachable — orders did **not** go in, and a Telegram
alert has already gone out.

```cron
35 15 * * 1-5  cd /home/ubuntu/strategy && ./venv/bin/python client/example_remote_strategy.py
```

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
