# Building a remote strategy — context for a fresh session

Everything a new Claude session needs to write a strategy that runs on a laptop and executes
through the EC2 executor. Read this first; it is shorter than the code and states the things
the code cannot tell you by being read in isolation.

## The shape of the system

```
your machine                       EC2 (ubuntu@executor)
─────────────                      ─────────────────────
strategy.py                        uvicorn api.server:app  -> 127.0.0.1:8000
  └─ RemoteStrategy                  ├─ CentralExecutor ── IB Gateway ── Interactive Brokers
      └─ ExecutorClient  ──HTTPS──▶  ├─ NettingCoordinator (pools books across strategies)
         (requests)      tailnet     ├─ RiskManager (fail-closed allowlist + caps)
                                     └─ SQLite (positions, fills, allocations, journal)
```

The executor is bound to loopback and published to the tailnet by `tailscale serve`, so the
only address that works is the tailnet one. There is no public endpoint, by design: 18 of the
server's endpoints need no API key, so anything that can reach port 8000 can read the whole
book.

```bash
source client/env.sh     # EXECUTOR_URL, EXECUTOR_API_KEY (from .env), STRATEGY_ID, Telegram
curl -s $EXECUTOR_URL/health
# {"connected":true,"killed":false,"market_open":false,"startup_degraded":false}
```

If `/health` doesn't answer: `tailscale status` on this machine, then `tailscale serve status`
on the EC2 box. Everything else is downstream of those two.

## Writing one

Subclass `RemoteStrategy` ([client/remote_strategy.py](client/remote_strategy.py)) and
implement `generate_book`. That is the entire required surface.

```python
from client.remote_strategy import RemoteStrategy

class MyStrategy(RemoteStrategy):
    strategy_id = "my_strategy"        # MUST exist in the executor's config.CONFIG
    require_market_open = True

    def generate_book(self, capital):
        # `capital` is this strategy's allocation, read live from the executor
        return [self.intent("AAPL", 78, 319.97)]

if __name__ == "__main__":
    raise SystemExit(MyStrategy.cli())
```

`run()` then does, in order: preflight → `should_run()` → allocation lookup → your
`generate_book()` → `validate()` → `POST /targets` → `on_submitted()` → journal entry.

Optional hooks: `describe(book)` (the journal one-liner — record *why*), `journal_detail(book)`
(scores, weights), `should_run(health)`, `on_submitted(result)`.
Class attributes: `mode` (`"book"` → `/targets`, `"orders"` → one `POST /orders` per intent),
`require_market_open`, `dry_run_capital` (or the `DRY_RUN_CAPITAL` env var).

## The five things that actually bite

**1. The book is authoritative.** `/targets` takes the strategy's ENTIRE desired book. Any
symbol you stop mentioning is CLOSED. This is the property that makes re-running safe and
self-healing, and it is also the one that will flatten a live strategy if you send it a
one-name test book. Include evaluated-and-rejected names explicitly with `quantity=0` — same
effect, but it says so in the journal instead of being silent.

**2. `expected_price` is load-bearing, not decoration.** The executor values each leg with it
to apply the allocation cap, and measures fill slippage against it. A missing, zero, negative
or NaN price mis-sizes the order *and* the limit meant to contain it. Both `validate()` here
and `RiskManager` there fail closed on it. Pass a real, current price.

**3. Futures legs need `instrument.multiplier`.** Without it the notional is understated by
the multiplier — a 1000x understatement on CL sails straight past the allocation cap.
`self.intent(..., sec_type="FUT", exchange="NYMEX", multiplier=1000)`, and use
`GET /resolve_front/{symbol}` for the front-month contract.

**4. `strategy_id` is a fail-closed allowlist.** An id not in `config.CONFIG` has every intent
rejected as "not active". See *Adding a strategy* below — you no longer need to edit config.py
and restart.

**5. Exit codes are the interface to cron.** `0` submitted or deliberately skipped, `1` the
executor refused (config or risk — fix the caller, retrying won't help), `2` unreachable, which
means the orders did **NOT** go in and a Telegram alert has already fired. Never collapse 2
into "probably fine" — that failure mode is exactly what `ExecutorClient` exists to prevent.

## Adding a strategy id

`config.CONFIG` in [src/config.py](src/config.py) is the allowlist. To register a new id
without editing it or restarting:

```
/addstrategy pairs_v2 150k 10%     (Telegram, then /confirm <token>)
```
```bash
curl -X POST $EXECUTOR_URL/strategies -H "X-API-Key: $EXECUTOR_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"strategy_id":"pairs_v2","capital_allocation":150000,"max_drawdown":0.10}'
```

It persists to SQLite before it takes effect and is restored into CONFIG at boot, so it does
not quietly stop existing at the next restart. `/delstrategy` removes one, and refuses while
it still holds a position. Changing an EXISTING strategy's capital is a different endpoint —
`POST /strategies/{id}/allocation` — because that one knows how to raise cash by selling.

Existing ids worth knowing: `test_suite_small_alloc` ($1k cap — use this for smoke tests),
`cross_sectional_momentum` ($100k), `ovn_volsurge` / `orb_breakout` / `kalman_rrg_combined`
($200k each), `kalman_vecm` ($2m, futures).

## Endpoints you will use

| Endpoint | Auth | What |
|---|---|---|
| `GET /health` | – | `connected`, `killed`, `market_open`, `startup_degraded` |
| `GET /strategies/{id}/allocation` | – | the capital to size against |
| `GET /strategies/{id}/book` | – | the strategy's current desired book |
| `GET /positions` `/pnl` `/equity` `/fills` | – | state |
| `GET /resolve_front/{symbol}?exchange=` | – | front-month futures contract |
| `POST /targets` | key | the whole book, absolute — **use this** |
| `POST /orders` | key | one intent (a domain rejection is HTTP 200 + `accepted:false`) |
| `POST /journal` | key | decision record |
| `POST /strategies` | key | register a new strategy id |

`ExecutorClient` ([client/executor_client.py](client/executor_client.py)) wraps all of these,
retries connection errors / timeouts / 502-503-504, never retries a 4xx, and alerts Telegram
directly when it gives up — the executor's own alerter cannot report the executor being down.

## The workflow

```bash
source client/env.sh
python3 client/local_strategy.py --dry-run                     # no executor contact at all
python3 client/local_strategy.py --symbol AAPL --notional 500  # a real $500 order
python3 client/local_strategy.py --symbol AAPL --flat          # close it
./venv_algotrade/bin/python -m pytest tests/ -q                # 451 tests, ~9s
```

Do the open/close cycle on `test_suite_small_alloc` from any new machine before pointing
anything at a real strategy. The $1,000 cap on that id is enforced server-side, so it is a
guard rail rather than a promise the script makes.

`--dry-run` skips preflight and the allocation lookup entirely, so it works before the tunnel
is up. The market being closed (`market_open: false`) makes a `require_market_open` strategy
exit 0 without trading — that is a skip, not a failure.

## Where things are

| Path | What |
|---|---|
| [client/remote_strategy.py](client/remote_strategy.py) | the base class — subclass this |
| [client/executor_client.py](client/executor_client.py) | HTTP layer, retries, alerting |
| [client/local_strategy.py](client/local_strategy.py) | one-name smoke test, real money, small |
| [client/example_remote_strategy.py](client/example_remote_strategy.py) | momentum template |
| [client/env.sh](client/env.sh) | `source` it; reads the key from the gitignored `.env` |
| [client/README.md](client/README.md) | the fuller version of this document |
| [src/config.py](src/config.py) | the allowlist + `GLOBAL` circuit breakers |
| [src/api/server.py](src/api/server.py) | every endpoint |
| [src/risk/risk_manager.py](src/risk/risk_manager.py) | what rejects your order and why |
| [src/execution/netting.py](src/execution/netting.py) | how books across strategies are pooled |
| [tools/telegram_control.py](tools/telegram_control.py) | the bot; `/help` lists commands |

## Conventions in this repo

- Tests are `pytest`, live in `tests/`, and run against fakes rather than IB. New behaviour
  gets a test — look at [tests/test_add_strategy.py](tests/test_add_strategy.py) for the
  house style: the docstring says which failure would be *silent*, and the tests pin that.
- Comments explain why a thing fails closed, not what the line does.
- Anything that changes state gets a journal entry, including the decision to do nothing.
- Run everything with `./venv_algotrade/bin/python`.
- `.env` holds the secrets and is gitignored; never write a key into a tracked file.

## Known rough edges

- The API key grants everything, including `/kill`, `/flatten` and `/strategies`. It is not
  scoped per strategy. Only put it on a host you trust as much as the executor's.
- Nothing here schedules anything. Use cron or a systemd timer on an always-on host — a
  laptop that sleeps will silently miss a rebalance.
- The dashboard and 18 read endpoints are unauthenticated. That is safe only because the
  executor is reachable on the tailnet alone; it stops being safe the moment it is published.
