# Algo-Trade

> Central execution for systematic trading strategies.

**Algo-Trade** is a Python-based trading execution platform designed to give multiple systematic strategies a single, controlled path to market. Strategies generate **order intents**; a central executor handles broker connectivity, validation, risk checks, order submission, position/ledger state, reconciliation, netting, monitoring, and operational controls.

The project is largely inspired by the idea of a centralised execution team: strategies should decide **what** they want to trade, while the execution layer decides **whether and how** those intentions can safely reach the broker.

> **Status:** Experimental / personal research project. The repository contains live-trading integrations and should not be treated as production-ready financial infrastructure without your own testing, review, and operational controls.

## Why this project?

A common failure mode in multi-strategy trading systems is letting each strategy own its own broker connection and execution state. That makes it difficult to coordinate exposure, deduplicate orders, recover after process failures, or enforce account-level safety rules.

Algo-Trade instead follows a central-executor model:

```text
                    ┌─────────────────────┐
                    │  Strategy processes  │
                    │                     │
                    └──────────┬──────────┘
                               │
                         Order Intents
                               │
                               ▼
                    ┌─────────────────────┐
                    │   FastAPI Server     │
                    │                     │
                    │ Auth / health gates │
                    │ Strategy controls   │
                    │ Reconciliation      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   CentralExecutor   │
                    │                     │
                    │ Risk • Ledger       │
                    │ Dedup • Orders      │
                    │ Recovery • Netting  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Interactive Brokers │
                    │  Gateway / TWS      │
                    └─────────────────────┘
```

The current repository includes test strategies such as cross-sectional momentum, Kalman/VECM futures trading, overnight volume-surges, opening-range breakouts, and a combined Kalman/RRG rotation strategy. 

## Core capabilities

### Centralised execution

All live strategy execution is designed around one `CentralExecutor`. The FastAPI application creates and owns that executor rather than allowing each strategy script to create its own independent execution state.

This gives the system one place to manage:

- Interactive Brokers connectivity
- order submission and status tracking
- client-order identifiers and deduplication
- internal position / average-cost state
- startup reconciliation against broker state
- strategy-level risk state
- account-level safety controls
- logging and decision journals

### Strategy-to-executor API

Strategies communicate with the executor through HTTP. The main execution endpoints include:

- `POST /orders` — submit a single order intent
- `POST /target` — set an absolute target for one symbol within a strategy
- `POST /targets` — submit an authoritative full-book target snapshot
- `POST /reconcile` — reconcile internal state with the broker
- `POST /flatten` — cancel open orders and flatten positions without enabling the kill switch
- `POST /kill` — activate the kill switch, optionally flattening positions
- `POST /reset_daily` — reset the daily portfolio circuit-breaker baseline
- `GET /health` — broker connectivity, kill-switch, and market-hours state
- `GET /positions` — current and strategy-attributed positions
- `GET /orders` — current live order state
- `GET /orders/{order_id}` — inspect a specific order
- `GET /pnl` — realized P&L by strategy
- `GET /net` — inspect pooled net exposure and desired strategy books
- `GET /strategies/{strategy_id}/status` — strategy risk status
- `GET /strategies/{strategy_id}/allocation` — strategy capital allocation and drawdown limit

Write endpoints are protected by an API key passed through the `X-API-Key` header.

## Risk and safety model

Risk controls are applied centrally rather than being left entirely to individual strategies.

### Strategy-level controls

Each configured strategy has a capital allocation and maximum drawdown. Strategy IDs are explicitly allow-listed; intents from unknown strategies are rejected rather than implicitly trusted.

The risk layer also supports strategy halting and reactivation, allowing a strategy to be stopped after a drawdown breach without taking down the entire executor.

### Portfolio-level controls

The configuration includes optional global safeguards such as:

- maximum daily portfolio loss
- maximum gross exposure across strategies
- fill-slippage alerts / halts
- pre-trade margin checks
- stale-mark protection for unrealized drawdown calculations
- automatic reconnect and recovery after unexpected Interactive Brokers disconnects

The repository's default configuration intentionally leaves some optional account-wide limits disabled until explicitly configured.

### Operational controls

The executor exposes both a **kill switch** and a **flatten** operation. The distinction is intentional:

- **Kill switch:** stops trading and can flatten the account.
- **Flatten:** cancels open orders and flattens positions without permanently setting the kill switch, so strategies can resume after review.

## Strategy modules

### Cross-sectional momentum

`models/xs_momentum.py` provides a cross-sectional momentum strategy that generates target-position intents for a defined equity universe. The runner is `run_strat.py`.

The runner supports two modes:

- `inspect` — generate intents locally without contacting the executor
- `live` — perform health / halt checks, reconcile broker state, obtain the server-authoritative allocation, generate intents, and submit them to the executor

The example universe in the runner contains large-cap U.S. equities including AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, XOM, JNJ, PG, KO, and WMT.

### Kalman VECM futures strategy

`models/vecm_strategy.py` wraps the Kalman VECM implementation under `src/vecm/` and emits target-position intents in futures contracts.

The current implementation models:

- WTI (`CL`)
- Brent (`BZ`)
- RBOB gasoline (`RB`)

It uses a Kalman-filtered VECM relationship, rolling volatility scaling, contract-aware sizing, and explicit per-leg / gross-notional risk limits. State is persisted between runs so the filter can advance incrementally from one daily cycle to the next.

### Overnight volume-surge equities

`models/equity_strategies.py` contains `OvernightVolSurgeStrategy` (`ovn_volsurge`). It looks for unusually high daily volume relative to recent history and expresses the signal as a target position. The strategy is designed around an enter-near-close / exit-at-open workflow.

### Opening-range breakout

`OrbBreakoutStrategy` (`orb_breakout`) is an intraday equities strategy based on opening-range breakout signals using 30-minute bars. It emits target positions rather than owning execution directly.

The configuration also contains an ATR-based execution layer for this strategy that can transform eligible market-order entries into pullback limit orders and cancel unfilled orders before the close.

### Kalman / RRG rotation

`models/rotation.py` implements a combined rotation strategy used by `run_rrg.py`. The example universe is a diversified cross-asset set:

```text
SPY  TLT  GLD  VEA  EEM  VNQ
DBC  SLV  LQD  HYG  SHY  IEF
```

The runner computes signals, journals the signal snapshot, and uses the `/targets` full-book endpoint so a symbol removed from the strategy's target book is automatically closed.

These strategies are meant purely for testing across different asset classes. 

## Netting and multi-strategy execution

The execution layer includes a `NettingCoordinator` for pooling strategy targets before sending residual exposure to the broker.

This makes it possible for multiple strategies to independently express desired positions while the account trades the **net** exposure. Where opposing strategy positions exist, the coordinator can internally cross those positions and only send the residual to Interactive Brokers.

The `/targets` endpoint is intentionally an authoritative full-book resynchronisation primitive: the submitted set represents the whole desired book for that strategy, so positions for names omitted from the snapshot can be closed automatically.

## Data sources

The repository currently uses several data approaches depending on the strategy:

- **Interactive Brokers** for broker connectivity, execution, and futures/equity market interaction.
- **Alpaca data** through `data/alpaca_data_provider.py` for the strategy runners that use that adapter.
- **yfinance** in several strategy adapters for historical equity / futures price data.

The project keeps data access separate from strategy logic where practical, which makes the strategy classes easier to test with injected data functions.

## Repository layout

```text
.
├── data/                  # Market-data provider adapters
├── db/                    # Runtime state / SQLite / strategy state (local)
├── deploy/                # Deployment-related files
├── models/                # Strategy models and adapters
├── src/
│   ├── api/               # FastAPI server and dashboard
│   ├── execution/         # Central executor, order flow, netting
│   ├── ledger/            # Position and P&L state
│   ├── monitoring/        # Logging and alerting
│   ├── pairs/             # Pair / relative-value components
│   ├── risk/              # Risk manager and safety checks
│   ├── vecm/              # Kalman VECM research / signal engine
│   ├── config.py          # Strategy and global risk configuration
│   ├── equity_signals.py  # Equity signal functions
│   └── indicators*.py     # Technical indicators
├── tests/                 # Automated tests and test fixtures
├── tools/                 # Utilities / operational scripts
├── main.py                # Standalone executor/recovery test harness
├── run_strat.py           # Cross-sectional momentum runner
├── run_rrg.py             # Kalman/RRG rotation runner
├── pyproject.toml         # Package metadata and core dependencies
├── requirements.txt       # Pinned runtime/dev environment
└── Dockerfile             # Container build for the FastAPI executor
```

## Requirements

The project targets **Python 3.12** in its Docker image. The package metadata declares core dependencies including:

- `ibapi`
- `pydantic`
- `pandas`
- `fastapi`
- `uvicorn`

The checked-in `requirements.txt` contains the broader pinned environment used by the repository.

You also need access to the external services used by the strategy you want to run, such as Interactive Brokers and, where applicable, Alpaca and Yahoo Finance data through the project adapters.

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/SJEGoh/Algo-Trade.git
cd Algo-Trade
```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

For the pinned repository environment:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For editable installation of the package metadata:

```bash
pip install -e .
```

### 4. Configure environment variables

Create a `.env` file in the repository root. At minimum, the executor expects an API key and Interactive Brokers connection settings.

Example:

```dotenv
# Executor API authentication
EXECUTOR_API_KEY=replace-with-a-long-random-secret

# Interactive Brokers
IB_HOST=127.0.0.1
IB_PORT=4002
IB_CLIENT_ID=8

# Strategy data provider credentials (when required)
ALPACA_KEY=your-alpaca-key
ALPACA_SECRET=your-alpaca-secret

# Optional Telegram alerting
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
TELEGRAM_CHAT_ID=your-telegram-chat-id
TELEGRAM_THREAD_ERRORS=123
TELEGRAM_THREAD_ORDERS=456
```

Do not commit `.env` or broker/API credentials. The repository's `.gitignore` already excludes `.env`, virtual environments, runtime databases, and logs.

### 5. Start Interactive Brokers Gateway / TWS

The application connects to an existing Interactive Brokers Gateway or TWS instance. The Dockerfile documents the commonly used ports:

```text
4002 = IB Gateway paper
4001 = IB Gateway live
7497 = TWS paper
```

Use paper trading while developing and testing.

### 6. Start the executor server

From the repository root:

```bash
PYTHONPATH=./src:. uvicorn api.server:app --host 127.0.0.1 --port 8000
```

The application performs startup reconciliation with Interactive Brokers and exposes the API plus the read-only dashboard at `/`.

Check the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

A healthy response reports broker connectivity, kill-switch status, and whether the market is open.

## Running strategies

### Cross-sectional momentum

Start the executor first, then run:

```bash
python run_strat.py
```

`run_strat.py` is configured for `live` mode by default. To inspect intents without network calls, change:

```python
MODE = "inspect"
```

The live flow deliberately performs health checks and broker reconciliation before generating targets and submitting intents.

### Kalman / RRG rotation

With the executor running and the strategy active:

```bash
python run_rrg.py
```

The runner checks executor health, confirms the strategy is active, obtains the allocation from the server, computes signals, journals them, and submits a full-book resynchronisation through `/targets`.

### Direct API usage

A minimal order-intent example looks like:

```bash
curl -X POST http://127.0.0.1:8000/orders \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $EXECUTOR_API_KEY" \
  -d '{
    "strategy_id": "cross_sectional_momentum",
    "client_order_id": "example-001",
    "timestamp": "2026-01-01T00:00:00+00:00",
    "schema_version": "1.0",
    "instrument": {
      "symbol": "AAPL",
      "asset_class": "equity",
      "exchange": "SMART"
    },
    "intent_type": "target_position",
    "target_quantity": 10,
    "order_type": "market",
    "expected_price": 200.0,
    "time_in_force": "day"
  }'
```

For production-like workflows, prefer strategy-level target APIs and the central executor rather than bypassing the system's reconciliation and risk gates.

## Docker

The repository includes a multi-stage Dockerfile for the FastAPI executor.

Build:

```bash
docker build -t algo_trade .
```

Run against an Interactive Brokers Gateway reachable from the host:

```bash
docker run --rm \
  --env-file .env \
  -e IB_HOST=host.docker.internal \
  -e IB_PORT=4002 \
  -p 127.0.0.1:8000:8000 \
  -v algo_db:/app/db \
  -v algo_logs:/app/logs \
  --add-host=host.docker.internal:host-gateway \
  algo_trade
```

The image exposes port `8000`, persists `/app/db` and `/app/logs` as volumes, runs as a non-root user, and uses a **single Uvicorn worker** because the application owns one shared executor and background state.

Run the container's test image with:

```bash
docker build --target test -t algo_trade:test .
docker run --rm --env-file .env algo_trade:test
```

## Testing

The project uses `pytest`.

Run the suite with:

```bash
pytest -q
```

`conftest.py` adds both the repository root and `src/` to `sys.path`, allowing the suite to import the application packages without manually setting `PYTHONPATH`.

The test suite includes coverage around execution flows, order handling, reconciliation, risk behaviour, and strategy components.

## State, persistence, and recovery

Runtime state is kept outside the package source tree under `db/` and `logs/`.

Important persistence patterns in the current implementation include:

- SQLite-backed execution / journal data
- strategy position attribution
- order history
- netting state
- persisted Kalman VECM state
- persisted rotation-strategy state

At startup the executor reconciles its internal state against the broker. This is a key design feature: the broker is treated as the source of truth for actual positions and live orders, while the application reconstructs internal state around that truth.

The repository also contains a standalone `main.py` harness specifically aimed at exercising recovery of open orders after a process restart.

## Monitoring and alerts

The monitoring layer supports Telegram alerts for operational events. Alerts are intentionally sent asynchronously so an unavailable Telegram API does not block the trading path.

The alerting system can route messages to separate Telegram topics/threads for:

- `errors`
- `orders`

Critical log records are also routed through the alerting handler.

## Design principles

### Strategies produce intent, not broker side effects

A strategy should describe its desired exposure. The execution layer is responsible for translating that desire into broker actions.

### One executor owns account state

There should be one authoritative execution/ledger state per account. This avoids double submission and conflicting local views of exposure.

### Reconcile before acting

Strategy runners should refresh internal state from broker truth before computing new targets whenever possible.

### Fail closed

Unknown strategies, missing credentials, inactive strategies, a tripped kill switch, unavailable broker connectivity, and other hard safety conditions should block trading rather than silently falling through.

### Target books are naturally idempotent

Absolute target positions make repeated strategy cycles easier to reason about than a stream of loosely coordinated buy/sell deltas. The full-book `/targets` endpoint also provides a mechanism to heal drift and close names that are no longer desired.

## Roadmap ideas

Potential directions for the project include:

- deeper broker abstraction so additional execution venues can be added cleanly
- richer backtesting and walk-forward evaluation for strategy modules
- portfolio-wide risk attribution and scenario analysis
- stronger configuration management and secret handling
- better observability around fills, slippage, and strategy contribution
- production deployment automation and service supervision
- more exhaustive integration tests against realistic broker failure modes

## Disclaimer

This software is provided for research and educational purposes. Algorithmic trading involves substantial financial risk, including the possibility of rapid and total loss of capital. Market data can be delayed, incomplete, or incorrect; broker APIs can disconnect; orders can be rejected or filled at unexpected prices; and software bugs can create unintended positions.

Do not connect this system to a live brokerage account until you understand the code, have tested it extensively in simulation / paper trading, and have implemented the operational and risk controls appropriate for your environment.

## License

No license file is currently specified in the repository. Unless and until a license is added, treat the code as **all rights reserved** and do not assume that it is licensed for redistribution or commercial use.

## Author

[Jun-En Samuel Goh](https://github.com/SJEGoh)

Repository: [github.com/SJEGoh/Algo-Trade](https://github.com/SJEGoh/Algo-Trade)
