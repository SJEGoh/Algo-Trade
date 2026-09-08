# client/env.sh — source this before running a strategy from a machine that is NOT the
# executor's host:  source client/env.sh
#
# The API key is read out of the repo's .env at source time rather than duplicated here,
# so there is one copy of it on this machine and it never lands in a file git can see.
# (.env is gitignored; this file is not, which is exactly why no secret is written in it.)

_here="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"

# The tailnet name from `tailscale serve status` on the executor. https, not http:
# serve publishes on 443 with a real cert, and the API key crosses this wire.
export EXECUTOR_URL="${EXECUTOR_URL:-https://executor.tail82fc30.ts.net}"

# tolerates the `KEY = value` spacing used in .env
_read_env() { sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*//p" "$_here/../.env" | tr -d '"'\''\r'; }

export EXECUTOR_API_KEY="$(_read_env EXECUTOR_API_KEY)"
export TELEGRAM_BOT_TOKEN="$(_read_env TELEGRAM_BOT_TOKEN)"
export TELEGRAM_CHAT_ID="$(_read_env TELEGRAM_CHAT_ID)"
export TELEGRAM_THREAD_ERRORS="$(_read_env TELEGRAM_THREAD_ERRORS)"

# Start on the id that config.CONFIG caps at $1,000. Switch to a real strategy only once
# you have watched a full open/close cycle go through.
export STRATEGY_ID="${STRATEGY_ID:-test_suite_small_alloc}"

unset _here
[ -n "$EXECUTOR_API_KEY" ] || echo "WARNING: EXECUTOR_API_KEY came back empty — check .env"
