# ── algo_trade trading app (FastAPI server + executor + coordinator) ──────────
# IB Gateway runs SEPARATELY; this app connects to it at $IB_HOST:$IB_PORT.
# Build:  docker build -t algo_trade .
# Run:    docker run --rm --env-file .env \
#           -e IB_HOST=host.docker.internal -e IB_PORT=4002 \
#           -p 127.0.0.1:8000:8000 \
#           -v algo_db:/app/db -v algo_logs:/app/logs \
#           --add-host=host.docker.internal:host-gateway \   # Linux only; Docker Desktop resolves it natively
#           algo_trade
#
# Change the IB port any time with  -e IB_PORT=<port>  (4002 paper / 4001 live Gateway; 7497 TWS paper).

# ---- build stage: compile deps (ibapi's legacy setup.py needs setuptools<60) ----
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# strip the editable self-package line; pin setuptools<60 so ibapi==9.81 builds
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && grep -v '^-e ' requirements.txt > /tmp/req.txt \
    && /opt/venv/bin/pip install -r /tmp/req.txt

# ---- runtime stage: slim, no compilers, non-root ----
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app:/app/src \
    IB_HOST=host.docker.internal \
    IB_PORT=4002 \
    IB_CLIENT_ID=8 \
    EXECUTOR_URL=http://127.0.0.1:8000
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY . .
RUN mkdir -p /app/db /app/logs && chown -R app:app /app
USER app
EXPOSE 8000
# db/ holds SQLite + netting.json + vecm_state.json; logs/ the run logs — keep them on volumes
VOLUME ["/app/db", "/app/logs"]
# /health is unauthenticated; reports IB connectivity
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1
# ONE worker only — the app owns a single executor + background threads; never scale workers.
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS test
USER root
COPY --chown=app:app tests/ tests/
USER app
CMD ["pytest", "-q", "tests/"]
