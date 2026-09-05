# TrainPilot Control Plane Gateway
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install uv for fast reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY examples ./examples

RUN uv sync --frozen --no-dev

ENV PYTHONPATH=/app/src \
    TRAINPILOT_BIND_HOST=0.0.0.0 \
    TRAINPILOT_HOST=0.0.0.0 \
    TRAINPILOT_PORT=28780

EXPOSE 28780

# NOTE: single-worker only (in-memory mailbox). Do not use --workers>1.
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "trainpilot.server.main:app", "--host", "0.0.0.0", "--port", "28780"]
