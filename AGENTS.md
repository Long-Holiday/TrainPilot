# AGENTS.md — TrainPilot

## Commands
- Install: `uv sync` (requires Python >=3.10, repo pins 3.13 in `.python-version`)
- Test all: `uv run pytest` (`pythonpath=["src","."]`, `testpaths=["tests"]` per `pyproject.toml`)
- Single test: `uv run pytest tests/test_mailbox.py -v` (or `-k <name>` for one case)
- E2E sim (needs server running): `./start.sh` then `uv run python examples/mock_training.py [gateway_url] [task_id]`
- Server: `./start.sh` (foreground) / `./start.sh --daemon` / `./start.sh --status` / `./start.sh --stop`; overrides: `-p <PORT>`, `-H <HOST>`
- Docker: `docker build -t trainpilot:latest .` + `docker run -d -p 28780:28780 -v $(pwd)/trainpilot.db:/app/trainpilot.db -v $(pwd)/.env:/app/.env:ro --restart always trainpilot:latest`
- No lint/typecheck/CI configured — pytest is the only verification.

## Architecture
- `src/trainpilot/server/` — FastAPI control plane; entrypoint `trainpilot.server.main:app` (`trainpilot-server`). Key files: `main.py` (mounts Streamable HTTP MCP at `/mcp`), `mcp_server.py` (6 tools: `report` [unified milestone/alert/completed/failed], `poll_instruction`, `ack_instruction`, `get_task_status`, `list_tasks`, `submit_decision`; `task_id` optional, falls back to `settings.task_id`), `mailbox.py` (in-memory scheduler + long-poll wakeup), `storage.py` (SQLite WAL), `watchdog.py` (pings each task's `gpu_host`), `ping.py` (ICMP ping helper), `config.py` (pydantic-settings, prefix `TRAINPILOT_`), `routes/`, `feishu/`.
- `src/trainpilot/agent/` — GPU-side client (`trainpilot-cli` → `agent/cli.py`): `client.py` (`TrainPilotClient`), `mcp_client.py`, `monitor.py` (`TrainingGuardian.check_and_handle_loss` freezes on NaN/OOM).
- `src/trainpilot/common/` — `states.py` (`TaskState`: RUNNING→WAITING→RESOLVED→RECOVERING→RUNNING/COMPLETED/FAILED; `ActionType`: `stop_training`/`self_resolve`/`custom`), `schemas.py`, `gateway.py` (address resolution, shared by server+agent).
- Endpoints: `/mcp` (MCP), `/docs`, `/health`, `/ready`, `/webhook/feishu`, `/api/tasks/...`.

## Gotchas
- **Single worker only.** Mailbox long-poll locks are in-process; never pass `--workers >1` to uvicorn (`start.sh` and `Dockerfile` already enforce this; scale via `task_id`-hash proxy).
- **`TRAINPILOT_HOST` has dual meaning** (`common/gateway.py`, `server/config.py`, `start.sh:resolve_bind_host`): server side = bind address (only local values `0.0.0.0`/`127.0.0.1`/`localhost` honored, public IP silently falls back to `0.0.0.0`); GPU side = gateway host. Server bind → use `TRAINPILOT_BIND_HOST`; client URL → use `TRAINPILOT_GATEWAY_URL` (highest priority) or `TRAINPILOT_HOST`+`TRAINPILOT_PORT`. Never copy a public IP into the server's `TRAINPILOT_HOST` expecting it to bind.
- **`.env` is per-machine, two modules** (`cp .env.example .env`; `start.sh` auto-creates it): Module 1 = public server, Module 2 = GPU client. Defaults: port `28780`, alert timeout `30s`, GPU ping timeout `3.0s`, long-poll `20s`. Liveness is **ping-based**: the server automatically extracts `gpu_host` by analyzing the GPU agent's network requests (or falls back to explicit `gpu_host`), and the watchdog pings it; there is no heartbeat tool or `last_heartbeat_at`.
- **Do not commit secrets/artifacts.** `.env` (has real tokens), `trainpilot.db*`, `*.log`, `.trainpilot.pid` are gitignored — never `git add` them.
- **Feishu without credentials is safe:** unconfigured → Mock mode logs cards to console. Use `TRAINPILOT_ENABLE_MOCK_FEISHU=true` for local tests.
- **Tests mutate global state:** most suites call `default_mailbox.reset()` (autouse fixture) and some override `settings.api_token=None` — always save/restore `settings` and reset mailbox; `test_e2e_simulation.py` boots live uvicorn on port `18925` and is slow.
- **Auth split:** MCP/REST use `Authorization: Bearer <TRAINPILOT_API_TOKEN>` (or `X-API-Token`); `/webhook/feishu` uses `TRAINPILOT_FEISHU_VERIFICATION_TOKEN` instead and is unaffected by API token.
