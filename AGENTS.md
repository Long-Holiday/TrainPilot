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
- `src/trainpilot/server/` — FastAPI control plane; entrypoint `trainpilot.server.main:app` (`trainpilot-server`). Key files: `main.py` (mounts Streamable HTTP MCP at `/mcp`), `mcp_server.py` (4 tools: `report` [unified milestone/alert/completed/failed], `poll_instruction`, `ack_instruction`, `get_task_status`; `task_id` optional, falls back to `settings.task_id`; no resources), `mailbox.py` (in-memory scheduler + long-poll wakeup), `storage.py` (SQLite WAL), `cleanup.py` (task & retention cleanup service), `config.py` (pydantic-settings, prefix `TRAINPILOT_`), `routes/` (operator REST only), `feishu/`.
- `src/trainpilot/agent/` — GPU-side MCP client: `mcp_client.py` (`TrainPilotMCPClient`) and `cli.py` (`trainpilot-cli`, subcommands `report`/`poll`/`ack`/`status`). There is no in-script HITL loop: the GPU-side AI agent (Claude Code / OpenCode) watches training and drives the protocol itself.
- `src/trainpilot/common/` — `states.py` (`TaskState`: RUNNING→WAITING→RESOLVED→RECOVERING→RUNNING/COMPLETED/FAILED; `ActionType`: `stop_training`/`self_resolve`/`custom`), `schemas.py`, `gateway.py` (address resolution, shared by server+agent).
- Endpoints: `/mcp` (MCP), `/docs`, `/health`, `/ready`, `/webhook/feishu`, `/api/tasks/...`.

## Gotchas
- **Single worker only.** Mailbox long-poll locks are in-process; never pass `--workers >1` to uvicorn (`start.sh` and `Dockerfile` already enforce this; scale via `task_id`-hash proxy).
- **`TRAINPILOT_HOST` has dual meaning** (`common/gateway.py`, `server/config.py`, `start.sh:resolve_bind_host`): server side = bind address (only local values `0.0.0.0`/`127.0.0.1`/`localhost` honored, public IP silently falls back to `0.0.0.0`); GPU side = gateway host. Server bind → use `TRAINPILOT_BIND_HOST`; client URL → use `TRAINPILOT_GATEWAY_URL` (highest priority) or `TRAINPILOT_HOST`+`TRAINPILOT_PORT`. Never copy a public IP into the server's `TRAINPILOT_HOST` expecting it to bind.
- **`.env` is per-machine, two modules** (`cp .env.example .env`; `start.sh` auto-creates it): Module 1 = public server, Module 2 = GPU client. Defaults: port `28780`, alert timeout `30s`, long-poll `20s`.
- **The agent-facing contract lives in exactly one place:** `mcp_server._INSTRUCTIONS` plus the four tool docstrings. README and AGENTS.md must point at it, never restate the rules — duplicated rule text is what drifts.
- **Task rows are created only by `report`/`record_event`.** Never build task state from a request path, header, or client IP; reading an unknown `task_id` must return not-found, not fabricate a `RUNNING` row.
- **Do not commit secrets/artifacts.** `.env` (has real tokens), `trainpilot.db*`, `*.log`, `.trainpilot.pid` are gitignored — never `git add` them.
- **Feishu without credentials is safe:** unconfigured → Mock mode logs cards to console. Use `TRAINPILOT_ENABLE_MOCK_FEISHU=true` for local tests.
- **Tests mutate global state:** `tests/conftest.py` only fixes `NO_PROXY`; the `default_mailbox.reset()` autouse fixture and the `settings.api_token = None` override are defined per test module, not repo-wide. Any new suite must do both, saving and restoring `settings`. `test_e2e_simulation.py` boots live uvicorn on port `18925`, runs the example in-process, and samples task state from a watcher thread to prove the HITL loop ran.
- **`examples/mock_training.py` is imported by a test** (`tests/test_e2e_simulation.py` → `run_mock_training`, resolved via `pythonpath=["src","."]`). Keep it importable and keep the `examples/` directory non-empty — the `Dockerfile` COPYs it.
- **Auth split:** MCP/REST use `Authorization: Bearer <TRAINPILOT_API_TOKEN>` (or `X-API-Token`); `/webhook/feishu` uses `TRAINPILOT_FEISHU_VERIFICATION_TOKEN` instead and is unaffected by API token.
