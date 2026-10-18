# AGENTS.md — TrainPilot

## Commands
- Install (control plane): `uv sync` (requires Python >=3.10; the repo pins 3.13 in `.python-version`). The GPU integration needs only a skill + remote MCP config: `python3 -I -S scripts/install_agent.py install` (stdlib only, no package install), or `agy plugin install https://github.com/Long-Holiday/TrainPilot`.
- Test all: `uv run pytest` (`pythonpath=["src","."]`, `testpaths=["tests"]` per `pyproject.toml`).
- Single test: `uv run pytest tests/test_mailbox.py -v` (or `-k <name>` for one case).
- E2E sim (needs the server running): `./start.sh`, then `uv run python examples/mock_training.py [gateway_url] [task_id]`.
- Regenerate the checked-in Antigravity plugin assets (stdlib only, no venv): `python3 -I -S scripts/build_plugins.py`.
- GPU install without installing packages: `python3 -I -S scripts/install_agent.py install`.
- Server: `./start.sh` (foreground) / `--daemon` / `--status` / `--stop`; overrides `-p <PORT>`, `-H <HOST>`.
- Docker: `docker build -t trainpilot:latest .` and run it with port `28780` and read-only `.env`.
- No lint/typecheck/CI configured — pytest is the only verification.

## Architecture
- `src/trainpilot/server/` — FastAPI control plane; entrypoint `trainpilot.server.main:app` (`trainpilot-server`). Key files: `main.py` (mounts Streamable HTTP MCP at `/mcp`), `mcp_server.py` (four tools: `report` for milestone/alert/completed/failed, `poll_instruction`, `ack_instruction`, `get_task_status`; `task_id` is optional and falls back to `settings.task_id`; no resources), `mailbox.py` (in-memory scheduler + long-poll wakeup), `storage.py` (SQLite WAL), `cleanup.py` (task & retention cleanup), `config.py` (pydantic-settings, prefix `TRAINPILOT_`), `routes/` (operator REST and the Feishu card webhook), `feishu/` (card builders and client; no inbound message parsing).
- `src/trainpilot/agent/` — GPU-side integration, and nothing else: the package is the control plane. The GPU node installs a skill plus one remote MCP entry; its native MCP client talks to `/mcp`. `agent/__init__.py` is a docstring only; everything here is `plugin/`.
- `src/trainpilot/agent/plugin/` — the GPU-side agent plugin, for Antigravity CLI only. `catalog.py` holds vendor facts; `templates.py` holds the only copy of the skill/rules prose; `render.py` is pure (context in, `Bundle` out); `install.py` writes into TrainPilot's own staging directory; `commands.py` is the `plugin` command group used by `scripts/install_agent.py`; `release.py` derives the checked-in `plugins/trainpilot/` from the same renderer.
- `src/trainpilot/common/` — `states.py` (`TaskState`: RUNNING→WAITING→RESOLVED→RECOVERING→RUNNING/COMPLETED/FAILED; `ActionType`: `cancel`/`self_resolve`/`custom`; `WAIT_FOR_USER` is the card-only direct-conversation control that keeps the task `WAITING`), `schemas.py`, `gateway.py` (address resolution, shared by server+agent).
- Endpoints: `/mcp` (MCP), `/docs`, `/health`, `/ready`, `/webhook/feishu`, `/api/tasks/...`.

## Contracts and Gotchas
- **Antigravity is the only supported GPU-side agent, and it spells "remote MCP server" as `serverUrl`; `url` and `httpUrl` are rejected.** `catalog.py` records that spelling and `tests/test_plugin_catalog.py` pins the manifest shape. Do not reintroduce another vendor's spelling "for compatibility".
- **The published plugin is the generated tree `plugins/trainpilot/`.** `scripts/build_plugins.py` regenerates it from the same renderer as the installer; do not hand-edit the checked-in assets. It must stay plugin-assets-only and credential-free. `tests/test_plugin_bootstrap.py` pins the tree as exactly the generator's output.
- **`commands/*.md` ships even though Antigravity's plugin page does not list a commands component.** `agy plugin install` processes them and converts them to skills named `<plugin>-cmd-<command>`. Argument substitution is undocumented, so each prompt also names its argument in words next to `$ARGUMENTS`.
- **The plugin must not restate the supervision protocol.** The contract lives in `mcp_server._INSTRUCTIONS` plus the tool docstrings. The skill and rules use the agent's native MCP connection and initialization `instructions` and name no TrainPilot CLI. Keep installation imports stdlib-only and test with `python -I -S`.
- **The installer writes only where the plugin is TrainPilot's own**: it renders into `~/.trainpilot/plugins/trainpilot`, then hands that directory to `agy plugin install`, which owns the copy under `~/.gemini/config/plugins/trainpilot`. Tests run under a throwaway `$HOME` with `agy` stubbed — never point them at a real one.
- **Server dependencies live in the `server` extra and the `server` dependency group.** The base dependency list is empty on purpose: the GPU side installs no Python. `tests/test_plugin_packaging.py` keeps the two lists equal; `start.sh` and the `Dockerfile` pass `--group server` explicitly.
- **Single worker only.** Mailbox long-poll locks are in-process; never pass `--workers > 1` to uvicorn. Scale via a `task_id`-hash proxy.
- **`TRAINPILOT_HOST` has dual meaning**: server side = bind address (only local values honored; use `TRAINPILOT_BIND_HOST`); GPU side = gateway host (or use `TRAINPILOT_GATEWAY_URL`).
- **`.env` is per-machine and has two modules** (`cp .env.example .env`; `start.sh` auto-creates it): public server and GPU client. Defaults: port `28780`, alert decision timeout `30s`, long-poll `20s`. Never read or commit a real `.env`.
- **The agent-facing contract lives in exactly one place:** `mcp_server._INSTRUCTIONS` plus the four tool docstrings. README and AGENTS.md point at it, never restate the rules.
- **No plan queue and no group-message ingestion.** `routes/webhook.py` consumes only card actions; message events are ignored. There is no `ack_plan`, `next_task_plans`, `#fix`/`#feat` handling, plan registry, or auto-resolve fallback while the engineer is in direct conversation.
- **Wait for user:** the button keeps the task `WAITING`; `poll_instruction` returns `awaiting_direct_conversation=true` and the agent stops polling. After the engineer replies directly, the agent ACKs `self_resolve` **without `instruction_id`** (optionally with `solution`/`step`/`epoch`/`metrics`); that ACK is the single recovery report.
- **Cancel:** the button means cancel this run and shut the machine down. The agent performs and verifies both, then ACKs success. A successful cancel ACK is terminal `FAILED` and sends one cancellation notification; do not send a separate failed report. The legacy `stop_training` wire name is accepted as an alias.
- **Terminal reports notify directly:** `report(completed)` / `report(failed)` each send one notification, with no follow-up report. A successful `self_resolve`/`custom` ACK sends one recovery card.
- **A card callback must always answer HTTP 200.** Feishu maps any non-200 to the opaque client code `200671`, so refusals are a `200` with an error toast, never a 4xx. Only the v2 `card.action.trigger` payload carries the Verification Token; `/feishu` and `/feishu/` are both routed.
- **Task rows are created only by `report`.** Never build task state from a request path, header, or client IP; reading an unknown `task_id` returns not-found.
- **Do not commit secrets/artifacts.** `.env` (real tokens), `trainpilot.db*`, `*.log`, `.trainpilot.pid` are gitignored — never `git add` them.
- **Feishu without credentials is safe:** unconfigured → Mock mode logs cards to console. Use `TRAINPILOT_ENABLE_MOCK_FEISHU=true` for local tests.
- **Test isolation is centralized in `tests/conftest.py`**: it redirects `settings.sqlite_path` to a temp DB before `trainpilot.server.mailbox` is imported, then resets `default_mailbox` and nulls tokens around each test. The default relative `trainpilot.db` must never be used by a test run.
- **`TaskMailboxManager` exposes only async long polling** (`get_instruction_async`). Callers inside sync tests must `await` it.
- **`examples/mock_training.py` is imported by tests** and drives the tools through the reference MCP SDK. Keep it importable and keep `examples/` non-empty (the `Dockerfile` COPYs it).
- **Auth split:** MCP/REST use `Authorization: Bearer <TRAINPILOT_API_TOKEN>` (or `X-API-Token`); `/webhook/feishu` uses `TRAINPILOT_FEISHU_VERIFICATION_TOKEN` and is unaffected by the API token.
