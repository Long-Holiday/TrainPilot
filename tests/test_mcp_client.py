"""Integration tests for TrainPilot MCP Client and CLI on GPU nodes."""

import threading
import time
import pytest
import uvicorn

from trainpilot.agent.cli import _parse_key_value_or_json, main as cli_main
from trainpilot.agent.mcp_client import TrainPilotMCPClient, _sanitize_for_json
from trainpilot.common.states import TaskState
from trainpilot.server.mailbox import default_mailbox
from trainpilot.server.main import app as server_app


@pytest.fixture(scope="module")
def live_mcp_server():
    """Run a live MCP server on a dedicated test port in a background thread."""
    port = 28795
    config = uvicorn.Config(server_app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to become responsive
    deadline = time.time() + 5.0
    import socket
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)

    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


@pytest.fixture(scope="module", autouse=True)
def decouple_env_auth():
    from trainpilot.server.config import settings
    old_token = settings.api_token
    settings.api_token = None
    yield
    settings.api_token = old_token


@pytest.fixture(autouse=True)
def clean_mailbox():
    default_mailbox.reset()
    yield
    default_mailbox.reset()


def test_sanitize_for_json():
    """Verify float sanitation handles NaN, Inf, and nested structures."""
    payload = {
        "loss": float("nan"),
        "grad_norm": float("inf"),
        "negative_inf": float("-inf"),
        "normal": 0.5,
        "nested": [float("nan"), 10],
    }
    cleaned = _sanitize_for_json(payload)
    assert cleaned["loss"] == "NaN"
    assert cleaned["grad_norm"] == "Infinity"
    assert cleaned["negative_inf"] == "-Infinity"
    assert cleaned["normal"] == 0.5
    assert cleaned["nested"] == ["NaN", 10]


def test_mcp_client_e2e_tools(live_mcp_server):
    """Test TrainPilotMCPClient invoking remote tools over SSE."""
    task_id = "test-live-gpu-task"
    client = TrainPilotMCPClient(server_url=live_mcp_server, task_id=task_id)

    # 1. Report milestone
    res = client.report(
        message="Step 100 milestone",
        event_type="milestone",
        step=100,
        epoch=1,
        metrics={"loss": 0.42},
        agent_note="Loss dropping smoothly.",
    )
    assert res["success"] is True
    assert res["state"] == TaskState.RUNNING.value

    # 2. Get task status
    status_res = client.get_task_status()
    assert status_res["success"] is True
    assert status_res["task"]["task_id"] == task_id
    assert status_res["task"]["latest_step"] == 100

    # 3. Report alert via unified report tool
    alert_res = client.report(
        message="Loss NaN encountered",
        event_type="alert",
        step=150,
        metrics={"loss": "NaN"},
    )
    assert alert_res["success"] is True
    assert alert_res["state"] == TaskState.WAITING.value

    # 4. A human decision is injected on the operator surface (REST), not by the agent
    instruction = default_mailbox.submit_decision(
        task_id=task_id, action="self_resolve", operator="reviewer"
    )
    instruction_id = instruction.instruction_id

    # 5. Poll instruction (consumes it, pop is implicit)
    poll_res = client.poll_instruction(wait_timeout=2.0)
    assert poll_res["has_instruction"] is True
    assert poll_res["action"] == "self_resolve"
    assert poll_res["instruction_id"] == instruction_id

    # 6. Ack instruction
    ack_res = client.ack_instruction(
        action="self_resolve",
        status="success",
        solution="Restarted from checkpoint at step 100",
        instruction_id=instruction_id,
    )
    assert ack_res["success"] is True
    assert ack_res["state"] == TaskState.RUNNING.value


def test_cli_parser_and_subcommands(live_mcp_server):
    """Test agent CLI argument parsing and tool calls."""
    # Test key-value parser
    parsed = _parse_key_value_or_json("loss=0.45,gpu=1,status=ok")
    assert parsed == {"loss": 0.45, "gpu": 1, "status": "ok"}

    parsed_json = _parse_key_value_or_json('{"loss": 0.33}')
    assert parsed_json == {"loss": 0.33}

    # Test CLI execution for report (milestone is the default event type)
    code = cli_main([
        "--server-url", live_mcp_server,
        "--task-id", "test-cli-task",
        "report",
        "--message", "CLI epoch completed",
        "--step", "50",
        "--metrics", "loss=0.35",
        "--agent-note", "All systems operational.",
    ])
    assert code == 0

    # Test unified report subcommand (alert event type)
    code_report = cli_main([
        "--server-url", live_mcp_server,
        "--task-id", "test-cli-task",
        "report",
        "--type", "alert",
        "--message", "CLI loss spike detected",
        "--step", "75",
        "--metrics", "loss=NaN",
    ])
    assert code_report == 0

    # Test CLI execution for status
    code_st = cli_main([
        "--server-url", live_mcp_server,
        "--task-id", "test-cli-task",
        "status",
    ])
    assert code_st == 0


def test_external_host_header_streamable_http_connection(live_mcp_server):
    """Verify that remote/external Host header (e.g. 35.202.16.245:28780) works over Streamable HTTP."""
    import asyncio
    import httpx
    from mcp.client.streamable_http import streamable_http_client
    from mcp.client.session import ClientSession

    async def _test():
        async with httpx.AsyncClient(headers={"Host": "35.202.16.245:28780"}, trust_env=False) as http_client:
            async with streamable_http_client(f"{live_mcp_server}/mcp", http_client=http_client) as (read, write):
                async with ClientSession(read, write) as session:
                    init_res = await session.initialize()
                    assert init_res.server_info.name == "TrainPilot"

    asyncio.run(_test())
