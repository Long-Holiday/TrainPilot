"""Integration tests for TrainPilot MCP Client and CLI on GPU nodes."""

import math
import threading
import time
import pytest
import uvicorn
from starlette.applications import Starlette

from trainpilot.agent.cli import _parse_key_value_or_json, main as cli_main
from trainpilot.agent.mcp_client import TrainPilotMCPClient, _sanitize_for_json
from trainpilot.agent.monitor import StopTrainingException, TrainingGuardian
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
    res = client.report_milestone(
        message="Step 100 milestone",
        step=100,
        epoch=1,
        metrics={"loss": 0.42},
        agent_note="Loss dropping smoothly.",
    )
    assert res["success"] is True
    assert res["state"] == TaskState.RUNNING.value

    # 2. Heartbeat
    hb_res = client.send_heartbeat(step=100, metrics={"gpu_mem": "80%"})
    assert hb_res["success"] is True

    # 3. Get task status
    status_res = client.get_task_status()
    assert status_res["success"] is True
    assert status_res["task"]["task_id"] == task_id
    assert status_res["task"]["latest_step"] == 100

    # 4. Report alert
    alert_res = client.report_alert(
        message="Loss NaN encountered",
        step=150,
        metrics={"loss": "NaN"},
    )
    assert alert_res["success"] is True
    assert alert_res["state"] == TaskState.WAITING.value

    # 5. Submit decision
    dec_res = client.submit_decision(action="self_resolve", operator="reviewer")
    assert dec_res["success"] is True
    instruction_id = dec_res["instruction"]["instruction_id"]

    # 6. Poll instruction
    poll_res = client.poll_instruction(wait_timeout=2.0, pop=True)
    assert poll_res["has_instruction"] is True
    assert poll_res["action"] == "self_resolve"
    assert poll_res["instruction_id"] == instruction_id

    # 7. Ack instruction
    ack_res = client.ack_instruction(
        action="self_resolve",
        status="success",
        solution="Restarted from checkpoint at step 100",
        instruction_id=instruction_id,
    )
    assert ack_res["success"] is True
    assert ack_res["state"] == TaskState.RUNNING.value


def test_training_guardian_with_mcp_client(live_mcp_server):
    """Test TrainingGuardian intercepting NaN and resolving via TrainPilotMCPClient."""
    task_id = "test-guardian-mcp-task"
    client = TrainPilotMCPClient(server_url=live_mcp_server, task_id=task_id)
    guardian = TrainingGuardian(client=client, poll_interval=0.5, poll_timeout=5.0)

    # In a separate thread, simulate engineer submitting decision after alert
    def _delayed_decision():
        time.sleep(1.0)
        client.submit_decision(
            action="self_resolve",
            payload={"solution": "Lowered learning rate"},
            operator="feishu_user",
        )

    t = threading.Thread(target=_delayed_decision, daemon=True)
    t.start()

    # Normal step
    guardian.check_and_handle_loss(0.5, step=1)

    # Anomaly step: Loss NaN
    guardian.check_and_handle_loss(float("nan"), step=2)

    # Verify task successfully restored to RUNNING
    status = client.get_task_status()
    assert status["task"]["state"] == TaskState.RUNNING.value


def test_cli_parser_and_subcommands(live_mcp_server):
    """Test agent CLI argument parsing and tool calls."""
    # Test key-value parser
    parsed = _parse_key_value_or_json("loss=0.45,gpu=1,status=ok")
    assert parsed == {"loss": 0.45, "gpu": 1, "status": "ok"}

    parsed_json = _parse_key_value_or_json('{"loss": 0.33}')
    assert parsed_json == {"loss": 0.33}

    # Test CLI execution for report-milestone
    code = cli_main([
        "--server-url", live_mcp_server,
        "--task-id", "test-cli-task",
        "report-milestone",
        "--message", "CLI epoch completed",
        "--step", "50",
        "--metrics", "loss=0.35",
        "--agent-note", "All systems operational.",
    ])
    assert code == 0

    # Test CLI execution for send-heartbeat
    code_hb = cli_main([
        "--server-url", live_mcp_server,
        "--task-id", "test-cli-task",
        "send-heartbeat",
        "--step", "50",
    ])
    assert code_hb == 0

    # Test CLI execution for get-status
    code_st = cli_main([
        "--server-url", live_mcp_server,
        "--task-id", "test-cli-task",
        "get-status",
    ])
    assert code_st == 0


def test_external_host_header_streamable_http_connection(live_mcp_server):
    """Verify that remote/external Host header (e.g. 35.202.16.245:28780) works over Streamable HTTP."""
    import asyncio
    import httpx2
    from mcp.client.streamable_http import streamable_http_client
    from mcp.client.session import ClientSession

    async def _test():
        async with httpx2.AsyncClient(headers={"Host": "35.202.16.245:28780"}) as http_client:
            async with streamable_http_client(f"{live_mcp_server}/mcp", http_client=http_client) as (read, write):
                async with ClientSession(read, write) as session:
                    init_res = await session.initialize()
                    assert init_res.server_info.name == "TrainPilot"

    asyncio.run(_test())
