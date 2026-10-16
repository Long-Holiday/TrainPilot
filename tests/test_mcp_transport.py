"""The wire path a native GPU-side agent actually uses.

TrainPilot ships no Python client: a GPU node installs a skill and a remote MCP entry,
and the agent's own client talks to ``/mcp``. These tests drive that endpoint with the
reference MCP SDK -- the same transport a built-in agent client uses -- so the contract
a remote agent depends on stays covered, including the external ``Host`` header a
public deployment arrives with.
"""

import asyncio
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from examples.mock_training import call_tool
from trainpilot.common.states import TaskState
from trainpilot.server.mailbox import default_mailbox
from trainpilot.server.main import app
from trainpilot.server.plans import default_plans

TEST_HOST = "127.0.0.1"
TEST_PORT = 18926
BASE_URL = f"http://{TEST_HOST}:{TEST_PORT}"


class UvicornTestServer(uvicorn.Server):
    def install_signal_handlers(self):
        pass


@pytest.fixture(scope="module")
def live_mcp_server():
    """A live control plane on a dedicated port. Auth is disabled by conftest."""
    config = uvicorn.Config(app=app, host=TEST_HOST, port=TEST_PORT, log_level="warning")
    server = UvicornTestServer(config=config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            with socket.create_connection((TEST_HOST, TEST_PORT), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)

    yield BASE_URL
    server.should_exit = True
    thread.join(timeout=2)


def test_the_tool_loop_round_trips_over_streamable_http(live_mcp_server):
    default_mailbox.reset()
    task_id = "wire-level-task"

    reported = call_tool(
        live_mcp_server,
        "report",
        {
            "message": "Step 100 milestone",
            "event_type": "milestone",
            "task_id": task_id,
            "step": 100,
            "epoch": 1,
            "metrics": {"loss": 0.42},
            "agent_note": "Loss dropping smoothly.",
        },
    )
    assert reported["success"] is True
    assert reported["state"] == TaskState.RUNNING.value

    status = call_tool(live_mcp_server, "get_task_status", {"task_id": task_id})
    assert status["success"] is True
    assert status["task"]["task_id"] == task_id
    assert status["task"]["latest_step"] == 100

    alert = call_tool(
        live_mcp_server,
        "report",
        {
            "message": "Loss NaN encountered",
            "event_type": "alert",
            "task_id": task_id,
            "step": 150,
            "metrics": {"loss": "NaN"},
        },
    )
    assert alert["success"] is True
    assert alert["state"] == TaskState.WAITING.value

    # The decision comes from the operator surface, never from the agent.
    instruction = default_mailbox.submit_decision(
        task_id=task_id, action="self_resolve", operator="reviewer"
    )

    polled = call_tool(
        live_mcp_server, "poll_instruction", {"task_id": task_id, "wait_timeout": 2.0}
    )
    assert polled["has_instruction"] is True
    assert polled["action"] == "self_resolve"
    assert polled["instruction_id"] == instruction.instruction_id

    acked = call_tool(
        live_mcp_server,
        "ack_instruction",
        {
            "task_id": task_id,
            "instruction_id": instruction.instruction_id,
            "action": "self_resolve",
            "status": "success",
            "solution": "Restarted from the checkpoint at step 100",
        },
    )
    assert acked["success"] is True
    assert acked["state"] == TaskState.RUNNING.value


def test_a_queued_plan_surfaces_on_a_report_and_is_acked_over_the_wire(live_mcp_server):
    default_mailbox.reset()
    task_id = "wire-level-plans"
    call_tool(
        live_mcp_server,
        "report",
        {"message": "started", "event_type": "milestone", "task_id": task_id, "step": 1},
    )

    plan, created = default_plans.register("重跑评测", task_id=task_id, created_by="ou_human")
    assert created

    reported = call_tool(
        live_mcp_server,
        "report",
        {"message": "still running", "event_type": "milestone", "task_id": task_id, "step": 2},
    )
    assert [item["plan_id"] for item in reported["next_task_plans"]] == [plan.plan_id]

    acked = call_tool(
        live_mcp_server,
        "ack_plan",
        {"plan_id": plan.plan_id, "status": "done", "note": "Reran the evaluation."},
    )
    assert acked["success"] is True


def test_an_external_host_header_still_connects(live_mcp_server):
    """A public deployment is reached by IP, so the Host header is never localhost."""
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async def _connect():
        async with httpx.AsyncClient(
            headers={"Host": "35.202.16.245:28780"}, trust_env=False
        ) as http_client:
            async with streamable_http_client(
                f"{live_mcp_server}/mcp", http_client=http_client
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    return await session.initialize()

    initialized = asyncio.run(_connect())
    assert initialized.server_info.name == "TrainPilot"
    assert "alert" in initialized.instructions
