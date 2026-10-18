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
from trainpilot.server.background import feishu_dispatcher
from trainpilot.server.feishu.client import default_feishu_client
from trainpilot.server.mailbox import default_mailbox
from trainpilot.server.main import app

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


@pytest.fixture(autouse=True)
def detach_real_feishu(monkeypatch):
    """Never post into a real group; start every test from an empty card log."""
    monkeypatch.setattr(default_feishu_client, "_lark_client", None)
    default_feishu_client._sent_cards_history.clear()


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
    # Plan ingestion is gone: the report response is the task state only.
    assert "next_task_plans" not in reported

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
    assert alert["next_action"] == {
        "tool": "poll_instruction",
        "arguments": {"task_id": task_id},
    }

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
    assert polled["awaiting_direct_conversation"] is False

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


def test_the_wire_surface_is_four_tools_and_has_no_plan_ingestion(live_mcp_server):
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async def _inspect():
        async with streamable_http_client(f"{live_mcp_server}/mcp") as (read, write):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                listed = await session.list_tools()
                return initialized, listed

    initialized, listed = asyncio.run(_inspect())
    assert initialized.server_info.name == "TrainPilot"
    assert {tool.name for tool in listed.tools} == {
        "report",
        "poll_instruction",
        "ack_instruction",
        "get_task_status",
    }

    instructions = initialized.instructions
    assert "wait_for_user" in instructions
    assert "no automatic fallback" in instructions
    assert "terminal FAILED" in instructions
    assert "never a recovery card" in instructions
    assert "next_task_plans" not in instructions
    assert "ack_plan" not in instructions
    assert "cancel only this run" in instructions


def test_wait_for_user_returns_control_and_closes_with_one_recovery_card(live_mcp_server):
    """⏳ 等待安排 -> wait_for_user over the wire -> one ACK, one recovery card."""
    default_mailbox.reset()
    task_id = "wire-direct-wait"

    alert = call_tool(
        live_mcp_server,
        "report",
        {
            "message": "Loss NaN encountered",
            "event_type": "alert",
            "task_id": task_id,
            "step": 150,
            "epoch": 1,
            "metrics": {"loss": "NaN"},
        },
    )
    assert alert["state"] == TaskState.WAITING.value

    # The engineer chose 「等待安排」 instead of a mailbox decision.
    default_mailbox.wait_for_direct_conversation(task_id, "ou_engineer")

    started = time.monotonic()
    polled = call_tool(
        live_mcp_server, "poll_instruction", {"task_id": task_id, "wait_timeout": 5.0}
    )
    elapsed = time.monotonic() - started
    assert elapsed < 4.0, "a direct-conversation poll must not wait for a mailbox instruction"
    assert polled["has_instruction"] is True
    assert polled["action"] == "wait_for_user"
    assert polled["awaiting_direct_conversation"] is True
    assert polled["instruction_id"] is None
    assert polled["status"] == TaskState.WAITING.value

    # The choice has no fallback: the ordinary auto-resolve timer refuses to fire.
    assert default_mailbox.try_auto_resolve(task_id, timeout_seconds=0) is None

    # The engineer gives the solution in the conversation; the ACK is the receipt.
    acked = call_tool(
        live_mcp_server,
        "ack_instruction",
        {
            "task_id": task_id,
            "action": "self_resolve",
            "status": "success",
            "solution": "按对话中的方案回滚到 step 4 并继续",
            "step": 160,
            "epoch": 2,
            "metrics": {"loss": 0.31},
        },
    )
    assert acked["success"] is True
    assert acked["state"] == TaskState.RUNNING.value

    status = call_tool(live_mcp_server, "get_task_status", {"task_id": task_id})
    assert status["task"]["state"] == TaskState.RUNNING.value
    assert status["task"]["awaiting_direct_conversation"] is False
    assert status["task"]["latest_step"] == 160
    assert status["task"]["latest_epoch"] == 2
    assert status["task"]["latest_metrics"] == {"loss": 0.31}

    assert feishu_dispatcher.wait_idle(2.0)
    history = default_feishu_client.sent_cards_history
    recovery = [
        card for card in history
        if card.get("type") == "recovery" and card.get("task_id") == task_id
    ]
    assert len(recovery) == 1
    rendered = str(recovery[0]["card"])
    assert "按对话中的方案回滚到 step 4 并继续" in rendered
    assert "Step `160`" in rendered
    assert "Epoch `2`" in rendered
    assert "0.31" in rendered
    assert not [
        card for card in history
        if card.get("type") == "failed" and card.get("task_id") == task_id
    ]


def test_cancel_ack_over_the_wire_is_terminal_failed_and_notifies_once(live_mcp_server):
    """Legacy stop_training ACK normalizes to cancel: FAILED + exactly one outcome."""
    default_mailbox.reset()
    task_id = "wire-cancel"

    call_tool(
        live_mcp_server,
        "report",
        {
            "message": "Loss exploded",
            "event_type": "alert",
            "task_id": task_id,
            "step": 10,
        },
    )
    default_mailbox.submit_decision(task_id=task_id, action="cancel", operator="ou_engineer")

    polled = call_tool(
        live_mcp_server, "poll_instruction", {"task_id": task_id, "wait_timeout": 0}
    )
    assert polled["action"] == "cancel"

    acked = call_tool(
        live_mcp_server,
        "ack_instruction",
        {
            "task_id": task_id,
            "instruction_id": polled["instruction_id"],
            "action": "stop_training",  # legacy wire alias
            "status": "success",
            "solution": "已取消本次运行并验证机器关机",
        },
    )
    assert acked["success"] is True
    assert acked["state"] == TaskState.FAILED.value

    status = call_tool(live_mcp_server, "get_task_status", {"task_id": task_id})
    assert status["task"]["state"] == TaskState.FAILED.value
    assert status["task"]["pending_instruction"] is None

    assert feishu_dispatcher.wait_idle(2.0)
    history = default_feishu_client.sent_cards_history
    outcomes = [
        card for card in history
        if card.get("type") == "failed" and card.get("task_id") == task_id
    ]
    assert len(outcomes) == 1
    rendered = str(outcomes[0]["card"])
    assert "【训练任务已失败/取消】" in rendered
    assert "已取消本次运行并验证机器关机" in rendered
    assert "已恢复正常运行" not in rendered
    assert not [
        card for card in history
        if card.get("type") == "recovery" and card.get("task_id") == task_id
    ]

    # Replaying an ACK after cancellation must not reopen the task or notify again.
    replay = call_tool(
        live_mcp_server,
        "ack_instruction",
        {
            "task_id": task_id,
            "instruction_id": polled["instruction_id"],
            "action": "cancel",
            "status": "success",
        },
    )
    assert replay["success"] is False
    assert "already terminal" in replay["error"]
    assert feishu_dispatcher.wait_idle(2.0)
    assert len([
        card for card in default_feishu_client.sent_cards_history
        if card.get("type") == "failed" and card.get("task_id") == task_id
    ]) == 1


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
    assert "wait_for_user" in initialized.instructions
    assert "cancel" in initialized.instructions
