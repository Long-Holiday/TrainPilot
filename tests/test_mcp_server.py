"""Unit and integration tests for TrainPilot MCP Server."""

import json
import pytest
from trainpilot.common.states import TaskState
from trainpilot.server.mailbox import default_mailbox
from trainpilot.server.mcp_server import (
    ack_instruction,
    get_task_status,
    list_tasks,
    mcp_server,
    poll_instruction,
    report_alert,
    report_milestone,
    resource_task_list,
    resource_task_status,
    send_heartbeat,
    submit_decision,
)


@pytest.fixture(autouse=True)
def clean_mailbox():
    """Ensure clean mailbox state for each test."""
    default_mailbox.reset()
    yield
    default_mailbox.reset()


def test_mcp_server_metadata_and_tools():
    """Verify MCP Server metadata and exposed tools."""
    assert mcp_server.name == "TrainPilot"


def test_report_milestone_tool():
    """Test report_milestone MCP tool."""
    res = report_milestone(
        task_id="test-mcp-1",
        message="Epoch 1 completed",
        step=500,
        epoch=1,
        metrics={"loss": 0.45, "val_loss": 0.48},
        agent_note="Training looks steady, recommend continuing.",
    )
    assert res["success"] is True
    assert res["task_id"] == "test-mcp-1"
    assert res["state"] == TaskState.RUNNING.value

    task = default_mailbox.get_task("test-mcp-1")
    assert task is not None
    assert task.latest_step == 500
    assert task.latest_epoch == 1


@pytest.mark.asyncio
async def test_report_alert_and_decision_lifecycle():
    """Test report_alert, submit_decision, poll_instruction, and ack_instruction tools."""
    task_id = "test-mcp-alert-cycle"

    # 1. Report alert -> transitions to WAITING
    alert_res = report_alert(
        task_id=task_id,
        message="Loss NaN encountered at step 1000",
        step=1000,
        metrics={"loss": "NaN"},
    )
    assert alert_res["success"] is True
    assert alert_res["state"] == TaskState.WAITING.value

    # Verify task state is WAITING
    status_res = get_task_status(task_id=task_id)
    assert status_res["success"] is True
    assert status_res["task"]["state"] == TaskState.WAITING.value

    # 2. Submit decision -> transitions to RESOLVED
    dec_res = submit_decision(
        task_id=task_id,
        action="self_resolve",
        operator="lead_engineer",
    )
    assert dec_res["success"] is True
    instruction_id = dec_res["instruction"]["instruction_id"]
    assert instruction_id is not None

    # 3. Poll instruction with pop=True -> transitions to RECOVERING
    poll_res = await poll_instruction(task_id=task_id, wait_timeout=0.0, pop=True)
    assert poll_res["has_instruction"] is True
    assert poll_res["action"] == "self_resolve"
    assert poll_res["instruction_id"] == instruction_id

    task = default_mailbox.get_task(task_id)
    assert task.state == TaskState.RECOVERING

    # 4. Acknowledge instruction -> transitions back to RUNNING
    ack_res = ack_instruction(
        task_id=task_id,
        instruction_id=instruction_id,
        action="self_resolve",
        status="success",
        solution="Rolled back to step 950 and reduced learning rate to 5e-5",
    )
    assert ack_res["success"] is True
    assert ack_res["state"] == TaskState.RUNNING.value

    task_after = default_mailbox.get_task(task_id)
    assert task_after.state == TaskState.RUNNING


def test_send_heartbeat_and_list_tasks():
    """Test send_heartbeat and list_tasks tools."""
    task_id = "test-mcp-hb"
    hb_res = send_heartbeat(
        task_id=task_id,
        step=200,
        metrics={"gpu_mem_used_pct": 78.5},
        status="running",
    )
    assert hb_res["success"] is True

    # List tasks
    list_res = list_tasks(limit=10)
    assert list_res["success"] is True
    assert list_res["count"] >= 1
    tids = [t["task_id"] for t in list_res["tasks"]]
    assert task_id in tids


def test_mcp_resources():
    """Test MCP resources for reading task state and list."""
    task_id = "test-mcp-res"
    report_milestone(task_id=task_id, message="Starting run", step=1)

    # Read status resource
    raw_status = resource_task_status(task_id=task_id)
    status_json = json.loads(raw_status)
    assert status_json["task_id"] == task_id
    assert status_json["state"] == TaskState.RUNNING.value

    # Read list resource
    raw_list = resource_task_list()
    list_json = json.loads(raw_list)
    assert isinstance(list_json, list)
    assert any(t["task_id"] == task_id for t in list_json)
