"""Unit and integration tests for TrainPilot MCP Server."""

import asyncio
from unittest.mock import Mock
import pytest
from trainpilot.common.states import EventType, TaskState
from trainpilot.server.mailbox import default_mailbox
from trainpilot.server.config import settings
import trainpilot.server.mcp_server as server
from trainpilot.server.mcp_server import (
    ack_instruction,
    get_task_status,
    mcp_server,
    poll_instruction,
    report,
)


@pytest.fixture(autouse=True)
def mock_notifications(monkeypatch):
    """Exercise delivery synchronously without external requests or live timers."""
    def submit(fn, *args, **kwargs):
        fn(*args, **kwargs)
        return True

    monkeypatch.setattr(server.feishu_dispatcher, "submit", submit)
    for name in (
        "send_alert",
        "send_milestone",
        "send_recovery",
        "send_outcome",
        "update_card_to_resolved",
    ):
        monkeypatch.setattr(server.default_feishu_client, name, Mock())
    monkeypatch.setattr(server, "_schedule_auto_self_resolve", Mock())


def test_report_milestone_tool():
    """Test the unified report MCP tool with default milestone event type."""
    res = report(
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
    """Test report(alert), decision injection, poll_instruction, and ack_instruction."""
    task_id = "test-mcp-alert-cycle"

    # 1. Report alert -> transitions to WAITING
    alert_res = report(
        task_id=task_id,
        message="Loss NaN encountered at step 1000",
        event_type="alert",
        step=1000,
        metrics={"loss": "NaN"},
    )
    assert alert_res["success"] is True
    assert alert_res["state"] == TaskState.WAITING.value
    assert alert_res["next_action"] == {
        "tool": "poll_instruction",
        "arguments": {"task_id": task_id},
    }
    server.default_feishu_client.send_alert.assert_called_once()
    server._schedule_auto_self_resolve.assert_called_once_with(task_id)

    # Verify task state is WAITING
    status_res = get_task_status(task_id=task_id)
    assert status_res["success"] is True
    assert status_res["task"]["state"] == TaskState.WAITING.value

    # 2. A human decision is injected (via the REST/operator surface) -> RESOLVED
    instruction = default_mailbox.submit_decision(
        task_id=task_id,
        action="self_resolve",
        operator="lead_engineer",
    )
    instruction_id = instruction.instruction_id
    assert instruction_id is not None

    # 3. Poll instruction -> consumes it and transitions to RECOVERING
    poll_res = await poll_instruction(task_id=task_id, wait_timeout=0.0)
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
    # The recovery card must carry the step/metrics from the last report: the agent is
    # no longer asked to re-send them on ack, so they can only come from the TaskRecord.
    server.default_feishu_client.send_recovery.assert_called_once_with(
        task_id=task_id,
        solution="Rolled back to step 950 and reduced learning rate to 5e-5",
        step=1000,
        epoch=None,
        metrics={"loss": "NaN"},
    )


def test_report_defaults_task_id_and_event_type():
    """Omitting task_id falls back to settings.task_id; event_type defaults to milestone."""
    old_task_id = settings.task_id
    settings.task_id = "test-default-task"
    try:
        res = report(message="Epoch 2 done", step=10)
        assert res["success"] is True
        assert res["task_id"] == "test-default-task"
        assert res["state"] == TaskState.RUNNING.value

        task = default_mailbox.get_task("test-default-task")
        assert task is not None
        assert task.latest_step == 10
    finally:
        settings.task_id = old_task_id


def test_report_rejects_invalid_event_type():
    """Invalid event_type is rejected without touching the mailbox."""
    res = report(task_id="test-bad-type", message="oops", event_type="banana")
    assert res["success"] is False
    assert "Invalid event_type" in res["error"]
    assert default_mailbox.get_task("test-bad-type") is None


def test_get_task_status_unknown_task_does_not_create_it():
    """Reading an unknown task reports not-found instead of fabricating a task row."""
    res = get_task_status(task_id="test-unknown-task")
    assert res["success"] is False
    assert res["task_id"] == "test-unknown-task"
    assert "not found" in res["error"]
    assert default_mailbox.get_task("test-unknown-task") is None
    assert default_mailbox.list_tasks() == []


@pytest.fixture(scope="module")
def mcp_tools():
    """The tool objects an MCP client actually sees, fetched once for both contract tests."""
    return {tool.name: tool for tool in asyncio.run(mcp_server.list_tools())}


def test_mcp_surface_metadata_and_guidance(mcp_tools):
    """Check metadata seen by AI clients, not just Python function docstrings."""
    assert mcp_server.name == "TrainPilot"

    instructions = mcp_server.instructions
    assert "report(event_type='alert')" in instructions
    assert "process exit status" in instructions
    assert "BEFORE repair, restart, or abandonment" in instructions
    assert "has_instruction=true" in instructions
    assert "instruction_id" in instructions
    assert "Connecting MCP does not monitor or pause GPU processes" in instructions
    # The wait choice returns a control action and must be able to end polling.
    assert "wait_for_user" in instructions
    assert "stop polling" in instructions
    assert "no automatic fallback" in instructions
    # Cancel is terminal and is the agent's verified shutdown, not the server's.
    assert "cancel only this run" in instructions
    assert "terminal FAILED" in instructions
    assert "never a recovery card" in instructions
    assert "next_task_plans" not in instructions
    assert "ack_plan" not in instructions

    assert set(mcp_tools) == {
        "report", "poll_instruction", "ack_instruction", "get_task_status",
    }
    # Behaviour rules belong to `instructions`; docstrings must point there, not restate them.
    assert "specified in the server instructions" in mcp_tools["report"].description
    assert "has_instruction" in mcp_tools["poll_instruction"].description
    assert "ACK is the one recovery report" in mcp_tools["ack_instruction"].description
    assert "instruction_id" in mcp_tools["ack_instruction"].description
    assert "instruction-mailbox summary" in mcp_tools["get_task_status"].description
    assert mcp_tools["report"].input_schema["properties"]["event_type"]["default"] == "milestone"


def test_tool_input_schemas_are_exactly_the_agent_contract(mcp_tools):
    """Pin the exposed parameter sets so signature drift cannot slip through.

    Every other test calls these tools as Python functions, which is blind to a
    leftover (or missing) parameter in the MCP schema.
    """
    expected = {
        "report": {"message", "event_type", "task_id", "step", "epoch", "metrics", "agent_note"},
        "poll_instruction": {"task_id", "wait_timeout"},
        "ack_instruction": {
            "task_id", "instruction_id", "action", "status", "solution",
            "step", "epoch", "metrics",
        },
        "get_task_status": {"task_id"},
    }
    for name, props in expected.items():
        assert set(mcp_tools[name].input_schema["properties"]) == props, name
    assert mcp_tools["report"].input_schema["required"] == ["message"]
    # An ACK may omit instruction_id for the direct-conversation path.
    assert mcp_tools["ack_instruction"].input_schema.get("required", []) == []

    # The agent surface is tools-only; the resource listings were removed.
    assert asyncio.run(mcp_server.list_resources()) == []


@pytest.mark.parametrize("message,metrics", [
    ("Loss is NaN", {"loss": "NaN"}),
    ("CUDA out of memory", {"error": "CUDA_OOM"}),
    ("Training process exited with code 1", {"exit_code": 1}),
])
def test_alert_delivery_and_next_action(message, metrics):
    result = report(
        task_id="test-agent-error", event_type="alert", message=message,
        step=42, metrics=metrics, agent_note="Observed in training output.",
    )
    assert result["success"] is True
    assert result["state"] == TaskState.WAITING.value
    assert result["next_action"] == {
        "tool": "poll_instruction",
        "arguments": {"task_id": "test-agent-error"},
    }
    assert "asynchronous and not confirmed" in result["guidance"]
    assert "has_instruction=true" in result["guidance"]
    assert "ack_instruction" in result["guidance"]
    server.default_feishu_client.send_alert.assert_called_once_with(
        task_id="test-agent-error", message=message, step=42, epoch=None,
        metrics=metrics, extra=None,
        timeout_seconds=settings.alert_decision_timeout_seconds,
        agent_note="Observed in training output.",
    )
    server.default_feishu_client.send_milestone.assert_not_called()
    server._schedule_auto_self_resolve.assert_called_once_with("test-agent-error")


def test_auto_self_resolve_callback_only_fires_past_the_timeout_window():
    """The timer callback must not resolve an alert that may still get a human decision."""
    task_id = "test-auto-resolve"
    report(task_id=task_id, message="Loss NaN", event_type="alert", step=10)

    # Alert is brand new: a 30s window has not elapsed, so nothing happens.
    server._auto_self_resolve_callback(task_id, 30)
    task = default_mailbox.get_task(task_id)
    assert task.state == TaskState.WAITING
    assert task.pending_instruction is None
    server.default_feishu_client.update_card_to_resolved.assert_not_called()

    # Past the window, the alert is auto-resolved and the Feishu card is patched.
    server._auto_self_resolve_callback(task_id, 0)
    task = default_mailbox.get_task(task_id)
    assert task.state == TaskState.RESOLVED
    assert task.pending_instruction is not None
    assert task.pending_instruction.action == "self_resolve"
    assert task.pending_instruction.payload["auto_resolved"] is True
    server.default_feishu_client.update_card_to_resolved.assert_called_once()
    assert server.default_feishu_client.update_card_to_resolved.call_args.kwargs["task_id"] == task_id


def test_auto_self_resolve_callback_skips_a_direct_conversation_wait():
    """The engineer chose ⏳ 等待用户: the 30s timer must no longer resolve the alert."""
    task_id = "test-direct-wait-auto-resolve"
    report(task_id=task_id, message="Loss NaN", event_type="alert", step=10)
    default_mailbox.wait_for_direct_conversation(task_id, "ou_engineer")

    server._auto_self_resolve_callback(task_id, 0)
    assert default_mailbox.get_task(task_id).state == TaskState.WAITING
    server.default_feishu_client.update_card_to_resolved.assert_not_called()


def test_schedule_auto_resolve_arms_the_default_window_and_can_be_disabled(monkeypatch):
    """The ordinary alert timeout is unchanged; a non-positive value disables it."""
    armed = []

    class FakeTimer:
        def __init__(self, interval, function, args=(), kwargs=None):
            armed.append((interval, function, args, kwargs or {}))
            self.daemon = False

        def start(self):
            pass

    monkeypatch.setattr(server.threading, "Timer", FakeTimer)

    server.schedule_auto_resolve("task-alert")
    assert [a[0] for a in armed] == [settings.alert_decision_timeout_seconds]
    assert armed[0][2] == ("task-alert", settings.alert_decision_timeout_seconds)

    monkeypatch.setattr(settings, "alert_decision_timeout_seconds", 0)
    server.schedule_auto_resolve("task-disabled")
    assert len(armed) == 1


def test_auto_self_resolve_callback_ignores_unknown_and_non_waiting_tasks():
    server._auto_self_resolve_callback("test-never-reported", 0)
    assert default_mailbox.get_task("test-never-reported") is None

    report(task_id="test-happy-task", message="Epoch 1 done", step=5)
    server._auto_self_resolve_callback("test-happy-task", 0)
    assert default_mailbox.get_task("test-happy-task").state == TaskState.RUNNING


@pytest.mark.parametrize("event_type,state", [
    ("milestone", TaskState.RUNNING),
    ("completed", TaskState.COMPLETED),
    ("failed", TaskState.FAILED),
])
def test_non_alert_reports_do_not_request_decision(event_type, state):
    """milestone/completed/failed are accepted and never open a decision window."""
    result = report(task_id="test-non-alert", message="Training update", event_type=event_type)
    assert result["success"] is True
    assert result["state"] == state.value
    assert "next_action" not in result
    server.default_feishu_client.send_alert.assert_not_called()
    server._schedule_auto_self_resolve.assert_not_called()


@pytest.mark.asyncio
async def test_direct_wait_returns_a_control_action_and_ack_closes_it():
    """Choosing ⏳ 等待用户 must end polling immediately, not keep it empty."""
    task_id = "test-direct-wait-mcp"
    report(task_id=task_id, message="Loss NaN", event_type="alert", step=10)
    default_mailbox.wait_for_direct_conversation(task_id, "ou_engineer")

    polled = await poll_instruction(task_id=task_id, wait_timeout=2.0)
    assert polled["has_instruction"] is True
    assert polled["action"] == "wait_for_user"
    assert polled["awaiting_direct_conversation"] is True
    assert polled["instruction_id"] is None

    # The engineer gives the solution in the conversation; the ACK is the receipt.
    acked = ack_instruction(
        task_id=task_id,
        instruction_id=None,
        action="self_resolve",
        status="success",
        solution="按对话中的方案回滚到 step 4 并继续",
        step=120,
        metrics={"loss": 0.31},
    )
    assert acked["success"] is True
    assert acked["state"] == TaskState.RUNNING.value

    task = default_mailbox.get_task(task_id)
    assert task.awaiting_direct_conversation is False
    # ACK values are persisted in the same snapshot, so the card and later reads agree.
    assert task.latest_step == 120
    assert task.latest_metrics == {"loss": 0.31}
    server.default_feishu_client.send_recovery.assert_called_once_with(
        task_id=task_id,
        solution="按对话中的方案回滚到 step 4 并继续",
        step=120,
        epoch=None,
        metrics={"loss": 0.31},
    )
    server.default_feishu_client.send_outcome.assert_not_called()


@pytest.mark.asyncio
async def test_ack_status_is_normalized_before_notifying():
    """The mailbox trims status; the notifier must use the same normalization."""
    task_id = "test-ack-status-normalized"
    report(task_id=task_id, message="Loss NaN", event_type="alert", step=10)
    default_mailbox.wait_for_direct_conversation(task_id, "ou_engineer")

    acked = ack_instruction(
        task_id=task_id,
        instruction_id=None,
        action="self_resolve",
        status=" Success ",
        solution="按对话方案恢复",
    )
    assert acked["success"] is True
    assert acked["state"] == TaskState.RUNNING.value
    server.default_feishu_client.send_recovery.assert_called_once()


@pytest.mark.asyncio
async def test_cancel_ack_is_terminal_and_notifies_once_without_a_recovery_card():
    task_id = "test-cancel-terminal-mcp"
    report(task_id=task_id, message="Loss NaN", event_type="alert", step=10)
    default_mailbox.submit_decision(task_id, action="cancel", operator="ou_engineer")
    polled = await poll_instruction(task_id=task_id, wait_timeout=0)
    assert polled["action"] == "cancel"

    acked = ack_instruction(
        task_id=task_id,
        instruction_id=polled["instruction_id"],
        action="cancel",
        status="success",
        solution="已取消本次运行并验证机器关机",
    )
    assert acked["success"] is True
    assert acked["state"] == TaskState.FAILED.value
    server.default_feishu_client.send_outcome.assert_called_once_with(
        task_id=task_id,
        event_type=EventType.FAILED,
        message="已取消本次运行并验证机器关机",
        step=None,
        epoch=None,
        metrics=None,
    )
    server.default_feishu_client.send_recovery.assert_not_called()

    # Replaying an ACK after cancellation must not reopen the terminal task or
    # send a second card.
    replay = ack_instruction(
        task_id=task_id,
        instruction_id=polled["instruction_id"],
        action="self_resolve",
        status="success",
    )
    assert replay["success"] is False
    assert "already terminal" in replay["error"]
    assert default_mailbox.get_task(task_id).state == TaskState.FAILED
    server.default_feishu_client.send_outcome.assert_called_once()
    server.default_feishu_client.send_recovery.assert_not_called()


@pytest.mark.asyncio
async def test_ack_rejects_bad_status_mismatched_action_and_replay():
    task_id = "test-ack-validation"
    report(task_id=task_id, message="Loss NaN", event_type="alert", step=10)
    default_mailbox.submit_decision(task_id, action="self_resolve", operator="ou_engineer")
    polled = await poll_instruction(task_id=task_id, wait_timeout=0)

    bad_status = ack_instruction(
        task_id=task_id,
        instruction_id=polled["instruction_id"],
        action="self_resolve",
        status="maybe",
    )
    assert bad_status["success"] is False
    assert "Invalid status" in bad_status["error"]

    mismatch = ack_instruction(
        task_id=task_id,
        instruction_id=polled["instruction_id"],
        action="cancel",
        status="success",
    )
    assert mismatch["success"] is False
    assert "Action mismatch" in mismatch["error"]

    bad_action = ack_instruction(
        task_id=task_id,
        instruction_id=polled["instruction_id"],
        action="wait_for_user",
        status="success",
    )
    assert bad_action["success"] is False
    assert "control action" in bad_action["error"]

    acked = ack_instruction(
        task_id=task_id,
        instruction_id=polled["instruction_id"],
        action="self_resolve",
        status="success",
    )
    assert acked["success"] is True
    replay = ack_instruction(
        task_id=task_id,
        instruction_id=polled["instruction_id"],
        action="self_resolve",
        status="success",
    )
    assert replay["success"] is False
    assert "already acknowledged" in replay["error"]
