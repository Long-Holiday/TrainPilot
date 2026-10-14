"""Unit and integration tests for TrainPilot MCP Server."""

import asyncio
from unittest.mock import Mock
import pytest
from trainpilot.common.states import TaskState
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
    for name in ("send_alert", "send_milestone", "send_recovery", "update_card_to_resolved"):
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
    assert "next_task_plans" in instructions
    assert "ack_plan" in instructions

    assert set(mcp_tools) == {
        "report", "poll_instruction", "ack_instruction", "ack_plan", "get_task_status",
    }
    # Behaviour rules belong to `instructions`; docstrings must point there, not restate them.
    assert "specified in the server instructions" in mcp_tools["report"].description
    assert "has_instruction" in mcp_tools["poll_instruction"].description
    assert "after you have executed it" in mcp_tools["ack_instruction"].description
    assert "instruction_id" in mcp_tools["ack_instruction"].description
    assert "specified in the server instructions" in mcp_tools["ack_plan"].description
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
        "ack_instruction": {"task_id", "instruction_id", "action", "status", "solution"},
        "ack_plan": {"plan_id", "status", "note"},
        "get_task_status": {"task_id"},
    }
    for name, props in expected.items():
        assert set(mcp_tools[name].input_schema["properties"]) == props, name
    assert mcp_tools["report"].input_schema["required"] == ["message"]
    assert mcp_tools["ack_plan"].input_schema["required"] == ["plan_id"]

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


def test_auto_self_resolve_callback_alert_stage_skips_a_deferred_task():
    """A human asked for time: the 30s timer must no longer resolve the alert."""
    task_id = "test-deferred-auto-resolve"
    report(task_id=task_id, message="Loss NaN", event_type="alert", step=10)
    default_mailbox.defer_alert(task_id, "ou_engineer")

    server._auto_self_resolve_callback(task_id, 0)
    assert default_mailbox.get_task(task_id).state == TaskState.WAITING
    server.default_feishu_client.update_card_to_resolved.assert_not_called()


def test_auto_self_resolve_callback_arrangement_stage_resolves_the_fallback():
    task_id = "test-arrangement-auto-resolve"
    report(task_id=task_id, message="Loss NaN", event_type="alert", step=10)

    # Not deferred yet: the arrangement stage has no window to close.
    server._auto_self_resolve_callback(task_id, 0, stage="arrangement")
    assert default_mailbox.get_task(task_id).state == TaskState.WAITING

    default_mailbox.defer_alert(task_id, "ou_engineer")
    # 0 = the window has elapsed, without waiting 30 minutes for it
    server._auto_self_resolve_callback(task_id, 0, stage="arrangement")

    task = default_mailbox.get_task(task_id)
    assert task.state == TaskState.RESOLVED
    assert task.pending_instruction.payload["arrangement"] is True
    assert "等待安排" in task.pending_instruction.decision_by
    # The card must not claim a 30-second timeout when the window is the long one
    patch_kwargs = server.default_feishu_client.update_card_to_resolved.call_args.kwargs
    assert "超过等待安排窗口" in patch_kwargs["trigger_note"]
    assert "30 秒" not in patch_kwargs["trigger_note"]
    # …and the alert-stage path keeps the default wording (no trigger_note)
    assert server.default_feishu_client.update_card_to_resolved.call_count == 1


def test_schedule_auto_resolve_picks_the_stage_window(monkeypatch):
    """Neither stage may arm a real timer here; capture what would have been armed."""
    armed = []

    class FakeTimer:
        def __init__(self, interval, function, args=(), kwargs=None):
            armed.append((interval, function, args, kwargs or {}))
            self.daemon = False

        def start(self):
            pass

    monkeypatch.setattr(server.threading, "Timer", FakeTimer)

    server.schedule_auto_resolve("task-alert", stage="alert")
    server.schedule_auto_resolve("task-arranged", stage="arrangement")
    assert [a[0] for a in armed] == [
        settings.alert_decision_timeout_seconds,
        settings.arrangement_timeout_seconds,
    ]
    assert [a[3]["stage"] for a in armed] == ["alert", "arrangement"]

    # A non-positive window disables that stage instead of arming an instant timer
    monkeypatch.setattr(settings, "arrangement_timeout_seconds", 0)
    server.schedule_auto_resolve("task-disabled", stage="arrangement")
    assert len(armed) == 2


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


def test_report_carries_only_matching_pending_plans():
    """Global plans reach every task; a bound plan only reaches its own task."""
    global_plan, _ = server.default_plans.register("用 5e-5 重跑", created_by="ou_human")
    bound_plan, _ = server.default_plans.register("评测 checkpoint", task_id="task-a")
    other_plan, _ = server.default_plans.register("别的任务", task_id="task-b")

    result = report(task_id="task-a", message="step 10 done", step=10)
    assert [p["plan_id"] for p in result["next_task_plans"]] == [
        global_plan.plan_id,
        bound_plan.plan_id,
    ]
    assert other_plan.plan_id not in [p["plan_id"] for p in result["next_task_plans"]]

    entry = result["next_task_plans"][1]
    assert set(entry) == {"plan_id", "text", "task_id", "created_by", "created_at"}
    assert entry["text"] == "评测 checkpoint"
    assert entry["task_id"] == "task-a"
    assert "next_task_plans_guidance" in result


@pytest.mark.parametrize("event_type", ["milestone", "completed", "failed"])
def test_reports_without_plans_carry_an_empty_list(event_type):
    """`[]` is the explicit 'I checked, nothing queued' signal; no guidance key."""
    result = report(task_id="test-no-plans", message="Training update", event_type=event_type)
    assert result["next_task_plans"] == []
    assert "next_task_plans_guidance" not in result


def test_alert_carries_plans_alongside_the_alert_contract():
    """Plan keys must not displace the alert-specific next_action/guidance pair."""
    server.default_plans.register("重跑评测", task_id="test-alert-with-plan")
    result = report(task_id="test-alert-with-plan", message="Loss NaN", event_type="alert", step=3)
    assert result["state"] == TaskState.WAITING.value
    assert result["next_action"] == {
        "tool": "poll_instruction",
        "arguments": {"task_id": "test-alert-with-plan"},
    }
    assert len(result["next_task_plans"]) == 1
    assert "Alert recorded" in result["guidance"]


def test_ack_plan_removes_the_plan_from_later_reports():
    plan, _ = server.default_plans.register("跑评测", task_id="test-ack-plan")

    assert len(report(task_id="test-ack-plan", message="m1")["next_task_plans"]) == 1

    ack = server.ack_plan(plan_id=plan.plan_id, status="done", note="已跑完评测")
    assert ack["success"] is True
    assert ack["status"] == "done"
    assert ack["acked_at"]

    after = report(task_id="test-ack-plan", message="m2")
    assert after["next_task_plans"] == []
    assert "next_task_plans_guidance" not in after


def test_ack_plan_rejects_unknown_duplicate_and_bad_status():
    bad_status = server.ack_plan(plan_id="plan_missing", status="whatever")
    assert bad_status["success"] is False
    assert "Invalid status 'whatever'" in bad_status["error"]

    unknown = server.ack_plan(plan_id="plan_missing")
    assert unknown["success"] is False
    assert "not found" in unknown["error"]

    plan, _ = server.default_plans.register("一次性任务", task_id="test-ack-plan-twice")
    assert server.ack_plan(plan_id=plan.plan_id)["success"] is True
    duplicate = server.ack_plan(plan_id=plan.plan_id)
    assert duplicate["success"] is False
    assert "already acknowledged" in duplicate["error"]
