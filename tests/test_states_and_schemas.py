"""Unit tests for states and schemas."""

from trainpilot.common.schemas import (
    EventNotifyRequest,
    InstructionResponse,
)
from trainpilot.common.states import ActionType, EventType, TaskState


def test_task_states_and_events():
    assert TaskState.RUNNING == "RUNNING"
    assert TaskState.WAITING == "WAITING"
    assert TaskState.RESOLVED == "RESOLVED"
    assert TaskState.RECOVERING == "RECOVERING"
    assert TaskState.COMPLETED == "COMPLETED"

    assert EventType.ALERT == "alert"
    assert EventType.MILESTONE == "milestone"
    assert [e.value for e in EventType] == ["alert", "milestone", "completed", "failed"]
    assert ActionType.STOP_TRAINING == "stop_training"
    assert ActionType.SELF_RESOLVE == "self_resolve"


def test_event_notify_serialization():
    req = EventNotifyRequest(
        task_id="task-001",
        event_type=EventType.ALERT,
        message="Loss NaN exploded",
        step=120,
        epoch=1,
        metrics={"loss": float("nan"), "lr": 1e-4},
    )
    assert req.task_id == "task-001"
    assert req.step == 120
    assert req.event_type == EventType.ALERT


def test_instruction_response_serialization():
    resp = InstructionResponse(
        has_instruction=True,
        status=TaskState.RESOLVED,
        instruction_id="inst_123",
        action="self_resolve",
        payload={"timeout_seconds": 30},
        decision_by="engineer_a",
    )
    assert resp.has_instruction is True
    assert resp.action == "self_resolve"
    assert resp.payload["timeout_seconds"] == 30


def test_instruction_response_reads_legacy_ready_json():
    """Persistence compatibility: instructions queued before the rename must still load.

    storage.py deserializes pending_instruction_json with model_validate_json and
    silently drops the instruction on failure, so an old row would otherwise
    disappear on restart.
    """
    legacy = (
        '{"ready": true, "status": "RESOLVED", "instruction_id": "inst_legacy",'
        ' "action": "self_resolve", "decision_by": "engineer_a"}'
    )
    resp = InstructionResponse.model_validate_json(legacy)
    assert resp.has_instruction is True
    assert resp.instruction_id == "inst_legacy"
    # Serialization uses the single canonical name going forward.
    assert "has_instruction" in resp.model_dump()
    assert "ready" not in resp.model_dump()
