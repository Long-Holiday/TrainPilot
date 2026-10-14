"""Unit tests for states and schemas."""

from trainpilot.common.schemas import (
    InstructionResponse,
)
from trainpilot.common.states import ActionType, EventType, PlanStatus, TaskState


def test_state_and_action_values_are_the_wire_contract():
    """These exact strings cross the MCP/REST/Feishu boundary; pin every member at once."""
    assert {s.name: s.value for s in TaskState} == {
        "RUNNING": "RUNNING",
        "WAITING": "WAITING",
        "RESOLVED": "RESOLVED",
        "RECOVERING": "RECOVERING",
        "COMPLETED": "COMPLETED",
        "FAILED": "FAILED",
    }
    assert {e.name: e.value for e in EventType} == {
        "ALERT": "alert",
        "MILESTONE": "milestone",
        "COMPLETED": "completed",
        "FAILED": "failed",
    }
    assert {a.name: a.value for a in ActionType} == {
        "STOP_TRAINING": "stop_training",
        "SELF_RESOLVE": "self_resolve",
        "CUSTOM": "custom",
    }
    assert {p.name: p.value for p in PlanStatus} == {
        "PENDING": "pending",
        "DONE": "done",
        "FAILED": "failed",
        "SKIPPED": "skipped",
    }


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
