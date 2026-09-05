"""Unit tests for states and schemas."""

import pytest
from trainpilot.common.schemas import (
    EventNotifyRequest,
    EventNotifyResponse,
    HeartbeatRequest,
    InstructionAckRequest,
    InstructionResponse,
    TaskDecisionRequest,
    TaskSummary,
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
        ready=True,
        status=TaskState.RESOLVED,
        instruction_id="inst_123",
        action="self_resolve",
        payload={"timeout_seconds": 30},
        decision_by="engineer_a",
    )
    assert resp.ready is True
    assert resp.action == "self_resolve"
    assert resp.payload["timeout_seconds"] == 30
