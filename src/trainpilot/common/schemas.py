"""Data contracts and Pydantic schemas for TrainPilot."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from trainpilot.common.states import ActionType, EventType, TaskState


def utc_now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


class EventNotifyRequest(BaseModel):
    """Event reported by GPU Agent to the Control Plane."""

    task_id: str = Field(..., description="Unique identifier for the training task")
    event_type: EventType = Field(..., description="Type of event: alert, milestone, heartbeat, etc.")
    message: str = Field(..., description="Human-readable event description")
    step: Optional[int] = Field(None, description="Current training step counter")
    epoch: Optional[int] = Field(None, description="Current epoch counter")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Loss, LR, accuracy, memory, etc.")
    extra: Optional[Dict[str, Any]] = Field(default=None, description="Arbitrary additional context")
    timestamp: str = Field(default_factory=utc_now_iso, description="ISO timestamp of event occurrence")


class EventNotifyResponse(BaseModel):
    """Response returned to GPU Agent upon event receipt."""

    success: bool = Field(True, description="Whether the event was successfully recorded")
    task_id: str = Field(..., description="Task identifier")
    state: TaskState = Field(..., description="Current task state after processing event")
    message: str = Field(..., description="Status summary message")
    timestamp: str = Field(default_factory=utc_now_iso)


class InstructionResponse(BaseModel):
    """Response payload when GPU Agent polls for pending instructions."""

    ready: bool = Field(..., description="True if human has made a decision, False if still waiting")
    status: TaskState = Field(..., description="Current task status in mailbox")
    instruction_id: Optional[str] = Field(None, description="Unique ID for the instruction to acknowledge")
    action: Optional[str] = Field(None, description="Action selected by human (e.g. self_resolve)")
    payload: Optional[Dict[str, Any]] = Field(default=None, description="Optional action parameters")
    decision_by: Optional[str] = Field(None, description="Operator identity who clicked the decision")
    decided_at: Optional[str] = Field(None, description="Timestamp of when decision was made")


class InstructionAckRequest(BaseModel):
    """Acknowledgment sent by GPU Agent after executing or failing an instruction."""

    task_id: str = Field(..., description="Task identifier")
    instruction_id: Optional[str] = Field(None, description="ID of the instruction executed")
    action: str = Field(..., description="The action executed")
    status: str = Field("success", description="Execution result: 'success' or 'failed'")
    message: Optional[str] = Field(None, description="Execution log or error message")
    timestamp: str = Field(default_factory=utc_now_iso)


class HeartbeatRequest(BaseModel):
    """Heartbeat reported by GPU Agent to signify liveness."""

    task_id: str = Field(..., description="Task identifier")
    step: Optional[int] = Field(None, description="Current training step")
    epoch: Optional[int] = Field(None, description="Current epoch")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Loss, GPU memory, etc.")
    timestamp: str = Field(default_factory=utc_now_iso)


class TaskDecisionRequest(BaseModel):
    """Direct decision injection request (e.g., from API, Web UI, or Feishu Webhook)."""

    task_id: str = Field(..., description="Task identifier")
    action: str = Field(..., description="Action to execute")
    payload: Optional[Dict[str, Any]] = Field(default=None, description="Optional action parameters")
    operator: Optional[str] = Field("human_operator", description="Name or ID of operator")


class FeishuCardActionPayload(BaseModel):
    """Payload embedded in Feishu interactive card button click value."""

    task_id: str = Field(..., description="Task identifier to steer")
    action: str = Field(..., description="Selected action")
    payload: Optional[Dict[str, Any]] = Field(default=None, description="Additional params")


class TaskSummary(BaseModel):
    """Complete summary of a training task tracked in the mailbox."""

    task_id: str
    state: TaskState
    created_at: str
    updated_at: str
    last_heartbeat_at: Optional[str] = None
    latest_step: Optional[int] = None
    latest_epoch: Optional[int] = None
    latest_metrics: Optional[Dict[str, Any]] = None
    latest_message: Optional[str] = None
    pending_instruction: Optional[InstructionResponse] = None
    events_count: int = 0
