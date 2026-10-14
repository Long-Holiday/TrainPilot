"""Data contracts and Pydantic schemas for TrainPilot."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import AliasChoices, BaseModel, Field

from trainpilot.common.states import EventType, PlanStatus, TaskState


def utc_now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


class EventNotifyRequest(BaseModel):
    """Event reported by GPU Agent to the Control Plane."""

    task_id: str = Field(..., description="Unique identifier for the training task")
    event_type: EventType = Field(..., description="Type of event: alert, milestone, completed, failed")
    message: str = Field(..., description="Human-readable event description")
    step: Optional[int] = Field(None, description="Current training step counter")
    epoch: Optional[int] = Field(None, description="Current epoch counter")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Loss, LR, accuracy, memory, etc.")
    extra: Optional[Dict[str, Any]] = Field(default=None, description="Arbitrary additional context")
    agent_note: Optional[str] = Field(
        default=None,
        description="外部 AI 智能体 (agy/opencode/claude-code) 针对当前实际情况自主生成的阶段点评, "
        "将渲染到飞书里程碑卡片的 🤖 Agent 智能点评 区块",
    )
    timestamp: str = Field(default_factory=utc_now_iso, description="ISO timestamp of event occurrence")


class InstructionResponse(BaseModel):
    """Response payload when GPU Agent polls for pending instructions."""

    has_instruction: bool = Field(
        ...,
        validation_alias=AliasChoices("has_instruction", "ready"),
        description="True if human has made a decision, False if still waiting",
    )
    status: TaskState = Field(..., description="Current task status in mailbox")
    instruction_id: Optional[str] = Field(None, description="Unique ID for the instruction to acknowledge")
    action: Optional[str] = Field(None, description="Action selected by human (e.g. self_resolve)")
    payload: Optional[Dict[str, Any]] = Field(default=None, description="Optional action parameters")
    decision_by: Optional[str] = Field(None, description="Operator identity who clicked the decision")
    decided_at: Optional[str] = Field(None, description="Timestamp of when decision was made")


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


class NextTaskPlan(BaseModel):
    """A 'next task' queued by a human through the Feishu group.

    Plans are read by the GPU Agent from every `report` response and stay pending --
    and therefore keep reappearing -- until acknowledged with `ack_plan` or cleared.
    A plan never creates or requires a task row: `task_id` is None for a global plan
    and otherwise only a routing filter for which reports carry it.
    """

    plan_id: str = Field(..., description="Unique ID used to acknowledge the plan")
    text: str = Field(..., description="Plan body as typed in Feishu, task tag stripped")
    task_id: Optional[str] = Field(None, description="Bound task id; None means every task")
    status: PlanStatus = Field(default=PlanStatus.PENDING, description="Pending until acknowledged")
    created_at: str = Field(default_factory=utc_now_iso, description="ISO timestamp of registration")
    created_by: Optional[str] = Field(None, description="Feishu open_id of the sender")
    source: str = Field(default="feishu", description="Where the plan was registered from")
    chat_id: Optional[str] = Field(None, description="Originating Feishu chat, used for the receipt card")
    source_message_id: Optional[str] = Field(None, description="Feishu message id, the idempotency key")
    acked_at: Optional[str] = Field(None, description="Timestamp of the acknowledgement")
    ack_note: Optional[str] = Field(None, description="What the agent reported after carrying it out")

    def brief(self) -> Dict[str, Any]:
        """The projection the GPU Agent sees inside a `report` response."""
        return {
            "plan_id": self.plan_id,
            "text": self.text,
            "task_id": self.task_id,
            "created_by": self.created_by,
            "created_at": self.created_at,
        }


class TaskSummary(BaseModel):
    """Complete summary of a training task tracked in the mailbox."""

    task_id: str
    state: TaskState
    created_at: str
    updated_at: str
    latest_step: Optional[int] = None
    latest_epoch: Optional[int] = None
    latest_metrics: Optional[Dict[str, Any]] = None
    latest_message: Optional[str] = None
    pending_instruction: Optional[InstructionResponse] = None
    events_count: int = 0
    # Set while an engineer asked for time to arrange a fix (alert auto-decision paused).
    arrangement_requested_at: Optional[str] = None
    arrangement_requested_by: Optional[str] = None
    awaiting_arrangement: bool = False
