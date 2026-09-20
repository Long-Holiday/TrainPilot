"""TrainPilot common definitions."""

from trainpilot.common.gateway import (
    detect_local_ip,
    resolve_bind_host,
    resolve_gateway_url,
    resolve_gpu_host,
    resolve_port,
    resolve_public_host,
)
from trainpilot.common.schemas import (
    EventNotifyRequest,
    EventNotifyResponse,
    FeishuCardActionPayload,
    InstructionAckRequest,
    InstructionResponse,
    TaskDecisionRequest,
    TaskSummary,
)
from trainpilot.common.states import ActionType, EventType, TaskState

__all__ = [
    "TaskState",
    "EventType",
    "ActionType",
    "resolve_bind_host",
    "resolve_gateway_url",
    "resolve_gpu_host",
    "detect_local_ip",
    "resolve_port",
    "resolve_public_host",
    "EventNotifyRequest",
    "EventNotifyResponse",
    "InstructionResponse",
    "InstructionAckRequest",
    "TaskDecisionRequest",
    "TaskSummary",
    "FeishuCardActionPayload",
]
