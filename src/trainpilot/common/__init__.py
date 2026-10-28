"""TrainPilot common definitions."""

from trainpilot.common.gateway import (
    resolve_bind_host,
    resolve_gateway_url,
    resolve_public_host,
)
from trainpilot.common.states import ActionType, EventType, TaskState

_SCHEMA_EXPORTS = (
    "EventNotifyRequest",
    "InstructionResponse",
    "TaskDecisionRequest",
    "TaskSummary",
    "FeishuCardActionPayload",
)


def __getattr__(name):
    # Schemas need pydantic, but the plugin generator runs without site-packages,
    # so they resolve on first use instead.
    if name in _SCHEMA_EXPORTS:
        from trainpilot.common import schemas

        return getattr(schemas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TaskState",
    "EventType",
    "ActionType",
    "resolve_bind_host",
    "resolve_gateway_url",
    "resolve_public_host",
    *_SCHEMA_EXPORTS,
]
