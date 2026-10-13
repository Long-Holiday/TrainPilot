"""TrainPilot - Distributed Deep Learning Human-in-the-Loop Gateway and Agent."""

__version__ = "0.1.1"

from trainpilot.common.states import ActionType, EventType, TaskState

__all__ = [
    "TaskState",
    "EventType",
    "ActionType",
]
