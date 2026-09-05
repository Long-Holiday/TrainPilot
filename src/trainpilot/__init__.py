"""TrainPilot - Distributed Deep Learning Human-in-the-Loop Gateway and Agent."""

__version__ = "0.1.0"

from trainpilot.agent.client import TrainPilotClient
from trainpilot.agent.hooks.pytorch import TrainPilotPyTorchHook
from trainpilot.agent.monitor import TrainingGuardian
from trainpilot.common.states import ActionType, EventType, TaskState

__all__ = [
    "TrainPilotClient",
    "TrainingGuardian",
    "TrainPilotPyTorchHook",
    "TaskState",
    "EventType",
    "ActionType",
]
