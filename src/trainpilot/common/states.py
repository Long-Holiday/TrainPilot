"""Core states and enums for TrainPilot."""

from enum import Enum


class TaskState(str, Enum):
    """Lifecycle states of a deep learning training task."""

    RUNNING = "RUNNING"  # Training cruising normally, periodic milestones
    WAITING = "WAITING"  # Anomaly detected; training paused waiting for human HITL decision
    RESOLVED = "RESOLVED"  # Human decided action; instruction ready in mailbox
    RECOVERING = "RECOVERING"  # GPU Agent executing recovery action (rollback/lr change)
    COMPLETED = "COMPLETED"  # Training finished successfully
    FAILED = "FAILED"  # Training terminated or failed fatally


class EventType(str, Enum):
    """Types of events reported from GPU Agent to Control Plane."""

    ALERT = "alert"  # Loss NaN, Loss Spike, CUDA OOM, Timeout, etc.
    MILESTONE = "milestone"  # Epoch completed, new metric high, checkpoint saved
    COMPLETED = "completed"  # Training gracefully finished
    FAILED = "failed"  # Training crashed or aborted


class ActionType(str, Enum):
    """Preset actions available for human-in-the-loop decisions."""

    CANCEL = "cancel"  # Agent cancels the run and verifies the machine shutdown
    SELF_RESOLVE = "self_resolve"  # Agent resolves by itself, continue training without intervention
    CUSTOM = "custom"  # Custom user-defined action with payload


# Card-only control action: the engineer asked to take the decision into a direct
# conversation with the agent. The task stays WAITING and the alert timer is suppressed.
WAIT_FOR_ARRANGEMENT = "wait_for_arrangement"

# Delivered by poll_instruction while that direct conversation is open. It is a control
# action, not a decision: it queues nothing and does not change the task state. The agent
# must stop polling and return control to the conversation.
WAIT_FOR_USER = "wait_for_user"
