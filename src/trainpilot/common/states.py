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
    RECOVERY = "recovery"  # Anomaly resolved by agent; reporting solution & resumed status
    HEARTBEAT = "heartbeat"  # Periodic health ping with lightweight resource usage
    COMPLETED = "completed"  # Training gracefully finished
    FAILED = "failed"  # Training crashed or aborted


class ActionType(str, Enum):
    """Preset actions available for human-in-the-loop decisions."""

    STOP_TRAINING = "stop_training"  # Terminate training safely and retain artifacts
    SELF_RESOLVE = "self_resolve"  # Agent resolves by itself, continue training without intervention
    CUSTOM = "custom"  # Custom user-defined action with payload
