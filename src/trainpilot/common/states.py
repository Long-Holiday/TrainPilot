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
    HEARTBEAT = "heartbeat"  # Periodic health ping with lightweight resource usage
    COMPLETED = "completed"  # Training gracefully finished
    FAILED = "failed"  # Training crashed or aborted


class ActionType(str, Enum):
    """Preset actions available for human-in-the-loop decisions."""

    REDUCE_LR_ROLLBACK = "reduce_lr_rollback"  # Cut LR and reload previous checkpoint
    SKIP_BATCH = "skip_batch"  # Drop current batch/step and resume
    STOP_TRAINING = "stop_training"  # Terminate training safely and retain artifacts
    RESUME = "resume"  # Continue as-is without modification
    CUSTOM = "custom"  # Custom user-defined action with payload
