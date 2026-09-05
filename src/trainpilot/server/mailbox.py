"""Thread-safe in-memory task mailbox and state machine management."""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trainpilot.common.schemas import (
    EventNotifyRequest,
    HeartbeatRequest,
    InstructionResponse,
    TaskSummary,
    utc_now_iso,
)
from trainpilot.common.states import EventType, TaskState

logger = logging.getLogger("trainpilot.mailbox")


@dataclass
class TaskRecord:
    """Internal task representation tracked in the mailbox."""

    task_id: str
    state: TaskState = TaskState.RUNNING
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    last_heartbeat_at: Optional[str] = None
    latest_step: Optional[int] = None
    latest_epoch: Optional[int] = None
    latest_metrics: Optional[Dict[str, Any]] = None
    latest_message: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    pending_instruction: Optional[InstructionResponse] = None
    latest_instruction: Optional[Dict[str, Any]] = None


class TaskMailboxManager:
    """Manages multi-task state machines, events, and instruction mailboxes safely."""

    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: Dict[str, TaskRecord] = {}

    def _get_or_create(self, task_id: str) -> TaskRecord:
        """Internal helper without lock acquisition."""
        if task_id not in self._tasks:
            self._tasks[task_id] = TaskRecord(task_id=task_id)
            logger.info("Initialized new task in mailbox: %s", task_id)
        return self._tasks[task_id]

    def record_event(self, req: EventNotifyRequest) -> TaskState:
        """Process an incoming event from the GPU Agent and transition states."""
        with self._lock:
            task = self._get_or_create(req.task_id)
            task.updated_at = utc_now_iso()
            task.latest_message = req.message
            if req.step is not None:
                task.latest_step = req.step
            if req.epoch is not None:
                task.latest_epoch = req.epoch
            if req.metrics is not None:
                task.latest_metrics = req.metrics

            event_entry = {
                "event_type": req.event_type.value,
                "message": req.message,
                "step": req.step,
                "epoch": req.epoch,
                "metrics": req.metrics,
                "timestamp": req.timestamp,
                "extra": req.extra,
            }
            task.events.append(event_entry)

            # State transitions
            if req.event_type == EventType.ALERT:
                task.state = TaskState.WAITING
                # Drop previous un-consumed instruction if any, waiting for fresh human input
                task.pending_instruction = None
                logger.warning("Task %s transitioned to WAITING due to alert: %s", req.task_id, req.message)
            elif req.event_type == EventType.MILESTONE:
                if task.state != TaskState.WAITING:
                    task.state = TaskState.RUNNING
                logger.info("Task %s recorded milestone at step %s", req.task_id, req.step)
            elif req.event_type == EventType.COMPLETED:
                task.state = TaskState.COMPLETED
                task.pending_instruction = None
                logger.info("Task %s completed successfully", req.task_id)
            elif req.event_type == EventType.FAILED:
                task.state = TaskState.FAILED
                task.pending_instruction = None
                logger.error("Task %s marked as FAILED", req.task_id)

            return task.state

    def record_heartbeat(self, req: HeartbeatRequest) -> None:
        """Update heartbeat timestamp and lightweight metrics."""
        with self._lock:
            task = self._get_or_create(req.task_id)
            now_iso = utc_now_iso()
            task.last_heartbeat_at = now_iso
            task.updated_at = now_iso
            if req.step is not None:
                task.latest_step = req.step
            if req.epoch is not None:
                task.latest_epoch = req.epoch
            if req.metrics is not None:
                task.latest_metrics = req.metrics

    def submit_decision(
        self,
        task_id: str,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
        operator: str = "human_operator",
    ) -> InstructionResponse:
        """Record human decision into the task mailbox and move state to RESOLVED."""
        with self._lock:
            task = self._get_or_create(task_id)
            inst_id = f"inst_{uuid.uuid4().hex[:12]}"
            now_iso = utc_now_iso()

            instruction = InstructionResponse(
                ready=True,
                status=TaskState.RESOLVED,
                instruction_id=inst_id,
                action=action,
                payload=payload,
                decision_by=operator,
                decided_at=now_iso,
            )

            task.pending_instruction = instruction
            task.state = TaskState.RESOLVED
            task.updated_at = now_iso
            logger.info("Decision submitted for task %s by %s: action=%s, id=%s",
                        task_id, operator, action, inst_id)
            return instruction

    def get_instruction(self, task_id: str, pop: bool = True) -> InstructionResponse:
        """Poll instructions for a task.

        If an instruction is pending and pop=True, moves state to RECOVERING and clears pending.
        """
        with self._lock:
            task = self._get_or_create(task_id)

            if task.pending_instruction and task.pending_instruction.ready:
                instruction = task.pending_instruction
                if pop:
                    task.pending_instruction = None
                    task.state = TaskState.RECOVERING
                    task.updated_at = utc_now_iso()
                    task.latest_instruction = {
                        "instruction_id": instruction.instruction_id,
                        "action": instruction.action,
                        "payload": instruction.payload,
                        "decision_by": instruction.decision_by,
                        "popped_at": task.updated_at,
                    }
                    logger.info("Task %s polled instruction %s (%s), transitioned to RECOVERING",
                                task_id, instruction.instruction_id, instruction.action)
                return instruction

            return InstructionResponse(
                ready=False,
                status=task.state,
                instruction_id=None,
                action=None,
                payload=None,
                decision_by=None,
                decided_at=None,
            )

    def ack_instruction(
        self,
        task_id: str,
        instruction_id: Optional[str],
        action: str,
        status: str,
        message: Optional[str] = None,
    ) -> TaskState:
        """Confirm execution of a recovery instruction and transition state back to RUNNING."""
        with self._lock:
            task = self._get_or_create(task_id)
            task.updated_at = utc_now_iso()

            ack_record = {
                "instruction_id": instruction_id,
                "action": action,
                "status": status,
                "message": message,
                "timestamp": task.updated_at,
            }
            if task.latest_instruction and task.latest_instruction.get("instruction_id") == instruction_id:
                task.latest_instruction["ack"] = ack_record

            if status.lower() == "success":
                task.state = TaskState.RUNNING
                logger.info("Task %s acknowledged instruction %s successfully -> RUNNING", task_id, instruction_id)
            else:
                task.state = TaskState.WAITING
                logger.warning("Task %s failed executing instruction %s -> WAITING: %s",
                               task_id, instruction_id, message)

            return task.state

    def get_task(self, task_id: str) -> Optional[TaskSummary]:
        """Get summary of a specific task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            return self._build_summary(task)

    def list_tasks(self) -> List[TaskSummary]:
        """List summaries of all tracked tasks."""
        with self._lock:
            return [self._build_summary(t) for t in self._tasks.values()]

    def reset(self) -> None:
        """Reset all tasks (mainly for testing)."""
        with self._lock:
            self._tasks.clear()

    def _build_summary(self, task: TaskRecord) -> TaskSummary:
        return TaskSummary(
            task_id=task.task_id,
            state=task.state,
            created_at=task.created_at,
            updated_at=task.updated_at,
            last_heartbeat_at=task.last_heartbeat_at,
            latest_step=task.latest_step,
            latest_epoch=task.latest_epoch,
            latest_metrics=task.latest_metrics,
            latest_message=task.latest_message,
            pending_instruction=task.pending_instruction,
            events_count=len(task.events),
        )


# Global default mailbox manager
default_mailbox = TaskMailboxManager()
