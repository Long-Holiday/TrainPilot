"""Thread-safe in-memory task mailbox and state machine management."""

import json
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

TERMINAL_STATES = frozenset({TaskState.COMPLETED, TaskState.FAILED})
STALE_CANDIDATE_STATES = frozenset({TaskState.RUNNING, TaskState.WAITING, TaskState.RECOVERING})


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _alert_fingerprint(req: EventNotifyRequest) -> str:
    try:
        metrics_str = json.dumps(req.metrics, sort_keys=True, default=str)
    except Exception:
        metrics_str = str(req.metrics)
    return f"{req.message}|{req.step}|{req.epoch}|{metrics_str}"


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
    last_alert_fingerprint: Optional[str] = None
    last_alert_at: Optional[str] = None


class TaskMailboxManager:
    """Manages multi-task state machines, events, and instruction mailboxes safely.

    NOTE: in-memory only. Restart loses state and multi-worker deployments
    diverge. For production use an external store; this class caps memory via
    per-task ring buffer (`max_events_per_task`).
    """

    def __init__(self, max_events_per_task: int = 500):
        self._lock = threading.Lock()
        self._tasks: Dict[str, TaskRecord] = {}
        self._max_events = max(1, int(max_events_per_task))

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
                "agent_note": req.agent_note,
            }
            task.events.append(event_entry)
            # Ring-buffer cap: drop oldest to bound memory on long trainings.
            if len(task.events) > self._max_events:
                task.events = task.events[-self._max_events:]

            # State transitions
            if req.event_type == EventType.ALERT:
                fp = _alert_fingerprint(req)
                is_retry = fp == task.last_alert_fingerprint
                task.last_alert_fingerprint = fp
                task.last_alert_at = task.updated_at
                task.state = TaskState.WAITING
                if is_retry:
                    # Network retry of identical alert: preserve pending human decision.
                    logger.info("Task %s duplicate alert ignored (idempotent): %s", req.task_id, req.message)
                else:
                    # Fresh anomaly supersedes stale decision awaiting consumption.
                    task.pending_instruction = None
                    logger.warning("Task %s transitioned to WAITING due to alert: %s", req.task_id, req.message)
            elif req.event_type == EventType.MILESTONE:
                if task.state != TaskState.WAITING:
                    task.state = TaskState.RUNNING
                logger.info("Task %s recorded milestone at step %s", req.task_id, req.step)
            elif req.event_type == EventType.RECOVERY:
                if task.state not in TERMINAL_STATES:
                    task.state = TaskState.RUNNING
                logger.info("Task %s self-recovered: %s", req.task_id, req.message)
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
            if task.state in TERMINAL_STATES:
                raise ValueError(f"Task '{task_id}' already in terminal state {task.state.value}, refusing new decision")
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

    def try_auto_resolve(
        self,
        task_id: str,
        action: str = "self_resolve",
        operator: str = "system_auto_resolve (30s timeout)",
        timeout_seconds: float = 30,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[InstructionResponse]:
        """Conditionally auto-resolve an alert that received no human decision in time.

        仅当任务仍处于 WAITING 且无待消费指令、且距离上次告警已超过
        timeout_seconds 时才写入自动“自行解决”决策。人工已决策时为 no-op，
        因此定时器与人工点击竞态时人工优先。
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            if task.state != TaskState.WAITING:
                return None
            if task.pending_instruction is not None:
                return None
            base_ts = task.last_alert_at or task.updated_at
            base_dt = _parse_iso(base_ts)
            if base_dt is None:
                return None
            now = datetime.now(timezone.utc)
            if (now - base_dt).total_seconds() < timeout_seconds:
                # 期间产生了更新鲜的告警，等待新一轮超时。
                return None
            inst_id = f"inst_{uuid.uuid4().hex[:12]}"
            now_iso = utc_now_iso()
            auto_payload = dict(payload or {})
            auto_payload.setdefault("auto_resolved", True)
            auto_payload.setdefault("timeout_seconds", timeout_seconds)
            instruction = InstructionResponse(
                ready=True,
                status=TaskState.RESOLVED,
                instruction_id=inst_id,
                action=action,
                payload=auto_payload,
                decision_by=operator,
                decided_at=now_iso,
            )
            task.pending_instruction = instruction
            task.state = TaskState.RESOLVED
            task.updated_at = now_iso
            logger.warning("Task %s auto-resolved to '%s' after %.1fs without human decision",
                           task_id, action, timeout_seconds)
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
            # Validate against last consumed instruction to catch stale/duplicate ACKs.
            if task.latest_instruction and instruction_id:
                expected = task.latest_instruction.get("instruction_id")
                if expected and instruction_id != expected:
                    raise ValueError(
                        f"Stale instruction ack: got {instruction_id}, expected {expected} for task '{task_id}'"
                    )
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

    def get_events(self, task_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Paginated chronological event history."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return []
            return list(task.events[offset:offset + limit])

    def get_events_count(self, task_id: str) -> Optional[int]:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            return len(task.events)

    def list_tasks(
        self,
        limit: int = 100,
        offset: int = 0,
        state: Optional[TaskState] = None,
        stale_only: bool = False,
        heartbeat_timeout_seconds: Optional[int] = None,
    ) -> List[TaskSummary]:
        """List summaries with pagination and optional state/stale filters."""
        with self._lock:
            tasks = list(self._tasks.values())
            if state is not None:
                tasks = [t for t in tasks if t.state == state]
            if stale_only:
                tasks = [t for t in tasks if self._is_stale_locked(t, heartbeat_timeout_seconds)]
            tasks = tasks[offset:offset + limit]
            return [self._build_summary(t) for t in tasks]

    def get_stale_tasks(self, timeout_seconds: int = 300) -> List[TaskSummary]:
        """Tasks without heartbeat beyond timeout (candidates for intervention)."""
        with self._lock:
            return [
                self._build_summary(t)
                for t in self._tasks.values()
                if self._is_stale_locked(t, timeout_seconds)
            ]

    def _is_stale_locked(self, task: TaskRecord, timeout_seconds: Optional[int]) -> bool:
        if timeout_seconds is None:
            try:
                from trainpilot.server.config import settings as _s
                timeout_seconds = _s.task_heartbeat_timeout_seconds
            except Exception:
                timeout_seconds = 300
        if task.state not in STALE_CANDIDATE_STATES:
            return False
        last = _parse_iso(task.last_heartbeat_at)
        if last is None:
            # Never heartbeated: use updated_at; brand-new tasks (<timeout) are not stale.
            last = _parse_iso(task.updated_at)
        if last is None:
            return False
        now = datetime.now(timezone.utc)
        return (now - last).total_seconds() > timeout_seconds

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
