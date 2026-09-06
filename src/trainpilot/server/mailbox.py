"""Thread-safe task mailbox and state machine management with SQLite persistence and Long Polling."""

import json
import logging
import threading
import uuid
from collections import defaultdict
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
    pending_instruction: Optional[InstructionResponse] = None
    latest_instruction: Optional[Dict[str, Any]] = None
    last_alert_fingerprint: Optional[str] = None
    last_alert_at: Optional[str] = None
    events_count: int = 0
    feishu_message_id: Optional[str] = None


class TaskMailboxManager:
    """Manages multi-task state machines, events, and instruction mailboxes safely.

    Features:
    - SQLite-first persistence: events stored strictly on-disk, minimal memory consumption.
    - Long Polling support (instant zero-delay wake up on human decisions).
    - Thread-safe state transitions.
    """

    def __init__(self, max_events_per_task: int = 500, storage: Optional[Any] = None):
        self._lock = threading.Lock()
        self._tasks: Dict[str, TaskRecord] = {}
        self._max_events = max(1, int(max_events_per_task))
        self._instruction_waiters: Dict[str, List[threading.Event]] = defaultdict(list)

        # Initialize SQLite Storage
        if storage is not None:
            self._storage = storage
        else:
            try:
                from trainpilot.server.config import settings as _s
                db_path = getattr(_s, "sqlite_path", "trainpilot.db")
                from trainpilot.server.storage import SQLiteStorage
                self._storage = SQLiteStorage(db_path)
            except Exception as e:
                logger.error("Failed to initialize SQLiteStorage from config: %s", e)
                from trainpilot.server.storage import SQLiteStorage
                self._storage = SQLiteStorage("trainpilot.db")

        # Warm up active tasks from SQLite storage (save RAM by not loading terminated tasks)
        try:
            persisted_tasks = self._storage.load_all_tasks(active_only=True)
            for tid, rec in persisted_tasks.items():
                rec.events_count = self._storage.get_events_count(tid)
                self._tasks[tid] = rec
            logger.info("Restored %d active tasks into memory from SQLite storage", len(persisted_tasks))
        except Exception as e:
            logger.error("Failed to restore tasks from SQLite: %s", e)

    def _evict_cold_tasks_locked(self, max_allowed: int = 150) -> None:
        """Evict completed or failed tasks from RAM when memory task count exceeds threshold."""
        if len(self._tasks) <= max_allowed:
            return
        to_evict = [
            tid for tid, rec in self._tasks.items()
            if rec.state in TERMINAL_STATES
        ]
        for tid in to_evict:
            if len(self._tasks) <= max_allowed:
                break
            self._tasks.pop(tid, None)

    def _get_or_create(self, task_id: str) -> TaskRecord:
        """Internal helper without lock acquisition, with SQLite lazy restoration."""
        if task_id not in self._tasks:
            # Lazy restore from SQLite if present to save continuous RAM
            if self._storage:
                existing = self._storage.get_task(task_id)
                if existing:
                    existing.events_count = self._storage.get_events_count(task_id)
                    self._tasks[task_id] = existing
                    return existing
            self._tasks[task_id] = TaskRecord(task_id=task_id)
            if self._storage:
                self._storage.save_task(self._tasks[task_id])
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
            task.events_count += 1

            # State transitions
            if req.event_type == EventType.ALERT:
                fp = _alert_fingerprint(req)
                is_retry = fp == task.last_alert_fingerprint
                task.last_alert_fingerprint = fp
                task.last_alert_at = task.updated_at
                task.state = TaskState.WAITING
                if is_retry:
                    logger.info("Task %s duplicate alert ignored (idempotent): %s", req.task_id, req.message)
                else:
                    task.pending_instruction = None
                    logger.warning("Task %s transitioned to WAITING due to alert: %s", req.task_id, req.message)
            elif req.event_type == EventType.MILESTONE:
                if task.state != TaskState.WAITING:
                    task.state = TaskState.RUNNING
                logger.info("Task %s recorded milestone at step %s", req.task_id, req.step)
            elif req.event_type == EventType.HEARTBEAT:
                task.last_heartbeat_at = req.timestamp or utc_now_iso()
            elif req.event_type == EventType.COMPLETED:
                task.state = TaskState.COMPLETED
                task.pending_instruction = None
                logger.info("Task %s completed successfully", req.task_id)
            elif req.event_type == EventType.FAILED:
                task.state = TaskState.FAILED
                task.pending_instruction = None
                logger.error("Task %s marked as FAILED", req.task_id)

            # Persist to SQLite
            try:
                self._storage.save_task(task)
                self._storage.append_event(req.task_id, event_entry, max_events=self._max_events)
            except Exception as e:
                logger.error("Failed to persist event to SQLite: %s", e)

            if req.event_type in (EventType.COMPLETED, EventType.FAILED):
                self._evict_cold_tasks_locked()

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

            try:
                self._storage.save_task(task, stale_alerted=False)
            except Exception as e:
                logger.error("Failed to update heartbeat in SQLite: %s", e)

    def _notify_instruction_waiters(self, task_id: str) -> None:
        """Wake up any pending long-polling requests waiting for this task's decision."""
        waiters = list(self._instruction_waiters.get(task_id, []))
        for ev in waiters:
            ev.set()

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

            try:
                self._storage.save_task(task)
            except Exception as e:
                logger.error("Failed to persist decision to SQLite: %s", e)

            logger.info("Decision submitted for task %s by %s: action=%s, id=%s",
                        task_id, operator, action, inst_id)

            # Wake up long-polling agents immediately
            self._notify_instruction_waiters(task_id)

            return instruction

    def try_auto_resolve(
        self,
        task_id: str,
        action: str = "self_resolve",
        operator: str = "system_auto_resolve (30s timeout)",
        timeout_seconds: float = 30,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[InstructionResponse]:
        """Conditionally auto-resolve an alert that received no human decision in time."""
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

            try:
                self._storage.save_task(task)
            except Exception as e:
                logger.error("Failed to persist auto-resolve to SQLite: %s", e)

            logger.warning("Task %s auto-resolved to '%s' after %.1fs without human decision",
                           task_id, action, timeout_seconds)

            # Wake up long-polling agents immediately
            self._notify_instruction_waiters(task_id)

            return instruction

    def get_instruction(
        self, task_id: str, pop: bool = True, wait_timeout: float = 0.0
    ) -> InstructionResponse:
        """Poll instructions for a task with Long Polling support.

        If wait_timeout > 0 and no instruction is ready, blocks up to wait_timeout
        seconds waiting for submit_decision or try_auto_resolve, waking up immediately.
        """
        # 1. Fast path: check if instruction is already ready
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
                    try:
                        self._storage.save_task(task)
                    except Exception as e:
                        logger.error("Failed to update pop state in SQLite: %s", e)
                    logger.info("Task %s polled instruction %s (%s), transitioned to RECOVERING",
                                task_id, instruction.instruction_id, instruction.action)
                return instruction

            if wait_timeout <= 0.0:
                return InstructionResponse(
                    ready=False,
                    status=task.state,
                    instruction_id=None,
                    action=None,
                    payload=None,
                    decision_by=None,
                    decided_at=None,
                )

            # Register long-poll waiter
            wait_event = threading.Event()
            self._instruction_waiters[task_id].append(wait_event)

        # 2. Long Polling wait (outside the global lock so others are not blocked)
        wait_event.wait(timeout=wait_timeout)

        # 3. Re-acquire lock and check again
        with self._lock:
            waiters = self._instruction_waiters.get(task_id, [])
            if wait_event in waiters:
                waiters.remove(wait_event)

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
                    try:
                        self._storage.save_task(task)
                    except Exception as e:
                        logger.error("Failed to update pop state in SQLite: %s", e)
                    logger.info("Task %s (long-polled) obtained instruction %s (%s) -> RECOVERING",
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

            try:
                self._storage.save_task(task)
            except Exception as e:
                logger.error("Failed to persist ack to SQLite: %s", e)

            return task.state

    def set_task_feishu_message_id(self, task_id: str, message_id: str) -> None:
        """Store the Feishu message_id in memory and SQLite (survives restarts)."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.feishu_message_id = message_id
            try:
                self._storage.set_task_feishu_message_id(task_id, message_id)
            except Exception as e:
                logger.error("Failed to set feishu_message_id in storage: %s", e)

    def get_task_feishu_message_id(self, task_id: str) -> Optional[str]:
        """Fetch the Feishu message_id for updating interactive cards."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.feishu_message_id:
                return task.feishu_message_id
            try:
                return self._storage.get_task_feishu_message_id(task_id)
            except Exception as e:
                logger.error("Failed to get feishu_message_id from storage: %s", e)
            return None

    def get_task(self, task_id: str) -> Optional[TaskSummary]:
        """Get summary of a specific task with lazy fallback to SQLite."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                if self._storage:
                    record = self._storage.get_task(task_id)
                    if record:
                        return self._build_summary(record)
                return None
            return self._build_summary(task)

    def get_tasks_count(self, state: Optional[TaskState] = None) -> int:
        """Fast count of total tasks from storage without instantiating objects in memory."""
        with self._lock:
            if self._storage:
                return self._storage.get_tasks_count(state=state)
            if state is not None:
                return sum(1 for t in self._tasks.values() if t.state == state)
            return len(self._tasks)

    def get_events(self, task_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Paginated chronological event history."""
        with self._lock:
            return self._storage.get_events(task_id, limit=limit, offset=offset)

    def get_events_count(self, task_id: str) -> Optional[int]:
        with self._lock:
            cnt = self._storage.get_events_count(task_id)
            # If 0 but task not exists, return None
            if cnt == 0 and task_id not in self._tasks:
                return None
            return cnt

    def list_tasks(
        self,
        limit: int = 100,
        offset: int = 0,
        state: Optional[TaskState] = None,
        stale_only: bool = False,
        heartbeat_timeout_seconds: Optional[int] = None,
    ) -> List[TaskSummary]:
        """List summaries with pagination and optional state/stale filters.

        Uses SQLite pagination directly when available to keep RAM consumption flat (O(limit) instead of O(N)).
        """
        with self._lock:
            if stale_only:
                # Stale tasks are strictly candidate non-terminal states tracked in memory
                tasks = [t for t in self._tasks.values() if self._is_stale_locked(t, heartbeat_timeout_seconds)]
                tasks = tasks[offset:offset + limit]
                return [self._build_summary(t) for t in tasks]

            # Storage pagination path: only loads current page rows into RAM
            if self._storage:
                records = self._storage.list_tasks(limit=limit, offset=offset, state=state)
                summaries = []
                for rec in records:
                    active = self._tasks.get(rec.task_id)
                    target = active if active else rec
                    summaries.append(self._build_summary(target))
                return summaries

            # Memory-only fallback (e.g. tests without storage)
            tasks = list(self._tasks.values())
            if state is not None:
                tasks = [t for t in tasks if t.state == state]
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
            last = _parse_iso(task.updated_at)
        if last is None:
            return False
        now = datetime.now(timezone.utc)
        return (now - last).total_seconds() > timeout_seconds

    def reset(self) -> None:
        """Reset all tasks (mainly for testing)."""
        with self._lock:
            self._tasks.clear()
            self._instruction_waiters.clear()
            try:
                self._storage.reset()
            except Exception as e:
                logger.warning("Failed to reset SQLite storage: %s", e)

    def _build_summary(self, task: TaskRecord) -> TaskSummary:
        count = task.events_count
        if count == 0:
            count = self._storage.get_events_count(task.task_id)
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
            events_count=count,
        )


# Global default mailbox manager
default_mailbox = TaskMailboxManager()
