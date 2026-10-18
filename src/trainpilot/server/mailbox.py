"""Thread-safe task mailbox and state machine management with SQLite persistence and Long Polling."""

import asyncio
import copy
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
    InstructionResponse,
    TaskSummary,
    utc_now_iso,
)
from trainpilot.common.states import EventType, TaskState, WAIT_FOR_USER

logger = logging.getLogger("trainpilot.mailbox")

TERMINAL_STATES = frozenset({TaskState.COMPLETED, TaskState.FAILED})


class TaskNotFoundError(ValueError):
    """Raised when a read/control operation references an unknown task."""


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
    # Set when a human clicked ⏳ 等待用户: the alert auto-decision is suppressed and
    # the agent is asked to stop polling and resolve the incident in a direct
    # conversation. There is deliberately no fallback timer.
    direct_conversation_requested_at: Optional[str] = None
    direct_conversation_requested_by: Optional[str] = None


class TaskMailboxManager:
    """Manages multi-task state machines, events, and instruction mailboxes safely.

    Features:
    - SQLite-first persistence: events stored strictly on-disk, minimal memory consumption.
    - Long Polling support (instant zero-delay wake up on human decisions).
    - Thread-safe state transitions.
    """

    def __init__(
        self,
        max_events_per_task: int = 500,
        storage: Optional[Any] = None,
        enable_persistence: Optional[bool] = None,
    ):
        self._lock = threading.Lock()
        self._tasks: Dict[str, TaskRecord] = {}
        self._max_events = max(1, int(max_events_per_task))
        self._async_instruction_waiters: Dict[str, List[asyncio.Event]] = defaultdict(list)

        # Initialize SQLite Storage
        if storage is not None:
            self._storage = storage
        else:
            try:
                from trainpilot.server.config import settings as _s
                if enable_persistence is None:
                    enable_persistence = _s.enable_sqlite
                db_path = _s.sqlite_path if enable_persistence else ":memory:"
                from trainpilot.server.storage import SQLiteStorage
                self._storage = SQLiteStorage(db_path)
            except Exception as e:
                logger.exception("Failed to initialize task storage")
                raise RuntimeError("Failed to initialize task storage") from e

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

    def _get_or_create(self, task_id: str, *, persist_new: bool = True) -> TaskRecord:
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
            if self._storage and persist_new:
                self._storage.save_task(self._tasks[task_id])
            logger.info("Initialized new task in mailbox: %s", task_id)
        return self._tasks[task_id]

    def _get_existing(self, task_id: str) -> TaskRecord:
        """Load an existing task without creating records for read/control calls."""
        task = self._tasks.get(task_id)
        if task is not None:
            return task
        if self._storage:
            task = self._storage.get_task(task_id)
            if task is not None:
                task.events_count = self._storage.get_events_count(task_id)
                self._tasks[task_id] = task
                return task
        raise TaskNotFoundError(f"Task '{task_id}' not found")

    def _persist_task_or_restore(
        self,
        task: TaskRecord,
        previous: TaskRecord,
        was_cached: bool,
        context: str,
    ) -> None:
        """Persist a mutation, restoring the in-memory snapshot on failure."""
        try:
            self._storage.save_task(task)
        except Exception:
            if was_cached:
                self._tasks[task.task_id] = previous
            else:
                self._tasks.pop(task.task_id, None)
            logger.exception("Failed to persist %s for task %s", context, task.task_id)
            raise

    def record_event(self, req: EventNotifyRequest) -> TaskState:
        """Process an incoming event from the GPU Agent and transition states."""
        with self._lock:
            was_cached = req.task_id in self._tasks
            task = self._get_or_create(req.task_id, persist_new=False)
            previous = copy.deepcopy(task)
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
                    # A fresh alert (not a retry of the same one) opens a new decision
                    # window: any direct-conversation request for the previous alert is over.
                    self._clear_direct_conversation(task)
                    logger.warning("Task %s transitioned to WAITING due to alert: %s", req.task_id, req.message)
            elif req.event_type == EventType.MILESTONE:
                # A milestone never dismisses a wait: the sole recovery receipt is the
                # ACK. Leaving WAITING also keeps the direct-conversation marker so a
                # periodic progress report cannot silently end the explicit wait.
                if task.state != TaskState.WAITING:
                    task.state = TaskState.RUNNING
                logger.info("Task %s recorded milestone at step %s", req.task_id, req.step)
            elif req.event_type == EventType.COMPLETED:
                task.state = TaskState.COMPLETED
                task.pending_instruction = None
                self._clear_direct_conversation(task)
                logger.info("Task %s completed successfully", req.task_id)
            elif req.event_type == EventType.FAILED:
                task.state = TaskState.FAILED
                task.pending_instruction = None
                self._clear_direct_conversation(task)
                logger.error("Task %s marked as FAILED", req.task_id)

            # Persist to SQLite
            try:
                self._storage.save_task_and_append_event(
                    task,
                    event_entry,
                    max_events=self._max_events,
                )
            except Exception:
                if was_cached:
                    self._tasks[req.task_id] = previous
                else:
                    self._tasks.pop(req.task_id, None)
                logger.exception("Failed to atomically persist task event")
                raise

            if req.event_type in (EventType.COMPLETED, EventType.FAILED):
                # A long poll may have started before the task became terminal.
                # Wake it so it can observe the terminal state without waiting
                # for its full timeout.
                self._notify_instruction_waiters(req.task_id)
                self._evict_cold_tasks_locked()

            return task.state

    def _notify_instruction_waiters(self, task_id: str) -> None:
        """Wake up any pending long-polling requests waiting for this task's decision."""
        async_waiters = list(self._async_instruction_waiters.get(task_id, []))
        for a_ev in async_waiters:
            try:
                loop = getattr(a_ev, "_loop", None)
                if loop and loop.is_running():
                    loop.call_soon_threadsafe(a_ev.set)
                else:
                    a_ev.set()
            except Exception as e:
                logger.debug("Failed setting async event for %s: %s", task_id, e)

    @staticmethod
    def _clear_direct_conversation(task: TaskRecord) -> None:
        """End the direct-conversation wait opened by the 「等待用户」 button.

        Any queued instruction, any fresh alert, and any terminal event close it.
        Lives here so the call sites cannot drift apart.
        """
        task.direct_conversation_requested_at = None
        task.direct_conversation_requested_by = None

    def _queue_instruction_locked(
        self,
        task: TaskRecord,
        action: str,
        payload: Optional[Dict[str, Any]],
        operator: str,
    ) -> InstructionResponse:
        """Install a decision on a task and move it to RESOLVED (caller holds the lock)."""
        now_iso = utc_now_iso()
        instruction = InstructionResponse(
            has_instruction=True,
            status=TaskState.RESOLVED,
            instruction_id=f"inst_{uuid.uuid4().hex[:12]}",
            action=action,
            payload=payload,
            decision_by=operator,
            decided_at=now_iso,
        )
        task.pending_instruction = instruction
        task.state = TaskState.RESOLVED
        task.updated_at = now_iso
        self._clear_direct_conversation(task)
        return instruction

    def submit_decision(
        self,
        task_id: str,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
        operator: str = "human_operator",
    ) -> InstructionResponse:
        """Record human decision into the task mailbox and move state to RESOLVED."""
        with self._lock:
            was_cached = task_id in self._tasks
            task = self._get_existing(task_id)
            previous = copy.deepcopy(task)
            if task.state in TERMINAL_STATES:
                raise ValueError(f"Task '{task_id}' already in terminal state {task.state.value}, refusing new decision")
            if task.state != TaskState.WAITING:
                # Only an open alert accepts a decision. This preserves the first
                # decision against duplicate card clicks while it is RESOLVED or
                # RECOVERING, instead of silently replacing it.
                raise ValueError(
                    f"Task '{task_id}' is in state {task.state.value}, "
                    "only a WAITING alert accepts a decision"
                )

            instruction = self._queue_instruction_locked(task, action, payload, operator)

            self._persist_task_or_restore(
                task, previous, was_cached, "operator decision"
            )

            logger.info("Decision submitted for task %s by %s: action=%s, id=%s",
                        task_id, operator, action, instruction.instruction_id)

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
        """Auto-resolve an alert that received no human decision in time.

        This is the ordinary N-second window started by `report`. It refuses to fire
        once the engineer chose ⏳ 等待用户: that choice hands the decision to a direct
        conversation with the agent and deliberately has no fallback.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            if task.state != TaskState.WAITING:
                return None
            if task.pending_instruction is not None:
                return None
            if task.direct_conversation_requested_at is not None:
                # The engineer is handling this directly with the agent; do not
                # auto-decide from under them.
                return None
            base_ts = task.last_alert_at or task.updated_at
            base_dt = _parse_iso(base_ts)
            if base_dt is None:
                return None
            now = datetime.now(timezone.utc)
            if (now - base_dt).total_seconds() < timeout_seconds:
                return None
            previous = copy.deepcopy(task)
            auto_payload = dict(payload or {})
            auto_payload.setdefault("auto_resolved", True)
            auto_payload.setdefault("timeout_seconds", timeout_seconds)
            instruction = self._queue_instruction_locked(task, action, auto_payload, operator)

            self._persist_task_or_restore(
                task, previous, True, "automatic resolution"
            )

            logger.warning("Task %s auto-resolved to '%s' after %.1fs without human decision",
                           task_id, action, timeout_seconds)

            # Wake up long-polling agents immediately
            self._notify_instruction_waiters(task_id)

            return instruction

    def wait_for_direct_conversation(self, task_id: str, operator: str) -> TaskSummary:
        """Put a waiting alert into 'the engineer is talking to the agent directly'.

        The task stays WAITING with no instruction queued and no fallback timer. The
        alert-stage auto-resolve timer is suppressed by the flag, and long-polling
        agents are woken so they can see `awaiting_direct_conversation` and yield
        control immediately instead of polling forever.
        """
        with self._lock:
            was_cached = task_id in self._tasks
            task = self._get_existing(task_id)
            previous = copy.deepcopy(task)
            if task.state != TaskState.WAITING:
                raise ValueError(
                    f"Task '{task_id}' is in state {task.state.value}, "
                    "only a WAITING alert can be handed to a direct conversation"
                )
            if task.direct_conversation_requested_at is None:
                # First click defines the request; the alert card is replaced in place,
                # so a duplicate click is not normally reachable.
                task.direct_conversation_requested_at = utc_now_iso()
                task.direct_conversation_requested_by = operator
                task.updated_at = task.direct_conversation_requested_at
            self._persist_task_or_restore(
                task, previous, was_cached, "direct-conversation request"
            )
            logger.warning(
                "Task %s is now waiting for a direct conversation with the agent (requested by %s)",
                task_id, task.direct_conversation_requested_by,
            )
            # A poll may already be in flight; wake it so it returns control at once.
            self._notify_instruction_waiters(task_id)
            return self._build_summary(task)

    @staticmethod
    def _empty_instruction(task: TaskRecord) -> InstructionResponse:
        """A no-instruction response for a normal poll."""
        return InstructionResponse(
            has_instruction=False,
            status=task.state,
            instruction_id=None,
            action=None,
            payload=None,
            decision_by=None,
            decided_at=None,
        )

    @staticmethod
    def _direct_conversation_instruction(task: TaskRecord) -> InstructionResponse:
        """Deliver the wait control action without queueing or changing state.

        This is a genuine `has_instruction=true` response: relying on an empty poll plus
        a flag makes it too easy for an agent to keep looping. The action tells the agent
        to stop polling and return control to the conversation. Nothing is popped and the
        task stays WAITING; the same response is returned if the agent polls again.
        """
        return InstructionResponse(
            has_instruction=True,
            status=task.state,
            instruction_id=None,
            action=WAIT_FOR_USER,
            payload={"reason": "engineer chose a direct conversation"},
            decision_by=task.direct_conversation_requested_by,
            decided_at=task.direct_conversation_requested_at,
            awaiting_direct_conversation=True,
        )

    async def get_instruction_async(
        self, task_id: str, pop: bool = True, wait_timeout: float = 0.0
    ) -> InstructionResponse:
        """Poll instructions for a task with native AsyncIO Long Polling.

        Does NOT block any OS thread or FastAPI/AnyIO thread pool.
        Wakes up immediately when submit_decision or try_auto_resolve is called.

        When the engineer chose ⏳ 等待用户, this returns immediately with
        ``awaiting_direct_conversation=True`` and never waits: no instruction will
        arrive through the mailbox, so the agent must return control to the
        conversation instead of polling.
        """
        # 1. Fast path: check if instruction is already available
        with self._lock:
            task = self._get_existing(task_id)
            if task.pending_instruction and task.pending_instruction.has_instruction:
                instruction = task.pending_instruction
                if pop:
                    previous = copy.deepcopy(task)
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
                    self._persist_task_or_restore(
                        task, previous, True, "instruction consumption"
                    )
                    logger.info("Task %s polled instruction %s (%s), transitioned to RECOVERING",
                                task_id, instruction.instruction_id, instruction.action)
                return instruction

            if task.direct_conversation_requested_at is not None:
                # Deliver wait_for_user immediately; never long-poll while the decision
                # is happening in the conversation.
                return self._direct_conversation_instruction(task)
            if wait_timeout <= 0.0:
                return self._empty_instruction(task)

            # Register async long-poll waiter
            wait_event = asyncio.Event()
            self._async_instruction_waiters[task_id].append(wait_event)

        # 2. Native AsyncIO wait (outside the lock, zero threadpool consumption!)
        try:
            await asyncio.wait_for(wait_event.wait(), timeout=wait_timeout)
        except (asyncio.TimeoutError, TimeoutError):
            pass
        finally:
            with self._lock:
                waiters = self._async_instruction_waiters.get(task_id, [])
                if wait_event in waiters:
                    waiters.remove(wait_event)

        # 3. Re-acquire lock and check again
        with self._lock:
            task = self._get_existing(task_id)
            if task.pending_instruction and task.pending_instruction.has_instruction:
                instruction = task.pending_instruction
                if pop:
                    previous = copy.deepcopy(task)
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
                    self._persist_task_or_restore(
                        task, previous, True, "instruction consumption"
                    )
                    logger.info("Task %s (async long-polled) obtained instruction %s (%s) -> RECOVERING",
                                task_id, instruction.instruction_id, instruction.action)
                return instruction

            if task.direct_conversation_requested_at is not None:
                return self._direct_conversation_instruction(task)
            return self._empty_instruction(task)

    def ack_instruction(
        self,
        task_id: str,
        instruction_id: Optional[str],
        action: str,
        status: str,
        message: Optional[str] = None,
        *,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> TaskState:
        """Confirm execution of a decision.

        A successful ``cancel`` is terminal (FAILED): cancel means the agent verified
        that the run is gone and the machine was shut down, so it must never report the
        task as RUNNING again. A successful ``self_resolve``/``custom`` returns to
        RUNNING. A failed ACK always leaves the incident open (WAITING), never RUNNING.

        ``instruction_id`` may be omitted after ⏳ 等待用户: the decision happened in a
        direct conversation, there is no mailbox instruction, and this ACK is how the
        agent closes the wait.

        Real post-recovery ``step``/``epoch``/``metrics`` supplied by the agent are
        stored on the same task snapshot as the state transition, so the recovery
        notification and later `get_task_status` reads agree.
        """
        valid_actions = {"self_resolve", "cancel", "custom"}
        normalized_action = (action or "").strip().lower()
        if normalized_action == "stop_training":  # legacy wire name
            normalized_action = "cancel"
        if normalized_action == WAIT_FOR_USER:
            raise ValueError(
                "wait_for_user is a control action, not a decision; stop polling and "
                "ACK self_resolve after resolving the incident with the engineer"
            )
        if normalized_action not in valid_actions:
            raise ValueError(
                f"Invalid action '{action}'. Use one of: {', '.join(sorted(valid_actions))}"
            )
        normalized_status = (status or "").strip().lower()
        if normalized_status not in ("success", "failed"):
            raise ValueError(
                f"Invalid status '{status}'. Use 'success' or 'failed'"
            )

        with self._lock:
            was_cached = task_id in self._tasks
            task = self._get_existing(task_id)
            previous = copy.deepcopy(task)

            if task.state in TERMINAL_STATES:
                raise ValueError(
                    f"Task '{task_id}' is already terminal ({task.state.value}); refusing ACK"
                )

            pending = task.pending_instruction if (
                task.pending_instruction and task.pending_instruction.has_instruction
            ) else None
            latest = task.latest_instruction
            direct_conversation = task.direct_conversation_requested_at is not None
            # ACKing a decision that was never polled is allowed; stage the record and
            # install it only after every validation below passes, so a rejected ACK
            # cannot mutate the cached task.
            promoted_from_pending: Optional[Dict[str, Any]] = None

            if instruction_id:
                if latest and latest.get("instruction_id") == instruction_id:
                    if latest.get("ack"):
                        raise ValueError(
                            f"Instruction '{instruction_id}' was already acknowledged for task '{task_id}'"
                        )
                    decision_action = latest.get("action")
                elif pending and pending.instruction_id == instruction_id:
                    decision_action = pending.action
                    promoted_from_pending = {
                        "instruction_id": pending.instruction_id,
                        "action": pending.action,
                        "payload": pending.payload,
                        "decision_by": pending.decision_by,
                        "popped_at": utc_now_iso(),
                    }
                else:
                    expected = (
                        (latest or {}).get("instruction_id")
                        or (pending.instruction_id if pending else None)
                    )
                    if expected is None:
                        raise ValueError(
                            f"Task '{task_id}' has no instruction to acknowledge; poll first "
                            "or report the event"
                        )
                    raise ValueError(
                        f"Stale instruction ack: got {instruction_id}, expected {expected} for task '{task_id}'"
                    )
                decision_action = (
                    "cancel" if (decision_action or "").strip().lower() == "stop_training"
                    else (decision_action or "").strip().lower()
                )
                if decision_action != normalized_action:
                    raise ValueError(
                        f"Action mismatch for task '{task_id}': decision was '{decision_action}', "
                        f"ACK said '{normalized_action}'"
                    )
                # Validation passed: it is now safe to record the unpolled decision.
                if promoted_from_pending is not None:
                    task.latest_instruction = promoted_from_pending
                    latest = task.latest_instruction
            else:
                # No id is only valid for the direct-conversation path opened by
                # ⏳ 等待用户. A concrete decision always carries its instruction_id.
                if not direct_conversation:
                    if latest and not latest.get("ack"):
                        raise ValueError(
                            f"Instruction '{latest.get('instruction_id')}' requires its "
                            "instruction_id in the ACK"
                        )
                    raise ValueError(
                        f"Task '{task_id}' has no instruction to acknowledge; poll first "
                        "or report the event"
                    )

            task.updated_at = utc_now_iso()
            ack_record = {
                "instruction_id": instruction_id,
                "action": normalized_action,
                "status": normalized_status,
                "message": message,
                "timestamp": task.updated_at,
            }
            if latest and (
                latest.get("instruction_id") == instruction_id or instruction_id is None
            ):
                if instruction_id is not None:
                    latest["ack"] = ack_record

            succeeded = normalized_status == "success"
            if succeeded:
                # The agent's verified post-recovery values become the task's current
                # values, in the same snapshot as the state transition below.
                if step is not None:
                    task.latest_step = step
                if epoch is not None:
                    task.latest_epoch = epoch
                if metrics is not None:
                    task.latest_metrics = metrics
            if succeeded and normalized_action == "cancel":
                # The agent performed and verified the shutdown; the control plane only
                # records the terminal outcome.
                task.state = TaskState.FAILED
                task.pending_instruction = None
                self._clear_direct_conversation(task)
                logger.info("Task %s acknowledged cancel successfully -> FAILED (terminal)",
                            task_id)
            elif succeeded:
                task.state = TaskState.RUNNING
                task.pending_instruction = None
                self._clear_direct_conversation(task)
                logger.info("Task %s acknowledged instruction %s successfully -> RUNNING",
                            task_id, instruction_id)
            else:
                # A failed ACK never means "resumed": the incident stays open. A failed
                # direct-conversation attempt keeps the wait open so the engineer can
                # give another instruction without a new card.
                task.state = TaskState.WAITING
                if not direct_conversation:
                    self._clear_direct_conversation(task)
                logger.warning("Task %s failed executing instruction %s -> WAITING: %s",
                               task_id, instruction_id, message)

            self._persist_task_or_restore(
                task, previous, was_cached, "instruction acknowledgment"
            )

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

    def clean_expired_tasks(self, max_age_seconds: float) -> List[str]:
        """Delete expired terminal tasks from storage and in-memory indexes."""
        with self._lock:
            if not self._storage:
                return []
            expired_ids = self._storage.delete_expired_tasks(max_age_seconds)
            for task_id in expired_ids:
                self._tasks.pop(task_id, None)
                self._async_instruction_waiters.pop(task_id, None)
            return expired_ids

    def get_events(self, task_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Paginated chronological event history."""
        with self._lock:
            return self._storage.get_events(task_id, limit=limit, offset=offset)

    def get_events_count(self, task_id: str) -> Optional[int]:
        with self._lock:
            cnt = self._storage.get_events_count(task_id)
            # If 0 but task not exists, return None
            if cnt == 0 and task_id not in self._tasks:
                if self._storage.get_task(task_id) is None:
                    return None
            return cnt

    def list_tasks(
        self,
        limit: int = 100,
        offset: int = 0,
        state: Optional[TaskState] = None,
    ) -> List[TaskSummary]:
        """List summaries with pagination and optional state filter.

        Uses SQLite pagination directly when available to keep RAM consumption flat (O(limit) instead of O(N)).
        """
        with self._lock:
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

    def reset(self) -> None:
        """Reset all tasks (mainly for testing)."""
        with self._lock:
            self._tasks.clear()
            self._async_instruction_waiters.clear()
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
            latest_step=task.latest_step,
            latest_epoch=task.latest_epoch,
            latest_metrics=task.latest_metrics,
            latest_message=task.latest_message,
            pending_instruction=task.pending_instruction,
            events_count=count,
            direct_conversation_requested_at=task.direct_conversation_requested_at,
            direct_conversation_requested_by=task.direct_conversation_requested_by,
            awaiting_direct_conversation=task.direct_conversation_requested_at is not None,
        )


# Global default mailbox manager. Honor both persistence and retention settings.
from trainpilot.server.config import settings as _settings

default_mailbox = TaskMailboxManager(
    max_events_per_task=_settings.max_events_per_task,
    enable_persistence=_settings.enable_sqlite,
)
