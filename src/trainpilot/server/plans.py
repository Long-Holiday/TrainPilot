"""Next-task plans queued from the Feishu group and read back by the GPU Agent.

A plan has no state machine, no long-poll wakeup and no task row: it is a one-table
CRUD log that lives outside `TaskMailboxManager` so "only `record_event` creates task
rows" stays trivially auditable. Persistence reuses the mailbox's SQLite connection
(same pattern as `cleanup.py`) to keep a single connection behind a single lock.

Every plan is bound to exactly one task, and that task must already exist: a plan
nobody's `report` ever carries is invisible and exempt from cleanup, so it would sit in
the table forever. The one task-row interaction here is a read-only existence probe --
nothing in this module writes task state.
"""

import logging
from typing import Any, List, Optional, Tuple
from uuid import uuid4

from trainpilot.common.schemas import NextTaskPlan, utc_now_iso
from trainpilot.common.states import PlanStatus

logger = logging.getLogger("trainpilot.plans")

# Cap on how many pending plans one report response carries.
MAX_PLANS_PER_REPORT = 20
MAX_PLAN_TEXT_CHARS = 2000
MAX_ACK_NOTE_CHARS = 500

# Statuses the agent may set; PENDING is server-owned.
ACK_STATUSES = (PlanStatus.DONE.value, PlanStatus.FAILED.value, PlanStatus.SKIPPED.value)


class PlanBindingError(ValueError):
    """A plan's binding is unusable: it named no task, or a task with no row behind it.

    `task_id` is None when the message named no task at all, so the caller can tell the
    two cases apart when it reports back. Defined here rather than reusing
    `mailbox.TaskNotFoundError`: importing the mailbox module from this one at import
    time would build `default_mailbox` before conftest redirects the database path,
    which is exactly what `_get_storage`'s lazy import avoids.
    """

    def __init__(self, task_id: Optional[str] = None):
        self.task_id = task_id
        super().__init__(f"Task '{task_id}' not found" if task_id else "Plan names no task")



class PlanRegistry:
    """Validates and persists next-task plans; never touches task rows."""

    def __init__(self, storage: Optional[Any] = None):
        self._storage_override = storage

    def _get_storage(self) -> Any:
        if self._storage_override is not None:
            return self._storage_override
        # Lazy import: mailbox imports nothing from this module, but resolving the
        # singleton at call time keeps the conftest DB redirect authoritative.
        from trainpilot.server.mailbox import default_mailbox

        return default_mailbox._storage

    def register(
        self,
        text: str,
        task_id: Optional[str] = None,
        created_by: Optional[str] = None,
        chat_id: Optional[str] = None,
        source_message_id: Optional[str] = None,
        source: str = "feishu",
    ) -> Tuple[NextTaskPlan, bool]:
        """Store one plan. Returns (plan, created); created=False for a retried message.

        Raises `PlanBindingError` when no task was named, or the named task has no row.
        """
        body = (text or "").strip()
        if not body:
            raise ValueError("Plan text must not be empty")
        if len(body) > MAX_PLAN_TEXT_CHARS:
            logger.warning(
                "Truncating next-task plan from %d to %d chars", len(body), MAX_PLAN_TEXT_CHARS
            )
            body = body[:MAX_PLAN_TEXT_CHARS]

        task_id = task_id or None
        if task_id is None:
            raise PlanBindingError()
        # Read-only probe of the task table: a plan bound to an id that never reports is
        # unreachable by every task and exempt from cleanup, so it would sit in the
        # table forever. Nothing here writes task state -- see the module docstring.
        if self._get_storage().get_task(task_id) is None:
            raise PlanBindingError(task_id)

        plan = NextTaskPlan(
            plan_id=f"plan_{uuid4().hex[:12]}",
            text=body,
            task_id=task_id,
            created_at=utc_now_iso(),
            created_by=created_by,
            source=source,
            chat_id=chat_id,
            source_message_id=source_message_id,
        )
        stored, created = self._get_storage().add_plan(plan)
        if created:
            logger.info(
                "Registered next-task plan %s (task=%s) from %s",
                stored.plan_id,
                stored.task_id or "global",
                created_by or "unknown",
            )
        return stored, created

    def list_pending(
        self, task_id: Optional[str] = None, limit: int = MAX_PLANS_PER_REPORT
    ) -> List[NextTaskPlan]:
        """Pending plans bound to one task; a caller with no task id sees none."""
        if not task_id:
            return []
        plans = self._get_storage().list_pending_plans(task_id, limit)
        if len(plans) >= limit:
            logger.warning(
                "Next-task plan backlog reached the %d-plan cap for task %s; older "
                "pending plans are not being returned",
                limit,
                task_id,
            )
        return plans

    def ack(self, plan_id: str, status: str = PlanStatus.DONE.value, note: str = "") -> NextTaskPlan:
        """Acknowledge a plan so it stops appearing in report responses."""
        normalized = (status or "").strip().lower()
        if normalized not in ACK_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. Use one of: {', '.join(ACK_STATUSES)}"
            )
        trimmed_note = (note or "").strip()[:MAX_ACK_NOTE_CHARS] or None
        return self._get_storage().ack_plan(
            plan_id, normalized, trimmed_note, utc_now_iso()
        )

    def clear_all_pending(self) -> int:
        """Drop every unacknowledged plan; returns how many were removed."""
        return self._get_storage().clear_all_pending_plans()

    def clear_pending_for_task(self, task_id: str) -> int:
        """Drop unacknowledged plans bound to one task; returns how many were removed."""
        return self._get_storage().clear_pending_plans_for_task(task_id)

    def delete_expired(self, max_age_seconds: float) -> int:
        """Reap acknowledged plans past retention; pending plans are never reaped."""
        return self._get_storage().delete_expired_plans(max_age_seconds)

    def reset(self) -> None:
        """Drop every plan row (tests)."""
        self._get_storage().clear_plans()


default_plans = PlanRegistry()
