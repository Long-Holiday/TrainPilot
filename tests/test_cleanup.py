"""Unit tests for TaskCleanupService and retention housekeeping."""

from trainpilot.common.states import TaskState
from trainpilot.server.cleanup import TaskCleanupService
from trainpilot.server.config import ServerSettings
from trainpilot.server.mailbox import TaskMailboxManager
from trainpilot.server.storage import SQLiteStorage


def test_scheduled_task_cleanup_runs(tmp_path):
    """Cleanup runs immediately, then respects its own interval."""
    storage = SQLiteStorage(str(tmp_path / "cleanup.db"))
    mailbox = TaskMailboxManager(storage=storage)
    custom_settings = ServerSettings(
        enable_task_cleanup=True,
        task_retention_hours=1,
        task_cleanup_interval_seconds=3600,
    )
    cleanup = TaskCleanupService(mailbox=mailbox, server_settings=custom_settings)

    first = mailbox._get_or_create("expired-first")
    first.state = TaskState.COMPLETED
    first.updated_at = "2020-01-01T00:00:00+00:00"
    storage.save_task(first)

    result = cleanup.run_cleanup_once()
    assert result["tasks_expired"] == 1
    assert mailbox.get_task("expired-first") is None
    assert cleanup.get_stats()["tasks_expired_total"] == 1
    assert cleanup.get_stats()["last_cleanup_at"] is not None

    second = mailbox._get_or_create("expired-second")
    second.state = TaskState.FAILED
    second.updated_at = "2020-01-01T00:00:00+00:00"
    storage.save_task(second)

    # A regular sweep before interval elapsed must not clean tasks
    assert cleanup.run_cleanup_once()["tasks_expired"] == 0
    assert mailbox.get_task("expired-second") is not None

    cleanup._last_cleanup_monotonic -= 3600
    assert cleanup.run_cleanup_once()["tasks_expired"] == 1
    assert mailbox.get_task("expired-second") is None
    storage.close()


def test_cleanup_expires_acked_plans_but_never_pending_ones(tmp_path):
    """A plan must not silently disappear before the agent ran it."""
    from trainpilot.common.schemas import EventNotifyRequest
    from trainpilot.common.states import EventType
    from trainpilot.server.plans import PlanRegistry

    storage = SQLiteStorage(str(tmp_path / "plans.db"))
    registry = PlanRegistry(storage=storage)
    mailbox = TaskMailboxManager(storage=storage)
    cleanup = TaskCleanupService(
        mailbox=mailbox,
        server_settings=ServerSettings(
            enable_task_cleanup=True,
            plan_retention_hours=168,
            task_cleanup_interval_seconds=3600,
        ),
    )

    # Plans are bound to a task that must already exist; a report is what creates one.
    mailbox.record_event(
        EventNotifyRequest(task_id="task-a", event_type=EventType.MILESTONE, message="step 1 done")
    )
    acked, _ = registry.register("旧计划", task_id="task-a")
    registry.ack(acked.plan_id)
    pending, _ = registry.register("还没做", task_id="task-a")

    # Force every acknowledged plan past its retention window.
    cleanup.settings.plan_retention_hours = -1
    result = cleanup.run_cleanup_once()

    assert result["plans_expired"] == 1
    assert cleanup.get_stats()["plans_expired_total"] == 1
    assert cleanup.get_stats()["plan_retention_hours"] == -1
    assert storage.get_plan(acked.plan_id) is None
    assert storage.get_plan(pending.plan_id) is not None
    storage.close()
