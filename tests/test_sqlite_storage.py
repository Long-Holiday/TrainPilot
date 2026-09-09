"""Unit tests for SQLite storage persistence and automatic event pruning."""

import os
import tempfile
import pytest

from trainpilot.common.schemas import EventNotifyRequest, HeartbeatRequest, InstructionResponse
from trainpilot.common.states import EventType, TaskState
from trainpilot.server.mailbox import TaskMailboxManager, TaskRecord
from trainpilot.server.storage import SQLiteStorage


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def test_sqlite_storage_crud_and_reload(temp_db):
    storage = SQLiteStorage(temp_db)

    # 1. Save task
    task = TaskRecord(
        task_id="task-test-01",
        state=TaskState.WAITING,
        latest_step=100,
        latest_epoch=2,
        latest_metrics={"loss": 0.35},
        latest_message="Loss spike detected",
        pending_instruction=InstructionResponse(
            ready=True,
            status=TaskState.RESOLVED,
            instruction_id="inst_123",
            action="self_resolve",
        ),
    )
    storage.save_task(task)

    # 2. Append events
    for i in range(1, 6):
        storage.append_event(
            task_id="task-test-01",
            event_entry={
                "event_type": "milestone",
                "message": f"Step {i} completed",
                "step": i,
                "epoch": 1,
                "metrics": {"loss": 0.5 - i * 0.05},
                "extra": None,
                "agent_note": f"Agent note {i}",
                "timestamp": f"2026-09-06T10:00:0{i}Z",
            },
            max_events=10,
        )

    assert storage.get_events_count("task-test-01") == 5
    events = storage.get_events("task-test-01", limit=3, offset=0)
    assert len(events) == 3
    assert events[0]["step"] == 1
    assert events[0]["agent_note"] == "Agent note 1"

    # 3. Reload from fresh storage instance (simulating server restart)
    storage.close()
    reloaded_storage = SQLiteStorage(temp_db)
    all_tasks = reloaded_storage.load_all_tasks()
    assert "task-test-01" in all_tasks
    loaded_task = all_tasks["task-test-01"]
    assert loaded_task.state == TaskState.WAITING
    assert loaded_task.latest_step == 100
    assert loaded_task.latest_metrics == {"loss": 0.35}
    assert loaded_task.pending_instruction is not None
    assert loaded_task.pending_instruction.action == "self_resolve"
    assert reloaded_storage.get_events_count("task-test-01") == 5
    reloaded_storage.close()


def test_sqlite_storage_automatic_retention_pruning(temp_db):
    """Verify that when event count exceeds max_events, old events are automatically deleted."""
    storage = SQLiteStorage(temp_db)
    max_events_cap = 5

    # Insert 12 events with max_events=5
    for i in range(1, 13):
        storage.append_event(
            task_id="task-pruning-01",
            event_entry={
                "event_type": "milestone",
                "message": f"Step {i}",
                "step": i,
                "epoch": 1,
                "metrics": {"loss": 0.1},
                "extra": None,
                "agent_note": None,
                "timestamp": f"2026-09-06T10:00:{i:02d}Z",
            },
            max_events=max_events_cap,
        )

    # Count must be capped at 5
    stored_count = storage.get_events_count("task-pruning-01")
    assert stored_count == max_events_cap

    # The oldest events (1..7) must have been deleted; only 8..12 should remain
    events = storage.get_events("task-pruning-01", limit=10, offset=0)
    assert len(events) == max_events_cap
    steps = [e["step"] for e in events]
    assert steps == [8, 9, 10, 11, 12]

    storage.close()


def test_mailbox_integrated_with_sqlite_storage(temp_db):
    """Test TaskMailboxManager integrated with SQLite storage."""
    storage = SQLiteStorage(temp_db)
    mailbox = TaskMailboxManager(max_events_per_task=4, storage=storage)

    # 1. Record milestone event
    req = EventNotifyRequest(
        task_id="task-mb-sql",
        event_type=EventType.MILESTONE,
        message="Milestone 1",
        step=50,
        epoch=1,
        metrics={"loss": 0.42},
    )
    mailbox.record_event(req)

    summary = mailbox.get_task("task-mb-sql")
    assert summary is not None
    assert summary.state == TaskState.RUNNING
    assert summary.events_count == 1

    # 2. Submit decision & poll
    mailbox.submit_decision("task-mb-sql", action="self_resolve", operator="tester")
    instruction = mailbox.get_instruction("task-mb-sql", pop=True)
    assert instruction.ready is True
    assert instruction.action == "self_resolve"
    assert mailbox.get_task("task-mb-sql").state == TaskState.RECOVERING

    # 3. Simulate restart: create another mailbox from same storage
    mailbox2 = TaskMailboxManager(max_events_per_task=4, storage=SQLiteStorage(temp_db))
    summary2 = mailbox2.get_task("task-mb-sql")
    assert summary2 is not None
    assert summary2.state == TaskState.RECOVERING
    assert summary2.latest_step == 50
    assert summary2.events_count == 1


def test_storage_checkpoint_and_vacuum(temp_db):
    """Verify PRAGMA wal_checkpoint and vacuum execution."""
    storage = SQLiteStorage(temp_db)
    res = storage.checkpoint_and_vacuum()
    assert "checkpoint" in res
    storage.close()


def test_storage_tasks_count_and_active_only(temp_db):
    """Verify get_tasks_count and active_only loading for memory saving."""
    storage = SQLiteStorage(temp_db)
    t1 = TaskRecord(task_id="active-task-1", state=TaskState.RUNNING)
    t2 = TaskRecord(task_id="completed-task-1", state=TaskState.COMPLETED)
    t3 = TaskRecord(task_id="failed-task-1", state=TaskState.FAILED)
    storage.save_task(t1)
    storage.save_task(t2)
    storage.save_task(t3)

    assert storage.get_tasks_count() == 3
    assert storage.get_tasks_count(TaskState.RUNNING) == 1
    assert storage.get_tasks_count(TaskState.COMPLETED) == 1

    active_tasks = storage.load_all_tasks(active_only=True)
    assert "active-task-1" in active_tasks
    assert "completed-task-1" not in active_tasks
    assert "failed-task-1" not in active_tasks
    storage.close()


def test_storage_clean_expired_tasks(temp_db):
    """Verify clean_expired_tasks deletes old completed/failed tasks."""
    storage = SQLiteStorage(temp_db)
    t_old = TaskRecord(
        task_id="old-completed",
        state=TaskState.COMPLETED,
        updated_at="2020-01-01T00:00:00+00:00",
    )
    t_new = TaskRecord(
        task_id="recent-completed",
        state=TaskState.COMPLETED,
        updated_at="2099-01-01T00:00:00+00:00",
    )
    storage.save_task(t_old)
    storage.save_task(t_new)

    deleted = storage.clean_expired_tasks(max_age_seconds=3600)
    assert deleted == 1
    assert storage.get_task("old-completed") is None
    assert storage.get_task("recent-completed") is not None
    storage.close()


def test_storage_cleanup_deletes_events_but_preserves_active_tasks(temp_db):
    """Expired terminal cleanup is atomic in scope and never touches active tasks."""
    storage = SQLiteStorage(temp_db)
    old_timestamp = "2020-01-01T00:00:00+00:00"
    expired = TaskRecord(
        task_id="expired-failed",
        state=TaskState.FAILED,
        updated_at=old_timestamp,
    )
    active = TaskRecord(
        task_id="old-but-running",
        state=TaskState.RUNNING,
        updated_at=old_timestamp,
    )
    storage.save_task(expired)
    storage.save_task(active)
    for task_id in (expired.task_id, active.task_id):
        storage.append_event(
            task_id,
            {
                "event_type": "milestone",
                "message": "event",
                "timestamp": old_timestamp,
            },
        )

    expired_ids = storage.delete_expired_tasks(max_age_seconds=3600)

    assert expired_ids == ["expired-failed"]
    assert storage.get_task(expired.task_id) is None
    assert storage.get_events_count(expired.task_id) == 0
    assert storage.get_task(active.task_id) is not None
    assert storage.get_events_count(active.task_id) == 1
    storage.close()


def test_mailbox_cleanup_evicts_expired_task_from_memory(temp_db):
    """A database cleanup must not leave a stale task in the mailbox cache."""
    storage = SQLiteStorage(temp_db)
    mailbox = TaskMailboxManager(storage=storage)
    task = mailbox._get_or_create("cached-completed")
    task.state = TaskState.COMPLETED
    task.updated_at = "2020-01-01T00:00:00+00:00"
    storage.save_task(task)

    expired_ids = mailbox.clean_expired_tasks(max_age_seconds=3600)

    assert expired_ids == ["cached-completed"]
    assert "cached-completed" not in mailbox._tasks
    assert mailbox.get_task("cached-completed") is None
    storage.close()


def test_task_and_event_write_roll_back_together(temp_db, monkeypatch):
    """A failed event insert must not leave the task snapshot half-updated."""
    storage = SQLiteStorage(temp_db)
    mailbox = TaskMailboxManager(storage=storage)
    mailbox.record_event(
        EventNotifyRequest(
            task_id="atomic-task",
            event_type=EventType.MILESTONE,
            message="before",
            step=1,
        )
    )

    original_append = storage.append_event

    def fail_append(*args, **kwargs):
        raise RuntimeError("simulated insert failure")

    monkeypatch.setattr(storage, "append_event", fail_append)
    with pytest.raises(RuntimeError, match="simulated insert failure"):
        mailbox.record_event(
            EventNotifyRequest(
                task_id="atomic-task",
                event_type=EventType.MILESTONE,
                message="after",
                step=2,
            )
        )

    monkeypatch.setattr(storage, "append_event", original_append)
    assert storage.get_task("atomic-task").latest_step == 1
    assert mailbox.get_task("atomic-task").latest_step == 1
    assert storage.get_events_count("atomic-task") == 1
    storage.close()


def test_disabled_persistence_uses_process_local_database():
    """Disabling persistence must not create or reuse the configured disk DB."""
    mailbox = TaskMailboxManager(enable_persistence=False)
    assert mailbox._storage.db_path == ":memory:"
    mailbox.record_heartbeat(HeartbeatRequest(task_id="memory-only", step=1))
    assert mailbox.get_task("memory-only") is not None
    mailbox._storage.close()
