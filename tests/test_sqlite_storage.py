"""Unit tests for SQLite storage persistence and automatic event pruning."""

import os
import tempfile
import pytest

from trainpilot.common.schemas import EventNotifyRequest, InstructionResponse
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
