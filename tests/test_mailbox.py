"""Unit tests for TaskMailboxManager and state machine transitions."""

import pytest
from trainpilot.common.schemas import (
    EventNotifyRequest,
    HeartbeatRequest,
)
from trainpilot.common.states import EventType, TaskState
from trainpilot.server.mailbox import TaskMailboxManager


@pytest.fixture
def mailbox():
    return TaskMailboxManager()


def test_mailbox_lifecycle_state_machine(mailbox: TaskMailboxManager):
    task_id = "test-task-lifecycle"

    # 1. Start with milestone -> RUNNING
    m_req = EventNotifyRequest(
        task_id=task_id,
        event_type=EventType.MILESTONE,
        message="Epoch 1 started",
        step=0,
    )
    st = mailbox.record_event(m_req)
    assert st == TaskState.RUNNING

    # 2. Trigger alert -> WAITING
    a_req = EventNotifyRequest(
        task_id=task_id,
        event_type=EventType.ALERT,
        message="Loss NaN at step 50",
        step=50,
        metrics={"loss": 9999.0},
    )
    st = mailbox.record_event(a_req)
    assert st == TaskState.WAITING

    # 3. Before decision, polling returns not ready
    inst = mailbox.get_instruction(task_id, pop=False)
    assert inst.ready is False
    assert inst.status == TaskState.WAITING

    # 4. Human submits decision -> RESOLVED
    inst_submitted = mailbox.submit_decision(
        task_id=task_id,
        action="self_resolve",
        operator="lead_researcher",
    )
    assert inst_submitted.ready is True
    assert inst_submitted.action == "self_resolve"

    task = mailbox.get_task(task_id)
    assert task.state == TaskState.RESOLVED

    # 5. Agent polls with pop=True -> RECOVERING
    inst_polled = mailbox.get_instruction(task_id, pop=True)
    assert inst_polled.ready is True
    assert inst_polled.action == "self_resolve"

    task_after_pop = mailbox.get_task(task_id)
    assert task_after_pop.state == TaskState.RECOVERING

    # Subsequent poll before ack returns not ready
    empty_poll = mailbox.get_instruction(task_id, pop=True)
    assert empty_poll.ready is False

    # 6. Agent acks success -> RUNNING
    st_after_ack = mailbox.ack_instruction(
        task_id=task_id,
        instruction_id=inst_polled.instruction_id,
        action="self_resolve",
        status="success",
        message="Self-resolved and continued",
    )
    assert st_after_ack == TaskState.RUNNING

    # 7. Complete task
    c_req = EventNotifyRequest(
        task_id=task_id,
        event_type=EventType.COMPLETED,
        message="Training finished",
        step=100,
    )
    st_final = mailbox.record_event(c_req)
    assert st_final == TaskState.COMPLETED


def test_heartbeat_updates_metadata(mailbox: TaskMailboxManager):
    task_id = "test-heartbeat"
    hb = HeartbeatRequest(task_id=task_id, step=25, metrics={"gpu_mem": 0.85})
    mailbox.record_heartbeat(hb)

    task = mailbox.get_task(task_id)
    assert task is not None
    assert task.latest_step == 25
    assert task.latest_metrics["gpu_mem"] == 0.85
    assert task.last_heartbeat_at is not None


def test_mailbox_tasks_count(mailbox: TaskMailboxManager):
    """Verify get_tasks_count behaves consistently."""
    initial_count = mailbox.get_tasks_count()
    mailbox._get_or_create("count-task-1")
    mailbox._get_or_create("count-task-2")
    assert mailbox.get_tasks_count() == initial_count + 2


def test_mailbox_lazy_load_from_storage(mailbox: TaskMailboxManager):
    """Verify get_task can lazily retrieve a task from storage even if evicted from RAM."""
    task_id = "test-lazy-evicted-task"
    rec = mailbox._get_or_create(task_id)
    rec.state = TaskState.COMPLETED
    mailbox._storage.save_task(rec)

    # Manually remove from RAM
    mailbox._tasks.pop(task_id, None)
    assert task_id not in mailbox._tasks

    # get_task should lazily restore summary from SQLite
    summary = mailbox.get_task(task_id)
    assert summary is not None
    assert summary.task_id == task_id
    assert summary.state == TaskState.COMPLETED
