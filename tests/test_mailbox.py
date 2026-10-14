"""Unit tests for TaskMailboxManager and state machine transitions."""

import pytest
from trainpilot.common.schemas import (
    EventNotifyRequest,
)
from trainpilot.common.states import EventType, TaskState
from trainpilot.server.mailbox import TaskMailboxManager


@pytest.fixture
def mailbox():
    return TaskMailboxManager()


@pytest.mark.asyncio
async def test_mailbox_lifecycle_state_machine(mailbox: TaskMailboxManager):
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

    # 3. Before decision, polling returns has_instruction=False
    inst = await mailbox.get_instruction_async(task_id, pop=False)
    assert inst.has_instruction is False
    assert inst.status == TaskState.WAITING

    # 4. Human submits decision -> RESOLVED
    inst_submitted = mailbox.submit_decision(
        task_id=task_id,
        action="self_resolve",
        operator="lead_researcher",
    )
    assert inst_submitted.has_instruction is True
    assert inst_submitted.action == "self_resolve"

    task = mailbox.get_task(task_id)
    assert task.state == TaskState.RESOLVED

    # 5. Agent polls with pop=True -> RECOVERING
    inst_polled = await mailbox.get_instruction_async(task_id, pop=True)
    assert inst_polled.has_instruction is True
    assert inst_polled.action == "self_resolve"

    task_after_pop = mailbox.get_task(task_id)
    assert task_after_pop.state == TaskState.RECOVERING

    # Subsequent poll before ack returns has_instruction=False
    empty_poll = await mailbox.get_instruction_async(task_id, pop=True)
    assert empty_poll.has_instruction is False

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


def test_mailbox_lazily_reloads_an_evicted_task_from_storage(mailbox: TaskMailboxManager):
    """Counts are served by storage, and get_task restores a task evicted from RAM."""
    before = mailbox.get_tasks_count()

    rec = mailbox._get_or_create("test-lazy-evicted-task")
    rec.state = TaskState.COMPLETED
    mailbox._storage.save_task(rec)
    assert mailbox.get_tasks_count() == before + 1
    assert mailbox.get_events_count(rec.task_id) == 0

    # Manually remove from RAM
    mailbox._tasks.pop(rec.task_id, None)
    assert rec.task_id not in mailbox._tasks

    # get_task should lazily restore summary from SQLite
    summary = mailbox.get_task(rec.task_id)
    assert summary is not None
    assert summary.task_id == rec.task_id
    assert summary.state == TaskState.COMPLETED
    assert mailbox.get_tasks_count() == before + 1


@pytest.mark.asyncio
async def test_control_calls_do_not_create_unknown_tasks(mailbox: TaskMailboxManager):
    """Read/control typos should return not-found without creating ghost tasks."""
    before = mailbox.get_tasks_count()

    with pytest.raises(ValueError, match="not found"):
        await mailbox.get_instruction_async("missing-task", wait_timeout=0)
    with pytest.raises(ValueError, match="not found"):
        mailbox.submit_decision("missing-task", action="self_resolve")
    with pytest.raises(ValueError, match="not found"):
        mailbox.ack_instruction(
            "missing-task",
            instruction_id=None,
            action="self_resolve",
            status="success",
        )

    assert mailbox.get_tasks_count() == before
