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


def _alert(mailbox: TaskMailboxManager, task_id: str, message: str = "Loss NaN") -> None:
    mailbox.record_event(
        EventNotifyRequest(task_id=task_id, event_type=EventType.ALERT, message=message, step=6)
    )


def test_defer_alert_keeps_waiting_and_queues_nothing(mailbox: TaskMailboxManager):
    """「等待安排」 pauses the countdown; the agent must keep long-polling."""
    task_id = "test-defer"
    _alert(mailbox, task_id)

    summary = mailbox.defer_alert(task_id, "ou_engineer")
    assert summary.awaiting_arrangement is True
    assert summary.arrangement_requested_by == "ou_engineer"
    assert summary.arrangement_requested_at is not None
    # Nothing was decided, so nothing may be handed to the agent
    assert summary.state == TaskState.WAITING
    assert summary.pending_instruction is None


def test_defer_alert_requires_a_waiting_alert(mailbox: TaskMailboxManager):
    task_id = "test-defer-running"
    mailbox.record_event(
        EventNotifyRequest(task_id=task_id, event_type=EventType.MILESTONE, message="epoch 1")
    )
    with pytest.raises(ValueError, match="only a WAITING alert"):
        mailbox.defer_alert(task_id, "ou_engineer")


def test_defer_alert_does_not_create_an_unknown_task(mailbox: TaskMailboxManager):
    before = mailbox.get_tasks_count()
    with pytest.raises(ValueError, match="not found"):
        mailbox.defer_alert("never-reported", "ou_engineer")
    assert mailbox.get_tasks_count() == before


def test_alert_stage_auto_resolve_is_disabled_while_deferred(mailbox: TaskMailboxManager):
    """The 30s timer may already be in flight when the human asks for time."""
    task_id = "test-deferred-no-timeout"
    _alert(mailbox, task_id)
    mailbox.defer_alert(task_id, "ou_engineer")

    assert mailbox.try_auto_resolve(task_id, timeout_seconds=0) is None
    assert mailbox.try_auto_resolve(task_id, timeout_seconds=0, stage="alert") is None
    assert mailbox.get_task(task_id).state == TaskState.WAITING


def test_arrangement_stage_auto_resolve_needs_an_open_window(mailbox: TaskMailboxManager):
    task_id = "test-arrangement-fallback"
    _alert(mailbox, task_id)

    # No deferral: the fallback stage must not fire on a plain alert
    assert mailbox.try_auto_resolve(task_id, timeout_seconds=0, stage="arrangement") is None

    mailbox.defer_alert(task_id, "ou_engineer")
    instruction = mailbox.try_auto_resolve(task_id, timeout_seconds=0, stage="arrangement")
    assert instruction is not None
    assert instruction.action == "self_resolve"
    assert mailbox.get_task(task_id).state == TaskState.RESOLVED
    # Closing the window is part of queueing the instruction
    assert mailbox.get_task(task_id).awaiting_arrangement is False


def test_auto_resolve_rejects_an_unknown_stage(mailbox: TaskMailboxManager):
    """A typo must fail loudly rather than silently defaulting to the short window."""
    _alert(mailbox, "test-bad-stage")
    with pytest.raises(ValueError, match="Unknown auto-resolve stage"):
        mailbox.try_auto_resolve("test-bad-stage", timeout_seconds=0, stage="arrangment")


@pytest.mark.asyncio
async def test_submit_fix_decision_dispatches_custom_and_wakes_the_poll(
    mailbox: TaskMailboxManager,
):
    task_id = "test-fix-dispatch"
    _alert(mailbox, task_id)
    mailbox.defer_alert(task_id, "ou_engineer")

    instruction, created = mailbox.submit_fix_decision(
        task_id=task_id,
        solution="回滚到 step4 检查点并把 lr 减半",
        operator="ou_engineer",
        source_message_id="om_1",
    )
    assert created is True
    assert instruction.action == "custom"
    assert instruction.payload["solution"] == "回滚到 step4 检查点并把 lr 减半"
    assert instruction.decision_by == "ou_engineer"

    polled = await mailbox.get_instruction_async(task_id, pop=True, wait_timeout=0)
    assert polled.has_instruction is True
    assert polled.action == "custom"
    assert polled.payload["solution"] == "回滚到 step4 检查点并把 lr 减半"
    assert mailbox.get_task(task_id).state == TaskState.RECOVERING


def test_submit_fix_decision_requires_a_waiting_task(mailbox: TaskMailboxManager):
    task_id = "test-fix-not-waiting"
    mailbox.record_event(
        EventNotifyRequest(task_id=task_id, event_type=EventType.MILESTONE, message="epoch 1")
    )
    with pytest.raises(ValueError, match="no alert waiting"):
        mailbox.submit_fix_decision(task_id, solution="重启", operator="ou_engineer")


def test_submit_fix_decision_never_creates_a_task_row(mailbox: TaskMailboxManager):
    before = mailbox.get_tasks_count()
    with pytest.raises(ValueError, match="not found"):
        mailbox.submit_fix_decision("never-reported", solution="重启", operator="ou_engineer")
    assert mailbox.get_tasks_count() == before


@pytest.mark.asyncio
async def test_submit_fix_decision_is_idempotent_per_message_id(mailbox: TaskMailboxManager):
    """Feishu redelivers an unconfirmed event; it must not decide twice."""
    task_id = "test-fix-idempotent"
    _alert(mailbox, task_id)

    first, created = mailbox.submit_fix_decision(
        task_id, solution="回滚到 step4", operator="ou_engineer", source_message_id="om_dup"
    )
    assert created is True

    # 1. Redelivery while the instruction is still queued
    again, created = mailbox.submit_fix_decision(
        task_id, solution="回滚到 step4", operator="ou_engineer", source_message_id="om_dup"
    )
    assert (again, created) == (None, False)
    assert mailbox.get_task(task_id).pending_instruction.instruction_id == first.instruction_id

    # 2. Redelivery after the agent already polled it away
    await mailbox.get_instruction_async(task_id, pop=True, wait_timeout=0)
    third, created = mailbox.submit_fix_decision(
        task_id, solution="回滚到 step4", operator="ou_engineer", source_message_id="om_dup"
    )
    assert (third, created) == (None, False)
    assert mailbox.get_task(task_id).state == TaskState.RECOVERING

    # 3. A genuinely new message still gets through (once the task waits again)
    mailbox.record_event(
        EventNotifyRequest(task_id=task_id, event_type=EventType.ALERT, message="Loss spike")
    )
    _, created = mailbox.submit_fix_decision(
        task_id, solution="换一条路", operator="ou_engineer", source_message_id="om_other"
    )
    assert created is True


@pytest.mark.asyncio
async def test_a_new_alert_clears_a_stale_arrangement_request(mailbox: TaskMailboxManager):
    """Otherwise the next alert's 30s window could never fire again."""
    task_id = "test-stale-defer"
    _alert(mailbox, task_id, message="Loss NaN")
    mailbox.defer_alert(task_id, "ou_engineer")
    assert mailbox.get_task(task_id).awaiting_arrangement is True

    # Same alert redelivered: the window belongs to this alert, so it stays open
    mailbox.record_event(
        EventNotifyRequest(task_id=task_id, event_type=EventType.ALERT, message="Loss NaN", step=6)
    )
    assert mailbox.get_task(task_id).awaiting_arrangement is True

    # A different alert opens a fresh decision window
    mailbox.record_event(
        EventNotifyRequest(task_id=task_id, event_type=EventType.ALERT, message="CUDA OOM", step=9)
    )
    summary = mailbox.get_task(task_id)
    assert summary.awaiting_arrangement is False
    assert summary.arrangement_requested_by is None
    assert mailbox.try_auto_resolve(task_id, timeout_seconds=0) is not None


def test_any_decision_closes_the_arrangement_window(mailbox: TaskMailboxManager):
    task_id = "test-decision-closes-window"
    _alert(mailbox, task_id)
    mailbox.defer_alert(task_id, "ou_engineer")

    mailbox.submit_decision(task_id, action="stop_training", operator="ou_engineer")
    assert mailbox.get_task(task_id).awaiting_arrangement is False


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
