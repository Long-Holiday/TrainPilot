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


def test_wait_for_direct_conversation_keeps_waiting_and_queues_nothing(
    mailbox: TaskMailboxManager,
):
    """「等待用户」 keeps the scene and hands the decision to a direct conversation."""
    task_id = "test-direct-wait"
    _alert(mailbox, task_id)

    summary = mailbox.wait_for_direct_conversation(task_id, "ou_engineer")
    assert summary.awaiting_direct_conversation is True
    assert summary.direct_conversation_requested_by == "ou_engineer"
    assert summary.direct_conversation_requested_at is not None
    # Nothing was decided, so nothing may be handed to the agent
    assert summary.state == TaskState.WAITING
    assert summary.pending_instruction is None


def test_wait_for_direct_conversation_requires_a_waiting_alert(mailbox: TaskMailboxManager):
    task_id = "test-direct-wait-running"
    mailbox.record_event(
        EventNotifyRequest(task_id=task_id, event_type=EventType.MILESTONE, message="epoch 1")
    )
    with pytest.raises(ValueError, match="only a WAITING alert"):
        mailbox.wait_for_direct_conversation(task_id, "ou_engineer")


def test_wait_for_direct_conversation_does_not_create_an_unknown_task(
    mailbox: TaskMailboxManager,
):
    before = mailbox.get_tasks_count()
    with pytest.raises(ValueError, match="not found"):
        mailbox.wait_for_direct_conversation("never-reported", "ou_engineer")
    assert mailbox.get_tasks_count() == before


def test_alert_auto_resolve_is_disabled_while_waiting_for_direct_conversation(
    mailbox: TaskMailboxManager,
):
    """The 30s timer may already be in flight when the human takes over the decision."""
    task_id = "test-direct-wait-no-timeout"
    _alert(mailbox, task_id)
    mailbox.wait_for_direct_conversation(task_id, "ou_engineer")

    assert mailbox.try_auto_resolve(task_id, timeout_seconds=0) is None
    assert mailbox.get_task(task_id).state == TaskState.WAITING


def test_the_default_alert_timeout_still_fires_without_a_wait(mailbox: TaskMailboxManager):
    """The ordinary 30s self-resolve is unchanged; only the wait choice suppresses it."""
    task_id = "test-default-timeout"
    _alert(mailbox, task_id)

    instruction = mailbox.try_auto_resolve(task_id, timeout_seconds=0)
    assert instruction is not None
    assert instruction.action == "self_resolve"
    assert mailbox.get_task(task_id).state == TaskState.RESOLVED


@pytest.mark.asyncio
async def test_wait_wakes_a_long_poll_and_returns_control(mailbox: TaskMailboxManager):
    """No instruction will arrive, so the poll must return at once instead of hanging."""
    import asyncio
    import time

    task_id = "test-direct-wait-wakes-poll"
    _alert(mailbox, task_id)

    async def _poll():
        started = time.monotonic()
        result = await mailbox.get_instruction_async(task_id, pop=True, wait_timeout=5.0)
        return result, time.monotonic() - started

    poll_task = asyncio.create_task(_poll())
    await asyncio.sleep(0.1)
    mailbox.wait_for_direct_conversation(task_id, "ou_engineer")

    result, elapsed = await asyncio.wait_for(poll_task, timeout=1.0)
    # The poll must deliver a control action, not merely time out with nothing:
    # otherwise the agent has no signal that it should stop polling.
    assert result.has_instruction is True
    assert result.action == "wait_for_user"
    assert result.instruction_id is None
    assert result.awaiting_direct_conversation is True
    assert result.status == TaskState.WAITING
    assert elapsed < 1.0
    # Delivering the control action does not change the state or queue a decision.
    task = mailbox.get_task(task_id)
    assert task.state == TaskState.WAITING
    assert task.pending_instruction is None


@pytest.mark.asyncio
async def test_ack_without_an_instruction_closes_the_direct_conversation(
    mailbox: TaskMailboxManager,
):
    """The engineer's direct solution reaches the agent outside the mailbox; the ACK is
    the execution receipt and must be able to end the wait."""
    task_id = "test-direct-wait-ack"
    _alert(mailbox, task_id)
    mailbox.wait_for_direct_conversation(task_id, "ou_engineer")

    state = mailbox.ack_instruction(
        task_id=task_id,
        instruction_id=None,
        action="self_resolve",
        status="success",
        message="按对话中的方案回滚到 step 4 并继续",
    )
    assert state == TaskState.RUNNING
    summary = mailbox.get_task(task_id)
    assert summary.awaiting_direct_conversation is False
    assert summary.direct_conversation_requested_at is None


def test_a_failed_direct_ack_keeps_the_wait_open_for_another_try(
    mailbox: TaskMailboxManager,
):
    """A failed attempt must not lock the task: the conversation is still the way out."""
    task_id = "test-direct-wait-failed-ack"
    _alert(mailbox, task_id)
    mailbox.wait_for_direct_conversation(task_id, "ou_engineer")

    state = mailbox.ack_instruction(
        task_id=task_id,
        instruction_id=None,
        action="self_resolve",
        status="failed",
        message="回滚后仍然 OOM",
    )
    assert state == TaskState.WAITING
    task = mailbox.get_task(task_id)
    assert task.awaiting_direct_conversation is True
    assert task.direct_conversation_requested_by == "ou_engineer"

    # A second solution from the conversation can close it successfully.
    state = mailbox.ack_instruction(
        task_id=task_id,
        instruction_id=None,
        action="custom",
        status="success",
        message="按对话方案缩小 batch size 后继续",
    )
    assert state == TaskState.RUNNING
    assert mailbox.get_task(task_id).awaiting_direct_conversation is False


@pytest.mark.asyncio
async def test_the_legacy_stop_training_decision_still_cancels(mailbox: TaskMailboxManager):
    """Old queued decisions must normalize to cancel rather than fall through."""
    task_id = "test-legacy-stop"
    _alert(mailbox, task_id)
    mailbox.submit_decision(task_id, action="stop_training", operator="ou_engineer")
    instruction = await mailbox.get_instruction_async(task_id, pop=True, wait_timeout=0)
    assert instruction.action == "stop_training"

    state = mailbox.ack_instruction(
        task_id=task_id,
        instruction_id=instruction.instruction_id,
        action="stop_training",
        status="success",
        message="已取消并验证关机",
    )
    assert state == TaskState.FAILED
    assert mailbox.get_task(task_id).state == TaskState.FAILED


@pytest.mark.asyncio
async def test_a_terminal_task_rejects_further_acks(mailbox: TaskMailboxManager):
    task_id = "test-terminal-ack"
    _alert(mailbox, task_id)
    mailbox.submit_decision(task_id, action="cancel", operator="ou_engineer")
    instruction = await mailbox.get_instruction_async(task_id, pop=True, wait_timeout=0)
    mailbox.ack_instruction(
        task_id=task_id,
        instruction_id=instruction.instruction_id,
        action="cancel",
        status="success",
    )

    with pytest.raises(ValueError, match="already terminal"):
        mailbox.ack_instruction(
            task_id=task_id,
            instruction_id=instruction.instruction_id,
            action="self_resolve",
            status="success",
        )
    assert mailbox.get_task(task_id).state == TaskState.FAILED


@pytest.mark.asyncio
async def test_a_rejected_ack_does_not_mutate_the_cached_task(
    mailbox: TaskMailboxManager,
):
    """Validation must run before an unpolled decision is promoted to latest."""
    task_id = "test-rejected-ack-no-mutation"
    _alert(mailbox, task_id)
    instruction = mailbox.submit_decision(
        task_id, action="self_resolve", operator="ou_engineer"
    )
    assert mailbox._get_existing(task_id).latest_instruction is None

    with pytest.raises(ValueError, match="Action mismatch"):
        mailbox.ack_instruction(
            task_id=task_id,
            instruction_id=instruction.instruction_id,
            action="cancel",
            status="success",
        )

    task = mailbox._get_existing(task_id)
    assert task.state == TaskState.RESOLVED
    assert task.latest_instruction is None
    assert task.pending_instruction is not None
    assert task.pending_instruction.instruction_id == instruction.instruction_id

    # The same decision is still executable with the matching action.
    state = mailbox.ack_instruction(
        task_id=task_id,
        instruction_id=instruction.instruction_id,
        action="self_resolve",
        status="success",
    )
    assert state == TaskState.RUNNING


def test_a_second_decision_is_rejected_while_the_first_is_pending(
    mailbox: TaskMailboxManager,
):
    """Duplicate card clicks must not overwrite the first decision."""
    task_id = "test-duplicate-decision"
    _alert(mailbox, task_id)
    first = mailbox.submit_decision(task_id, action="self_resolve", operator="alice")

    with pytest.raises(ValueError, match="only a WAITING alert accepts a decision"):
        mailbox.submit_decision(task_id, action="cancel", operator="bob")

    task = mailbox._get_existing(task_id)
    assert task.state == TaskState.RESOLVED
    assert task.pending_instruction.instruction_id == first.instruction_id
    assert task.pending_instruction.action == "self_resolve"


def test_a_milestone_during_a_direct_wait_does_not_dismiss_it(
    mailbox: TaskMailboxManager,
):
    """Only the ACK receipt may end the wait; periodic progress must not."""
    task_id = "test-milestone-during-wait"
    _alert(mailbox, task_id)
    mailbox.wait_for_direct_conversation(task_id, "ou_engineer")

    mailbox.record_event(
        EventNotifyRequest(
            task_id=task_id, event_type=EventType.MILESTONE, message="still alive", step=7
        )
    )

    summary = mailbox.get_task(task_id)
    assert summary.state == TaskState.WAITING
    assert summary.awaiting_direct_conversation is True
    assert summary.direct_conversation_requested_by == "ou_engineer"
    # The suppressed timeout stays suppressed after the progress report.
    assert mailbox.try_auto_resolve(task_id, timeout_seconds=0) is None


def test_ack_without_an_instruction_and_without_a_wait_is_rejected(
    mailbox: TaskMailboxManager,
):
    task_id = "test-ack-nothing"
    mailbox.record_event(
        EventNotifyRequest(task_id=task_id, event_type=EventType.MILESTONE, message="epoch 1")
    )
    with pytest.raises(ValueError, match="no instruction to acknowledge"):
        mailbox.ack_instruction(
            task_id=task_id,
            instruction_id=None,
            action="self_resolve",
            status="success",
        )


@pytest.mark.asyncio
async def test_cancel_ack_is_terminal_and_never_returns_to_running(
    mailbox: TaskMailboxManager,
):
    task_id = "test-cancel-terminal"
    _alert(mailbox, task_id)
    mailbox.submit_decision(task_id, action="cancel", operator="ou_engineer")
    instruction = await mailbox.get_instruction_async(task_id, pop=True, wait_timeout=0)
    assert instruction.action == "cancel"

    state = mailbox.ack_instruction(
        task_id=task_id,
        instruction_id=instruction.instruction_id,
        action="cancel",
        status="success",
        message="已取消本次运行并验证机器关机",
    )
    assert state == TaskState.FAILED
    assert mailbox.get_task(task_id).state == TaskState.FAILED


@pytest.mark.asyncio
async def test_a_failed_cancel_ack_does_not_claim_the_run_is_running(
    mailbox: TaskMailboxManager,
):
    """A shared machine cannot be shut down for one run; the incident must stay open."""
    task_id = "test-cancel-failed"
    _alert(mailbox, task_id)
    mailbox.submit_decision(task_id, action="cancel", operator="ou_engineer")
    instruction = await mailbox.get_instruction_async(task_id, pop=True, wait_timeout=0)

    state = mailbox.ack_instruction(
        task_id=task_id,
        instruction_id=instruction.instruction_id,
        action="cancel",
        status="failed",
        message="机器由其他任务共享，未执行关机",
    )
    assert state == TaskState.WAITING
    assert mailbox.get_task(task_id).state == TaskState.WAITING


@pytest.mark.asyncio
async def test_a_new_alert_clears_a_stale_direct_conversation_request(
    mailbox: TaskMailboxManager,
):
    """Otherwise the next alert's 30s window could never fire again."""
    task_id = "test-stale-direct-wait"
    _alert(mailbox, task_id, message="Loss NaN")
    mailbox.wait_for_direct_conversation(task_id, "ou_engineer")
    assert mailbox.get_task(task_id).awaiting_direct_conversation is True

    # Same alert redelivered: the request belongs to this alert, so it stays open
    mailbox.record_event(
        EventNotifyRequest(task_id=task_id, event_type=EventType.ALERT, message="Loss NaN", step=6)
    )
    assert mailbox.get_task(task_id).awaiting_direct_conversation is True

    # A different alert opens a fresh decision window
    mailbox.record_event(
        EventNotifyRequest(task_id=task_id, event_type=EventType.ALERT, message="CUDA OOM", step=9)
    )
    summary = mailbox.get_task(task_id)
    assert summary.awaiting_direct_conversation is False
    assert summary.direct_conversation_requested_by is None
    assert mailbox.try_auto_resolve(task_id, timeout_seconds=0) is not None


def test_any_decision_closes_the_direct_conversation(mailbox: TaskMailboxManager):
    task_id = "test-decision-closes-wait"
    _alert(mailbox, task_id)
    mailbox.wait_for_direct_conversation(task_id, "ou_engineer")

    mailbox.submit_decision(task_id, action="cancel", operator="ou_engineer")
    assert mailbox.get_task(task_id).awaiting_direct_conversation is False


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
