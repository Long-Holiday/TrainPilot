"""Unit tests for TaskMailboxManager native AsyncIO long polling."""

import asyncio
import time
import pytest

from trainpilot.common.schemas import EventNotifyRequest
from trainpilot.common.states import EventType, TaskState
from trainpilot.server.mailbox import default_mailbox


@pytest.fixture(autouse=True)
def clean_mailbox():
    default_mailbox.reset()
    yield
    default_mailbox.reset()


@pytest.mark.asyncio
async def test_async_long_polling_immediate_return():
    """Async fast path returns immediately when instruction is already ready."""
    task_id = "task-async-01"
    default_mailbox.record_event(
        EventNotifyRequest(task_id=task_id, event_type=EventType.ALERT, message="NaN")
    )
    default_mailbox.submit_decision(task_id, action="self_resolve", operator="lead_eng")

    start = time.time()
    res = await default_mailbox.get_instruction_async(task_id, pop=True, wait_timeout=5.0)
    elapsed = time.time() - start

    assert res.ready is True
    assert res.action == "self_resolve"
    assert res.decision_by == "lead_eng"
    assert elapsed < 0.2


@pytest.mark.asyncio
async def test_async_long_polling_timeout():
    """Async wait times out cleanly without blocking any OS thread."""
    task_id = "task-async-02"
    default_mailbox.record_event(
        EventNotifyRequest(task_id=task_id, event_type=EventType.ALERT, message="NaN")
    )

    start = time.time()
    res = await default_mailbox.get_instruction_async(task_id, pop=True, wait_timeout=0.5)
    elapsed = time.time() - start

    assert res.ready is False
    assert 0.45 <= elapsed < 1.0


@pytest.mark.asyncio
async def test_async_long_polling_instant_wake_up():
    """Async long-polling coroutine is woken up immediately when decision arrives."""
    task_id = "task-async-03"
    default_mailbox.record_event(
        EventNotifyRequest(task_id=task_id, event_type=EventType.ALERT, message="NaN")
    )

    async def poll_worker():
        start = time.time()
        inst = await default_mailbox.get_instruction_async(task_id, pop=True, wait_timeout=5.0)
        elapsed = time.time() - start
        return inst, elapsed

    poll_task = asyncio.create_task(poll_worker())
    await asyncio.sleep(0.1)

    # Submit decision
    default_mailbox.submit_decision(task_id, action="stop_training", operator="safety_agent")

    inst, elapsed = await poll_task
    assert inst.ready is True
    assert inst.action == "stop_training"
    assert inst.decision_by == "safety_agent"
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_concurrent_async_long_polling():
    """Multiple tasks can long-poll concurrently without thread exhaustion."""
    num_tasks = 20
    tasks = [f"task-concurrent-{i}" for i in range(num_tasks)]

    for tid in tasks:
        default_mailbox.record_event(
            EventNotifyRequest(task_id=tid, event_type=EventType.ALERT, message="NaN")
        )

    results = {}

    async def worker(tid):
        inst = await default_mailbox.get_instruction_async(tid, pop=True, wait_timeout=5.0)
        results[tid] = inst

    coros = [asyncio.create_task(worker(tid)) for tid in tasks]
    await asyncio.sleep(0.1)

    # Concurrently submit decisions for all tasks
    for i, tid in enumerate(tasks):
        default_mailbox.submit_decision(tid, action=f"action_{i}", operator=f"operator_{i}")

    await asyncio.gather(*coros)

    assert len(results) == num_tasks
    for i, tid in enumerate(tasks):
        assert results[tid].ready is True
        assert results[tid].action == f"action_{i}"
