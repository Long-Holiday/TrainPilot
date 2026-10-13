"""Unit tests for mailbox long-polling wakeup semantics."""

import threading
import time
import pytest

from trainpilot.common.schemas import EventNotifyRequest
from trainpilot.common.states import EventType
from trainpilot.server.mailbox import default_mailbox


@pytest.fixture(autouse=True)
def clean_mailbox():
    from trainpilot.server.config import settings
    default_mailbox.reset()
    old_api_token = settings.api_token
    settings.api_token = None
    yield
    settings.api_token = old_api_token
    default_mailbox.reset()


def test_long_polling_immediate_return_when_available():
    """If decision is already in mailbox, long-poll returns immediately without delay."""
    default_mailbox.record_event(
        EventNotifyRequest(
            task_id="task-lp-01",
            event_type=EventType.ALERT,
            message="Loss NaN",
        )
    )
    default_mailbox.submit_decision("task-lp-01", action="self_resolve", operator="human")

    start = time.time()
    res = default_mailbox.get_instruction("task-lp-01", pop=True, wait_timeout=5.0)
    elapsed = time.time() - start

    assert res.has_instruction is True
    assert res.action == "self_resolve"
    assert elapsed < 0.2  # Immediate return


def test_long_polling_times_out_when_no_decision():
    """If no decision is submitted, holds connection until wait_timeout expires and returns has_instruction=False."""
    default_mailbox.record_event(
        EventNotifyRequest(
            task_id="task-lp-02",
            event_type=EventType.ALERT,
            message="Loss NaN",
        )
    )

    start = time.time()
    res = default_mailbox.get_instruction("task-lp-02", pop=True, wait_timeout=0.6)
    elapsed = time.time() - start

    assert res.has_instruction is False
    assert 0.5 <= elapsed < 1.0


def test_long_polling_instant_wake_up():
    """A blocking long-poll call is immediately awakened when decision is submitted."""
    task_id = "task-lp-03"
    default_mailbox.record_event(
        EventNotifyRequest(
            task_id=task_id,
            event_type=EventType.ALERT,
            message="Loss NaN",
        )
    )

    results = []

    def waiter():
        start = time.time()
        inst = default_mailbox.get_instruction(task_id, pop=True, wait_timeout=5.0)
        elapsed = time.time() - start
        results.append((inst, elapsed))

    t = threading.Thread(target=waiter)
    t.start()

    # Sleep briefly to ensure waiter is blocked inside event.wait
    time.sleep(0.3)
    default_mailbox.submit_decision(task_id, action="stop_training", operator="supervisor")

    t.join(timeout=2.0)
    assert len(results) == 1
    inst, elapsed = results[0]
    assert inst.has_instruction is True
    assert inst.action == "stop_training"
    # Awakened quickly upon submit_decision (well below the 5.0s timeout)
    assert elapsed < 1.0
