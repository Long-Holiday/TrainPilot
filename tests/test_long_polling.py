"""Unit tests for Server and Client HTTP Long Polling."""

import threading
import time
import pytest
from fastapi.testclient import TestClient

from trainpilot.agent.client import TrainPilotClient
from trainpilot.common.schemas import EventNotifyRequest, TaskDecisionRequest
from trainpilot.common.states import EventType, TaskState
from trainpilot.server.mailbox import TaskMailboxManager, default_mailbox
from trainpilot.server.main import app


@pytest.fixture(autouse=True)
def clean_mailbox():
    from trainpilot.server.config import settings
    default_mailbox.reset()
    old_api_token = settings.api_token
    settings.api_token = None
    yield
    settings.api_token = old_api_token
    default_mailbox.reset()


def test_long_polling_immediate_return_when_ready():
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

    assert res.ready is True
    assert res.action == "self_resolve"
    assert elapsed < 0.2  # Immediate return


def test_long_polling_times_out_when_no_decision():
    """If no decision is submitted, holds connection until wait_timeout expires and returns ready=False."""
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

    assert res.ready is False
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
    assert inst.ready is True
    assert inst.action == "stop_training"
    # Awakened quickly upon submit_decision (well below the 5.0s timeout)
    assert elapsed < 1.0


def test_long_polling_api_endpoint():
    """Test HTTP GET /api/tasks/{task_id}/instruction with wait_timeout via TestClient."""
    client = TestClient(app)
    task_id = "task-api-lp"

    client.post(
        "/api/tasks/notify",
        json={"task_id": task_id, "event_type": "alert", "message": "NaN loss"},
    )

    # Fast check with wait_timeout=0 returns ready=false
    resp = client.get(f"/api/tasks/{task_id}/instruction?wait_timeout=0")
    assert resp.status_code == 200
    assert resp.json()["ready"] is False

    # Wake up test via threading
    res_box = []

    def long_poll_req():
        r = client.get(f"/api/tasks/{task_id}/instruction?wait_timeout=3.0&pop=true")
        res_box.append(r)

    t = threading.Thread(target=long_poll_req)
    t.start()

    time.sleep(0.3)
    # Submit decision
    client.post(
        f"/api/tasks/{task_id}/decision",
        json={"task_id": task_id, "action": "self_resolve"},
    )

    t.join(timeout=2.0)
    assert len(res_box) == 1
    data = res_box[0].json()
    assert data["ready"] is True
    assert data["action"] == "self_resolve"
