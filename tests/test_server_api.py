"""Integration tests for TrainPilot FastAPI Server endpoints.

Agents reach the control plane over MCP; the REST surface covered here is the
operator/ops one (decision injection, status, event history) plus the Feishu webhook.
"""

import pytest
from fastapi.testclient import TestClient

from trainpilot.common.schemas import EventNotifyRequest
from trainpilot.common.states import EventType
from trainpilot.server.background import feishu_dispatcher
from trainpilot.server.feishu.client import default_feishu_client
from trainpilot.server.main import app
from trainpilot.server.mailbox import default_mailbox
from trainpilot.server.mcp_server import ack_instruction


@pytest.fixture
def client():
    return TestClient(app)


def _record(task_id, event_type=EventType.ALERT, message="Loss NaN at step 100", **kwargs):
    return default_mailbox.record_event(
        EventNotifyRequest(
            task_id=task_id, event_type=event_type, message=message, **kwargs
        )
    )


def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_decision_and_recovery_card_flow(client: TestClient):
    """Alert -> operator decision over REST -> agent poll/ack over MCP -> recovery card."""
    task_id = "test-api-task"

    # 1. Agent reports an alert -> WAITING
    assert _record(task_id, message="Loss exploded", step=60, epoch=1).value == "WAITING"

    # 2. Nothing decided yet -> no instruction
    idle_poll = await default_mailbox.get_instruction_async(task_id, pop=False, wait_timeout=0.0)
    assert idle_poll.has_instruction is False
    assert idle_poll.status.value == "WAITING"

    # 3. Human submits a decision via the operator API
    dec_resp = client.post(
        f"/api/tasks/{task_id}/decision",
        json={"task_id": task_id, "action": "self_resolve", "operator": "Dr. Wang"},
    )
    assert dec_resp.status_code == 200
    assert dec_resp.json()["action"] == "self_resolve"

    # 4. Poll after decision -> has_instruction: true, pops into RECOVERING
    poll = await default_mailbox.get_instruction_async(task_id, pop=True, wait_timeout=0.0)
    assert poll.has_instruction is True
    assert poll.instruction_id is not None
    assert client.get(f"/api/tasks/{task_id}/status").json()["state"] == "RECOVERING"

    # 5. Agent acks over MCP -> back to RUNNING; step/epoch are backfilled from the TaskRecord
    ack = ack_instruction(
        task_id=task_id,
        instruction_id=poll.instruction_id,
        action="self_resolve",
        status="success",
        solution="已跳过异常 Batch 并恢复正常训练。",
    )
    assert ack["success"] is True
    assert ack["state"] == "RUNNING"

    # 6. The recovery card carries the step from the last report, not from the ack
    assert feishu_dispatcher.wait_idle(timeout=2.0)
    rec_cards = [
        c for c in default_feishu_client.sent_cards_history
        if c.get("type") == "recovery" and c.get("task_id") == task_id
    ]
    assert len(rec_cards) == 1
    card_dict = rec_cards[0]["card"]
    assert "【异常已自行解决】" in card_dict["header"]["title"]["content"]
    assert "Step `60`" in str(card_dict)
    assert "已跳过异常 Batch" in str(card_dict)


def test_decision_rejected_for_unknown_task(client: TestClient):
    resp = client.post(
        "/api/tasks/never-reported/decision",
        json={"task_id": "never-reported", "action": "self_resolve"},
    )
    assert resp.status_code == 404
    assert default_mailbox.get_task("never-reported") is None


def test_task_status_and_events_endpoints(client: TestClient):
    task_id = "test-api-events"
    _record(task_id, event_type=EventType.MILESTONE, message="Epoch 1 done", step=50)
    _record(task_id, message="Loss NaN", step=60)

    status_resp = client.get(f"/api/tasks/{task_id}/status")
    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["state"] == "WAITING"
    assert body["latest_step"] == 60
    assert body["events_count"] == 2

    events_resp = client.get(f"/api/tasks/{task_id}/events", params={"limit": 10, "offset": 0})
    assert events_resp.status_code == 200
    events = events_resp.json()
    assert events["total"] == 2
    assert [e["message"] for e in events["events"]] == ["Epoch 1 done", "Loss NaN"]

    assert client.get("/api/tasks/never-reported/status").status_code == 404
    assert client.get("/api/tasks/never-reported/events").status_code == 404

    listed = client.get("/api/tasks").json()
    assert [t["task_id"] for t in listed] == [task_id]


def test_removed_agent_endpoints_are_gone(client: TestClient):
    """The MCP tools replaced /notify, /instruction and /ack; they must not linger."""
    for method, path, kwargs in (
        ("post", "/api/tasks/notify", {"json": {}}),
        ("get", "/api/tasks/some-task/instruction", {}),
        ("post", "/api/tasks/some-task/ack", {"json": {}}),
    ):
        resp = getattr(client, method)(path, **kwargs)
        assert resp.status_code in (404, 405), f"{method.upper()} {path} -> {resp.status_code}"


def test_feishu_webhook_flow(client: TestClient):
    task_id = "feishu-test-task"

    # 1. Test URL Verification handshake challenge
    challenge_payload = {
        "challenge": "sample_challenge_token_123",
        "token": "test_token",
        "type": "url_verification",
    }
    handshake = client.post("/webhook/feishu", json=challenge_payload)
    assert handshake.status_code == 200
    assert handshake.json() == {"challenge": "sample_challenge_token_123"}

    # 2. A task alerts first
    _record(task_id, message="Out of memory")

    # 3. Feishu interactive card click event
    card_action_webhook = {
        "open_id": "ou_feishu_engineer",
        "action": {
            "tag": "button",
            "value": {
                "task_id": task_id,
                "action": "stop_training",
            },
        },
    }
    wh_resp = client.post("/webhook/feishu", json=card_action_webhook)
    assert wh_resp.status_code == 200
    wh_data = wh_resp.json()
    assert "toast" in wh_data
    assert "card" in wh_data
    assert wh_data["card"]["type"] == "raw"
    assert "data" in wh_data["card"]
    assert "elements" in wh_data["card"]["data"]
    assert wh_data["card"]["data"]["header"]["template"] == "turquoise"

    # Check that mailbox received the decision
    inst = client.get(f"/api/tasks/{task_id}/status").json()["pending_instruction"]
    assert inst["has_instruction"] is True
    assert inst["action"] == "stop_training"
    assert inst["decision_by"] == "ou_feishu_engineer"
