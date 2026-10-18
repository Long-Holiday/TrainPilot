"""Integration tests for TrainPilot FastAPI Server endpoints.

Agents reach the control plane over MCP; the REST surface covered here is the
operator/ops one (decision injection, status, event history) plus the Feishu webhook.
"""

import json

import pytest
from fastapi.testclient import TestClient

from trainpilot.common.schemas import EventNotifyRequest
from trainpilot.common.states import EventType
from trainpilot.server.background import feishu_dispatcher
from trainpilot.server.config import settings
from trainpilot.server.feishu.client import default_feishu_client
from trainpilot.server.main import app
from trainpilot.server.mailbox import default_mailbox
from trainpilot.server.mcp_server import ack_instruction


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def detach_real_feishu(monkeypatch):
    """Never let a test post into a real group, and start from an empty card log.

    With a populated local `.env` the lark client is built at import time, so
    `enable_mock_feishu` cannot help here; drop the client for the whole module. The
    card history is a process-wide deque that no other fixture resets, so card-count
    assertions need it cleared per test.
    """
    monkeypatch.setattr(default_feishu_client, "_lark_client", None)
    default_feishu_client._sent_cards_history.clear()


def _message_event(text, *, message_id="om_plan_1", chat_type="group", mentions=True):
    return {
        "schema": "2.0",
        "header": {"event_id": "evt_1", "event_type": "im.message.receive_v1"},
        "event": {
            "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_engineer"}},
            "message": {
                "message_id": message_id,
                "chat_id": "oc_test_group",
                "chat_type": chat_type,
                "message_type": "text",
                "content": json.dumps({"text": text}),
                "mentions": [{"key": "@_user_1", "id": {"open_id": "ou_bot"}, "name": "TrainPilot"}]
                if mentions
                else [],
            },
        },
    }


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

    # 5. Agent acks over MCP -> back to RUNNING. The ACK carries the real
    # post-recovery step/epoch/metrics, and those persist on the task snapshot.
    ack = ack_instruction(
        task_id=task_id,
        instruction_id=poll.instruction_id,
        action="self_resolve",
        status="success",
        solution="已跳过异常 Batch 并恢复正常训练。",
        step=80,
        epoch=2,
        metrics={"loss": 0.21},
    )
    assert ack["success"] is True
    assert ack["state"] == "RUNNING"
    status = client.get(f"/api/tasks/{task_id}/status").json()
    assert status["latest_step"] == 80
    assert status["latest_epoch"] == 2
    assert status["latest_metrics"] == {"loss": 0.21}

    # 6. Exactly one recovery card, built from the ACK's step/epoch/metrics.
    assert feishu_dispatcher.wait_idle(timeout=2.0)
    rec_cards = [
        c for c in default_feishu_client.sent_cards_history
        if c.get("type") == "recovery" and c.get("task_id") == task_id
    ]
    assert len(rec_cards) == 1
    card_dict = rec_cards[0]["card"]
    assert "【异常已自行解决】" in card_dict["header"]["title"]["content"]
    rendered = str(card_dict)
    assert "Step `80`" in rendered
    assert "Epoch `2`" in rendered
    assert "0.21" in rendered
    assert "已跳过异常 Batch" in rendered
    assert not [
        c for c in default_feishu_client.sent_cards_history
        if c.get("type") == "failed" and c.get("task_id") == task_id
    ]


def test_cancel_ack_is_terminal_failed_and_notifies_once(client: TestClient):
    """A successful cancel ACK is FAILED and sends one outcome, never a recovery card."""
    task_id = "test-api-cancel"
    _record(task_id, message="Loss NaN", step=6)
    instruction = default_mailbox.submit_decision(
        task_id=task_id, action="cancel", operator="ou_engineer"
    )

    ack = ack_instruction(
        task_id=task_id,
        instruction_id=instruction.instruction_id,
        action="stop_training",  # legacy wire alias for cancel
        status="success",
        solution="已取消本次运行并验证机器关机。",
    )
    assert ack["success"] is True
    assert ack["state"] == "FAILED"
    status = client.get(f"/api/tasks/{task_id}/status").json()
    assert status["state"] == "FAILED"
    assert status["pending_instruction"] is None

    assert feishu_dispatcher.wait_idle(timeout=2.0)
    history = default_feishu_client.sent_cards_history
    outcomes = [
        c for c in history
        if c.get("type") == "failed" and c.get("task_id") == task_id
    ]
    assert len(outcomes) == 1
    card_dict = outcomes[0]["card"]
    assert "【训练任务已失败/取消】" in card_dict["header"]["title"]["content"]
    rendered = str(card_dict)
    assert "已取消本次运行并验证机器关机" in rendered
    assert "已恢复正常运行" not in rendered
    assert not [
        c for c in history
        if c.get("type") == "recovery" and c.get("task_id") == task_id
    ]


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


@pytest.mark.parametrize("text,mentions", [
    ("@_user_1 用 5e-5 重跑 10 个 epoch #demo-llm-pretrain", True),
    ("@_user_1 #fix(demo-llm-pretrain) 回滚到 step4", True),
    ("@_user_1 #feat(demo-llm-pretrain) 重跑评测", True),
    ("大家中午吃什么", False),
])
def test_feishu_message_events_are_ignored(client: TestClient, text, mentions):
    """There is no plan/#fix/#feat ingestion: chat text never queues a decision."""
    task_id = "demo-llm-pretrain"
    _record(task_id, event_type=EventType.MILESTONE, message="epoch 1 done")
    before = [task.task_id for task in default_mailbox.list_tasks()]

    resp = client.post("/webhook/feishu", json=_message_event(text, mentions=mentions))
    assert resp.status_code == 200
    assert resp.json() == {"code": 0, "msg": "Ignored non-card-action event"}
    # Nothing was stored or created, and no mailbox instruction was dispatched.
    assert [task.task_id for task in default_mailbox.list_tasks()] == before
    assert client.get(f"/api/tasks/{task_id}/status").json()["pending_instruction"] is None
    assert feishu_dispatcher.wait_idle(2.0)
    assert default_feishu_client.sent_cards_history == []


def test_feishu_message_event_rejects_a_bad_token(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "feishu_verification_token", "expected_token")
    payload = _message_event("@_user_1 跑评测")
    payload["header"]["token"] = "wrong"

    resp = client.post("/webhook/feishu", json=payload)
    assert resp.status_code == 403
    assert default_mailbox.list_tasks() == []


def _card_values_webhook(client, task_id, action):
    return client.post("/webhook/feishu", json={
        "open_id": "ou_engineer",
        "action": {"tag": "button", "value": {"task_id": task_id, "action": action}},
    })


def _capture_auto_resolve(monkeypatch):
    """Record any ordinary alert timer the webhook might arm; the wait path arms none."""
    from trainpilot.server import mcp_server

    armed = []
    monkeypatch.setattr(
        mcp_server, "schedule_auto_resolve", lambda task_id: armed.append(task_id)
    )
    return armed


def test_feishu_wait_for_user_hands_off_to_direct_conversation(client: TestClient, monkeypatch):
    armed = _capture_auto_resolve(monkeypatch)
    task_id = "feishu-direct-wait"
    _record(task_id, message="Loss NaN at step 6", step=6)

    resp = _card_values_webhook(client, task_id, "wait_for_arrangement")
    assert resp.status_code == 200
    body = resp.json()
    assert body["card"]["type"] == "raw"
    card = body["card"]["data"]
    assert card["header"]["template"] == "orange"
    assert "【等待与 Agent 直接沟通】" in card["header"]["title"]["content"]
    rendered = str(card)
    assert "没有自动兜底" in rendered
    assert "直接对话" in rendered
    assert "ACK" in rendered
    assert "#fix" not in rendered

    # Still WAITING, nothing dispatched, and no fallback timer was armed.
    status = client.get(f"/api/tasks/{task_id}/status").json()
    assert status["state"] == "WAITING"
    assert status["awaiting_direct_conversation"] is True
    assert status["direct_conversation_requested_by"] == "ou_engineer"
    assert status["pending_instruction"] is None
    assert armed == []
    # The ordinary auto-resolve refuses while the conversation owns the decision.
    assert default_mailbox.try_auto_resolve(task_id, timeout_seconds=0) is None

    assert feishu_dispatcher.wait_idle(2.0)
    updates = [
        c for c in default_feishu_client.sent_cards_history
        if c["type"] == "awaiting_user_update"
    ]
    assert len(updates) == 1
    assert updates[0]["data"]["action"] == "wait_for_arrangement"


def test_feishu_wait_for_user_is_rejected_outside_an_alert(client: TestClient, monkeypatch):
    armed = _capture_auto_resolve(monkeypatch)
    task_id = "feishu-direct-wait-no-alert"
    _record(task_id, event_type=EventType.MILESTONE, message="epoch 1 done")

    resp = _card_values_webhook(client, task_id, "wait_for_arrangement")
    # A refused click is still HTTP 200: Feishu renders any non-200 from a card callback
    # as the opaque client error 200671, so the refusal travels in the toast instead.
    assert resp.status_code == 200
    assert resp.json()["toast"]["type"] == "error"
    assert task_id in resp.json()["toast"]["content"]
    assert "card" not in resp.json()
    status = client.get(f"/api/tasks/{task_id}/status").json()
    assert status["awaiting_direct_conversation"] is False
    assert armed == []


def test_feishu_card_action_for_a_finished_task_is_a_toast_not_an_http_error(client: TestClient):
    task_id = "feishu-click-after-completion"
    _record(task_id, event_type=EventType.COMPLETED, message="training finished")

    resp = _card_values_webhook(client, task_id, "cancel")

    assert resp.status_code == 200
    assert resp.json()["toast"]["type"] == "error"
    assert "COMPLETED" in resp.json()["toast"]["content"]
    assert client.get(f"/api/tasks/{task_id}/status").json()["state"] == "COMPLETED"


@pytest.mark.parametrize("action", ["cancel", "stop_training", "wait_for_arrangement"])
def test_feishu_card_action_for_an_unknown_task_never_creates_one(client: TestClient, action):
    resp = _card_values_webhook(client, "feishu-ghost", action)

    assert resp.status_code == 200
    assert resp.json()["toast"]["type"] == "error"
    assert client.get("/api/tasks/feishu-ghost/status").status_code == 404


def test_feishu_card_action_with_a_bad_token_answers_200_and_dispatches_nothing(
    client: TestClient, monkeypatch
):
    monkeypatch.setattr(settings, "feishu_verification_token", "expected_token")
    task_id = "feishu-card-bad-token"
    _record(task_id, message="Loss NaN at step 6")

    payload = {
        "schema": "2.0",
        "header": {"event_type": "card.action.trigger", "token": "wrong"},
        "event": {
            "operator": {"open_id": "ou_engineer"},
            "action": {"tag": "button", "value": {"task_id": task_id, "action": "self_resolve"}},
        },
    }
    resp = client.post("/webhook/feishu", json=payload)

    assert resp.status_code == 200
    assert resp.json()["toast"]["type"] == "error"
    status = client.get(f"/api/tasks/{task_id}/status").json()
    assert status["state"] == "WAITING"
    assert status["pending_instruction"] is None


def test_feishu_legacy_card_callback_reports_the_shape_instead_of_a_bare_403(
    client: TestClient, monkeypatch
):
    """The v1 payload's `token` is the c-xxxx card credential, not the verification token."""
    monkeypatch.setattr(settings, "feishu_verification_token", "expected_token")

    resp = _card_values_webhook(client, "feishu-legacy-shape", "self_resolve")

    assert resp.status_code == 200
    assert "旧版卡片回调" in resp.json()["toast"]["content"]


def test_feishu_card_callback_url_with_a_trailing_slash_is_served(client: TestClient):
    """A stray "/" in the console URL used to be swallowed by the MCP mount and 404ed."""
    task_id = "feishu-trailing-slash"
    _record(task_id, message="Loss NaN at step 6")

    resp = client.post("/webhook/feishu/", json={
        "open_id": "ou_engineer",
        "action": {"tag": "button", "value": {"task_id": task_id, "action": "self_resolve"}},
    })

    assert resp.status_code == 200
    assert resp.json()["toast"]["type"] == "success"
