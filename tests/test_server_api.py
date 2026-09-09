"""Integration tests for TrainPilot FastAPI Server endpoints."""

import pytest
from fastapi.testclient import TestClient

from trainpilot.server.main import app
from trainpilot.server.mailbox import default_mailbox


@pytest.fixture(autouse=True)
def reset_mailbox():
    from trainpilot.server.config import settings
    default_mailbox.reset()
    old_token = settings.feishu_verification_token
    old_api_token = settings.api_token
    # 测试必须与本地 .env 解耦: 强制关闭 API 鉴权与 Webhook 校验
    settings.feishu_verification_token = None
    settings.api_token = None
    yield
    settings.feishu_verification_token = old_token
    settings.api_token = old_api_token
    default_mailbox.reset()


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_notify_alert_and_decision_flow(client: TestClient):
    task_id = "test-api-task"

    # 1. Notify Alert
    alert_payload = {
        "task_id": task_id,
        "event_type": "alert",
        "message": "Loss NaN at step 100",
        "step": 100,
        "epoch": 1,
        "metrics": {"loss": 10000.0},
    }
    resp = client.post("/api/tasks/notify", json=alert_payload)
    assert resp.status_code == 200
    assert resp.json()["state"] == "WAITING"

    # 2. Poll before decision -> ready: false
    poll_resp = client.get(f"/api/tasks/{task_id}/instruction")
    assert poll_resp.status_code == 200
    assert poll_resp.json()["ready"] is False
    assert poll_resp.json()["status"] == "WAITING"

    # 3. Simulate human decision via API
    dec_resp = client.post(
        f"/api/tasks/{task_id}/decision",
        json={"task_id": task_id, "action": "self_resolve", "operator": "Dr. Wang"},
    )
    assert dec_resp.status_code == 200
    assert dec_resp.json()["action"] == "self_resolve"

    # 4. Poll after decision -> ready: true, pops into RECOVERING
    poll_resp2 = client.get(f"/api/tasks/{task_id}/instruction")
    assert poll_resp2.status_code == 200
    data2 = poll_resp2.json()
    assert data2["ready"] is True
    assert data2["action"] == "self_resolve"
    assert data2["instruction_id"] is not None

    # Verify task state is now RECOVERING
    status_resp = client.get(f"/api/tasks/{task_id}/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["state"] == "RECOVERING"

    # 5. Agent Acks instruction -> status back to RUNNING
    ack_resp = client.post(
        f"/api/tasks/{task_id}/ack",
        json={
            "task_id": task_id,
            "instruction_id": data2["instruction_id"],
            "action": "self_resolve",
            "status": "success",
            "message": "Self-resolved and continued training",
        },
    )
    assert ack_resp.status_code == 200
    assert ack_resp.json()["state"] == "RUNNING"


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

    # 2. Trigger task alert first
    client.post("/api/tasks/notify", json={
        "task_id": task_id,
        "event_type": "alert",
        "message": "Out of memory",
    })

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
    inst = client.get(f"/api/tasks/{task_id}/instruction").json()
    assert inst["ready"] is True
    assert inst["action"] == "stop_training"
    assert inst["decision_by"] == "ou_feishu_engineer"


def test_ack_recovery_card_flow(client: TestClient):
    from trainpilot.server.feishu.client import default_feishu_client

    task_id = "ack-recovery-test-task"

    # 1. 触发告警
    alert_resp = client.post(
        "/api/tasks/notify",
        json={
            "task_id": task_id,
            "event_type": "alert",
            "message": "Loss exploded",
        },
    )
    assert alert_resp.status_code == 200
    assert alert_resp.json()["state"] == "WAITING"

    # 2. 模拟下发决策
    dec_resp = client.post(
        f"/api/tasks/{task_id}/decision",
        json={
            "task_id": task_id,
            "action": "self_resolve",
        },
    )
    assert dec_resp.status_code == 200
    assert dec_resp.json()["action"] == "self_resolve"

    # 3. 客户端消费指令并 ACK (携带 solution)
    resp = client.post(
        f"/api/tasks/{task_id}/ack",
        json={
            "task_id": task_id,
            "action": "self_resolve",
            "status": "success",
            "solution": "已跳过异常 Batch 并恢复正常训练。",
            "step": 60,
            "epoch": 1,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["state"] == "RUNNING"

    # 4. 验证 FeishuClient 接收到恢复卡片
    from trainpilot.server.background import feishu_dispatcher
    assert feishu_dispatcher.wait_idle(timeout=2.0)

    history = default_feishu_client.sent_cards_history
    rec_cards = [c for c in history if c.get("type") == "recovery" and c.get("task_id") == task_id]
    assert len(rec_cards) == 1
    card_dict = rec_cards[0]["card"]
    assert "【异常已自行解决】" in card_dict["header"]["title"]["content"]


