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
from trainpilot.server.plans import default_plans


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


def _seed_task(task_id):
    """Create a task row the supported way -- a report. Plans only bind to live tasks."""
    return _record(task_id, event_type=EventType.MILESTONE, message="epoch 1 done")


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


def _plan_cards(card_type):
    return [c for c in default_feishu_client.sent_cards_history if c["type"] == card_type]


def test_feishu_group_at_message_registers_a_plan(client: TestClient):
    """@-mentioning the bot is the whole trigger; the tag binds it to one live task."""
    _seed_task("demo-llm-pretrain")
    resp = client.post("/webhook/feishu", json=_message_event(
        "@_user_1 用 5e-5 重跑 10 个 epoch #demo-llm-pretrain"
    ))
    assert resp.status_code == 200
    assert resp.json() == {"code": 0, "msg": "Plan registered"}

    plans = default_plans.list_pending("demo-llm-pretrain")
    assert len(plans) == 1
    assert plans[0].text == "用 5e-5 重跑 10 个 epoch"
    assert plans[0].task_id == "demo-llm-pretrain"
    assert plans[0].created_by == "ou_engineer"

    assert feishu_dispatcher.wait_idle(2.0)
    cards = _plan_cards("plan_registered")
    assert len(cards) == 1
    assert cards[0]["chat_id"] == "oc_test_group"
    card_text = cards[0]["card"]["elements"][0]["text"]["content"]
    assert plans[0].plan_id in card_text
    assert "用 5e-5 重跑 10 个 epoch" in card_text
    assert "`demo-llm-pretrain`" in card_text


def test_feishu_plan_for_an_unknown_task_is_refused(client: TestClient):
    """A plan nobody's report ever carries would be invisible for good, so it is refused."""
    _seed_task("demo-llm-pretrain")
    resp = client.post("/webhook/feishu", json=_message_event(
        "@_user_1 用 5e-5 重跑 #demo-llm-pretrain-typo"
    ))
    assert resp.status_code == 200
    assert resp.json() == {"code": 0, "msg": "Plan not registered"}

    # Nothing was stored, and no task row was fabricated for the typo either.
    assert default_plans.list_pending("demo-llm-pretrain-typo") == []
    assert default_plans.list_pending("demo-llm-pretrain") == []
    assert default_mailbox.get_task("demo-llm-pretrain-typo") is None

    assert feishu_dispatcher.wait_idle(2.0)
    assert _plan_cards("plan_registered") == []
    rejects = _plan_cards("plan_rejected")
    assert len(rejects) == 1
    card_text = rejects[0]["card"]["elements"][0]["text"]["content"]
    assert "不存在" in card_text
    assert "`demo-llm-pretrain-typo`" in card_text
    # The tasks that do exist are listed, so the corrected name is one copy-paste away.
    assert "`demo-llm-pretrain`" in card_text


def test_feishu_plan_without_a_task_name_is_refused(client: TestClient):
    """There are no global plans: every plan names the task whose reports carry it."""
    _seed_task("demo-llm-pretrain")
    resp = client.post("/webhook/feishu", json=_message_event("@_user_1 跑评测"))
    assert resp.json() == {"code": 0, "msg": "Plan not registered"}
    assert default_plans.list_pending("demo-llm-pretrain") == []

    assert feishu_dispatcher.wait_idle(2.0)
    rejects = _plan_cards("plan_rejected")
    assert len(rejects) == 1
    assert "缺少任务名" in rejects[0]["card"]["elements"][0]["text"]["content"]


def test_feishu_group_message_without_mention_is_ignored(client: TestClient):
    """Group chatter that does not @ the bot must never become a task."""
    _seed_task("demo-llm-pretrain")
    resp = client.post("/webhook/feishu", json=_message_event("大家中午吃什么", mentions=False))
    assert resp.status_code == 200
    assert default_plans.list_pending("demo-llm-pretrain") == []


def test_feishu_retried_message_event_registers_only_one_plan(client: TestClient):
    """Feishu redelivers events it cannot confirm; the second delivery must be a no-op."""
    _seed_task("demo-llm-pretrain")
    payload = _message_event("@_user_1 跑评测 #demo-llm-pretrain")
    first = client.post("/webhook/feishu", json=payload)
    second = client.post("/webhook/feishu", json=payload)

    assert first.json()["msg"] == "Plan registered"
    assert second.json()["msg"] == "Duplicate plan message ignored"
    assert len(default_plans.list_pending("demo-llm-pretrain")) == 1

    assert feishu_dispatcher.wait_idle(2.0)
    assert len(_plan_cards("plan_registered")) == 1


def test_feishu_clear_command_drops_pending_plans(client: TestClient):
    _seed_task("demo-llm-pretrain")
    client.post("/webhook/feishu", json=_message_event("@_user_1 跑评测 #demo-llm-pretrain"))
    client.post("/webhook/feishu", json=_message_event(
        "@_user_1 再跑一次 #demo-llm-pretrain", message_id="om_plan_2"
    ))
    assert len(default_plans.list_pending("demo-llm-pretrain")) == 2

    resp = client.post("/webhook/feishu", json=_message_event("@_user_1 清除", message_id="om_plan_3"))
    assert resp.json() == {"code": 0, "msg": "Cleared pending next-task plans"}
    assert default_plans.list_pending("demo-llm-pretrain") == []

    assert feishu_dispatcher.wait_idle(2.0)
    cards = _plan_cards("plan_cleared")
    assert len(cards) == 1
    assert "`2`" in cards[0]["card"]["elements"][0]["text"]["content"]


def test_feishu_message_event_rejects_a_bad_token(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "feishu_verification_token", "expected_token")
    payload = _message_event("@_user_1 跑评测")
    payload["header"]["token"] = "wrong"

    resp = client.post("/webhook/feishu", json=payload)
    assert resp.status_code == 403
    assert default_plans.list_pending("demo-llm-pretrain") == []


def _card_values_webhook(client, task_id, action):
    return client.post("/webhook/feishu", json={
        "open_id": "ou_engineer",
        "action": {"tag": "button", "value": {"task_id": task_id, "action": action}},
    })


def _defer_timers(monkeypatch):
    """Capture the arrangement fallback timer instead of arming a real 30-minute one."""
    from trainpilot.server import mcp_server

    armed = []
    monkeypatch.setattr(
        mcp_server, "schedule_auto_resolve", lambda task_id, stage="alert": armed.append((task_id, stage))
    )
    return armed


def test_feishu_wait_for_arrangement_pauses_the_countdown(client: TestClient, monkeypatch):
    armed = _defer_timers(monkeypatch)
    task_id = "feishu-defer-task"
    _record(task_id, message="Loss NaN at step 6", step=6)

    resp = _card_values_webhook(client, task_id, "wait_for_arrangement")
    assert resp.status_code == 200
    body = resp.json()
    assert body["card"]["type"] == "raw"
    deferred = body["card"]["data"]
    assert deferred["header"]["template"] == "orange"
    assert "【等待人工安排】" in deferred["header"]["title"]["content"]
    assert f"#fix({task_id})" in str(deferred)

    # Still waiting, nothing dispatched, and the long window is now the only timer
    status = client.get(f"/api/tasks/{task_id}/status").json()
    assert status["state"] == "WAITING"
    assert status["awaiting_arrangement"] is True
    assert status["pending_instruction"] is None
    assert armed == [(task_id, "arrangement")]

    assert feishu_dispatcher.wait_idle(2.0)
    updates = [c for c in default_feishu_client.sent_cards_history if c["type"] == "arrangement_update"]
    assert len(updates) == 1
    assert updates[0]["data"]["action"] == "wait_for_arrangement"


def test_feishu_wait_for_arrangement_is_rejected_outside_an_alert(client: TestClient, monkeypatch):
    armed = _defer_timers(monkeypatch)
    task_id = "feishu-defer-no-alert"
    _record(task_id, event_type=EventType.MILESTONE, message="epoch 1 done")

    resp = _card_values_webhook(client, task_id, "wait_for_arrangement")
    # A refused click is still HTTP 200: Feishu renders any non-200 from a card callback
    # as the opaque client error 200671, so the refusal travels in the toast instead.
    assert resp.status_code == 200
    assert resp.json()["toast"]["type"] == "error"
    assert task_id in resp.json()["toast"]["content"]
    assert "card" not in resp.json()
    assert client.get(f"/api/tasks/{task_id}/status").json()["awaiting_arrangement"] is False
    assert armed == []


def test_feishu_card_action_for_a_finished_task_is_a_toast_not_an_http_error(client: TestClient):
    task_id = "feishu-click-after-completion"
    _record(task_id, event_type=EventType.COMPLETED, message="training finished")

    resp = _card_values_webhook(client, task_id, "stop_training")

    assert resp.status_code == 200
    assert resp.json()["toast"]["type"] == "error"
    assert "COMPLETED" in resp.json()["toast"]["content"]
    assert client.get(f"/api/tasks/{task_id}/status").json()["state"] == "COMPLETED"


@pytest.mark.parametrize("action", ["stop_training", "wait_for_arrangement"])
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


def test_feishu_fix_message_dispatches_a_custom_decision(client: TestClient):
    task_id = "feishu-fix-task"
    _record(task_id, message="Loss NaN at step 6", step=6, epoch=1)
    _card_values_webhook(client, task_id, "wait_for_arrangement")

    resp = client.post("/webhook/feishu", json=_message_event(
        f"@_user_1 #fix({task_id}) 回滚到 step4 检查点并把 lr 减半"
    ))
    assert resp.json() == {"code": 0, "msg": "Fix dispatched"}

    inst = client.get(f"/api/tasks/{task_id}/status").json()["pending_instruction"]
    assert inst["has_instruction"] is True
    assert inst["action"] == "custom"
    assert inst["payload"]["solution"] == "回滚到 step4 检查点并把 lr 减半"
    assert inst["decision_by"] == "ou_engineer"
    # Dispatching a decision closes the arrangement window
    assert client.get(f"/api/tasks/{task_id}/status").json()["awaiting_arrangement"] is False

    assert feishu_dispatcher.wait_idle(2.0)
    receipts = _plan_cards("fix_dispatched")
    assert len(receipts) == 1
    assert receipts[0]["chat_id"] == "oc_test_group"
    receipt_text = str(receipts[0]["card"])
    assert f"`{task_id}`" in receipt_text
    assert "回滚到 step4 检查点并把 lr 减半" in receipt_text

    # The alert / waiting card is replaced in place
    updates = [c for c in default_feishu_client.sent_cards_history if c["type"] == "resolved_update"]
    assert len(updates) == 1
    assert updates[0]["data"]["action"] == "custom"
    assert "【决策已闭环】" in updates[0]["data"]["card"]["header"]["title"]["content"]


def test_feishu_fix_works_inside_the_thirty_second_window(client: TestClient):
    """A human who types fast need not click 等待安排 first."""
    task_id = "feishu-fix-fast"
    _record(task_id, message="Loss exploded", step=7)

    resp = client.post("/webhook/feishu", json=_message_event(
        f"@_user_1 #fix({task_id}) 降低 lr 后继续"
    ))
    assert resp.json()["msg"] == "Fix dispatched"
    assert client.get(f"/api/tasks/{task_id}/status").json()["state"] == "RESOLVED"


def test_feishu_fix_for_an_unknown_task_creates_no_row(client: TestClient):
    """`#fix` may never fabricate a task: an unknown id is refused with a receipt."""
    before = default_mailbox.list_tasks()

    resp = client.post("/webhook/feishu", json=_message_event(
        "@_user_1 #fix(never-reported) 重启训练"
    ))
    assert resp.json() == {"code": 0, "msg": "Fix not dispatched"}
    assert default_mailbox.get_task("never-reported") is None
    assert default_mailbox.list_tasks() == before

    assert feishu_dispatcher.wait_idle(2.0)
    rejects = _plan_cards("fix_rejected")
    assert len(rejects) == 1
    assert "不会为它创建任务行" in str(rejects[0]["card"])


def test_feishu_fix_is_rejected_when_the_task_is_not_waiting(client: TestClient):
    task_id = "feishu-fix-not-waiting"
    _record(task_id, event_type=EventType.MILESTONE, message="epoch 1 done")

    resp = client.post("/webhook/feishu", json=_message_event(
        f"@_user_1 #fix({task_id}) 重启"
    ))
    assert resp.json() == {"code": 0, "msg": "Fix not dispatched"}

    assert feishu_dispatcher.wait_idle(2.0)
    rejects = _plan_cards("fix_rejected")
    assert len(rejects) == 1
    assert "RUNNING" in str(rejects[0]["card"])


def test_feishu_fix_redelivery_dispatches_once_and_sends_one_receipt(client: TestClient):
    """Feishu retries unconfirmed events; the second delivery must be a silent no-op."""
    task_id = "feishu-fix-retry"
    _record(task_id, message="Loss NaN", step=6)
    payload = _message_event(f"@_user_1 #fix({task_id}) 回滚到 step4")

    first = client.post("/webhook/feishu", json=payload)
    second = client.post("/webhook/feishu", json=payload)
    assert first.json()["msg"] == "Fix dispatched"
    assert second.json()["msg"] == "Duplicate fix message ignored"

    assert feishu_dispatcher.wait_idle(2.0)
    assert len(_plan_cards("fix_dispatched")) == 1
    assert _plan_cards("fix_rejected") == []
    # The instruction is the first one, not a second decision
    assert client.get(f"/api/tasks/{task_id}/status").json()["pending_instruction"]["action"] == "custom"


def test_feishu_fix_without_a_task_name_is_refused_with_a_hint(client: TestClient):
    resp = client.post("/webhook/feishu", json=_message_event("@_user_1 #fix 回滚到 step4"))
    assert resp.json()["msg"] == "Fix not dispatched"
    assert default_plans.list_pending("fix") == []

    assert feishu_dispatcher.wait_idle(2.0)
    rejects = _plan_cards("fix_rejected")
    assert len(rejects) == 1
    assert "缺少任务名" in str(rejects[0]["card"])


def test_feishu_feat_message_registers_a_plan_bound_to_the_task(client: TestClient):
    _seed_task("demo-llm-pretrain")
    resp = client.post("/webhook/feishu", json=_message_event(
        "@_user_1 #feat(demo-llm-pretrain) 用 5e-5 重跑 10 个 epoch"
    ))
    assert resp.json() == {"code": 0, "msg": "Plan registered"}

    plans = default_plans.list_pending("demo-llm-pretrain")
    assert len(plans) == 1
    assert plans[0].task_id == "demo-llm-pretrain"
    assert plans[0].text == "用 5e-5 重跑 10 个 epoch"


def test_feishu_fix_is_disabled_with_next_task_plans(client: TestClient, monkeypatch):
    """One kill switch governs all @-bot messages, fixes included."""
    monkeypatch.setattr(settings, "enable_next_task_plans", False)
    task_id = "feishu-fix-disabled"
    _record(task_id, message="Loss NaN", step=6)

    resp = client.post("/webhook/feishu", json=_message_event(f"@_user_1 #fix({task_id}) 重启"))
    assert resp.json() == {"code": 0, "msg": "Next-task plans are disabled"}
    assert client.get(f"/api/tasks/{task_id}/status").json()["pending_instruction"] is None


def test_feishu_plans_can_be_disabled(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "enable_next_task_plans", False)
    resp = client.post("/webhook/feishu", json=_message_event("@_user_1 跑评测"))
    assert resp.json() == {"code": 0, "msg": "Next-task plans are disabled"}
    assert default_plans.list_pending("demo-llm-pretrain") == []
