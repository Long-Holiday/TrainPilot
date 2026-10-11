"""Unit tests for Client IP analysis from network requests."""

import pytest
from fastapi.testclient import TestClient

from trainpilot.common.states import TaskState
from trainpilot.server.mailbox import default_mailbox
from trainpilot.server.main import app


@pytest.fixture(autouse=True)
def clean_env():
    from trainpilot.server.config import settings
    old_token = settings.api_token
    settings.api_token = None
    default_mailbox.reset()
    yield
    default_mailbox.reset()
    settings.api_token = old_token


def test_auto_detects_ip_from_http_request_without_agent_reporting():
    """Agent 不需要汇报 IP，服务端通过分析网络请求自动获取 IP。"""
    client = TestClient(app)
    resp = client.post(
        "/api/tasks/notify",
        json={
            "task_id": "auto-ip-task",
            "event_type": "milestone",
            "message": "Epoch 1 started",
            "step": 10,
        },
    )
    assert resp.status_code == 200

    task = default_mailbox.get_task("auto-ip-task")
    assert task is not None
    assert task.gpu_host in ("testclient", "127.0.0.1")


def test_detects_ip_from_x_forwarded_for():
    """服务端通过 X-Forwarded-For 分析提取 GPU 真实 IP。"""
    client = TestClient(app)
    resp = client.post(
        "/api/tasks/notify",
        headers={"X-Forwarded-For": "192.168.10.88, 10.0.0.1"},
        json={
            "task_id": "xff-task",
            "event_type": "milestone",
            "message": "Step 1",
            "step": 1,
        },
    )
    assert resp.status_code == 200
    task = default_mailbox.get_task("xff-task")
    assert task is not None
    assert task.gpu_host == "192.168.10.88"


def test_detects_ip_from_x_real_ip():
    """服务端通过 X-Real-IP 分析提取 GPU 真实 IP。"""
    client = TestClient(app)
    resp = client.post(
        "/api/tasks/notify",
        headers={"X-Real-IP": "10.20.30.40"},
        json={
            "task_id": "real-ip-task",
            "event_type": "milestone",
            "message": "Step 1",
            "step": 1,
        },
    )
    assert resp.status_code == 200
    task = default_mailbox.get_task("real-ip-task")
    assert task is not None
    assert task.gpu_host == "10.20.30.40"


def test_updates_ip_on_subsequent_request():
    """当 Agent IP 发生变动（例如 DHCP 分配或迁移）时，后续网络请求会自动刷新 IP。"""
    client = TestClient(app)
    client.post(
        "/api/tasks/notify",
        headers={"X-Real-IP": "10.0.0.1"},
        json={"task_id": "roaming-task", "event_type": "milestone", "message": "Step 1"},
    )
    assert default_mailbox.get_task("roaming-task").gpu_host == "10.0.0.1"

    # Agent 在第二个请求中 IP 变更为 10.0.0.2
    client.post(
        "/api/tasks/notify",
        headers={"X-Real-IP": "10.0.0.2"},
        json={"task_id": "roaming-task", "event_type": "milestone", "message": "Step 2"},
    )
    assert default_mailbox.get_task("roaming-task").gpu_host == "10.0.0.2"


def test_network_request_ip_via_instruction_poll_or_status():
    """通过 poll_instruction 或 status 等日常请求也能分析并记录客户端 IP。"""
    client = TestClient(app)
    t = default_mailbox._get_or_create("poll-task")
    t.state = TaskState.RUNNING
    assert t.gpu_host is None

    client.get(
        "/api/tasks/poll-task/instruction?wait_timeout=0",
        headers={"X-Real-IP": "172.16.50.5"},
    )
    assert default_mailbox.get_task("poll-task").gpu_host == "172.16.50.5"
