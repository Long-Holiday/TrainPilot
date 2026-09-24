"""Unit tests for ServerWatchdog ping-based liveness and health metrics."""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from trainpilot.common.schemas import EventNotifyRequest
from trainpilot.common.states import EventType, TaskState
from trainpilot.server.config import ServerSettings
from trainpilot.server.feishu.client import default_feishu_client
from trainpilot.server.mailbox import TaskMailboxManager, default_mailbox
from trainpilot.server.main import app
from trainpilot.server.storage import SQLiteStorage
from trainpilot.server.watchdog import ServerWatchdog


@pytest.fixture(autouse=True)
def clean_env():
    from trainpilot.server.config import settings
    old_token = settings.api_token
    settings.api_token = None
    default_mailbox.reset()
    default_feishu_client.sent_cards_history.clear()
    yield
    default_mailbox.reset()
    default_feishu_client.sent_cards_history.clear()
    settings.api_token = old_token


def _register(mailbox: TaskMailboxManager, task_id: str, gpu_host: str = "127.0.0.1") -> None:
    mailbox.record_event(
        EventNotifyRequest(
            task_id=task_id,
            event_type=EventType.MILESTONE,
            message="Task started",
            step=0,
            gpu_host=gpu_host,
        )
    )


def test_watchdog_detects_unreachable_and_alerts():
    """Ping 失败的任务被判失联并告警一次(去重)。"""
    custom_settings = ServerSettings(
        watchdog_interval_seconds=1,
        enable_watchdog=True,
    )
    mailbox = TaskMailboxManager(storage=None)
    watchdog = ServerWatchdog(
        mailbox=mailbox,
        server_settings=custom_settings,
        ping_func=lambda host, timeout: False,
    )

    _register(mailbox, "task-unreachable-01", gpu_host="10.0.0.1")

    check_res = watchdog.run_check_once()
    assert check_res["stale_count"] == 1
    assert check_res["alerts_sent"] == 1
    assert check_res["ping_checked"] == 1

    cards = default_feishu_client.sent_cards_history
    assert len(cards) >= 1
    last_card = cards[-1]
    assert last_card["type"] == "stale_alert"
    assert last_card["task_id"] == "task-unreachable-01"

    check_res2 = watchdog.run_check_once()
    assert check_res2["stale_count"] == 1
    assert check_res2["alerts_sent"] == 0


def test_watchdog_recovers_when_ping_succeeds():
    """Ping 恢复后任务不再失联, 去重标记被清除。"""
    custom_settings = ServerSettings(enable_watchdog=True)
    mailbox = TaskMailboxManager(storage=None)
    results = {"ok": False}
    watchdog = ServerWatchdog(
        mailbox=mailbox,
        server_settings=custom_settings,
        ping_func=lambda host, timeout: results["ok"],
    )

    _register(mailbox, "task-recovering-01", gpu_host="10.0.0.2")

    res1 = watchdog.run_check_once()
    assert res1["stale_count"] == 1

    results["ok"] = True
    res2 = watchdog.run_check_once()
    assert res2["stale_count"] == 0

    # 再次失联应重新告警(去重已清除)
    results["ok"] = False
    res3 = watchdog.run_check_once()
    assert res3["stale_count"] == 1
    assert res3["alerts_sent"] == 1


def test_watchdog_flags_task_without_gpu_host():
    """从未上报 GPU IP 的任务无法 ping, 直接判失联。"""
    custom_settings = ServerSettings(enable_watchdog=True)
    mailbox = TaskMailboxManager(storage=None)
    watchdog = ServerWatchdog(
        mailbox=mailbox,
        server_settings=custom_settings,
        ping_func=lambda host, timeout: True,
    )

    task = mailbox._get_or_create("task-no-gpu-ip")
    task.state = TaskState.RUNNING
    assert task.gpu_host is None

    res = watchdog.run_check_once()
    assert res["stale_count"] == 1
    assert res["ping_checked"] == 0


def test_watchdog_pings_shared_host_once():
    """同一 GPU IP 的多个任务一轮只 ping 一次。"""
    custom_settings = ServerSettings(enable_watchdog=True)
    mailbox = TaskMailboxManager(storage=None)
    calls: list = []
    watchdog = ServerWatchdog(
        mailbox=mailbox,
        server_settings=custom_settings,
        ping_func=lambda host, timeout: calls.append(host) or True,
    )

    _register(mailbox, "task-shared-01", gpu_host="10.0.0.3")
    _register(mailbox, "task-shared-02", gpu_host="10.0.0.3")

    res = watchdog.run_check_once()
    assert res["stale_count"] == 0
    assert res["ping_checked"] == 1
    assert calls == ["10.0.0.3"]


def test_watchdog_in_health_endpoint():
    """Verify /health exposes watchdog stats."""
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "watchdog" in data
    assert "checks_count" in data["watchdog"]
    assert "stale_detected_total" in data["watchdog"]
    assert data["sqlite_enabled"] is True


def test_scheduled_task_cleanup_runs_independently_of_ping_watchdog(tmp_path):
    """Cleanup runs immediately, then respects its own interval."""
    storage = SQLiteStorage(str(tmp_path / "cleanup.db"))
    mailbox = TaskMailboxManager(storage=storage)
    custom_settings = ServerSettings(
        enable_watchdog=False,
        enable_task_cleanup=True,
        task_retention_hours=1,
        task_cleanup_interval_seconds=3600,
    )
    watchdog = ServerWatchdog(mailbox=mailbox, server_settings=custom_settings)

    first = mailbox._get_or_create("expired-first")
    first.state = TaskState.COMPLETED
    first.updated_at = "2020-01-01T00:00:00+00:00"
    storage.save_task(first)

    result = watchdog.run_check_once()
    assert result["tasks_expired"] == 1
    assert mailbox.get_task("expired-first") is None
    assert watchdog.get_stats()["tasks_expired_total"] == 1
    assert watchdog.get_stats()["last_cleanup_at"] is not None

    second = mailbox._get_or_create("expired-second")
    second.state = TaskState.FAILED
    second.updated_at = "2020-01-01T00:00:00+00:00"
    storage.save_task(second)

    # A regular sweep must not turn into an early cleanup sweep.
    assert watchdog.run_check_once()["tasks_expired"] == 0
    assert mailbox.get_task("expired-second") is not None

    watchdog._last_cleanup_monotonic -= 3600
    assert watchdog.run_check_once()["tasks_expired"] == 1
    assert mailbox.get_task("expired-second") is None
    storage.close()


def test_running_watchdog_does_not_block_on_slow_alert_delivery(tmp_path, monkeypatch):
    """Feishu network latency must not block a maintenance sweep."""
    storage = SQLiteStorage(str(tmp_path / "alerts.db"))
    mailbox = TaskMailboxManager(storage=storage)
    custom_settings = ServerSettings(
        enable_watchdog=True,
        enable_task_cleanup=False,
    )
    watchdog = ServerWatchdog(
        mailbox=mailbox,
        server_settings=custom_settings,
        ping_func=lambda host, timeout: False,
    )
    _register(mailbox, "slow-alert", gpu_host="10.0.0.4")

    entered = threading.Event()
    release = threading.Event()

    def slow_send(**kwargs):
        entered.set()
        release.wait(timeout=2)
        return {"ok": True}

    monkeypatch.setattr(default_feishu_client, "send_stale_alert", slow_send)
    # An alive thread marks this as the service path without starting its loop.
    watchdog._thread = threading.current_thread()
    started_at = time.monotonic()
    result = watchdog.run_check_once()
    elapsed = time.monotonic() - started_at
    watchdog._thread = None

    assert result["stale_count"] == 1
    assert entered.wait(timeout=1)
    assert elapsed < 0.5
    release.set()

    deadline = time.monotonic() + 1
    while watchdog.get_stats()["stale_alerts_sent_total"] == 0:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    storage.close()


def test_watchdog_auto_detects_ip_from_http_request_without_agent_reporting():
    """Agent 不需要汇报 IP，服务端通过分析网络请求获取 IP，看门狗对其 ping 判活。"""
    client = TestClient(app)
    # Agent 发送请求但完全不传递 gpu_host
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
    # TestClient 默认 client.host 为 testclient
    assert task.gpu_host in ("testclient", "127.0.0.1")

    # Watchdog 可以 ping 通该 IP
    pinged_hosts = []
    watchdog = ServerWatchdog(
        mailbox=default_mailbox,
        server_settings=ServerSettings(enable_watchdog=True),
        ping_func=lambda host, timeout: pinged_hosts.append(host) or True,
    )
    res = watchdog.run_check_once()
    assert res["stale_count"] == 0
    assert pinged_hosts == [task.gpu_host]


def test_watchdog_detects_ip_from_x_forwarded_for():
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


def test_watchdog_detects_ip_from_x_real_ip():
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


def test_watchdog_updates_ip_on_subsequent_request():
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
    # 创建一个纯任务（尚无 IP）
    t = default_mailbox._get_or_create("poll-task")
    t.state = TaskState.RUNNING
    assert t.gpu_host is None

    # Agent 发送 poll_instruction 请求，服务端从网络请求分析出 IP
    client.get(
        "/api/tasks/poll-task/instruction?wait_timeout=0",
        headers={"X-Real-IP": "172.16.50.5"},
    )
    assert default_mailbox.get_task("poll-task").gpu_host == "172.16.50.5"
