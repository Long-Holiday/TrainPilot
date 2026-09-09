"""Unit tests for ServerWatchdog background service and health metrics."""

import time
import threading
from datetime import datetime, timedelta, timezone
import pytest
from fastapi.testclient import TestClient

from trainpilot.common.schemas import EventNotifyRequest, HeartbeatRequest
from trainpilot.common.states import EventType, TaskState
from trainpilot.server.config import ServerSettings
from trainpilot.server.feishu.client import default_feishu_client
from trainpilot.server.mailbox import TaskMailboxManager, default_mailbox
from trainpilot.server.main import app
from trainpilot.server.storage import SQLiteStorage
from trainpilot.server.watchdog import ServerWatchdog


@pytest.fixture(autouse=True)
def clean_env():
    default_mailbox.reset()
    default_feishu_client.sent_cards_history.clear()
    yield
    default_mailbox.reset()
    default_feishu_client.sent_cards_history.clear()


def test_watchdog_detects_stale_and_alerts():
    """Verify watchdog detects a task whose last update is beyond timeout and alerts."""
    custom_settings = ServerSettings(
        task_heartbeat_timeout_seconds=2,
        watchdog_interval_seconds=1,
        enable_watchdog=True,
    )
    mailbox = TaskMailboxManager(storage=None)
    watchdog = ServerWatchdog(mailbox=mailbox, server_settings=custom_settings)

    # 1. Create a task with updated_at in the past (> 2s ago)
    past_iso = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    task = mailbox._get_or_create("task-stalled-01")
    task.updated_at = past_iso
    task.last_heartbeat_at = past_iso
    task.state = TaskState.RUNNING

    # 2. Run check once
    check_res = watchdog.run_check_once()
    assert check_res["stale_count"] == 1
    assert check_res["alerts_sent"] == 1

    # Verify stale card sent
    cards = default_feishu_client.sent_cards_history
    assert len(cards) >= 1
    last_card = cards[-1]
    assert last_card["type"] == "stale_alert"
    assert last_card["task_id"] == "task-stalled-01"

    # 3. Second check sweep: debouncing prevents duplicate alerts
    check_res2 = watchdog.run_check_once()
    assert check_res2["stale_count"] == 1
    assert check_res2["alerts_sent"] == 0  # No duplicate alert sent!


def test_watchdog_recovers_after_heartbeat():
    """When a stale task sends a fresh heartbeat, it is no longer stale."""
    custom_settings = ServerSettings(
        task_heartbeat_timeout_seconds=2,
        enable_watchdog=True,
    )
    mailbox = TaskMailboxManager(storage=None)
    watchdog = ServerWatchdog(mailbox=mailbox, server_settings=custom_settings)

    past_iso = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    task = mailbox._get_or_create("task-recovering-01")
    task.updated_at = past_iso
    task.last_heartbeat_at = past_iso

    # Detect stale
    res1 = watchdog.run_check_once()
    assert res1["stale_count"] == 1

    # Send fresh heartbeat
    mailbox.record_heartbeat(
        HeartbeatRequest(
            task_id="task-recovering-01",
            step=100,
        )
    )

    # Detect again: no longer stale
    res2 = watchdog.run_check_once()
    assert res2["stale_count"] == 0


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


def test_scheduled_task_cleanup_runs_independently_of_heartbeat_watchdog(tmp_path):
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

    # A regular heartbeat sweep must not turn into an early cleanup sweep.
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
        task_heartbeat_timeout_seconds=1,
    )
    watchdog = ServerWatchdog(mailbox=mailbox, server_settings=custom_settings)
    stale = mailbox._get_or_create("slow-alert")
    stale.updated_at = "2020-01-01T00:00:00+00:00"
    stale.last_heartbeat_at = stale.updated_at
    storage.save_task(stale)

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
