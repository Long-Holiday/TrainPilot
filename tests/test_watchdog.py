"""Unit tests for ServerWatchdog background service and health metrics."""

import time
from datetime import datetime, timedelta, timezone
import pytest
from fastapi.testclient import TestClient

from trainpilot.common.schemas import EventNotifyRequest, HeartbeatRequest
from trainpilot.common.states import EventType, TaskState
from trainpilot.server.config import ServerSettings
from trainpilot.server.feishu.client import default_feishu_client
from trainpilot.server.mailbox import TaskMailboxManager, default_mailbox
from trainpilot.server.main import app
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
