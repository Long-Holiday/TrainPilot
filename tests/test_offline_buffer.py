"""Unit tests for TrainPilot GPU Client offline buffering and fault self-healing."""

from unittest.mock import MagicMock, patch
import requests
import pytest

from trainpilot.agent.client import TrainPilotClient


def test_offline_buffering_on_network_failure():
    """Routine milestone reporting must never crash training when gateway is offline."""
    client = TrainPilotClient(
        gateway_url="http://127.0.0.1:9999",
        task_id="test-offline-task",
        enable_offline_buffering=True,
    )

    # Mock session.post to simulate a connection error
    with patch.object(client.session, "post", side_effect=requests.ConnectionError("Failed to connect")):
        res = client.notify_milestone(message="Epoch 1 done", step=100, epoch=1)
        assert res["buffered"] is True
        assert res["success"] is True
        assert client.get_buffered_count() == 1

        client.notify_milestone(message="Epoch 2 done", step=200, epoch=2)
        assert client.get_buffered_count() == 2


def test_flush_offline_buffer_on_reconnection():
    """Buffered offline events are flushed in strict FIFO chronological order upon reconnection."""
    client = TrainPilotClient(
        gateway_url="http://127.0.0.1:9999",
        task_id="test-offline-task",
        enable_offline_buffering=True,
    )

    # 1. First, buffer 3 events
    with patch.object(client.session, "post", side_effect=requests.ConnectionError("Offline")):
        client.notify_milestone("Step 100", step=100)
        client.notify_milestone("Step 200", step=200)
        client.notify_milestone("Step 300", step=300)
        assert client.get_buffered_count() == 3

    # 2. Network recovers: mock successful post response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"success": True, "state": "RUNNING"}

    sent_steps = []
    def fake_post(url, json=None, timeout=None):
        if json:
            sent_steps.append(json.get("step"))
        return mock_resp

    with patch.object(client.session, "post", side_effect=fake_post):
        flushed = client.flush_offline_buffer()
        assert flushed == 3
        assert client.get_buffered_count() == 0
        # Verify chronological order
        assert sent_steps == [100, 200, 300]


def test_heartbeat_drains_offline_buffer():
    """Successful heartbeat pings automatically drain queued offline events."""
    client = TrainPilotClient(
        gateway_url="http://127.0.0.1:9999",
        task_id="test-offline-task",
        enable_offline_buffering=True,
    )

    # Buffer 2 events
    with patch.object(client.session, "post", side_effect=requests.ConnectionError("Offline")):
        client.notify_milestone("Step 10", step=10)
        client.notify_milestone("Step 20", step=20)
        assert client.get_buffered_count() == 2

    # Heartbeat succeeds
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"success": True}

    with patch.object(client.session, "post", return_value=mock_resp):
        ok = client.send_heartbeat(step=25)
        assert ok is True
        # Buffer was drained automatically by heartbeat
        assert client.get_buffered_count() == 0


def test_notify_alert_retry_with_backoff():
    """Critical alerts retry with exponential backoff on transient network failures."""
    client = TrainPilotClient(
        gateway_url="http://127.0.0.1:9999",
        task_id="test-offline-task",
    )

    attempts = 0
    mock_success = MagicMock()
    mock_success.status_code = 200
    mock_success.json.return_value = {"success": True, "state": "WAITING"}

    def transient_failure(url, json=None, timeout=None):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise requests.ConnectionError("Transient glitch")
        return mock_success

    with patch.object(client.session, "post", side_effect=transient_failure):
        res = client.notify_alert(message="Loss NaN", step=50, max_retries=3)
        assert res["success"] is True
        assert attempts == 3
