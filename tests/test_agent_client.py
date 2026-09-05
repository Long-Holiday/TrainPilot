"""Unit tests for TrainPilot Agent client and TrainingGuardian."""

import math
from unittest.mock import MagicMock, patch
import pytest

from trainpilot.agent.client import TrainPilotClient
from trainpilot.agent.hooks.pytorch import TrainPilotPyTorchHook
from trainpilot.agent.monitor import (
    StopTrainingException,
    TrainingGuardian,
)


@pytest.fixture
def mock_client():
    client = MagicMock(spec=TrainPilotClient)
    client.task_id = "test-agent-task"
    return client


def test_guardian_loss_anomaly_trigger(mock_client):
    guardian = TrainingGuardian(client=mock_client)

    recovered = False

    def on_self_resolve(payload):
        nonlocal recovered
        recovered = True

    guardian.register_action_handler("self_resolve", on_self_resolve)

    # Configure mock poll_instruction to return self_resolve
    mock_client.poll_instruction.return_value = {
        "ready": True,
        "action": "self_resolve",
        "instruction_id": "inst_xyz",
    }

    # Pass NaN loss
    guardian.check_and_handle_loss(loss_val=float("nan"), step=42, epoch=1)

    # Verify client called alert
    mock_client.notify_alert.assert_called_once()
    alert_args = mock_client.notify_alert.call_args[1]
    assert "NaN" in alert_args["message"]
    assert alert_args["step"] == 42

    # Verify poll_instruction called
    mock_client.poll_instruction.assert_called_once()

    # Verify custom handler was executed
    assert recovered is True

    # Verify ack sent
    mock_client.ack_instruction.assert_called_once_with(
        action="self_resolve",
        instruction_id="inst_xyz",
        status="success",
        message="Action 'self_resolve' executed successfully",
    )


def test_guardian_stop_training_exception(mock_client):
    guardian = TrainingGuardian(client=mock_client)
    mock_client.poll_instruction.return_value = {
        "ready": True,
        "action": "stop_training",
        "instruction_id": "inst_stop",
    }

    with pytest.raises(StopTrainingException):
        guardian.check_and_handle_loss(loss_val=float("inf"), step=10)

    # Ensure ack was still dispatched before exception raised
    mock_client.ack_instruction.assert_called_once_with(
        action="stop_training",
        instruction_id="inst_stop",
        status="success",
        message="Control signal 'stop_training' handled",
    )


def test_pytorch_hook_milestone_and_heartbeat(mock_client):
    with patch("trainpilot.agent.hooks.pytorch.TrainPilotClient", return_value=mock_client):
        hook = TrainPilotPyTorchHook(
            task_id="hook-task",
            milestone_step_interval=5,
            heartbeat_step_interval=2,
        )
        # Step 2: should trigger heartbeat
        hook.on_step_end(step=2, loss=0.5)
        mock_client.send_heartbeat.assert_called_once()

        # Step 5: should trigger milestone
        hook.on_step_end(step=5, loss=0.3)
        mock_client.notify_milestone.assert_called_once()
