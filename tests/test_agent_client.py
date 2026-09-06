"""Unit tests for TrainPilot Agent client and TrainingGuardian."""

import math
from unittest.mock import MagicMock
import pytest

from trainpilot.agent.client import TrainPilotClient
from trainpilot.agent.monitor import (
    StopTrainingException,
    TrainingGuardian,
)


@pytest.fixture
def mock_client():
    client = MagicMock(spec=TrainPilotClient)
    client.task_id = "test-agent-task"
    return client


def test_guardian_loss_anomaly_and_ack(mock_client):
    guardian = TrainingGuardian(client=mock_client)
    mock_client.poll_instruction.return_value = {
        "ready": True,
        "action": "self_resolve",
        "instruction_id": "inst_123",
    }

    # 触发 NaN
    guardian.check_and_handle_loss(loss_val=float("nan"), step=10, epoch=1)

    # 校验是否上报告警
    mock_client.notify_alert.assert_called_once()
    alert_args = mock_client.notify_alert.call_args[1]
    assert "NaN" in alert_args["message"]
    assert alert_args["step"] == 10
    assert alert_args["epoch"] == 1

    # 校验是否长轮询等待指令
    mock_client.poll_instruction.assert_called_once()

    # 校验是否正确发送 ACK 并附带自愈总结
    mock_client.ack_instruction.assert_called_once()
    ack_args = mock_client.ack_instruction.call_args[1]
    assert ack_args["action"] == "self_resolve"
    assert ack_args["instruction_id"] == "inst_123"
    assert ack_args["status"] == "success"
    assert "已跳过异常 Batch" in ack_args["solution"]
    assert ack_args["step"] == 10
    assert ack_args["epoch"] == 1


def test_guardian_custom_action_handler(mock_client):
    guardian = TrainingGuardian(client=mock_client)
    mock_client.poll_instruction.return_value = {
        "ready": True,
        "action": "self_resolve",
        "instruction_id": "inst_custom",
    }
    guardian.register_action_handler("self_resolve", lambda payload: "自定义恢复逻辑成功执行")

    guardian.check_and_handle_loss(loss_val=float("inf"), step=20)

    # 校验告警中的 Inf
    mock_client.notify_alert.assert_called_once()
    alert_args = mock_client.notify_alert.call_args[1]
    assert "Inf" in alert_args["message"]
    assert alert_args["step"] == 20

    mock_client.ack_instruction.assert_called_once()
    ack_args = mock_client.ack_instruction.call_args[1]
    assert ack_args["solution"] == "自定义恢复逻辑成功执行"
    assert ack_args["step"] == 20


def test_guardian_stop_training_exception(mock_client):
    guardian = TrainingGuardian(client=mock_client)
    mock_client.poll_instruction.return_value = {
        "ready": True,
        "action": "stop_training",
        "instruction_id": "inst_stop",
    }

    with pytest.raises(StopTrainingException):
        guardian.check_and_handle_loss(loss_val=float("nan"), step=30)

    # 确保在抛出异常前向服务端发送 ACK
    mock_client.ack_instruction.assert_called_once_with(
        action="stop_training",
        instruction_id="inst_stop",
        status="success",
        message="Control signal 'stop_training' handled",
    )


def test_guardian_loss_spike_trigger(mock_client):
    guardian = TrainingGuardian(client=mock_client, loss_spike_threshold=100.0)
    mock_client.poll_instruction.return_value = {
        "ready": True,
        "action": "self_resolve",
        "instruction_id": "inst_spike",
    }

    # 正常 loss 不触发
    guardian.check_and_handle_loss(loss_val=50.0, step=1)
    mock_client.notify_alert.assert_not_called()

    # 超过阈值触发
    guardian.check_and_handle_loss(loss_val=200.0, step=2)
    mock_client.notify_alert.assert_called_once()
    alert_args = mock_client.notify_alert.call_args[1]
    assert "exploded" in alert_args["message"]
    assert alert_args["step"] == 2
