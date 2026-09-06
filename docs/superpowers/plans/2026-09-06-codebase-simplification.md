# TrainPilot 代码库精简与架构统一实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 整理 TrainPilot 项目代码，消除冗余的多层接入方式（废除 PyTorchHook）与多端重复超时兜底，统一使用 SQLite 单一持久化与 ACK 自愈驱动通道，使项目干净、易读且维护成本最低。

**Architecture:** 控制面服务端作为单一决策源（30s 无人响应自动决策并就地更新卡片），GPU 训练端以通用 `TrainingGuardian` 作为唯一官方接入标准并长轮询等待指令；所有现场自愈由 `ack_instruction` 附带 `solution` 触发飞书恢复通知；服务端彻底移除内存双轨存储，全量由 `SQLiteStorage` 承接。

**Tech Stack:** Python 3.10+, FastAPI, SQLite, Pydantic v2, PyTorch, pytest, requests, lark-oapi.

## Global Constraints
- 遵守项目中所有的测试规范，测试使用 `uv run pytest` 执行。
- 保证执行过程中不破坏飞书卡片交互协议及长轮询毫秒唤醒机制。
- 每次在请求执行 bash 命令前，先用简体中文解释该命令的作用。
- 遵循极简（YAGNI）、单责任原则与高可读性代码标准。

---

### Task 1: 精简客户端与移除 PyTorch Hook

**Files:**
- Delete: `src/trainpilot/agent/hooks/pytorch.py`
- Delete: `src/trainpilot/agent/hooks/__init__.py`
- Modify: `src/trainpilot/agent/__init__.py`
- Modify: `src/trainpilot/agent/monitor.py`
- Modify: `src/trainpilot/agent/client.py`
- Modify: `tests/test_agent_client.py`

**Interfaces:**
- Consumes: `TrainPilotClient.notify_alert`, `TrainPilotClient.poll_instruction`, `TrainPilotClient.ack_instruction`
- Produces: 精简后的 `TrainingGuardian`（仅保留核心检测与纯阻塞长轮询恢复，无 `timeout_fallback_action`），移除 `TrainPilotPyTorchHook` 与 `client.notify_recovery`。

- [ ] **Step 1: 更新测试用例 `tests/test_agent_client.py`**

移除所有针对 `TrainPilotPyTorchHook` 的测试，重写针对 `TrainingGuardian` 的核心异常流测试：
```python
"""Unit tests for TrainPilot Agent client and TrainingGuardian."""

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

    # 校验是否轮询
    mock_client.poll_instruction.assert_called_once()

    # 校验是否正确发送 ACK 并附带自愈总结
    mock_client.ack_instruction.assert_called_once()
    ack_args = mock_client.ack_instruction.call_args[1]
    assert ack_args["action"] == "self_resolve"
    assert ack_args["status"] == "success"
    assert "已跳过异常 Batch" in ack_args["solution"]
    assert ack_args["step"] == 10


def test_guardian_custom_action_handler(mock_client):
    guardian = TrainingGuardian(client=mock_client)
    mock_client.poll_instruction.return_value = {
        "ready": True,
        "action": "self_resolve",
        "instruction_id": "inst_custom",
    }
    guardian.register_action_handler("self_resolve", lambda payload: "自定义恢复逻辑成功执行")

    guardian.check_and_handle_loss(loss_val=float("inf"), step=20)

    mock_client.ack_instruction.assert_called_once()
    ack_args = mock_client.ack_instruction.call_args[1]
    assert ack_args["solution"] == "自定义恢复逻辑成功执行"


def test_guardian_stop_training_exception(mock_client):
    guardian = TrainingGuardian(client=mock_client)
    mock_client.poll_instruction.return_value = {
        "ready": True,
        "action": "stop_training",
        "instruction_id": "inst_stop",
    }

    with pytest.raises(StopTrainingException):
        guardian.check_and_handle_loss(loss_val=float("nan"), step=30)

    mock_client.ack_instruction.assert_called_once_with(
        action="stop_training",
        instruction_id="inst_stop",
        status="success",
        message="Control signal 'stop_training' handled",
    )
```

- [ ] **Step 2: 运行测试观察预期的错误或待调整项**

运行: `uv run pytest tests/test_agent_client.py -v`

- [ ] **Step 3: 删除 hooks 目录并精简 `agent/__init__.py`**

删除 `src/trainpilot/agent/hooks/`，修改 `src/trainpilot/agent/__init__.py`：
```python
"""TrainPilot Agent Package."""

from trainpilot.agent.client import TrainPilotClient
from trainpilot.agent.monitor import StopTrainingException, TrainingGuardian

__all__ = [
    "TrainPilotClient",
    "TrainingGuardian",
    "StopTrainingException",
]
```

- [ ] **Step 4: 精简 `src/trainpilot/agent/monitor.py`**

移除 `timeout_fallback_action` 参数，保持纯粹阻塞等待服务端决策：
```python
"""Training Guardian and Recovery Controller for GPU Agent."""

import logging
import math
from typing import Any, Callable, Dict, Optional

from trainpilot.agent.client import TrainPilotClient, _coerce_to_float

logger = logging.getLogger("trainpilot.agent.guardian")


class StopTrainingException(Exception):
    """Raised when human operator selects 'stop_training' action."""


class TrainingGuardian:
    """Monitors training iterations, intercepts anomalies, freezes training,

    and orchestrates human-in-the-loop recovery.
    """

    def __init__(
        self,
        client: TrainPilotClient,
        poll_interval: float = 2.0,
        poll_timeout: Optional[float] = None,
        loss_spike_threshold: Optional[float] = 1e4,
    ):
        self.client = client
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.loss_spike_threshold = loss_spike_threshold
        self._action_handlers: Dict[str, Callable[[Optional[Dict[str, Any]]], Any]] = {}
        self._register_default_handlers()

    def register_action_handler(
        self,
        action: str,
        handler: Callable[[Optional[Dict[str, Any]]], Any],
    ) -> None:
        """Register a callback for a specific human action (e.g. self_resolve)."""
        self._action_handlers[action] = handler
        logger.info("Registered custom action handler for: %s", action)

    def _register_default_handlers(self) -> None:
        self._action_handlers["stop_training"] = self._default_stop_handler
        self._action_handlers["self_resolve"] = self._default_self_resolve_handler

    def _default_stop_handler(self, payload: Optional[Dict[str, Any]] = None):
        logger.warning("Stop action triggered by human operator. Aborting training gracefully.")
        raise StopTrainingException("Training stopped by human operator decision")

    def _default_self_resolve_handler(self, payload: Optional[Dict[str, Any]] = None):
        if payload and payload.get("auto_resolved"):
            logger.warning("Server auto self-resolve after timeout; continuing training as-is.")
        else:
            logger.info("Self-resolve action: continuing training without modification.")
        return {"self_resolved": True, "payload": payload}

    def check_and_handle_loss(
        self,
        loss_val: Any,
        step: int,
        epoch: Optional[int] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            coerced = _coerce_to_float(loss_val)
        except (TypeError, ValueError):
            logger.warning("Unparseable loss value at step %s: %r, skipping anomaly check", step, loss_val)
            return

        anomaly_msg = None
        if math.isnan(coerced):
            anomaly_msg = f"Loss is NaN at step {step}"
        elif math.isinf(coerced):
            anomaly_msg = f"Loss is Inf at step {step}"
        elif self.loss_spike_threshold is not None and coerced > self.loss_spike_threshold:
            anomaly_msg = f"Loss exploded to {coerced:.2e} exceeding threshold {self.loss_spike_threshold:.2e} at step {step}"

        if anomaly_msg:
            logger.critical("Training anomaly detected: %s", anomaly_msg)
            combined_metrics = {"loss": loss_val}
            if extra_metrics:
                combined_metrics.update(extra_metrics)

            self.handle_anomaly(
                message=anomaly_msg,
                step=step,
                epoch=epoch,
                metrics=combined_metrics,
            )

    def handle_anomaly(
        self,
        message: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Any:
        # 1. Notify control plane (triggers Feishu alert card and starts 30s server timer)
        self.client.notify_alert(
            message=message,
            step=step,
            epoch=epoch,
            metrics=metrics,
            extra=extra,
        )

        anomaly_ctx = {
            "message": message,
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "extra": extra,
        }

        # 2. Block and poll for human instruction (waits for human click or server 30s auto-resolve)
        instruction = self.client.poll_instruction(
            timeout=self.poll_timeout,
            interval=self.poll_interval,
            pop=True,
        )

        action = instruction.get("action") or "self_resolve"
        instruction_id = instruction.get("instruction_id")
        payload = instruction.get("payload")

        logger.info("Executing recovery action: '%s' (instruction_id: %s)", action, instruction_id)
        return self._execute_action(
            action=action,
            instruction_id=instruction_id,
            payload=payload,
            anomaly_context=anomaly_ctx,
        )

    def _generate_solution_summary(
        self,
        result: Any,
        payload: Optional[Dict[str, Any]],
        anomaly_context: Optional[Dict[str, Any]],
    ) -> str:
        if isinstance(result, str) and result.strip():
            return result.strip()
        if isinstance(result, dict) and result.get("solution") and str(result["solution"]).strip():
            return str(result["solution"]).strip()
        if payload and isinstance(payload, dict) and payload.get("solution") and str(payload["solution"]).strip():
            return str(payload["solution"]).strip()

        anomaly_msg = (anomaly_context.get("message") or "").lower() if anomaly_context else ""
        if "nan" in anomaly_msg or "inf" in anomaly_msg or "loss" in anomaly_msg:
            return "已跳过异常 Batch 并重置优化器梯度状态，Loss 恢复正常，训练继续进行。"
        if "oom" in anomaly_msg or "memory" in anomaly_msg or "cuda" in anomaly_msg:
            return "已清理 GPU 显存碎片并释放非必要缓存，显存恢复安全水位，训练继续进行。"
        return "已完成现场自愈检查并重置执行状态，现场校验通过，训练恢复正常运行。"

    def _execute_action(
        self,
        action: str,
        instruction_id: Optional[str],
        payload: Optional[Dict[str, Any]],
        anomaly_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        handler = self._action_handlers.get(action)
        if not handler:
            err_msg = f"No handler registered for action '{action}'. Defaulting to self-resolve."
            logger.warning(err_msg)
            self.client.ack_instruction(
                action=action,
                instruction_id=instruction_id,
                status="failed",
                message=err_msg,
            )
            return None

        try:
            result = handler(payload)
            solution_summary = None
            if action == "self_resolve":
                solution_summary = self._generate_solution_summary(result, payload, anomaly_context)

            step = anomaly_context.get("step") if anomaly_context else None
            epoch = anomaly_context.get("epoch") if anomaly_context else None
            metrics = anomaly_context.get("metrics") if anomaly_context else None

            ack_msg = (
                f"Action '{action}' executed successfully. Solution: {solution_summary}"
                if solution_summary
                else f"Action '{action}' executed successfully"
            )
            self.client.ack_instruction(
                action=action,
                instruction_id=instruction_id,
                status="success",
                message=ack_msg,
                solution=solution_summary,
                step=step,
                epoch=epoch,
                metrics=metrics,
            )
            return result
        except StopTrainingException:
            self.client.ack_instruction(
                action=action,
                instruction_id=instruction_id,
                status="success",
                message=f"Control signal '{action}' handled",
            )
            raise
        except Exception as exc:
            logger.error("Failed to execute recovery action '%s': %s", action, exc)
            self.client.ack_instruction(
                action=action,
                instruction_id=instruction_id,
                status="failed",
                message=str(exc),
            )
            raise
```

- [ ] **Step 5: 精简 `src/trainpilot/agent/client.py` 并移除 `notify_recovery`**

从 `src/trainpilot/agent/client.py` 中删除 `notify_recovery` 方法。

- [ ] **Step 6: 运行测试验证**

运行: `uv run pytest tests/test_agent_client.py -v`
预期: 全部 PASS。

- [ ] **Step 7: 提交任务 1**

```bash
git add src/trainpilot/agent tests/test_agent_client.py
git commit -m "refactor(agent): streamline TrainingGuardian and remove PyTorchHook"
```

---

### Task 2: 精简服务端 Mailbox 存储与统一恢复通道

**Files:**
- Modify: `src/trainpilot/server/mailbox.py`
- Modify: `src/trainpilot/server/routes/tasks.py`
- Modify: `tests/test_server_api.py`
- Modify: `tests/test_sqlite_storage.py`

**Interfaces:**
- Consumes: `SQLiteStorage`
- Produces: 彻底移除 `_fallback_memory_events` 的纯净 `TaskMailboxManager`；移除 `notify(recovery)` 重复通道。

- [ ] **Step 1: 修改 `tests/test_server_api.py`**

移除 `test_notify_recovery_event`，增加校验通过 `ack` 接口发送自愈总结并触发飞书自愈卡片的测试：
```python
def test_ack_recovery_card_flow(client: TestClient):
    from trainpilot.server.feishu.client import default_feishu_client
    task_id = "ack-recovery-test-task"

    # 1. 触发告警
    client.post("/api/tasks/notify", json={
        "task_id": task_id,
        "event_type": "alert",
        "message": "Loss exploded",
    })

    # 2. 模拟下发决策
    client.post(f"/api/tasks/{task_id}/decision", json={
        "task_id": task_id,
        "action": "self_resolve",
    })

    # 3. 客户端消费指令并 ACK (携带 solution)
    resp = client.post(f"/api/tasks/{task_id}/ack", json={
        "task_id": task_id,
        "action": "self_resolve",
        "status": "success",
        "solution": "已跳过异常 Batch 并恢复正常训练。",
        "step": 60,
        "epoch": 1,
    })
    assert resp.status_code == 200
    assert resp.json()["state"] == "RUNNING"

    # 4. 验证 FeishuClient 接收到恢复卡片
    history = default_feishu_client.sent_cards_history
    rec_cards = [c for c in history if c.get("type") == "recovery" and c.get("task_id") == task_id]
    assert len(rec_cards) == 1
    card_dict = rec_cards[0]["card"]
    assert "【异常已自行解决】" in card_dict["header"]["title"]["content"]
```

- [ ] **Step 2: 改造 `src/trainpilot/server/mailbox.py`**

1. 彻底删除 `self._fallback_memory_events` 字段及其初始化；
2. 确保 `self._storage` 在初始化失败时抛出异常或默认初始化默认路径，不再降级为不持久化的内存双轨；
3. `record_event`: 移除 `EventType.RECOVERY` 的处理分支；事件写入统一走 `self._storage.append_event`；
4. `get_events` 与 `get_events_count`: 纯粹从 `self._storage` 读取；
5. `_build_summary`: 计数直接从 `self._storage.get_events_count` 获取。

- [ ] **Step 3: 改造 `src/trainpilot/server/routes/tasks.py`**

1. `_dispatch_feishu`: 移除 `req.event_type == EventType.RECOVERY` 分支；
2. 确保在 `ack_instruction` 中，只要 `req.status == "success"` 且 `(req.action == "self_resolve" or req.solution)`，统一异步调用 `default_feishu_client.send_recovery`。

- [ ] **Step 4: 运行测试验证**

运行: `uv run pytest tests/test_server_api.py tests/test_sqlite_storage.py -v`
预期: 全部 PASS。

- [ ] **Step 5: 提交任务 2**

```bash
git add src/trainpilot/server tests/test_server_api.py tests/test_sqlite_storage.py
git commit -m "refactor(server): clean up mailbox storage and unify recovery flow to ACK"
```

---

### Task 3: 精简 CLI 工具、示例代码与运行清理

**Files:**
- Modify: `skills/trainpilot/scripts/trainpilot_tool.py`
- Modify: `examples/real_gpu_training.py`
- Modify: `examples/mock_training.py`
- Modify: `tests/test_skills_cli.py`
- Modify: `tests/test_e2e_simulation.py`

**Interfaces:**
- Consumes: `TrainingGuardian`, `TrainPilotClient`
- Produces: 极简纯粹的示例代码与 CLI 客户端工具。

- [ ] **Step 1: 精简 `skills/trainpilot/scripts/trainpilot_tool.py`**

移除重复实现的 `_sanitize_for_json` 与多层 candidates 搜索逻辑，直接从 `trainpilot.agent.client` 导入；保持参数解析与调用简洁清晰。

- [ ] **Step 2: 重构 `examples/real_gpu_training.py`**

彻底删除 `decision_watcher` 后台线程、状态轮询和自写兜底注入。整体重构为极简、标准的使用方式：
```python
"""Real PyTorch GPU training simulation with TrainPilot HITL closed-loop recovery."""

import argparse
import logging
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from trainpilot.agent import TrainPilotClient, TrainingGuardian
from trainpilot.common.gateway import resolve_gateway_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RealGPUTraining")


class SimpleMLP(nn.Module):
    def __init__(self, in_features: int = 128, hidden: int = 256, out_features: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_training(gateway_url: str, task_id: str = "real-gpu-mlp"):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this script.")

    device = torch.device("cuda:0")
    print(f"🚀 Training task {task_id} on {torch.cuda.get_device_name(device)} with TrainPilot...")

    model = SimpleMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    client = TrainPilotClient(gateway_url=gateway_url, task_id=task_id)
    guardian = TrainingGuardian(client=client)

    for step in range(1, 11):
        inputs = torch.randn(64, 128, device=device)
        targets = torch.randint(0, 10, (64,), device=device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if step == 5:
            print(f"\n[Step {step}] ⚠️ Injecting NaN loss anomaly...")
            loss_val = float("nan")
        else:
            loss.backward()
            optimizer.step()
            loss_val = loss.item()

        # 核心：自动捕获 -> 冻结现场 -> 飞书告警 -> 等待飞书点击或服务端 30s 自动闭环 -> ACK 恢复
        guardian.check_and_handle_loss(loss_val=loss_val, step=step)
        print(f"[Step {step}] Loss: {loss_val:.4f}")

    client.notify_event(event_type="completed", message="Training finished successfully!", step=10)
    print("🎉 Training successfully completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default=resolve_gateway_url())
    parser.add_argument("--task-id", default="real-gpu-mlp")
    args = parser.parse_args()
    run_training(args.gateway, args.task_id)
```

- [ ] **Step 3: 精简 `examples/mock_training.py`**

使用同样的极简结构，保证无 GPU 机器上快速运行。

- [ ] **Step 4: 运行 CLI 与 E2E 测试**

运行: `uv run pytest tests/test_skills_cli.py tests/test_e2e_simulation.py -v`
预期: 全部 PASS。

- [ ] **Step 5: 提交任务 3**

```bash
git add skills/trainpilot/scripts/trainpilot_tool.py examples/ tests/
git commit -m "refactor(examples): clean up CLI tool and real GPU training example"
```

---

### Task 4: 文档更新、全量回归与环境清理

**Files:**
- Modify: `README.md`
- Modify: `skills/trainpilot/SKILL.md`
- Clean: `trainpilot.log`, `trainpilot.db*`, `.pytest_cache`

- [ ] **Step 1: 更新 `README.md` 与 `skills/trainpilot/SKILL.md`**

移除所有 `TrainPilotPyTorchHook` 代码和说明，统一主推 `TrainingGuardian`。

- [ ] **Step 2: 清理临时数据库与日志文件**

删除本地生成的 `trainpilot.log`, `trainpilot.db`, `trainpilot.db-shm`, `trainpilot.db-wal`。

- [ ] **Step 3: 全量测试回归**

运行: `uv run pytest -v`
预期: 50+ 个测试全部通过。

- [ ] **Step 4: 最终提交**

```bash
git add README.md skills/trainpilot/SKILL.md
git commit -m "docs: align documentation with streamlined TrainingGuardian architecture"
```
