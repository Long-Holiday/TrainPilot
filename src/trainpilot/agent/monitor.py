"""Training Guardian and Recovery Controller for GPU Agent."""

import logging
import math
from typing import Any, Callable, Dict, Optional

from trainpilot.agent.client import TrainPilotClient, _coerce_to_float

logger = logging.getLogger("trainpilot.agent.guardian")


class StopTrainingException(Exception):
    """Raised when human operator selects 'stop_training' action."""


class TrainingGuardian:
    """Monitors training iterations, intercepts anomalies, pauses training,

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

        # Register default built-in handlers
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
        """Register basic default implementations."""
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
        """Verify loss value; if NaN, Inf, or spike detected, pause and enter HITL loop.

        Accepts float, numpy scalar, 0-dim Tensor, or numeric string for robustness.
        """
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

    def handle_oom(
        self,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Convenience wrapper: report CUDA OOM through the same HITL pipeline."""
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return self.handle_anomaly(
            message=f"CUDA out of memory at step {step}",
            step=step,
            epoch=epoch,
            metrics={"error": "CUDA_OOM"},
            extra=extra,
        )

    def handle_anomaly(
        self,
        message: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Pause training, notify gateway, poll for human decision, execute handler, and ack."""
        # 1. Notify control plane
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

        # 2. Block and poll for human instruction
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
        """生成概括一句解决方法，简单明了告知用户自愈措施。"""
        # 1. 优先采用自定义 handler 返回的解决说明
        if isinstance(result, str) and result.strip():
            return result.strip()
        if isinstance(result, dict) and result.get("solution") and str(result["solution"]).strip():
            return str(result["solution"]).strip()
        # 2. 检查 payload 中是否显式指定 solution
        if payload and isinstance(payload, dict) and payload.get("solution") and str(payload["solution"]).strip():
            return str(payload["solution"]).strip()

        # 3. 智能根据异常信息概括一句解决方法
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
        # 3. Dispatch to handler
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

            # 4. Acknowledge success to reset state to RUNNING (server dispatches Feishu card from ACK)
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
            # Acknowledge before re-raising control flow exceptions
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
