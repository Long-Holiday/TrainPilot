"""Training Guardian and Recovery Controller for GPU Agent."""

import logging
import math
from typing import Any, Callable, Dict, Optional

from trainpilot.agent.client import TrainPilotClient, _coerce_to_float

logger = logging.getLogger("trainpilot.agent.guardian")


class StopTrainingException(Exception):
    """Raised when human operator selects 'stop_training' action."""


class SkipBatchException(Exception):
    """Raised when human operator selects 'skip_batch' action."""


class TrainingGuardian:
    """Monitors training iterations, intercepts anomalies, pauses training,

    and orchestrates human-in-the-loop recovery.
    """

    def __init__(
        self,
        client: TrainPilotClient,
        poll_interval: float = 2.0,
        poll_timeout: Optional[float] = 600.0,
        loss_spike_threshold: Optional[float] = 1e4,
        timeout_fallback_action: Optional[str] = None,
    ):
        self.client = client
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.loss_spike_threshold = loss_spike_threshold
        # When polling times out, auto-execute this action instead of crashing.
        # e.g. timeout_fallback_action="resume" keeps training alive; None re-raises.
        self.timeout_fallback_action = timeout_fallback_action
        self._action_handlers: Dict[str, Callable[[Optional[Dict[str, Any]]], Any]] = {}

        # Register default built-in handlers
        self._register_default_handlers()

    def register_action_handler(
        self,
        action: str,
        handler: Callable[[Optional[Dict[str, Any]]], Any],
    ) -> None:
        """Register a callback for a specific human action (e.g. reduce_lr_rollback)."""
        self._action_handlers[action] = handler
        logger.info("Registered custom action handler for: %s", action)

    def _register_default_handlers(self) -> None:
        """Register basic default implementations."""
        self._action_handlers["stop_training"] = self._default_stop_handler
        self._action_handlers["skip_batch"] = self._default_skip_handler
        self._action_handlers["resume"] = lambda payload: logger.info("Resuming training without modification")
        # Most-clicked card button needs a safe default so ACK is not 'failed'
        # when users forget to register their own checkpoint logic.
        self._action_handlers.setdefault("reduce_lr_rollback", self._default_rollback_handler)

    def _default_stop_handler(self, payload: Optional[Dict[str, Any]] = None):
        logger.warning("Stop action triggered by human operator. Aborting training gracefully.")
        raise StopTrainingException("Training stopped by human operator decision")

    def _default_skip_handler(self, payload: Optional[Dict[str, Any]] = None):
        logger.info("Skip batch action triggered by human operator.")
        raise SkipBatchException("Current batch skipped by human operator decision")

    def _default_rollback_handler(self, payload: Optional[Dict[str, Any]] = None):
        logger.warning(
            "reduce_lr_rollback requested but no custom handler registered. "
            "Override via register_action_handler() with checkpoint reload + LR decay. Resuming as-is."
        )
        return {"fallback": "resume_without_checkpoint", "payload": payload}

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

        # 2. Block and poll for human instruction (with timeout fallback)
        try:
            instruction = self.client.poll_instruction(
                timeout=self.poll_timeout,
                interval=self.poll_interval,
                pop=True,
            )
        except TimeoutError as exc:
            if self.timeout_fallback_action:
                logger.warning("HITL poll timed out (%s); executing fallback '%s'",
                               exc, self.timeout_fallback_action)
                return self._execute_action(
                    action=self.timeout_fallback_action,
                    instruction_id=None,
                    payload={"timeout_fallback": True, "original_message": message},
                )
            raise

        action = instruction.get("action") or "resume"
        instruction_id = instruction.get("instruction_id")
        payload = instruction.get("payload")

        logger.info("Executing recovery action: '%s' (instruction_id: %s)", action, instruction_id)
        return self._execute_action(action=action, instruction_id=instruction_id, payload=payload)

    def _execute_action(
        self,
        action: str,
        instruction_id: Optional[str],
        payload: Optional[Dict[str, Any]],
    ) -> Any:
        # 3. Dispatch to handler
        handler = self._action_handlers.get(action)
        if not handler:
            err_msg = f"No handler registered for action '{action}'. Defaulting to resume."
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
            # 4. Acknowledge success to reset state to RUNNING
            self.client.ack_instruction(
                action=action,
                instruction_id=instruction_id,
                status="success",
                message=f"Action '{action}' executed successfully",
            )
            return result
        except (StopTrainingException, SkipBatchException):
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
