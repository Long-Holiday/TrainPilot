"""PyTorch training hook and helper integration."""

import logging
from typing import Any, Callable, Dict, Optional

from trainpilot.agent.client import TrainPilotClient
from trainpilot.agent.monitor import (
    SkipBatchException,
    StopTrainingException,
    TrainingGuardian,
)

logger = logging.getLogger("trainpilot.agent.hooks.pytorch")


class TrainPilotPyTorchHook:
    """Helper hook for integrating into PyTorch training loops or trainer callbacks."""

    def __init__(
        self,
        task_id: str,
        gateway_url: Optional[str] = None,
        milestone_step_interval: int = 100,
        heartbeat_step_interval: int = 20,
        api_token: Optional[str] = None,
        **guardian_kwargs: Any,
    ):
        self.client = TrainPilotClient(gateway_url=gateway_url, task_id=task_id, api_token=api_token)
        self.guardian = TrainingGuardian(client=self.client, **guardian_kwargs)
        self.milestone_step_interval = milestone_step_interval
        self.heartbeat_step_interval = heartbeat_step_interval
        self.current_step = 0
        self.current_epoch = 0

    def register_recovery_callback(
        self,
        action: str,
        callback: Callable[[Optional[Dict[str, Any]]], Any],
    ) -> None:
        """Register specific recovery logic for human actions like reduce_lr_rollback."""
        self.guardian.register_action_handler(action, callback)

    def on_step_end(
        self,
        step: int,
        loss: Any,
        epoch: Optional[int] = None,
        lr: Optional[float] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Invoked after each optimizer step.

        Performs:
        1. Loss validity check (NaN / Inf / Explosion). If invalid, triggers HITL alert.
        2. Periodic lightweight heartbeat.
        3. Periodic milestone reporting.
        """
        from trainpilot.agent.client import _coerce_to_float

        self.current_step = step
        if epoch is not None:
            self.current_epoch = epoch

        metrics: Dict[str, Any] = {"loss": loss}
        if lr is not None:
            metrics["lr"] = lr
        if extra_metrics:
            metrics.update(extra_metrics)

        # 1. Anomaly check (pauses & polls if anomaly detected; Tensor/numpy-safe)
        self.guardian.check_and_handle_loss(
            loss_val=loss,
            step=step,
            epoch=epoch,
            extra_metrics=metrics,
        )

        # 2. Periodic heartbeat (failures only warn inside client)
        if self.heartbeat_step_interval and step % self.heartbeat_step_interval == 0:
            self.client.send_heartbeat(step=step, epoch=epoch, metrics=metrics)

        # 3. Periodic milestone report (loss formatting must not crash on Tensor/NaN)
        if self.milestone_step_interval and step > 0 and step % self.milestone_step_interval == 0:
            try:
                loss_f = _coerce_to_float(loss)
                loss_str = f"{loss_f:.4f}"
            except Exception:
                loss_str = str(loss)
            self.client.notify_milestone(
                message=f"Cruising normally at step {step} (loss: {loss_str})",
                step=step,
                epoch=epoch,
                metrics=metrics,
            )

    def on_epoch_end(
        self,
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Report epoch milestone."""
        self.client.notify_milestone(
            message=f"Epoch {epoch} finished successfully",
            step=step,
            epoch=epoch,
            metrics=metrics,
        )

    def on_train_end(self, final_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Report successful training completion."""
        self.client.notify_event(
            event_type="completed",
            message="Training completed successfully!",
            step=self.current_step,
            epoch=self.current_epoch,
            metrics=final_metrics,
        )
