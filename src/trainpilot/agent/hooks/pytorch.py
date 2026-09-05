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
        gateway_url: str = "http://localhost:8000",
        milestone_step_interval: int = 100,
        heartbeat_step_interval: int = 20,
    ):
        self.client = TrainPilotClient(gateway_url=gateway_url, task_id=task_id)
        self.guardian = TrainingGuardian(client=self.client)
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
        loss: float,
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
        self.current_step = step
        if epoch is not None:
            self.current_epoch = epoch

        metrics = {"loss": loss}
        if lr is not None:
            metrics["lr"] = lr
        if extra_metrics:
            metrics.update(extra_metrics)

        # 1. Anomaly check (pauses & polls if anomaly detected)
        self.guardian.check_and_handle_loss(
            loss_val=loss,
            step=step,
            epoch=epoch,
            extra_metrics=metrics,
        )

        # 2. Periodic heartbeat
        if self.heartbeat_step_interval and step % self.heartbeat_step_interval == 0:
            self.client.send_heartbeat(step=step, epoch=epoch, metrics=metrics)

        # 3. Periodic milestone report
        if self.milestone_step_interval and step > 0 and step % self.milestone_step_interval == 0:
            self.client.notify_milestone(
                message=f"Cruising normally at step {step} (loss: {loss:.4f})",
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
