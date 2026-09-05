"""PyTorch training hook and helper integration."""

import logging
from typing import Any, Callable, Dict, Optional

from trainpilot.agent.client import TrainPilotClient
from trainpilot.agent.monitor import (
    StopTrainingException,
    TrainingGuardian,
)

logger = logging.getLogger("trainpilot.agent.hooks.pytorch")


def build_default_agent_note(
    step: int,
    loss: Any = None,
    epoch: Optional[int] = None,
    lr: Optional[float] = None,
    extra_metrics: Optional[Dict[str, Any]] = None,
    prev_loss: Optional[float] = None,
) -> str:
    """基于当前实际指标生成一条规则版 Agent 点评 (无 LLM 时的兜底)。

    外部 AI 智能体 (agy / opencode / claude-code) 应优先自己结合
    训练日志、指标趋势、显存占用等实际情况自主撰写 ``agent_note`` 并上报;
    本函数仅在未提供时提供一条可读的默认文案, 避免卡片 ``🤖 Agent 智能点评``
    区块缺失。
    """
    from trainpilot.agent.client import _coerce_to_float

    try:
        loss_f: Optional[float] = _coerce_to_float(loss) if loss is not None else None
    except Exception:
        loss_f = None

    import math

    parts = []
    if loss_f is not None and not (isinstance(loss_f, float) and (math.isnan(loss_f) or math.isinf(loss_f))):
        if prev_loss is not None:
            try:
                delta = loss_f - float(prev_loss)
                trend = "下降" if delta < 0 else ("持平" if abs(delta) < 1e-9 else "上升")
                parts.append(f"loss {prev_loss:.4f}→{loss_f:.4f} ({trend} {abs(delta):.4f})")
            except Exception:
                parts.append(f"当前 loss {loss_f:.4f}")
        else:
            parts.append(f"当前 loss {loss_f:.4f}")
    elif loss_f is not None:
        parts.append("当前 loss 异常 (NaN/Inf), 已触发告警流程")
    else:
        parts.append(f"已推进至 step {step}")

    if epoch is not None:
        parts.append(f"epoch {epoch}")
    if lr is not None:
        try:
            parts.append(f"lr {float(lr):.2e}")
        except Exception:
            pass
    if extra_metrics:
        # 挑选常见关键指标透出, 避免过长
        for k in ("val_loss", "acc", "val_acc", "grad_norm", "gpu_mem"):
            if k in extra_metrics:
                parts.append(f"{k}={extra_metrics[k]}")

    base = "，".join(parts) if parts else f"step {step} 运行正常"
    # 给出趋势判断 + 下一步建议, 保持 1-2 句
    if loss_f is not None and prev_loss is not None:
        try:
            if float(loss_f) < float(prev_loss):
                return f"{base}；训练收敛趋势正常，建议保持当前超参与数据管线继续观察。"
            return f"{base}；指标未继续下降，建议关注学习率与数据批次质量，必要时检查日志。"
        except Exception:
            pass
    return f"{base}；训练巡航正常，建议继续按当前计划推进。"


class TrainPilotPyTorchHook:
    """Helper hook for integrating into PyTorch training loops or trainer callbacks."""

    def __init__(
        self,
        task_id: str,
        gateway_url: Optional[str] = None,
        milestone_step_interval: int = 100,
        heartbeat_step_interval: int = 20,
        api_token: Optional[str] = None,
        auto_agent_note: bool = True,
        **guardian_kwargs: Any,
    ):
        self.client = TrainPilotClient(gateway_url=gateway_url, task_id=task_id, api_token=api_token)
        self.guardian = TrainingGuardian(client=self.client, **guardian_kwargs)
        self.milestone_step_interval = milestone_step_interval
        self.heartbeat_step_interval = heartbeat_step_interval
        self.auto_agent_note = auto_agent_note
        self.current_step = 0
        self.current_epoch = 0
        self._prev_loss: Optional[float] = None

    def register_recovery_callback(
        self,
        action: str,
        callback: Callable[[Optional[Dict[str, Any]]], Any],
    ) -> None:
        """Register specific recovery logic for human actions like self_resolve."""
        self.guardian.register_action_handler(action, callback)

    def on_step_end(
        self,
        step: int,
        loss: Any,
        epoch: Optional[int] = None,
        lr: Optional[float] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
        agent_note: Optional[str] = None,
    ) -> None:
        """Invoked after each optimizer step.

        Performs:
        1. Loss validity check (NaN / Inf / Explosion). If invalid, triggers HITL alert.
        2. Periodic lightweight heartbeat.
        3. Periodic milestone reporting (附带 ``🤖 Agent 智能点评``)。

        ``agent_note`` 为调用方 (外部 AI 智能体或训练脚本) 针对当前实际情况
        自主撰写的点评; 为 None 且 ``auto_agent_note=True`` 时自动基于指标生成兜底文案。
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
            resolved_note = agent_note
            if resolved_note is None and self.auto_agent_note:
                try:
                    resolved_note = build_default_agent_note(
                        step=step,
                        loss=loss,
                        epoch=epoch,
                        lr=lr,
                        extra_metrics=extra_metrics,
                        prev_loss=self._prev_loss,
                    )
                except Exception:
                    resolved_note = None
            self.client.notify_milestone(
                message=f"Cruising normally at step {step} (loss: {loss_str})",
                step=step,
                epoch=epoch,
                metrics=metrics,
                agent_note=resolved_note,
            )

        # 记录本次 loss 供下次生成趋势点评
        try:
            self._prev_loss = _coerce_to_float(loss)
        except Exception:
            pass

    def on_epoch_end(
        self,
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, Any]] = None,
        agent_note: Optional[str] = None,
    ) -> None:
        """Report epoch milestone."""
        resolved_note = agent_note
        if resolved_note is None and self.auto_agent_note:
            try:
                loss_val = (metrics or {}).get("loss")
                resolved_note = build_default_agent_note(
                    step=step,
                    loss=loss_val,
                    epoch=epoch,
                    extra_metrics=metrics,
                    prev_loss=self._prev_loss,
                )
            except Exception:
                resolved_note = None
        self.client.notify_milestone(
            message=f"Epoch {epoch} finished successfully",
            step=step,
            epoch=epoch,
            metrics=metrics,
            agent_note=resolved_note,
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
