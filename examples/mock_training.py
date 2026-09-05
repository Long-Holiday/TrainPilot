"""Simulation script of a PyTorch training task encountering Loss NaN and recovering via TrainPilot HITL."""

import math
import os
import sys
import threading
import time
from typing import Optional
import requests

# Ensure src is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from trainpilot.agent import TrainPilotClient, TrainingGuardian


def run_mock_training(gateway_url: Optional[str] = None, task_id: str = "demo-llm-pretrain"):
    if gateway_url is None:
        from trainpilot.common.gateway import resolve_gateway_url

        gateway_url = resolve_gateway_url()
    print(f"\n==========================================")
    print(f"🚀 Starting Simulated Training for Task: {task_id}")
    print(f"🔗 Gateway: {gateway_url}")
    print(f"==========================================\n")

    client = TrainPilotClient(gateway_url=gateway_url, task_id=task_id)
    guardian = TrainingGuardian(client=client, poll_interval=1.0, poll_timeout=30.0,
                                timeout_fallback_action="self_resolve")

    # Simulated training state
    training_state = {
        "lr": 1e-4,
        "step": 0,
        "loss": 2.5,
    }

    # 自行解决：无需复杂恢复逻辑，默认 handler 直接继续训练，Agent 将自动发送卡片概括解决方法并告知任务恢复正常。

    # Simulate steps
    total_steps = 10
    simulated_human_triggered = False

    while training_state["step"] < total_steps:
        training_state["step"] += 1
        current_step = training_state["step"]

        # Simulate checkpoint at step 4
        if current_step == 4:
            client.notify_milestone(
                message="Checkpoint saved at step 4",
                step=current_step,
                epoch=1,
                metrics={"loss": 1.18, "lr": training_state["lr"]},
            )

        # Simulate Loss NaN anomaly at step 6
        if current_step == 6 and not simulated_human_triggered:
            simulated_human_triggered = True
            current_loss = float("nan")
            print(f"[Step {current_step}] ⚠️ Triggering anomaly: Loss is NaN!")

            # Start a background timer to simulate human engineer clicking "self_resolve" on Feishu after 3 seconds
            def simulate_human_click():
                time.sleep(3.0)
                print("\n[Human Simulator] 👨‍💻 Engineer clicked 'self_resolve' on Feishu interactive card!")
                try:
                    res = requests.post(
                        f"{gateway_url}/api/tasks/{task_id}/decision",
                        json={
                            "task_id": task_id,
                            "action": "self_resolve",
                            "operator": "Senior ML Engineer (Feishu)",
                        },
                        timeout=5,
                    )
                    res.raise_for_status()
                except Exception as err:
                    print(f"[Human Simulator Error] {err}")

            t = threading.Thread(target=simulate_human_click, daemon=True)
            t.start()
        else:
            # Normal descent
            current_loss = max(0.2, training_state["loss"] - 0.1 * current_step)

        # Check loss with guardian (will pause and block if NaN until human decides!)
        guardian.check_and_handle_loss(
            loss_val=current_loss,
            step=current_step,
            epoch=1,
            extra_metrics={"lr": training_state["lr"]},
        )

        print(f"[Step {training_state['step']}] Normal iteration completed. Loss: {current_loss:.4f}, LR: {training_state['lr']:.2e}")
        time.sleep(0.3)

    # Training completed
    client.notify_event(
        event_type="completed",
        message="Simulated training completed successfully!",
        step=training_state["step"],
        epoch=1,
        metrics={"final_loss": 0.25},
    )
    print("\n🎉 Training run successfully concluded!\n")


if __name__ == "__main__":
    from trainpilot.common.gateway import resolve_gateway_url

    url = sys.argv[1] if len(sys.argv) > 1 else resolve_gateway_url()
    task = sys.argv[2] if len(sys.argv) > 2 else "demo-llm-pretrain"
    run_mock_training(url, task)
