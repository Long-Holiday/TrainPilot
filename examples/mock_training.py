"""Simulated training run driven the way a GPU-side AI agent drives TrainPilot.

Demonstrates the full HITL closed loop over MCP:

    report(milestone) -> report(alert) -> poll_instruction -> ack_instruction -> report(completed)

The "GPU work" here is a fake loss curve; substitute your real training loop for
the `_step()` call. The point is the supervision protocol, not the model.

Run against a live server:

    ./start.sh --daemon
    uv run python examples/mock_training.py [gateway_url] [task_id]
"""

import os
import sys
import threading
import time
from typing import Optional

import httpx

# Ensure src is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from trainpilot.agent.mcp_client import TrainPilotMCPClient
from trainpilot.common.gateway import resolve_gateway_url

TOTAL_STEPS = 10
CHECKPOINT_STEP = 4
ANOMALY_STEP = 6


def _simulate_operator_decision(gateway_url: str, task_id: str, delay: float = 3.0) -> None:
    """Stand-in for an engineer clicking a button on the Feishu alert card.

    In production nobody runs this: the human taps the card and Feishu calls back
    into the same `/api/tasks/{id}/decision` endpoint.
    """

    def _click() -> None:
        time.sleep(delay)
        try:
            resp = httpx.post(
                f"{gateway_url}/api/tasks/{task_id}/decision",
                json={
                    "task_id": task_id,
                    "action": "self_resolve",
                    "operator": "Senior ML Engineer (Feishu)",
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            print(f"[operator] clicked 'self_resolve' on the Feishu card for {task_id}")
        except Exception as exc:
            print(f"[operator] failed to submit decision: {exc}")

    threading.Thread(target=_click, daemon=True).start()


def _wait_for_decision(client: TrainPilotMCPClient, task_id: str) -> dict:
    """Long-poll until a human decision arrives, then hand back the instruction."""
    while True:
        res = client.poll_instruction(task_id=task_id, wait_timeout=5.0)
        if not res.get("success", True):
            raise RuntimeError(f"poll_instruction failed: {res.get('error')}")
        if res.get("has_instruction"):
            return res
        print(f"[agent] no decision yet (task state: {res.get('status')}), still waiting...")


def run_mock_training(gateway_url: Optional[str] = None, task_id: str = "demo-llm-pretrain") -> None:
    if gateway_url is None:
        gateway_url = resolve_gateway_url()
    print("\n==========================================")
    print(f"Starting simulated training for task: {task_id}")
    print(f"Gateway: {gateway_url}")
    print("==========================================\n")

    client = TrainPilotMCPClient(server_url=gateway_url, task_id=task_id)
    lr = 1e-4
    loss = 2.5

    for step in range(1, TOTAL_STEPS + 1):
        # ---- 1. Supervision: watch the run, report what you see ----
        if step == CHECKPOINT_STEP:
            res = client.report(
                message=f"Checkpoint saved at step {step}",
                event_type="milestone",
                task_id=task_id,
                step=step,
                epoch=1,
                metrics={"loss": 1.18, "lr": lr},
            )
            print(f"[agent] milestone reported -> state={res['state']}")

        if step == ANOMALY_STEP:
            # ---- 2. Anomaly: pause the job, then alert BEFORE repairing ----
            print(f"\n[step {step}] training loss went NaN - pausing the GPU job")
            _simulate_operator_decision(gateway_url, task_id)

            res = client.report(
                message="Loss became NaN at step 6; paused the job pending a decision",
                event_type="alert",
                task_id=task_id,
                step=step,
                epoch=1,
                metrics={"loss": "NaN", "lr": lr},
                agent_note="Loss diverged after the LR warmup finished; a rollback to the "
                           "step-4 checkpoint with a lower LR looks sufficient.",
            )
            print(f"[agent] alert reported -> state={res['state']}")

            # ---- 3. Poll until the human decides, then execute and ACK ----
            instruction = _wait_for_decision(client, task_id)
            print(f"[agent] decision received: action={instruction['action']} "
                  f"by {instruction['decision_by']}")

            if instruction["action"] == "stop_training":
                client.ack_instruction(
                    task_id=task_id,
                    instruction_id=instruction["instruction_id"],
                    action="stop_training",
                    status="success",
                    solution="Stopped the run as instructed.",
                )
                client.report(
                    message="Run stopped by operator decision at step 6",
                    event_type="failed",
                    task_id=task_id,
                    step=step,
                )
                print("\nRun halted by operator. Exiting.\n")
                return

            # self_resolve: actually recover, verify, then ACK success
            lr = 5e-5
            loss = 1.4
            client.ack_instruction(
                task_id=task_id,
                instruction_id=instruction["instruction_id"],
                action=instruction["action"],
                status="success",
                solution="Rolled back to the step-4 checkpoint and halved the LR to 5e-5; "
                         "loss is finite again after one probe step.",
            )
            print("[agent] recovery acknowledged; resuming training\n")
            continue

        # ---- The actual (simulated) training step ----
        loss = max(0.2, loss - 0.1 * step) if loss > 0.2 else loss
        print(f"[step {step}/{TOTAL_STEPS}] loss={loss:.4f} lr={lr:.2e}")
        time.sleep(0.3)

    # ---- 4. Terminal state: only when the run is definitively over ----
    client.report(
        message="Simulated training completed successfully",
        event_type="completed",
        task_id=task_id,
        step=TOTAL_STEPS,
        epoch=1,
        metrics={"final_loss": round(loss, 4)},
    )
    print("\nTraining run completed.\n")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else resolve_gateway_url()
    task = sys.argv[2] if len(sys.argv) > 2 else "demo-llm-pretrain"
    run_mock_training(url, task)
