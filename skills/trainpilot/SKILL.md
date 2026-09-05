---
name: trainpilot
description: Use when running, supervising, or recovering deep learning training tasks on GPU servers, needing to report milestones, alert on anomalies (Loss NaN, CUDA OOM, loss explosion), poll human-in-the-loop decisions, or execute recovery actions via the TrainPilot gateway.
---

# TrainPilot Agent Skill

## Overview

**TrainPilot** bridges isolated GPU cluster training jobs (running PyTorch/Deepspeed/Slurm) with a lightweight public-facing Control Plane gateway and Feishu HITL (Human-In-The-Loop) interactive cards.

The GPU Agent maintains a goal-driven state machine:
- **RUNNING**: Silently cruises, periodically reporting milestone metrics.
- **WAITING**: On anomalies (Loss NaN, OOM, loss explosion), freezes training, reports alerts to the gateway, and enters polling mode.
- **RESOLVED -> RECOVERING**: Fetches human decisions made via Feishu interactive cards or Webhook/API, executes hot-recovery (e.g. `reduce_lr_rollback`, `skip_batch`), and ACKs back to resume `RUNNING`.

## When to Use

Use this skill when:
- Running or monitoring distributed GPU training jobs on private clusters (no public IP required on GPU nodes).
- Need to report periodic training milestones (`step`, `epoch`, `loss`, `metrics`) to team communication channels.
- Need to intercept critical training crashes (`Loss NaN`, `Loss Inf`, `CUDA OOM`, gradient explosion), freeze process state, and request human engineer intervention via interactive buttons.
- Need to poll human decisions from the central gateway and trigger local recovery procedures (reloading checkpoints, lowering learning rate, skipping corrupt batches).

Do NOT use this skill for:
- Routine inference tasks or static batch evaluations where no human intervention is desired.

---

## Quick Reference (CLI for AI Agents)

The skill provides an executable script `scripts/trainpilot_tool.py` runnable directly by AI Agents (`agy`, `opencode`, `claude-code`) via bash or subagents:

| Goal | CLI Command |
|---|---|
| **Report Milestone** | `python3 skills/trainpilot/scripts/trainpilot_tool.py report-milestone --task-id <ID> --step <N> --message "Epoch 1 done" --metrics "loss=0.35,val_loss=0.42"` |
| **Report Anomaly & Alert** | `python3 skills/trainpilot/scripts/trainpilot_tool.py report-alert --task-id <ID> --step <N> --message "Loss NaN at step 1450" --metrics "loss=NaN"` |
| **Poll Human Decision** | `python3 skills/trainpilot/scripts/trainpilot_tool.py poll-instruction --task-id <ID> --wait --wait-timeout 300` |
| **Acknowledge Recovery** | `python3 skills/trainpilot/scripts/trainpilot_tool.py ack-instruction --task-id <ID> --action reduce_lr_rollback --status success` |
| **Send Heartbeat** | `python3 skills/trainpilot/scripts/trainpilot_tool.py send-heartbeat --task-id <ID> --step <N> --metrics "gpu_mem=85%"` |
| **Query Task Status** | `python3 skills/trainpilot/scripts/trainpilot_tool.py get-status --task-id <ID>` |
| **Mock Decision (Dev/Test)** | `python3 skills/trainpilot/scripts/trainpilot_tool.py mock-decision --task-id <ID> --action reduce_lr_rollback` |

> [!NOTE]
> Environment variables `TRAINPILOT_GATEWAY_URL` (default: `http://localhost:8000`) and `TRAINPILOT_TASK_ID` can be set to omit `--gateway` and `--task-id` flags in scripts.

---

## Detailed Tool Usage

### 1. Milestone Notification
When training crosses an epoch boundary or reaches a new evaluation metric high:
```bash
python3 skills/trainpilot/scripts/trainpilot_tool.py report-milestone \
  --gateway "http://control-plane.example.com:8000" \
  --task-id "qwen2-7b-sft-0905" \
  --step 1000 \
  --epoch 1 \
  --message "Epoch 1 finished, validation loss reached new minimum" \
  --metrics '{"loss": 0.421, "val_loss": 0.450, "learning_rate": 0.0001}'
```

### 2. Anomaly Alert & Freezing
When NaN or sudden loss spike is intercepted:
```bash
python3 skills/trainpilot/scripts/trainpilot_tool.py report-alert \
  --gateway "http://control-plane.example.com:8000" \
  --task-id "qwen2-7b-sft-0905" \
  --step 1450 \
  --message "Loss NaN detected at step 1450 (previous loss was 0.35)" \
  --metrics '{"loss": "NaN", "step": 1450}'
```

### 3. Waiting & Polling for Human Decision
After sending the alert, wait for an engineer to click a button on the Feishu interactive card:
```bash
python3 skills/trainpilot/scripts/trainpilot_tool.py poll-instruction \
  --gateway "http://control-plane.example.com:8000" \
  --task-id "qwen2-7b-sft-0905" \
  --wait \
  --interval 2 \
  --wait-timeout 600
```
Output JSON will indicate the selected strategy:
```json
{
  "ready": true,
  "status": "RECOVERING",
  "instruction_id": "inst_b12fa09c",
  "action": "reduce_lr_rollback",
  "decision_by": "ou_3429810a",
  "decided_at": "2026-09-05T09:45:00Z"
}
```

### 4. Acknowledging Recovery
Once the recovery procedure finishes (e.g. reload checkpoint, reduce learning rate by 50%):
```bash
python3 skills/trainpilot/scripts/trainpilot_tool.py ack-instruction \
  --gateway "http://control-plane.example.com:8000" \
  --task-id "qwen2-7b-sft-0905" \
  --instruction-id "inst_b12fa09c" \
  --action "reduce_lr_rollback" \
  --status "success" \
  --message "Reloaded checkpoint_step_1400.pt and reduced LR to 5e-5"
```

---

## Python In-Process API Usage

For PyTorch training code, use the Python SDK directly without extra dependencies beyond `requests`:

```python
from trainpilot.agent import TrainPilotClient, TrainingGuardian

# 1. Initialize lightweight client
client = TrainPilotClient(
    gateway_url="http://control-plane.example.com:8000",
    task_id="qwen2-7b-sft-0905",
)

# 2. Setup guardian and register recovery handlers
guardian = TrainingGuardian(client=client)

def on_reduce_lr_rollback(payload):
    print("Reloading previous checkpoint and scaling down LR...")
    # your_model.load_state_dict(...)
    # optimizer.param_groups[0]['lr'] *= 0.5

guardian.register_action_handler("reduce_lr_rollback", on_reduce_lr_rollback)

# 3. In your training step loop:
for step, (inputs, targets) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Check loss for NaN / Inf / Explosion.
    # If abnormal: freezes, sends Feishu alert, polls human decision, executes handler, and ACKs!
    guardian.check_and_handle_loss(loss.item(), step=step)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Or plug in `TrainPilotPyTorchHook`:
```python
from trainpilot.agent.hooks.pytorch import TrainPilotPyTorchHook

hook = TrainPilotPyTorchHook(
    task_id="qwen2-7b-sft-0905",
    gateway_url="http://control-plane.example.com:8000",
    milestone_step_interval=100,
)
hook.register_recovery_callback("reduce_lr_rollback", my_rollback_fn)

# In loop:
hook.on_step_end(step=step, loss=loss_val, lr=current_lr)
```

---

## State Machine

```text
[ RUNNING ] ───(milestone)───► [ Feishu Milestone Card (Green, Read-only) ]
     │
     ▼ (Loss NaN / OOM)
[ WAITING ] ───(notify_alert)─► [ Feishu Alert Card (Red, with Action Buttons) ]
     │
     ▼ (Engineer clicks button on Feishu)
[ RESOLVED ] ──(webhook)─────► [ Card updated in-place to Green (Anti-duplicate) ]
     │
     ▼ (Agent polls instruction)
[ RECOVERING ]
     │ (Execute rollback/recovery)
     ▼ (ack_instruction: success)
[ RUNNING ]
```

## Common Mistakes & Best Practices

1. **Do NOT poll without pop consideration**: Default polling pops the pending instruction and transitions status to `RECOVERING`. If you only want to inspect status, use `--no-pop` or `get-status`.
2. **Always ACK after recovery**: If an agent fails to ACK the instruction, the gateway state will remain in `RECOVERING` rather than transitioning back to `RUNNING`.
3. **No Feishu credentials needed on GPU machines**: The GPU agent only connects to the control plane via standard HTTP requests.
