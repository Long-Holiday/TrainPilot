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
- **RESOLVED -> RECOVERING**: Fetches human decisions made via Feishu interactive cards or Webhook/API (`stop_training` / `self_resolve`, 30s timeout auto `self_resolve`), and ACKs back to `RUNNING`.

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
| **Report Milestone** | `python3 skills/trainpilot/scripts/trainpilot_tool.py report-milestone --task-id <ID> --step <N> --message "Epoch 1 done" --metrics "loss=0.35,val_loss=0.42" --agent-note "Loss 连续下降，收敛平稳，建议保持 lr"` |
| **Report Anomaly & Alert** | `python3 skills/trainpilot/scripts/trainpilot_tool.py report-alert --task-id <ID> --step <N> --message "Loss NaN at step 1450" --metrics "loss=NaN"` |
| **Poll Human Decision** | `python3 skills/trainpilot/scripts/trainpilot_tool.py poll-instruction --task-id <ID> --wait --wait-timeout 300` |
| **Acknowledge Recovery** | `python3 skills/trainpilot/scripts/trainpilot_tool.py ack-instruction --task-id <ID> --action self_resolve --status success` |
| **Send Heartbeat** | `python3 skills/trainpilot/scripts/trainpilot_tool.py send-heartbeat --task-id <ID> --step <N> --metrics "gpu_mem=85%"` |
| **Query Task Status** | `python3 skills/trainpilot/scripts/trainpilot_tool.py get-status --task-id <ID>` |
| **Mock Decision (Dev/Test)** | `python3 skills/trainpilot/scripts/trainpilot_tool.py mock-decision --task-id <ID> --action self_resolve` |

> [!IMPORTANT]
> **上报里程碑务必附带 `--agent-note` (AI Agent 自主点评)**：由外部 AI 智能体
> （agy / opencode / claude-code）**针对当前实际情况自主撰写 1-3 句点评**并随里程碑
> 一并上报，最终以 ``🤖 Agent 智能点评`` 区块呈现在飞书里程碑卡片中（紧随
> ``📝 阶段描述`` 之后）。点评应基于真实的训练日志与指标，涵盖：当前收敛/震荡趋势、
> 关键指标解读、存在的风险点（如 val 过拟合、梯度范数偏大）、以及给工程师的下一步建议。
> 不要逐字复述 `--message`，不要编造指标。示例：
> `--agent-note "val_loss 降至 0.42 为当前新低，train/val 差距约 0.08 未见明显过拟合；lr 按 schedule 衰减中，建议本 epoch 后做一次 eval 存档。"`

> [!NOTE]
> 网关地址解析 (GPU 侧): `--gateway` 参数 > 环境变量 `TRAINPILOT_GATEWAY_URL` (完整 URL)
> > `http://<TRAINPILOT_HOST>:<TRAINPILOT_PORT>` 拼接 > 默认 `http://127.0.0.1:28780`。
> 因此 GPU 机器只需设置 `TRAINPILOT_HOST=<Web 公网 IP/域名>` (如 `35.202.16.245`) 即可,
> 无需拼完整 URL。`TRAINPILOT_TASK_ID` 可用于省略 `--task-id`。
> 注意: `TRAINPILOT_HOST` 在 Web 侧表示绑定地址 (应为 `0.0.0.0`), 在 GPU 侧表示网关地址,
> 两台机器必须分别配置, 不可直接共用同一 `.env`。

---

## Detailed Tool Usage

### 1. Milestone Notification
When training crosses an epoch boundary or reaches a new evaluation metric high:
```bash
python3 skills/trainpilot/scripts/trainpilot_tool.py report-milestone \
  --gateway "http://control-plane.example.com:28780" \
  --task-id "qwen2-7b-sft-0905" \
  --step 1000 \
  --epoch 1 \
  --message "Epoch 1 finished, validation loss reached new minimum" \
  --metrics '{"loss": 0.421, "val_loss": 0.450, "learning_rate": 0.0001}' \
  --agent-note "val_loss 0.450 为当前新低，train/val 差距稳定未见过拟合；loss 曲线平滑，建议保持当前超参并在下一里程碑前做一次完整 eval + checkpoint 存档。"
```
> **`--agent-note` 由你自主撰写**：请作为执行上报的 AI 智能体，根据此刻掌握的
> 训练日志、loss 曲线、显存、数据批次等实际情况，写出 1-3 句**独立于 `--message`**
> 的阶段点评（趋势判断 / 指标解读 / 风险提示 / 建议动作），不要逐字复述 message。
> 飞书卡片将把 message 显示在 `📝 阶段描述`，把 agent-note 显示在 `🤖 Agent 智能点评`。

### 2. Anomaly Alert & Freezing
When NaN or sudden loss spike is intercepted:
```bash
python3 skills/trainpilot/scripts/trainpilot_tool.py report-alert \
  --gateway "http://control-plane.example.com:28780" \
  --task-id "qwen2-7b-sft-0905" \
  --step 1450 \
  --message "Loss NaN detected at step 1450 (previous loss was 0.35)" \
  --metrics '{"loss": "NaN", "step": 1450}'
```

### 3. Waiting & Polling for Human Decision
After sending the alert, wait for an engineer to click a button on the Feishu interactive card:
```bash
python3 skills/trainpilot/scripts/trainpilot_tool.py poll-instruction \
  --gateway "http://control-plane.example.com:28780" \
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
  "action": "self_resolve",
  "decision_by": "ou_3429810a",
  "decided_at": "2026-09-05T09:45:00Z"
}
```

### 4. Acknowledging Recovery
Once the self-resolve continuation is confirmed:
```bash
python3 skills/trainpilot/scripts/trainpilot_tool.py ack-instruction \
  --gateway "http://control-plane.example.com:28780" \
  --task-id "qwen2-7b-sft-0905" \
  --instruction-id "inst_b12fa09c" \
  --action "self_resolve" \
  --status "success" \
  --message "Self-resolved and continued training"
```

---

## Python In-Process API Usage

For PyTorch training code, use the Python SDK directly without extra dependencies beyond `requests`:

```python
from trainpilot.agent import TrainPilotClient, TrainingGuardian

# 1. Initialize lightweight client
client = TrainPilotClient(
    gateway_url="http://control-plane.example.com:28780",
    task_id="qwen2-7b-sft-0905",
)

# 2. Setup guardian (defaults: 30s timeout auto self_resolve)
guardian = TrainingGuardian(client=client)

def on_self_resolve(payload):
    print("Self-resolved: continuing training without modification...")

guardian.register_action_handler("self_resolve", on_self_resolve)

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

# 4. Periodically report milestones with autonomous AI Agent note:
client.notify_milestone(
    message="Epoch 1 completed, val_loss reached 0.450",
    step=1000,
    epoch=1,
    metrics={"loss": 0.421, "val_loss": 0.450},
    agent_note="val_loss 0.450 为当前新低，收敛平稳未见明显过拟合；建议保持学习率继续推进下个 epoch。",
)
```

> [!IMPORTANT]
> **TrainingGuardian 与 AI Agent 自主点评标准模式**：
> 1. **统一异常守卫**：在训练主循环中使用 `TrainingGuardian` 守卫异常（`NaN` / `Inf` / 数值突增），自动完成“现场冻结 ➔ 飞书告警 ➔ 决策轮询 ➔ 执行回调 ➔ ACK 确认”完整闭环。
> 2. **AI Agent 自主撰写 `agent_note`**：在阶段里程碑（如 Epoch 完成或周期评测）时，由外部 AI 智能体（如 agy / opencode / claude-code）结合实时训练日志与指标走势，**自主撰写 1-3 句专业阶段点评**并随 `client.notify_milestone(..., agent_note="...")` 上报。飞书卡片会在 `🤖 Agent 智能点评` 区块中独立展示，为人类工程师提供清晰的收敛判断与决策建议。

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
4. **Milestone 上报尽量附 agent-note**: 它是飞书里程碑卡片中 `🤖 Agent 智能点评` 的内容来源, 由上报的 AI 智能体根据真实情况自主撰写; 请勿编造指标, 也勿逐字复刻 `--message`。
5. **向后兼容**: `agent_note` 为可选字段。不传时卡片只显示 `📝 阶段描述`, 行为与旧版本完全一致。
