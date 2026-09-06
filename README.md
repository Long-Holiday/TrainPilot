# TrainPilot: 深度学习训练集群的公网网关与 HITL 闭环决策系统

> 专为大规模深度学习分布式训练（PyTorch / DeepSpeed / Slurm / K8s）设计，采用 **“集中式公网网关（Control Plane） + 边缘轻量 Agent（Worker Pull 模式）”** 解耦架构。

TrainPilot 彻底解决了内网 GPU 集群无公网 IP 与无外网凭据的痛点，将训练异常捕获、现场冻结与飞书交互卡片原生打通，实现**人类在回路（Human-In-The-Loop, HITL）**的一键闭环决策与热修复恢复。

---

## ✨ 核心特性

- 🛡️ **异常安全上报**：内置浮点递归清洗机制，自动转换 `NaN` / `Inf`，杜绝 JSON 序列化崩溃，确保告警 100% 可靠送达。
- 🔄 **双向 ACK 状态机**：严格的 `RUNNING -> WAITING -> RESOLVED -> RECOVERING -> RUNNING` 状态流转与原子消费，杜绝重复执行与重放。
- ⚡ **长轮询毫秒唤醒**：客户端指令拉取支持服务端挂起（默认 20s），一旦人类决策提交毫秒级唤醒下发，空轮询减少 90% 以上。
- 📱 **飞书卡片就地防呆**：Webhook 3 秒内即刻响应，点击后立即就地更新卡片为【已处理】状态并收起按钮，防止多人误触。
- 💾 **轻量持久化与自修剪**：内置 SQLite WAL 模式，控制面重启无缝恢复；自动淘汰历史事件，防内存与磁盘无界膨胀。
- 🐕 **失联看门狗 (Watchdog)**：后台守护巡检线程定时检测长时间无心跳的僵死/失联任务，主动向飞书群推送失联预警。
- 🤖 **Agent 原生友好**：内置标准项目级 Skill（[`skills/trainpilot`](skills/trainpilot/SKILL.md)），无缝接入 `agy`、`opencode`、`claude-code` 等。
- 🪶 **GPU 节点零凭证依赖**：内网 GPU 节点仅依赖原生 `requests`，无需任何飞书凭据与公网 IP。

---

## 🏛 架构总览

```text
[ GPU 内网训练集群 ]                  [ 公网云服务器 ]                     [ 飞书客户端 ]
 (无公网 IP / 零飞书凭证)          (FastAPI 控制面网关)                   (人类在回路 HITL)
        │                                  │                                  │
  PyTorch 训练 ─── 1. POST /notify ───────►│                                  │
  (Guardian 监控)   (上报异常或里程碑)       │─── 2. 发送富文本/交互卡片 ───────►│
        │                                  │        (带停止训练/自行解决按钮)     │ (展示异常与按钮)
        │                                  │◄── 3. 点击按钮 (Webhook) ─────────│ (人工点击决策)
        │                                  │    (写入信箱并就地置为“已处理”)
  GPU Agent ◄─── 4. Long-poll /instruction ┤
   (执行恢复) ─── 5. POST /ack ────────────►│ (确认恢复，重回 RUNNING)
```

---

## 🚀 快速上手

本项目基于高性能 Python 管理工具 [uv](https://docs.astral.sh/uv/)。

### 1. 环境准备
```bash
git clone https://github.com/Long-Holiday/TrainPilot.git && cd TrainPilot
uv sync --extra feishu  # 一键安装核心依赖与飞书 SDK
```

### 2. 双端分离部署

TrainPilot 将公网控制面与内网训练节点物理解耦，两端独立配置、各司其职：

| 节点角色 | 部署位置 | 配置文件模板 | 启动/安装命令 | 核心环境变量 |
| :--- | :--- | :--- | :--- | :--- |
| **Web 控制面** | 公网云服务器 | `.env.web.example` | `./start.sh`<br>(`--daemon` 后台守护) | `TRAINPILOT_BIND_HOST=0.0.0.0`<br>`TRAINPILOT_PORT=28780`<br>飞书凭据（未配置自动降级为 Mock 模式） |
| **GPU 节点** | 内网训练集群 | `.env.gpu.example` | `./setup_skills.sh` | `TRAINPILOT_HOST=<Web公网IP/域名>`<br>`TRAINPILOT_PORT=28780`<br>`TRAINPILOT_TASK_ID` |

> ⚠️ **注意**：GPU 节点为客户端，`TRAINPILOT_HOST` 必须配置为 Web 端真实公网 IP 或域名，严禁配置为 `0.0.0.0`。

---

## 💻 使用示例

### 1. Python 训练脚本接入 (零外部凭据)

在 PyTorch 训练主循环中注入 `TrainingGuardian`，当检测到 `NaN` / `Inf` / 数值突增时自动冻结现场并等待飞书端处理：

```python
from trainpilot.agent import TrainPilotClient, TrainingGuardian

client = TrainPilotClient(
    gateway_url="http://<Web公网IP>:28780",
    task_id="llama3-8b-lora",
)
guardian = TrainingGuardian(client=client)

# 注册飞书卡片决策处理回调 (stop_training / self_resolve)
guardian.register_action_handler("self_resolve", lambda payload: print("自行解决：继续训练..."))

for step, batch in enumerate(dataloader):
    loss = model(batch)
    
    # 自动监测异常：冻结现场 -> 飞书告警 -> 轮询决策 -> 执行回调 -> ACK 恢复
    guardian.check_and_handle_loss(loss.item(), step=step)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 上报里程碑（支持外部 AI Agent 自主点评）
client.notify_milestone(
    message="Epoch 1 finished",
    step=1000,
    epoch=1,
    metrics={"loss": 0.41, "val_loss": 0.43},
    agent_note="收敛平稳，train/val 差距健康，建议保持当前超参继续观察。",
)
```

### 2. AI Agent 命令行工具 (CLI)

GPU 节点上的智能体可通过标准终端调用工具脚本（完整规范详见 [`skills/trainpilot/SKILL.md`](skills/trainpilot/SKILL.md)）：

```bash
# 1. 上报训练阶段里程碑 (飞书绿色卡片 + Agent 智能点评)
python3 skills/trainpilot/scripts/trainpilot_tool.py report-milestone \
  --task-id "task-01" --step 1000 --epoch 1 --metrics "loss=0.41" \
  --agent-note "val_loss 为本轮新低，未见过拟合，建议保持 lr。"

# 2. 上报训练异常并挂起现场 (飞书红色告警卡片，等待专家决策)
python3 skills/trainpilot/scripts/trainpilot_tool.py report-alert \
  --task-id "task-01" --step 1450 --message "Loss NaN" --metrics "loss=NaN"

# 3. 阻塞长轮询信箱等待决策，执行后向网关发送 ACK 确认并恢复训练
python3 skills/trainpilot/scripts/trainpilot_tool.py poll-instruction --task-id "task-01" --wait
python3 skills/trainpilot/scripts/trainpilot_tool.py ack-instruction --task-id "task-01" --action self_resolve --status success
```

---

## 📡 核心 API 契约

| 接口路径 | 方法 | 调用方 | 功能说明 |
|---|---|---|---|
| `/api/tasks/notify` | `POST` | GPU Agent | 上报事件（`alert` 告警、`milestone` 里程碑、`completed` 完成） |
| `/api/tasks/{task_id}/instruction` | `GET` | GPU Agent | 长轮询信箱获取决策（`pop=true` 消费并进入 `RECOVERING`） |
| `/api/tasks/{task_id}/ack` | `POST` | GPU Agent | 确认指令执行结果，任务状态恢复为 `RUNNING` |
| `/api/tasks/{task_id}/heartbeat` | `POST` | GPU Agent | 运行时心跳保活与资源利用率上报 |
| `/api/tasks/{task_id}/status` | `GET` | GPU / 运维 | 查询任务当前状态详情与信箱概况 |
| `/api/tasks/{task_id}/decision` | `POST` | 控制台/测试 | 直接注入人工决策（支持本地开发或 Web 控制台） |
| `/webhook/feishu` | `POST` | 飞书客户端 | 飞书 URL 握手校验与卡片交互点击回调（`card.action.trigger`） |

---

## 🧪 自动化测试

```bash
# 运行全量单元与集成测试套件
uv run pytest -v

# 运行完整的“遇异常 -> 冻结 -> 飞书决策 -> 恢复”端到端闭环模拟
uv run pytest tests/test_e2e_simulation.py -v -s
```
