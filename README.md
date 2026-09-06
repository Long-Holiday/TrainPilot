# TrainPilot: 深度学习训练集群的公网 MCP Server 与 HITL 闭环决策系统

> 专为大规模深度学习分布式训练（PyTorch / DeepSpeed / Slurm / K8s）设计，采用 **“公网 MCP Server（Control Plane） + 内网 GPU 客户端（Worker Pull 模式）”** 现代化解耦架构。

TrainPilot 彻底解决了内网 GPU 集群无公网 IP 与无外网凭据的痛点，通过标准 **Model Context Protocol (MCP)** 将训练异常捕获、现场冻结与飞书交互卡片原生打通，实现**人类在回路（Human-In-The-Loop, HITL）**的一键闭环决策与自愈恢复。

---

## ✨ 核心特性

- 🌐 **公网 MCP Server 原生驱动**：公网服务器作为标准 Model Context Protocol (MCP) Server，通过 SSE 远程协议向外暴露训练监控、指标上报、决策轮询与自愈确认等标准化 Tools 与 Resources。
- 🪶 **内网 GPU 节点纯客户端**：内网 GPU 节点无公网 IP、无外网凭证，作为客户端主动发起连接调用公网 MCP Server，完全不再使用脆弱的本地 Skills 脚本。
- 🛡️ **浮点安全与异常拦截**：内置浮点递归清洗机制，自动转换 `NaN` / `Inf`（支持 PyTorch Tensor、NumPy、Decimal），杜绝 JSON 序列化崩溃；`TrainingGuardian` 自动拦截 Loss 突增或发散，就地冻结训练。
- 🔄 **双向 ACK 与自愈闭环**：严格的 `RUNNING -> WAITING -> RESOLVED -> RECOVERING -> RUNNING` 状态机。客户端执行自愈后在 ACK 中携带 `solution` 方案，控制面原子流转回 `RUNNING` 并自动向飞书推送【自愈成功】卡片。
- ⏱️ **服务端单点超时决策**：告警触发后，支持飞书端人类专家即时介入；若超过预设时间（默认 30s）未决策或处于 Mock 模式，控制面统一派发自动决策，消除客户端竞态抢跑。
- ⚡ **长轮询毫秒唤醒**：客户端指令拉取支持服务端长连接挂起（默认 20s），一旦人类决策提交毫秒级唤醒下发，空轮询减少 90% 以上。
- 📱 **飞书卡片就地防呆**：Webhook 3 秒内即刻响应，点击后立即就地更新卡片为【已处理】状态并收起按钮，防止多人误触与重放。
- 💾 **纯净 SQLite WAL 持久化**：任务与事件统一由 SQLite (WAL 模式) 存储，支持重启无损恢复与自修剪淘汰，保证单机稳定性。
- 🐕 **失联看门狗 (Watchdog)**：后台守护巡检线程定时检测长时间无心跳的僵死/失联任务，主动向飞书群推送失联预警。
- 🤖 **通用 Agent 原生集成**：外部智能体（Claude Code、Cursor、OpenCode、Gemini CLI 等）只需配置公网 MCP Server 地址，即可开箱获得所有训练监控管控工具。

---

## 🏛 架构总览

```text
[ 内网 GPU 训练集群 ]                   [ 公网云服务器 ]                     [ 飞书客户端 ]
 (无公网 IP / 零飞书凭证)           (FastAPI + MCP Server)                 (人类在回路 HITL)
        │                                  │                                  │
  PyTorch / GPU 客户端                     │                                  │
  (Guardian / MCP Client)                  │                                  │
        │─── 1. call: report_alert ───────►│ (置为 WAITING 冻结现场)          │
        │      (NaN/Inf/Loss激增)          │─── 2. 推送交互告警卡片 ─────────►│
        │                                  │        (带停止训练/自愈按钮)     │ (展示异常与操作按钮)
        │                                  │◄── 3. 点击按钮 (Webhook) ─────────│ (人工点击决策)
        │                                  │    (或服务端 30s 超时自动决策)    │ (卡片就地置为“已处理”)
        │                                  │                                  │
        │◄── 4. call: poll_instruction ────┤ (长轮询毫秒唤醒获取决策)          │
  执行自愈恢复 (回退/调低lr)                 │                                  │
        │─── 5. call: ack_instruction ────►│ (流转回 RUNNING)                 │
        │      (携带 solution 方案)        │─── 6. 异步推送自愈卡片 ──────────►│ (展示已自愈与方案详情)
        │                                  │                                  │
        │─── 7. call: report_milestone ───►│─── 8. 异步推送里程碑卡片 ────────►│ (包含 Agent 智能点评)
```

---

## 📂 项目结构

```text
TrainPilot/
├── src/trainpilot/
│   ├── server/              # 公网控制面与 MCP Server
│   │   ├── mcp_server.py    # 标准 MCP Server 实现 (Tools & Resources)
│   │   ├── main.py          # FastAPI 服务端入口 (挂载 /mcp 与 Webhook)
│   │   ├── mailbox.py       # Mailbox 状态机核心调度
│   │   ├── watchdog.py      # 失联看门狗守护线程
│   │   ├── storage/         # SQLite WAL 单一持久化与自动修剪
│   │   ├── routes/          # 任务管理与 Webhook 路由
│   │   └── feishu/          # 飞书交互卡片适配器
│   ├── agent/               # 内网 GPU 客户端组件
│   │   ├── mcp_client.py    # TrainPilotMCPClient (原生 MCP SSE 客户端)
│   │   ├── client.py        # TrainPilotClient (轻量客户端)
│   │   ├── monitor.py       # TrainingGuardian (深度学习现场冻结与自愈守卫)
│   │   └── cli.py           # 命令行客户端 (trainpilot-cli)
│   └── common/              # 核心数据契约与状态机定义
├── examples/                # 训练模拟与真实 PyTorch GPU 演示范例
└── tests/                   # 完整自动化测试套件 (56 项全部通过)
```

---

## 🚀 快速上手

本项目基于高性能 Python 包管理工具 [uv](https://docs.astral.sh/uv/)。

### 1. 环境准备

```bash
git clone https://github.com/Long-Holiday/TrainPilot.git && cd TrainPilot
uv sync --extra feishu
```

### 2. 双端分离部署

| 节点角色 | 部署位置 | 配置文件模板 | 启动方式 | 核心说明 |
| :--- | :--- | :--- | :--- | :--- |
| **MCP 服务端** | 公网云服务器 (有公网 IP) | `.env.example`<br>(配置模块一) | `./start.sh`<br>(`--daemon` 后台守护) | 监听 `0.0.0.0:28780`，暴露 `/mcp` Streamable HTTP 端点与飞书回调 |
| **GPU 客户端** | 内网训练集群 (无公网 IP) | `.env.example`<br>(配置模块二) | Python 代码直接调用或 CLI | 配置公网服务端 IP/域名，作为客户端主动发起连接 |

---

## 💻 使用指南

### 1. 深度学习训练脚本接入 (内网 GPU 节点)

在 PyTorch 训练主循环中引入 `TrainingGuardian`，当检测到 `NaN` / `Inf` / 数值突增时自动冻结现场并通过公网 MCP Server 告警：

```python
from trainpilot.agent import (
    TrainPilotClient,
    TrainingGuardian,
    StopTrainingException,
)

# 初始化客户端 (指向公网 MCP Server)
client = TrainPilotClient(
    gateway_url="http://<公网服务器IP>:28780",
    task_id="llama3-8b-lora",
)

# 初始化守卫 (默认内置 stop_training 与 self_resolve 自动闭环)
guardian = TrainingGuardian(client=client)

try:
    for step, batch in enumerate(dataloader):
        loss = model(batch)
        
        # 核心监控：检测异常 -> 冻结现场 -> 飞书告警 -> 长轮询决策 -> 执行恢复 -> 自动 ACK 闭环
        guardian.check_and_handle_loss(loss.item(), step=step)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
except StopTrainingException:
    print("收到飞书人工停止决策，安全保存 Checkpoint 并退出训练。")

# 阶段上报 (支持 AI Agent 自主点评)
client.notify_milestone(
    message="Epoch 1 completed",
    step=1000,
    epoch=1,
    metrics={"loss": 0.41, "val_loss": 0.43},
    agent_note="收敛平稳，train/val 差距健康，建议保持当前超参继续观察。",
)
```

---

### 2. 外部 AI Agent 接入 (作为 MCP Client)

任何支持 MCP 协议的智能体（如 Claude Code、Cursor、OpenCode、Gemini CLI 等）可直接在配置文件中添加公网 MCP Server (采用最新的 Streamable HTTP 协议)：

```json
{
  "mcpServers": {
    "trainpilot": {
      "url": "http://<公网服务器IP>:28780/mcp"
    }
  }
}
```

配置后，Agent 将自动在上下文中拥有全部 TrainPilot 管控工具，无需在本地安装任何 skills！

---

### 3. GPU 节点命令行客户端 (CLI)

内网节点上可通过 `trainpilot-cli` 直接调用公网 MCP Server 的工具：

```bash
# 1. 上报训练阶段里程碑 (飞书绿色卡片 + Agent 智能点评)
uv run trainpilot-cli report-milestone \
  --task-id "task-01" --step 1000 --epoch 1 --metrics "loss=0.41" \
  --agent-note "val_loss 降至新低，未见过拟合，建议保持 lr 继续训练。"

# 2. 上报训练异常并就地冻结 (飞书红色告警卡片，带交互决策按钮)
uv run trainpilot-cli report-alert \
  --task-id "task-01" --step 1450 --message "Loss NaN" --metrics "loss=NaN"

# 3. 阻塞长轮询信箱等待专家决策
uv run trainpilot-cli poll-instruction --task-id "task-01" --wait

# 4. 执行恢复后向服务端发送 ACK 确认 (流转回 RUNNING 并推送自愈卡片)
uv run trainpilot-cli ack-instruction --task-id "task-01" --action self_resolve --status success --solution "回退至上一步Checkpoint并降低学习率"

# 5. 查看服务端所有 MCP 工具列表
uv run trainpilot-cli list-tools
```

---

## 📡 MCP Tools 核心契约

公网 MCP Server 对外提供以下标准工具：

| 工具名称 | 类别 | 核心入参 | 功能说明 |
|---|---|---|---|
| `report_milestone` | 监控上报 | `task_id`, `message`, `step`, `epoch`, `metrics`, `agent_note` | 上报训练里程碑，自动向飞书推送绿色卡片及 Agent 智能点评 |
| `report_alert` | 异常处理 | `task_id`, `message`, `step`, `epoch`, `metrics` | 拦截训练异常并就地冻结，推送飞书红色交互卡片，启动超时自动决策 |
| `poll_instruction` | HITL 决策 | `task_id`, `wait_timeout`, `pop` | 长轮询信箱等待人类或自动决策（支持毫秒级挂起唤醒） |
| `ack_instruction` | 状态自愈 | `task_id`, `action`, `status`, `solution`, `message` | 确认恢复方案执行，状态机流转回 RUNNING，推送飞书自愈成功卡片 |
| `send_heartbeat` | 保活巡检 | `task_id`, `step`, `epoch`, `metrics`, `status` | 上报 GPU 节点心跳与显存利用率，供 Watchdog 看门狗巡检 |
| `get_task_status` | 状态查询 | `task_id` | 查询指定任务当前状态、步数与信箱概况 |
| `list_tasks` | 集群汇总 | `limit`, `offset`, `state`, `stale_only` | 分页查询集群所有跟踪任务列表 |
| `submit_decision` | 控制台干预 | `task_id`, `action`, `payload`, `operator` | 运维或管理员直接注入人工决策 |

---

## 🧪 自动化测试

```bash
# 运行完整自动化测试套件 (56 项全部通过)
uv run pytest -v

# 单独运行 MCP Server 与 MCP Client 专项测试
uv run pytest tests/test_mcp_server.py tests/test_mcp_client.py -v
```
