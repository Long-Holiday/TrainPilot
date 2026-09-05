# TrainPilot: 深度学习训练集群的公网网关与 HITL 闭环决策系统

TrainPilot 专为大规模深度学习分布式训练（PyTorch / DeepSpeed / Slurm / K8s）设计，采用 **“集中式公网网关（Control Plane） + 边缘轻量 Agent（Worker Pull 模式）”** 的解耦架构。

它彻底解决了内网 GPU 算力集群物理隔离、缺乏公网 IP 与无外网凭据约束的痛点，同时将训练异常捕获、现场冻结与飞书交互式卡片原生打通，实现**人类在回路（Human-In-The-Loop, HITL）**的一键闭环决策与热修复恢复。

---

## 🌟 核心设计与实际落地优化

相较于初步草案，本项目在实际生产落地中进行了如下关键工程修正与深度优化：

1. **浮点异常安全序列化（Sanitization）**：
   - 在真实 PyTorch 训练中，Loss 爆炸常表现为 `float('nan')` 或 `float('inf')`。由于标准 JSON（RFC 7159/8259）不支持原生 NaN，直接序列化会导致 Python `requests` 崩溃。
   - TrainPilot 客户端与 CLI 内置递归清洗机制，自动将数值安全转换为 `"NaN"` / `"Infinity"`，确保异常告警百分之百可靠送达公网。
2. **状态机闭环与双向 ACK 确认机制**：
   - 确立了严格的状态流转：`RUNNING -> WAITING -> RESOLVED -> RECOVERING -> RUNNING`。
   - 指令拉取支持原子消费（Pop）与完成回执（ACK），避免指令重放、漏跑或网络重试导致的重复回滚。
3. **飞书交互卡片“就地防呆”更新**：
   - 飞书 Webhook 接收到操作点击后，立即更新原卡片为绿色【已处理】状态并移除操作按钮，杜绝多人协作场景下的误触与重复执行。
   - Webhook 3 秒内即刻响应，彻底解耦耗时数分钟的模型 Checkpoint 回滚与显存清理。
4. **轻量与优雅降级（Mock 模式）**：
   - 网关控制面支持未配置飞书凭证时的本地 Mock 运行模式，支持通过 API / Webhook 直接注入决策，保障无外网环境和自动化测试顺畅执行。
   - GPU 节点 Agent 保持**零外网凭据、零飞书依赖、仅用原生 `requests`**。
5. **项目级别 AI Agent Skills（适配 `agy`、`opencode`）**：
   - 提炼出符合 Agent 技能规范的项目级 Skill（`skills/trainpilot`），供智能体自主上报指标、捕获异常、轮询信箱和执行自动化运维。

---

## 🏛 系统架构总览

```text
  [ GPU 训练集群 / 内网节点 ]                   [ 轻量公网云服务器 ]                     [ 飞书移动端 / PC 客户端 ]
   (无需公网 IP / 零飞书凭证)                    (FastAPI + lark_oapi)                       (人类专家在回路 HITL)
             │                                            │                                            │
             │                                            │                                            │
   ┌─────────┴─────────┐                                  │                                            │
   │  PyTorch 训练任务  │                                  │                                            │
   │        │          │                                  │                                            │
   │ (Hook 捕获异常/指标) │                                  │                                            │
   │        ▼          │                                  │                                            │
   │   GPU Agent 进程  │─── 1. POST /api/tasks/notify ───►│                                            │
   │ (纯原生 requests)  │    (上报异常或阶段里程碑)        │─── 2. lark_oapi 发送富文本卡片 ──────────►│
   │                   │                                  │       (带 [降LR] [跳过] [终止] 按钮)        │ (卡片展示异常与按钮)
   │                   │                                  │                                            │
   │                   │                                  │◄── 3. 点击按钮: card.action.trigger ───────│ (工程师点击方案)
   │                   │                                  │    (写入任务信箱，并就地将卡片置为“已处理”) │
   │                   │                                  │                                            │
   │                   │─── 4. GET .../instruction ──────►│                                            │
   │                   │    (轮询信箱，获取决策)          │                                            │
   │                   │◄── 5. 返回 {"action": "..."} ────│                                            │
   │        │          │                                  │                                            │
   │ (执行恢复策略与ACK)│─── 6. POST .../ack ─────────────►│ (确认恢复，状态重回 RUNNING)                 │
   └───────────────────┘                                  │                                            │
```

### 状态机流转

```text
 ┌──────────────┐      里程碑达成 (Epoch完成/指标新高/心跳)
 │   RUNNING    ├────────────────────────────────► [单向推送绿色概览卡片，静默记录]
 └──────┬───────┘
        │ 检测到偏离 Goal (Loss NaN, Loss Spike, OOM, 进程僵死)
        ▼
 ┌──────────────┐      POST /api/tasks/notify (alert)
 │   WAITING    ├────────────────────────────────► [推送红色告警卡片，冻结现场并提供交互按钮]
 └──────┬───────┘
        │ 工程师在飞书客户端点击决策按钮 (Webhook 回调)
        ▼
 ┌──────────────┐
 │   RESOLVED   ├────────────────────────────────► [就地更新卡片为已处理并收起按钮，决策入信箱]
 └──────┬───────┘
        │ GPU Agent 轮询读取到 action (pop=true)
        ▼
 ┌──────────────┐
 │  RECOVERING  ├───► (回滚上一 Checkpoint、缩减 LR、跳过 Batch)
 └──────┬───────┘
        │ Agent 执行完成，POST /api/tasks/{task_id}/ack
        ▼
 ┌──────────────┐
 │   RUNNING    ├───► [重回静默巡航继续训练]
 └──────────────┘
```

---

## 🛠 使用 uv 进行项目环境依赖管理

本项目采用高性能 Python 包管理器 `uv` 进行环境与依赖管理。

### 1. 安装与同步环境
```bash
# 1. 克隆或进入项目目录
cd TrainPilot

# 2. 一键创建虚拟环境并同步所有核心依赖与开发依赖
uv sync

# 3. 若需要安装飞书官方 lark-oapi SDK
uv sync --extra feishu
```

### 2. 一键启动与专属 Skills 生成 (推荐)
本项目提供了已拆分解耦的一键脚本：

```bash
# 1. 独立生成 / 同步全局 Agent Skills（默认安装到 ~/.config/opencode 与 ~/.agents，任意目录均可发现）
./setup_skills.sh

# 2. 一键启动 Web 控制面网关服务 (默认采用小众端口 28780 避免冲突，自动检查端口可用性)
./start.sh

# 支持常用参数：
./start.sh --daemon   # 后台守护进程启动
./start.sh --status   # 查看运行状态与健康检查
./start.sh --stop     # 停止后台服务
./start.sh -p 29580   # 临时指定其它端口
```

### 3. 运行开发服务与测试
```bash
# 启动控制面公网网关服务
uv run python examples/run_server.py

# 运行全量自动化测试套件
uv run pytest -v

# 运行端到端训练与 HITL 恢复闭环模拟
uv run pytest tests/test_e2e_simulation.py -v -s
```

---

## 🤖 项目级 Agent Skills 规范（支持 agy 与 opencode）

为让运行在 GPU 服务器上的 AI Agent（如 `agy`、`opencode`、`claude-code`）能够自主调用网关的 HTTP 接口，本项目在项目根目录构建了标准规范的 Skills 包：

- **技能规范定义**：[`skills/trainpilot/SKILL.md`](skills/trainpilot/SKILL.md)
- **独立可执行工具**：[`skills/trainpilot/scripts/trainpilot_tool.py`](skills/trainpilot/scripts/trainpilot_tool.py)

### CLI 常用操作示例

智能体可在 GPU 节点通过标准终端直接执行（支持配置环境变量 `TRAINPILOT_GATEWAY_URL` 与 `TRAINPILOT_TASK_ID`）：

```bash
# 1. 上报训练阶段里程碑 (静默记录，推送飞书只读绿色卡片)
python3 skills/trainpilot/scripts/trainpilot_tool.py report-milestone \
  --gateway "http://127.0.0.1:28780" \
  --task-id "qwen2-7b-sft-0905" \
  --step 1000 \
  --epoch 1 \
  --message "Epoch 1 finished successfully" \
  --metrics "loss=0.41,val_loss=0.43"

# 2. 上报训练异常并挂起现场 (推送飞书红色告警卡片，等待工程师决策)
python3 skills/trainpilot/scripts/trainpilot_tool.py report-alert \
  --gateway "http://127.0.0.1:28780" \
  --task-id "qwen2-7b-sft-0905" \
  --step 1450 \
  --message "Loss NaN detected at step 1450" \
  --metrics "loss=NaN"

# 3. 阻塞轮询信箱，等待人类专家在飞书端下发的决策指令
python3 skills/trainpilot/scripts/trainpilot_tool.py poll-instruction \
  --gateway "http://127.0.0.1:28780" \
  --task-id "qwen2-7b-sft-0905" \
  --wait \
  --wait-timeout 600

# 4. 执行本地修复后，向网关确认 ACK，将任务状态重置为 RUNNING
python3 skills/trainpilot/scripts/trainpilot_tool.py ack-instruction \
  --gateway "http://127.0.0.1:28780" \
  --task-id "qwen2-7b-sft-0905" \
  --instruction-id "inst_b12fa09c" \
  --action "reduce_lr_rollback" \
  --status "success" \
  --message "Rolled back to checkpoint step 1400 and halved learning rate"

# 5. 上报运行时心跳
python3 skills/trainpilot/scripts/trainpilot_tool.py send-heartbeat \
  --gateway "http://127.0.0.1:28780" \
  --task-id "qwen2-7b-sft-0905" \
  --step 1500 \
  --metrics "gpu_mem=88%"

# 6. 查询当前任务完整信箱与状态看板
python3 skills/trainpilot/scripts/trainpilot_tool.py get-status \
  --gateway "http://127.0.0.1:28780" \
  --task-id "qwen2-7b-sft-0905"
```

---

## 🐍 Python 原生 SDK（零凭证极简依赖）

在训练脚本中，可以直接导入 `trainpilot.agent` 原生 Python 模块：

```python
from trainpilot.agent import TrainPilotClient, TrainingGuardian

# 初始化轻量客户端（仅依赖标准 requests）
client = TrainPilotClient(
    gateway_url="http://your-control-plane:28780",
    task_id="llama3-8b-lora",
)

guardian = TrainingGuardian(client=client)

# 注册针对飞书卡片决策的回调函数
def on_reduce_lr_rollback(payload):
    print("加载上一可用 Checkpoint，并将学习率下调 50%...")
    # model.load_state_dict(...)
    # optimizer.param_groups[0]['lr'] *= 0.5

guardian.register_action_handler("reduce_lr_rollback", on_reduce_lr_rollback)

# 在训练主循环中调用：
for step, batch in enumerate(dataloader):
    loss = model(batch)
    
    # 自动监测 NaN / Inf / 突发数值爆炸
    # 异常时自动：挂起现场 -> 推送飞书卡片 -> 轮询决策 -> 执行注册回调 -> ACK 恢复！
    guardian.check_and_handle_loss(loss.item(), step=step)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

或者使用通用训练 Hook：
```python
from trainpilot.agent.hooks.pytorch import TrainPilotPyTorchHook

hook = TrainPilotPyTorchHook(
    task_id="llama3-8b-lora",
    gateway_url="http://your-control-plane:28780",
    milestone_step_interval=500, # 每 500 步自动上报里程碑
    heartbeat_step_interval=50,  # 每 50 步上报心跳
)
hook.register_recovery_callback("reduce_lr_rollback", my_rollback_logic)

# 训练步结束时：
hook.on_step_end(step=step, loss=loss_val, lr=current_lr)
```

---

## 📡 核心接口契约 (API Contracts)

| 请求方 | 接口路径 | 方法 | 作用说明 |
|---|---|---|---|
| **GPU Agent** | `/api/tasks/notify` | `POST` | 上报事件（`alert` 告警、`milestone` 里程碑、`completed` 完成） |
| **GPU Agent** | `/api/tasks/{task_id}/instruction` | `GET` | 轮询信箱获取人类决策（参数 `pop=true` 消费并进入 `RECOVERING`） |
| **GPU Agent** | `/api/tasks/{task_id}/ack` | `POST` | 确认指令执行结果，恢复状态至 `RUNNING` |
| **GPU Agent** | `/api/tasks/{task_id}/heartbeat` | `POST` | 定期上报保活心跳与显存/资源利用率 |
| **GPU Agent** | `/api/tasks/{task_id}/status` | `GET` | 查询指定任务详情与历史事件记录 |
| **GPU Agent** | `/api/tasks/{task_id}/events?limit=100&offset=0` | `GET` | 分页查询任务事件历史（审计用） |
| **控制台/管理** | `/api/tasks?limit=100&offset=0&state=RUNNING&stale_only=false` | `GET` | 分页列出任务，支持按状态/失联过滤 |
| **控制台/管理** | `/api/tasks/{task_id}/decision` | `POST` | 直接注入人工决策（可用于测试或 Web UI 控制台） |
| **飞书客户端** | `/webhook/feishu` | `POST` | 飞书应用事件握手（`url_verification`）与交互卡片点击回调（`card.action.trigger`） |

> **部署约束（v0.1.1+）**：
> - 信箱为进程内存 + 环形裁剪（默认每任务保留 500 事件），重启丢失；勿用 `--workers>1`，生产建议外置 Redis/SQLite。
> - 公网部署请设置 `TRAINPILOT_API_TOKEN`，Agent 侧配置同值 `TRAINPILOT_API_TOKEN` 或 `Authorization: Bearer`；Webhook 另用飞书 `verification_token` 严格校验（缺失也拒绝）。
> - `/notify` 的飞书推送已改为后台任务，不阻塞训练循环；失联任务可通过 `/health` 的 `stale_tasks_count` 或 `stale_only=true` 发现。
> - Docker：`docker build -t trainpilot . && docker run -p 28780:28780 --env-file .env trainpilot`（单副本）。

---

## 🧪 自动化测试验证

项目内置全面的测试覆盖（包含单元测试与端到端模拟测试）：

```bash
uv run pytest -v
```

测试集清单：
- `tests/test_states_and_schemas.py`：生命周期状态枚举与 Pydantic 数据契约校验
- `tests/test_mailbox.py`：多任务并发安全信箱、状态机流转与超时心跳测试
- `tests/test_feishu_cards.py`：飞书富文本、交互式告警卡片与防呆更新卡片生成
- `tests/test_server_api.py`：FastAPI 核心端点与飞书 Webhook 交互测试
- `tests/test_agent_client.py`：Agent 客户端、NaN/Inf 安全清洗、PyTorch Hook 与异常处理测试
- `tests/test_skills_cli.py`：项目级 Agent Skill CLI 命令行工具端到端调用测试
- `tests/test_e2e_simulation.py`：完整模拟训练遇 NaN、冻结、飞书决策、回滚、恢复至完成的端到端闭环测试
