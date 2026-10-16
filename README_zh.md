# TrainPilot：GPU 训练集群的 HITL 控制面

<p align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python Version" />
  <img src="https://img.shields.io/badge/Protocol-Model%20Context%20Protocol%20(MCP)-orange" alt="MCP Protocol" />
  <img src="https://img.shields.io/badge/License-MIT-purple" alt="License" />
</p>

面向**无公网 IP、无法接收外部 Webhook** 的内网 GPU 节点：GPU 侧的 AI Agent（Claude Code /
Antigravity / Gemini CLI / Codex / OpenCode 等）通过 MCP 上报训练进度与异常，TrainPilot 把异常变成
飞书审批卡片，并把人的一键决策实时回传给 Agent。GPU 侧一条命令即可安装，见
[接入 GPU 侧 Agent](#接入-gpu-侧-agent)。

**TrainPilot 不监控、也不暂停你的训练进程。** 盯日志和控制作业是 Agent 的职责；控制面只负责任务
状态机、审批闭环与审计留痕。

---

## 架构

```mermaid
flowchart TB
    subgraph GPU["内网 GPU 节点（仅出网）"]
        Job["训练作业"]
        Agent["AI Agent（Claude Code / Antigravity / Gemini CLI / Codex / OpenCode）"]
        Job -->|日志、退出状态| Agent
    end

    subgraph Plane["控制面（公网）"]
        MCP["MCP Server<br/>/mcp（Streamable HTTP）"]
        Mailbox["任务信箱<br/>长轮询唤醒"]
        DB[("SQLite WAL")]
        MCP <--> Mailbox
        Mailbox <--> DB
    end

    subgraph Human["人（HITL）"]
        Card["飞书卡片"]
        Engineer["ML 工程师"]
        Card <--> Engineer
    end

    Agent ==>|"1. report / poll / ack"| MCP
    MCP ==>|"2. 推送卡片"| Card
    Engineer -.->|"3. 点击按钮（Webhook）"| MCP
    Mailbox -.->|"4. 长轮询返回指令"| Agent
```

```mermaid
stateDiagram-v2
    [*] --> RUNNING: 首次 report
    RUNNING --> RUNNING: milestone
    RUNNING --> WAITING: alert
    WAITING --> RESOLVED: 人工决策（点卡片或超时）
    RESOLVED --> RECOVERING: poll_instruction
    RECOVERING --> RUNNING: ack 成功
    RECOVERING --> WAITING: ack 失败
    RUNNING --> COMPLETED: completed
    RUNNING --> FAILED: failed
    WAITING --> FAILED: stop_training
    COMPLETED --> [*]
    FAILED --> [*]
```

---

## 快速开始

这里装的是**控制面**，跑在公网服务器上。GPU 侧是一套小得多的独立安装，见
[接入 GPU 侧 Agent](#接入-gpu-侧-agent)。

需要 Python >= 3.10，推荐使用 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/Long-Holiday/TrainPilot.git && cd TrainPilot
uv sync
cp .env.example .env     # 缺失时 start.sh 会自动创建
```

控制面的依赖（FastAPI、uvicorn、lark-oapi）放在 `server` 依赖组而不是基础依赖里，所以克隆本仓库的
GPU 节点可以直接跳过它们：

```bash
uv sync --no-group server --no-group dev    # GPU 侧：只装 mcp + httpx + pydantic
```

```bash
./start.sh               # 前台运行，实时日志
./start.sh --daemon      # 后台运行
./start.sh --status      # 健康检查
./start.sh --stop
```

端点：`/mcp`（MCP，Streamable HTTP）· `/docs`（Swagger）· `/health` · `/ready` ·
`/webhook/feishu` · `/api/tasks/...`。

本机验证完整闭环（无需 GPU、无需飞书应用——卡片会打印到控制台）：

```bash
uv run python examples/mock_training.py
```

---

## 接入 GPU 侧 Agent

GPU 节点上**不需要装 TrainPilot 的代码**也能用控制面：受支持的 Agent 都自带远程 MCP 客户端，
直连 `/mcp`。它们猜不到的是各家对「这里有一个远程 MCP server」的写法，而**写错的后果通常是静默的**
—— Agent 正常启动，就是永远看不到那几个工具。所以用一条命令把 TrainPilot 渲染成你面前这个 Agent
认识的格式，写进去，再交给该 Agent 自己的 CLI 去注册：

```bash
# 尚未发布到 PyPI —— 直接从仓库装 GPU 侧客户端。
# tool 安装只会带上基础依赖，那正好就是 GPU 侧。
uv tool install "git+https://github.com/Long-Holiday/TrainPilot"
export TRAINPILOT_GATEWAY_URL=http://<公网服务器IP>:28780
export TRAINPILOT_API_TOKEN=<服务端配了就要填一致的值>

trainpilot-cli plugin install all      # 装到本机检测到的每一个 Agent
trainpilot-cli plugin install codex    # 或指定一个
```

| Agent | 下发给它的远程 server key | 装了什么 |
|:---|:---|:---|
| `claude-code` | `"type": "http"` + `url` | 插件 + marketplace 落在 `~/.trainpilot/plugins/claude-code`，随后 `claude plugin install` |
| `antigravity` | `serverUrl` | 插件（`plugin.json`、`mcp_config.json`、skill），用 `agy plugin install` 注册 |
| `gemini-cli` | `httpUrl` | 扩展（`gemini-extension.json`、`GEMINI.md`、TOML 命令、skill），用 `gemini extensions link` 链接 |
| `opencode` | `mcp.servers.<name>`，`type: remote` | 合并进 `opencode.json`，外加 skill 与命令 |
| `codex` | `[mcp_servers.<name>]` TOML 表 | 合并进 `config.toml`，外加 skill 与 `AGENTS.md` 指引 |
| `generic` | `mcpServers` JSON | 跨厂商 skill 落到 `.agents/skills/`，并打印一份可粘贴的片段 |

Antigravity 明确不接受 `url` 与 `httpUrl`，Gemini CLI 则把 `url` 当作 SSE —— 这就是为什么这里需要
一个安装器而不是一段 README 片段。再加一家 Agent = 在
`src/trainpilot/agent/plugin/catalog.py` 加一条 + 写一个渲染函数，两者的对应关系由测试钉住。

常用命令：

```bash
trainpilot-cli plugin list                # 支持的 Agent，以及本机装了哪些
trainpilot-cli plugin render codex        # 只打印将要写入的内容，不落盘
trainpilot-cli plugin install codex --dry-run
trainpilot-cli plugin uninstall codex     # 外科手术式卸载：你自己的键与段落原样保留
trainpilot-cli plugin doctor              # 一次检查网关、token、协议、task id
trainpilot-cli plugin protocol            # 打印当前生效的监督协议
```

出问题时先跑 `plugin doctor`：它能把「网关连不上」「token 不对」「任务还没有行」这三件在 Agent
内部看起来一模一样的事区分开。

### 行为规则从哪来

规则（何时告警、什么算「已验证的恢复」、如何轮询）由服务端通过 MCP 的 `instructions` 字段和每个工具
的 docstring 直接下发，**无需**往 Agent 的 system prompt 里拷贝任何内容。装进去的 skill **刻意不复述**
这些规则：它只交代怎么连接、怎么绑定任务、怎么回执，然后让 Agent 去跑 `trainpilot-cli plugin
protocol` —— 那条命令在一次真实握手里读出契约。因此今天装的插件，不会对之后升级的服务端过时。
装完重启 Agent。

安装器改的是你自己的文件（`config.toml`、`opencode.json`、`AGENTS.md`），所以它只做**合并**而非覆盖：
你自己的键和段落原样保留，改动前的字节留在同目录的 `.trainpilot.bak`，解析不了的 JSON 会被拒绝而
不是被覆盖，`uninstall` 会把文档还给你。

任何工具的 `task_id` 都可省略，此时回退到 `TRAINPILOT_TASK_ID`。

---

## MCP 接口

共 5 个工具：前四个覆盖任务闭环，`ack_plan` 用于确认人类在飞书群里排的下一步计划。

| 工具 | 入参 | 效果 |
|:---|:---|:---|
| **`report`** | `message`、`event_type`（`milestone`/`alert`/`completed`/`failed`，默认 `milestone`）、`task_id`、`step`、`epoch`、`metrics`、`agent_note` | `milestone` → 绿色卡片并展示 Agent 自撰的 `agent_note`；`alert` → 状态置 `WAITING`、推送红色审批卡片并启动 30s 自动决策定时器；`completed`/`failed` → 终态。每次响应都携带 `next_task_plans`（没有计划时为空列表） |
| **`poll_instruction`** | `task_id`、`wait_timeout`（默认 20s） | 长轮询挂起直至决策到达；消费该指令并把任务推进到 `RECOVERING` |
| **`ack_instruction`** | `task_id`、`instruction_id`、`action`、`status`、`solution` | 记录已核实的执行结果，任务回到 `RUNNING`，并推送由任务最近一次上报的 step/metrics 构建的自愈卡片 |
| **`get_task_status`** | `task_id` | 当前状态、最近指标、待处理指令、事件计数 |
| **`ack_plan`** | `plan_id`、`status`（`done`/`failed`/`skipped`，默认 `done`）、`note` | 确认某条下一步计划已处理，之后它不再出现在 `report` 响应里 |

REST 面（`/api/tasks`）面向运维与看板，不面向 Agent：
`GET ""`、`GET /{task_id}/status`、`GET /{task_id}/events`、`POST /{task_id}/decision`。

---

## 飞书卡片

| 卡片 | 头部 | 触发 | 内容 |
|:---|:---:|:---|:---|
| **异常告警** | 红色 | `report(event_type="alert")` | 指标、异常详情、**Agent 自撰的异常判断**、30s 倒计时、`[终止训练]` / `[自行解决]` / `[⏳ 等待安排]` 按钮 |
| **训练里程碑** | 绿色 | `report(event_type="milestone")` | 阶段进度、关键指标、**Agent 点评** |
| **决策闭环** | 青色 | 点击按钮、`#fix` 消息或超时 | 就地替换原告警卡片；展示操作人、所选动作、解决时间；移除按钮 |
| **等待人工安排** | 橙色 | 点击 `[⏳ 等待安排]` | 就地替换原告警卡片；自动决策已暂停、`#fix` 用法、兜底窗口 |
| **自愈成功** | 绿色 | `ack_instruction(status="success")` | Agent 实际执行的处理方案 |
| **下一步计划已登记** | 蓝色 | 群里 `@机器人 <计划>` | 计划编号、完整正文、绑定对象（某个任务或全局）、提交人 |
| **修复方案已下发** | 青色 | 群里 `@机器人 #fix(<任务>) <方案>` | 方案下发到了哪个任务、方案正文、提交人、指令编号 |
| **修复方案未下发** | 红色 | 下发失败的 `#fix` 消息 | 未下发的原因（任务不存在、没有等待中的告警、写法有误） |
| **待办计划已清除** | 黄色 | `@机器人 清除` | 清除范围、清掉几条、操作人 |

点击按钮时服务端返回 `type: "raw"`，飞书会在数秒内就地替换原卡片，避免并发重复点击；任务一旦
离开 `WAITING`，后续决策一律拒绝——但拒绝也是 **HTTP 200 + 错误 toast**，绝不返回 4xx：飞书会把
卡片回调的任何非 200 应答变成客户端那句没有信息量的 `200671`。ack 必须携带匹配的 `instruction_id`。

### 在群里排下一步任务

群里 @机器人 就是全部触发条件，**没有关键字、没有前缀**，消息正文即一条「下一步计划」：

| 群消息 | 效果 |
|:---|:---|
| `@机器人 用 5e-5 重跑 10 个 epoch` | 全局计划 —— 每个任务的 `report` 都看得到 |
| `@机器人 用 5e-5 重跑 #demo-llm-pretrain` | 绑定到 `demo-llm-pretrain`；`#…` 标签会从正文里剥掉 |
| `@机器人 用 5e-5 重跑 #feat(demo-llm-pretrain)` | 与裸 `#…` 等价，`#feat(...)` 是显式写法 |
| `@机器人 #fix(demo-llm-pretrain) 回滚到 step4` | 下发**修复方案**而不是排计划 —— 见下节 |
| `@机器人 清除`（`取消` / `clear` 同样有效） | 清空所有未确认计划 |
| `@机器人 清除 #demo-llm-pretrain` | 只清空该任务名下的未确认计划 |

每次 `report` 都会在 `next_task_plans` 里带回适用于该任务的计划（绑定本任务的 + 全局的），
服务端的 MCP instructions 会指示 Agent 在安全点执行、然后用 `ack_plan` 确认。未确认的计划会
**每次 report 都重新出现**——这正是它能扛住 Agent 崩溃/重启的原因，不是重复指令。计划**永远不创建
task 行**，也不要求被绑定的任务已经存在，所以可以先把活排上、再等开跑。

机器人会在原群回一张确认卡片，回显 `plan_id` 与绑定对象；这是唯一的反馈通道，也是你发现
「正文里的 `#123` 本来只是引用 issue，却把计划悄悄绑到了一个永远不会汇报的任务上」的方式。
飞书对 ~3s 内没确认的事件会重投；`message_id` 是唯一键，所以重投既不会登记出第二条计划，也不会
重发卡片。待确认的计划不会被清理服务回收（只有已确认的才会），清除除了「能在群里 @机器人」之外
没有任何权限校验。

### 在群里回应告警

计划是给**下一轮**用的，救不了眼前卡住的告警。要自己写处理方案交给 Agent，先点告警卡片上的
**⏳ 等待安排**，再在群里发 `@机器人 #fix(<任务名>) <你的处理方案>`：

- 点击只暂停 30s 自动决策：不下发任何指令，任务仍是 `WAITING`，Agent 的长轮询继续挂着。告警卡片
  会被就地换成橙色卡片，重复原告警信息，并写明该发什么 `#fix` 命令。
- `#fix` 正文即成为该任务的决策（`action: "custom"`，方案在指令 payload 里），并唤醒 Agent 的
  长轮询——与点按钮完全同一条路径。告警卡片随即被换成「决策闭环」，群里另回一张回执确认下发到了
  哪个任务。
- `#fix` 只对**已存在且正在等待告警**的任务生效。其余情况（任务名不存在、任务在跑、已经在恢复、
  写法有误）一律拒绝：回一张红色回执，且不产生任何决策；`#fix` **永远不会创建 task 行**。
- 不必先点按钮：在最初的 30s 窗口内直接发 `#fix` 同样有效。
- 兜底：点击后 `TRAINPILOT_ARRANGEMENT_TIMEOUT_SECONDS`（默认 30 分钟）内仍没有 `#fix`，就照旧
  自动 `self_resolve`，任务不会永久卡死。
- 飞书对 ~3s 内未确认的事件会重投；重投的 `#fix` 既不会下发第二条指令，也不会重发第二张回执。

`#fix` 与下一步计划共用同一个开关（`TRAINPILOT_ENABLE_NEXT_TASK_PLANS`），关掉计划也就同时关掉了
`#fix`。标签大小写不敏感；若一条消息同时带 `#fix` 与 `#feat`/裸标签，按 `#fix` 处理，另一个标签
照常从正文里剥掉。

**应用开通：** 在[飞书开放平台](https://open.feishu.cn/)创建企业自建应用 → 从「凭证与基础信息」
复制 `App ID` / `App Secret` / `Verification Token` → 把事件订阅 URL 与卡片回调 URL 都设为
`http://<公网服务器IP>:28780/webhook/feishu` → 申请 `im:message` 权限，另外为「接收群里 @机器人
的消息」申请 `im:message.group_at_msg` → 把机器人拉进群并复制 `chat_id`（或使用个人 `open_id`）。
「加密策略」的 Encrypt Key 请留空：webhook 不解密事件。若不希望群里所有人都能排/清计划，请自行
限制谁能 @机器人。

未配置凭据时服务端进入 Mock 模式，卡片直接打印到控制台。

---

## 部署

```text
./start.sh [OPTIONS]
  （无参数）                前台运行（默认）
  --daemon, -d             后台运行
  --stop                   停止后台进程
  --status                 通过 /health 检查健康状态
  --port, -p <PORT>        临时覆盖端口（默认 28780）
  --host, --bind, -H <IP>  临时覆盖绑定地址（默认 0.0.0.0）
```

```bash
docker build -t trainpilot:latest .
docker run -d -p 28780:28780 -v $(pwd)/trainpilot.db:/app/trainpilot.db \
  -v $(pwd)/.env:/app/.env:ro --restart always trainpilot:latest
```

> [!WARNING]
> **必须单 Worker 运行。** 信箱的长轮询锁是进程内的，uvicorn 绝不能使用 `--workers > 1`
> （`start.sh` 与 `Dockerfile` 已强制约束）。需要横向扩展时，在前置反向代理上按 `task_id`
> 哈希路由。

配置位于 `.env`（`cp .env.example .env`）：公网服务端与内网 GPU 客户端两个模块的完整变量说明
已逐条写在文件内。最关键的几项：

| 变量 | 默认值 | 含义 |
|:---|:---|:---|
| `TRAINPILOT_PORT` | `28780` | 监听端口 |
| `TRAINPILOT_BIND_HOST` | `0.0.0.0` | 绑定网卡（**服务端请用它，而不是 `TRAINPILOT_HOST`**） |
| `TRAINPILOT_API_TOKEN` | – | MCP/REST 的 Bearer Token；留空为开放模式 |
| `TRAINPILOT_FEISHU_APP_ID` / `_APP_SECRET` / `_RECEIVER_ID` | – | 卡片推送；未配置则进入 Mock 模式 |
| `TRAINPILOT_ALERT_DECISION_TIMEOUT_SECONDS` | `30` | 超过该时长无人决策则自动 `self_resolve` |
| `TRAINPILOT_ARRANGEMENT_TIMEOUT_SECONDS` | `1800` | 点「等待安排」后的兜底窗口；`0` 表示关闭该兜底 |
| `TRAINPILOT_ENABLE_NEXT_TASK_PLANS` | `true` | 飞书群消息（`@机器人` 计划与 `#fix` 方案）；`false` 则忽略所有消息事件 |
| `TRAINPILOT_LONG_POLL_TIMEOUT_SECONDS` | `20.0` | 服务端长轮询上限 |
| `TRAINPILOT_SQLITE_PATH` | `trainpilot.db` | SQLite WAL 文件（`TRAINPILOT_ENABLE_SQLITE=false` 则仅内存） |
| `TRAINPILOT_TASK_ID` | `train-task-default` | 工具调用省略 `task_id` 时的回退值 |
| `TRAINPILOT_GATEWAY_URL` | – | 仅客户端：控制面完整 URL（优先级高于 `TRAINPILOT_HOST`+`TRAINPILOT_PORT`） |

> `TRAINPILOT_HOST` 有两种含义：客户端侧是网关地址；服务端侧是绑定地址，且只认本地值
> （`0.0.0.0` / `127.0.0.1` / `localhost`）。服务端绑定请使用 `TRAINPILOT_BIND_HOST`。

---

## 项目结构

```text
src/trainpilot/
├── server/     # 控制面：MCP 工具、信箱、SQLite WAL、飞书卡片、REST
├── agent/      # GPU 侧 MCP 客户端 + trainpilot-cli
│   └── plugin/ # 把 TrainPilot 渲染成各家 Agent 的原生插件格式
└── common/     # Schemas、状态机、网关地址解析
examples/       # mock_training.py —— 无需 GPU 即可跑通完整 HITL 闭环
```

## License

[MIT](https://opensource.org/licenses/MIT)。欢迎提交 Issue 与 Pull Request。
