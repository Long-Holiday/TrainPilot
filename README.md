# TrainPilot: Public MCP Server & HITL Closed-Loop Decision System for Deep Learning Training Clusters

<p align="center">
  <b>English</b> | <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python Version" />
  <img src="https://img.shields.io/badge/Protocol-Model%20Context%20Protocol%20(MCP)-orange" alt="MCP Protocol" />
  <img src="https://img.shields.io/badge/Framework-FastAPI%20%7C%20PyTorch-green" alt="Framework" />
  <img src="https://img.shields.io/badge/Storage-SQLite%20WAL%20(Zero--RAM)-brightgreen" alt="Storage" />
  <img src="https://img.shields.io/badge/License-MIT-purple" alt="License" />
</p>

> Specifically designed for large-scale distributed deep learning training, featuring a modern decoupled architecture: **"Intranet GPU AI Agent + Public MCP Server Control Plane + Feishu/Lark User Notifications"**.

TrainPilot thoroughly resolves the longstanding pain points of intranet GPU training clusters—such as lacking public IPs, inability to receive external inbound webhooks, absent external credential delivery channels, and delayed intervention during training anomalies:
1. **Agent Executes Training Tasks on GPU Servers**: AI Agents on intranet GPU nodes (e.g., Claude Code, OpenCode, custom agents, or training scripts) manage and execute model training workloads.
2. **Reports Task Status to Public MCP Server**: The GPU-side Agent actively reports training progress, real-time metrics (Loss/LR/VRAM), and autonomous assessment insights (`agent_note`) to the public MCP server via standard Model Context Protocol (MCP) remote transport.
3. **Notifies Human Users in Real Time with Closed-Loop Decisions**: Upon receiving updates, the public MCP server immediately renders rich text cards and pushes them to Feishu/Lark. When anomalies occur (such as NaN or OOM), it freezes the training execution on-site and provides interactive buttons. Users make one-click decisions in Feishu/Lark, and instructions are streamed back to the GPU-side Agent in real time for self-healing and recovery.

---

## ✨ Key Features

- 🤖 **Intranet GPU Pure-Outbound Agent**: Architected for intranet GPU clusters with no public IP, no inbound ports, and zero external credentials. The GPU-side Agent operates as a pure outbound client syncing state via standard Model Context Protocol (Streamable HTTP `/mcp`) with zero firewall configuration.
- 📱 **Feishu/Lark HITL One-Click Self-Healing Loop**: Freezes training execution on-site upon sudden NaN or OOM anomalies and pushes interactive cards within seconds; ML engineers make one-click decisions in Feishu/Lark, and self-healing instructions are streamed back to the GPU in real time for recovery.

---

## 🏛 System Architecture & Workflow

### Network Topology

```mermaid
flowchart TB
    %% Style definitions
    classDef compStyle fill:#FFFFFF,stroke:#3B82F6,stroke-width:1.5px;
    classDef serverCompStyle fill:#FFFFFF,stroke:#2563EB,stroke-width:1.5px;
    classDef userCompStyle fill:#FFFFFF,stroke:#16A34A,stroke-width:1.5px;
    classDef subStyle fill:#F8FAFC,stroke:#94A3B8,stroke-width:1.5px,stroke-dasharray: 4 4;

    subgraph GPU_Side["🏢 Intranet GPU Training Cluster [No Public IP / Pure Client]"]
        direction LR
        TrainingJob["🚀 PyTorch / DeepSpeed<br/>Distributed Training Loop"]:::compStyle
        Guardian["🛡️ TrainingGuardian<br/>Floating-Point Sanitizer & On-Site Freeze"]:::compStyle
        GPUAgent["🤖 GPU AI Agent<br/>Claude Code / OpenCode / Script"]:::compStyle

        TrainingJob -->|"Monitor Loss & VRAM"| Guardian
        Guardian -->|"Anomaly Analysis & Status Sync"| GPUAgent
    end

    subgraph Control_Plane["☁️ Cloud Server [TrainPilot Control Plane]"]
        direction LR
        MCPServer["🌐 Public MCP Server Gateway<br/>Streamable HTTP /mcp Endpoint"]:::serverCompStyle
        Mailbox["📬 Task Mailbox & Scheduler<br/>Millisecond Long-Poll Wakeup"]:::serverCompStyle
        Storage[("💾 SQLite WAL Engine<br/>Cold/Hot Task Tiering / Ultra-low RAM")]:::serverCompStyle

        MCPServer <-->|"Status Sync & Instruction Dispatch"| Mailbox
        Mailbox <-->|"Local Persistence & Recovery"| Storage
    end

    subgraph User_Side["📱 Human Expert & Collaboration [Human-in-the-Loop]"]
        direction LR
        FeishuCard["💬 Feishu/Lark Interactive Card<br/>Milestone Display / Anomaly Alert"]:::userCompStyle
        Human["👨‍💻 ML Engineer / MLOps Expert<br/>Real-time Tracking / One-click Decision"]:::userCompStyle

        FeishuCard <-->|"Card Push & Interaction"| Human
    end

    %% Cross-tier Data and Control Flow
    GPUAgent ==> |"1. Call MCP tools to report status & metrics"| MCPServer
    MCPServer ==> |"2. Asynchronously push Feishu cards to user"| FeishuCard
    Human -.-> |"3. Click Feishu button - Webhook decision"| MCPServer
    Mailbox -.-> |"4. Long-poll returns instruction - Agent self-heals"| GPUAgent

    class GPU_Side,Control_Plane,User_Side subStyle;
```

### Task Lifecycle State Machine

```mermaid
stateDiagram-v2
    [*] --> RUNNING: Agent launches training task
    RUNNING --> RUNNING: Heartbeat send_heartbeat / Milestone report_milestone
    RUNNING --> WAITING: Anomaly captured & execution frozen report_alert
    WAITING --> RESOLVED: User decision in Feishu or timeout fallback
    RESOLVED --> RECOVERING: Agent polls instruction poll_instruction
    RECOVERING --> RUNNING: Self-healing success ack_instruction - success
    RECOVERING --> WAITING: Self-healing failure ack_instruction - failed
    RUNNING --> COMPLETED: Training completed successfully
    RUNNING --> FAILED: Process crashed or stopped by user stop_training
    WAITING --> FAILED: User clicks [Stop Training] in Feishu
    COMPLETED --> [*]
    FAILED --> [*]
```

---

## 📂 Project Structure

```text
TrainPilot/
├── src/trainpilot/
│   ├── server/          # Public MCP Server & Control Plane Gateway (FastAPI / State Machine / Feishu Cards / SQLite WAL)
│   ├── agent/           # GPU Client Components (TrainingGuardian / MCP Client / trainpilot-cli)
│   └── common/          # Core Data Contracts (Schemas) & State Machine Definitions (States)
├── examples/            # Simulation & Real GPU Training Examples (mock_training.py, real_gpu_training.py)
├── Dockerfile           # Container build specification for control plane
├── start.sh             # Server management script (--daemon / --status / --stop)
└── pyproject.toml       # Project configuration & dependencies
```

---

## 🚀 Quick Start

This project recommends using the modern high-performance Python package manager [uv](https://docs.astral.sh/uv/).

### 1. Installation & Dependency Sync

```bash
# Clone the repository
git clone https://github.com/Long-Holiday/TrainPilot.git
cd TrainPilot

# Install dependencies
uv sync
```

### 2. Public Server Management (`start.sh`)

The control plane script [`start.sh`](file:///home/default_user/TrainPilot/start.sh) manages the MCP Server lifecycle, handling environment verification and process supervision automatically:

```bash
# 1. Start interactively in foreground (for debugging & viewing real-time logs)
./start.sh

# 2. Start in background daemon mode (recommended for production)
./start.sh --daemon

# 3. Check service status & /health probe
./start.sh --status

# 4. Gracefully stop the background service
./start.sh --stop
```

Key server endpoints:
- **MCP Service Endpoint**: `http://<PUBLIC_IP>:28780/mcp` (Streamable HTTP transport)
- **Interactive API Documentation**: `http://<PUBLIC_IP>:28780/docs` (Swagger UI)
- **Health Check Probe**: `http://<PUBLIC_IP>:28780/health`
- **Feishu/Lark Event Webhook**: `http://<PUBLIC_IP>:28780/webhook/feishu`

### 3. Quick End-to-End Simulation

Quickly verify the full closed loop on your local machine: **Agent runs task -> Reports anomaly to MCP Server -> Feishu notifies user -> User decides -> Agent self-heals**:

```bash
uv run python examples/mock_training.py
```

---

## 🤖 GPU-Side Agent Integration & Status Reporting

AI Agents running on intranet GPU servers (e.g., Claude Code, OpenCode, Cursor, or autonomous algorithmic agents) simply need to configure the public MCP server endpoint to autonomously report status, real-time metrics, and model reviews during training, as well as receive user intervention instructions.

### 1. GPU Agent MCP Client Configuration

In your AI Agent configuration file on the GPU server (e.g., `~/.claude.json`, `.cursor/mcp.json`), configure the public MCP Server's Streamable HTTP endpoint:

```json
{
  "mcpServers": {
    "trainpilot": {
      "url": "http://<PUBLIC_SERVER_IP>:28780/mcp",
      "headers": {
        "Authorization": "Bearer your_optional_api_token"
      }
    }
  }
}
```

Once configured, the Agent running on the GPU cluster can directly invoke TrainPilot's standardized MCP tools while running training scripts or troubleshooting.

---

### 2. Common GPU Agent Workflows

When executing training workloads, the Agent interacts with the public MCP Server through the following tools:

1. **Report Milestones & Notify Users**:
   The Agent regularly reports progress (e.g., at each epoch or checkpoint) along with autonomous analysis in `agent_note`. The MCP Server immediately delivers a green card to the Feishu/Lark group:
   ```json
   {
     "tool": "report_milestone",
     "arguments": {
       "task_id": "llama3-8b-sft",
       "step": 5000,
       "epoch": 1,
       "metrics": {"loss": 0.385, "gpu_mem": "82%"},
       "message": "Epoch 1 warmup completed successfully. Checkpoint saved.",
       "agent_note": "Loss convergence rate is stable with no signs of overfitting on the validation set. Recommend continuing training with current learning rate."
     }
   }
   ```

2. **Detect Anomalies, Freeze Execution & Request User Intervention**:
   When the Agent detects `Loss NaN`, extreme metric spikes, or `CUDA OOM`, it reports an alert. The control plane sets the task state to `WAITING`, freezes execution on-site, and sends a red alert card with [🛑 Stop Training] and [✅ Self Resolve] action buttons to Feishu/Lark:
   ```json
   {
     "tool": "report_alert",
     "arguments": {
       "task_id": "llama3-8b-sft",
       "step": 5420,
       "metrics": {"loss": "NaN"},
       "message": "Loss turned into NaN",
       "agent_note": "Suspected gradient overflow triggered by anomalous corrupted samples in the batch. Training execution frozen."
     }
   }
   ```

3. **Poll User Decisions from Feishu & Execute Self-Healing**:
   The Agent invokes `poll_instruction` to suspend via long polling. Once the user clicks an action in Feishu/Lark (or when the 30-second timeout auto-decision triggers), the Agent receives the instruction in milliseconds:
   ```json
   {
     "tool": "poll_instruction",
     "arguments": {
       "task_id": "llama3-8b-sft",
       "wait_timeout": 30.0
     }
   }
   ```

4. **Acknowledge Recovery & Notify Users of Self-Healing Result**:
   After performing self-healing actions on the GPU node (e.g., skipping bad batches, rolling back to checkpoint, lowering learning rate), the Agent sends an ACK. The control plane transitions the task state back to `RUNNING` and automatically pushes a "Self-Healing Succeeded" card to Feishu/Lark:
   ```json
   {
     "tool": "ack_instruction",
     "arguments": {
       "task_id": "llama3-8b-sft",
       "action": "self_resolve",
       "status": "success",
       "solution": "Skipped corrupted batch, loaded step 5000 checkpoint, and reset optimizer state. Training resumed successfully."
     }
   }
   ```

---

## 📡 MCP Core Interface Contracts

The public control plane strictly adheres to the MCP specification, providing standardized **Tools** and **Resources**.

### MCP Tools (8 Standard Tools)

| Tool Name | Primary Role | Input Parameters Summary | Behavior & Effects |
|:---|:---|:---|:---|
| **`report_milestone`** | Status Reporting | `task_id`, `message`, `step`, `epoch`, `metrics`, `agent_note`, `extra` | Records training milestone, updates step & metrics, pushes green card with Agent analysis to Feishu/Lark |
| **`report_alert`** | Anomaly Interception | `task_id`, `message`, `step`, `epoch`, `metrics`, `agent_note`, `extra` | Sets state to `WAITING`, freezes execution, pushes red interactive card to user, activates 30s timeout timer |
| **`poll_instruction`** | Decision Polling | `task_id`, `wait_timeout` (default 20s), `pop` (default True) | Agent suspends waiting for user decision; returns instruction when received and transitions state to `RECOVERING` |
| **`ack_instruction`** | Self-Healing ACK | `task_id`, `instruction_id`, `action`, `status`, `solution`, `message` | Validates instruction, transitions state back to `RUNNING`, pushes self-healing success card to Feishu/Lark |
| **`send_heartbeat`** | Node Keepalive | `task_id`, `step`, `epoch`, `metrics`, `status` | Updates `last_heartbeat_at`, resets disconnection flags, reports to watchdog |
| **`get_task_status`** | Status Query | `task_id` | Retrieves current lifecycle state, latest metrics, pending mailbox instructions, and event statistics |
| **`list_tasks`** | Cluster Summary | `limit`, `offset`, `state` (filter), `stale_only` (unreachable only) | Paginated list of all tracked cluster tasks with state and stale filtering |
| **`submit_decision`** | User Intervention | `task_id`, `action`, `payload`, `operator` | Injects operational decision, updates mailbox, and marks Feishu card as resolved on-site |

### MCP Resources (Read-Only Resources)

In addition to callable tools, TrainPilot exposes the following standard resources to clients:

| Resource URI | MIME Type | Description |
|:---|:---|:---|
| **`tasks://status/{task_id}`** | `application/json` | Full status snapshot of the specified task in JSON format (including mailbox and event statistics) |
| **`tasks://list`** | `application/json` | Concise summary list of all currently tracked training tasks in JSON format |

---

## 📱 Feishu/Lark Interactive Cards & HITL Closed Loop

### 5 Card Types Matrix

TrainPilot features meticulously designed, structured Feishu/Lark cards for various deep learning scenarios:

| Card Type | Header Color | Trigger Condition | Core Content & Interaction Elements |
|:---|:---:|:---|:---|
| **⚠️ Training Anomaly Alert Card** | 🔴 Red | Agent triggers `report_alert` (e.g., Loss NaN) | Real-time metrics, anomaly details, **🤖 Agent Anomaly Assessment**, 30s countdown notice; provides [🛑 Stop Training] and [✅ Self Resolve] interactive buttons |
| **🚀 Training Milestone Card** | 🟢 Green | Agent triggers `report_milestone` (e.g., Epoch finished) | Stage progress, key metrics, **🤖 Agent Intelligent Review** (dedicated block showing optimization suggestions) |
| **✅ Decision Closed-Loop Card** | 🟦 Turquoise | User clicks action button or 30s timeout triggers | Replaces original alert card on-site; displays **Operator**, **Selected Action**, and **Resolution Time**; hides buttons to prevent re-clicks |
| **🛠️ Self-Healing Success Card** | 🟢 Green | Agent sends `ack_instruction` after self-healing | Summarizes applied solution (e.g., checkpoint rollback, LR reduction), informing the team that training has recovered |
| **⚠️ Disconnection Warning Card** | 🟠 Orange | Watchdog detects heartbeat timeout (default >300s) | Disconnection duration, last known step & heartbeat time; prompts investigation of power loss, node crash, or NCCL deadlock |

### Card Idempotency & Foolproofing Safeguards

1. **On-Site In-Place Replacement**: When a user clicks an action button on mobile or desktop, the server returns a new card with `type: "raw"`. Feishu/Lark updates the original card to [Decision Closed] within 3 seconds and completely removes action buttons, preventing duplicate clicks in concurrent team environments.
2. **State Machine Strict Locking**: Even if network jitter causes duplicate requests, once a task leaves the `WAITING` state, subsequent repeated decisions are rejected. When acknowledging, the client must also provide a matching `instruction_id`.

### Feishu/Lark Custom App Setup Guide

To enable real Feishu/Lark card notifications and interactive callbacks, create an enterprise custom app on the [Feishu Open Platform](https://open.feishu.cn/) (or Lark Open Platform):

1. **Obtain Credentials**: Navigate to app details -> "Credentials & Basic Info" to obtain `App ID` and `App Secret`.
2. **Event Subscription & Callback Configuration**:
   - Set Request URL to: `http://<PUBLIC_SERVER_IP>:28780/webhook/feishu`
   - Obtain and configure `Verification Token` and optional `Encrypt Key`.
3. **Card Interaction Callback Configuration**:
   - Enable bot capabilities in "App Features" -> "Bot".
   - In "Messages & Groups" or card interaction settings, set the message card interaction callback URL to `http://<PUBLIC_SERVER_IP>:28780/webhook/feishu`.
4. **Permissions (Scopes)**:
   - Request `im:message` related read and write permissions (send/receive messages in direct chats and group chats).
5. **Obtain Target Receiver IDs**:
   - Group notifications: Add the bot to your training monitoring group and copy the `chat_id` (e.g., `oc_xxxx`) from group settings.
   - Private notifications: Obtain the user's `open_id` (e.g., `ou_xxxx`).

> [!NOTE]
> If a Feishu/Lark app is not yet configured, the system defaults to safe Mock mode: all cards are logged in formatted visual blocks on the server console without interrupting training workflows.

---

## ⚙️ Deployment & Environment Configuration

### Start Script Advanced Options

[`start.sh`](file:///home/default_user/TrainPilot/start.sh) provides comprehensive command-line arguments:

```text
Usage: ./start.sh [OPTIONS]

Options:
  (no arguments)           Start control plane gateway interactively in foreground (default)
  --daemon, -d             Start in background daemon mode (managed via nohup/setsid)
  --stop                   Gracefully stop running background gateway process
  --status                 Check service health status and call /health probe
  --port, -p <PORT>        Temporarily override listening port (default: 28780)
  --host, --bind, -H <IP>  Temporarily override bind IP address (default: 0.0.0.0)
  --help, -h               Show help message
```

### Docker Container Deployment

A lightweight multi-stage Dockerfile is provided (based on `python:3.11-slim` and `uv`):

```bash
# 1. Build image
docker build -t trainpilot:latest .

# 2. Run container (map port 28780, mount persistent database and configuration)
docker run -d \
  --name trainpilot-gateway \
  -p 28780:28780 \
  -v $(pwd)/trainpilot.db:/app/trainpilot.db \
  -v $(pwd)/.env:/app/.env:ro \
  --restart always \
  trainpilot:latest
```

> [!WARNING]
> **Important Note on Concurrent Workers**: Because the control plane maintains an in-memory mailbox scheduler and millisecond-level long-poll event suspension, Uvicorn **must run in single-worker mode** (`--workers 1`). Do NOT run with multiple worker processes; otherwise, long-poll wait locks cannot be shared across processes. For horizontal scaling, use a unified reverse proxy with hash-based routing on `task_id`.

### Environment Variable Reference

Copy and customize the configuration file: `cp .env.example .env`

#### Module 1: Public Server Configuration (Runs on Public Server)

| Variable | Default | Type | Description |
|:---|:---|:---:|:---|
| `TRAINPILOT_PORT` | `28780` | int | Server listening port (uses high-range port to avoid conflicts) |
| `TRAINPILOT_BIND_HOST` | `0.0.0.0` | str | Network interface IP to bind |
| `TRAINPILOT_DEBUG` | `false` | bool | Enable debug-level logging |
| `TRAINPILOT_FEISHU_APP_ID` | - | str | Feishu/Lark App ID (`cli_xxxx`) |
| `TRAINPILOT_FEISHU_APP_SECRET` | - | str | Feishu/Lark App Secret |
| `TRAINPILOT_FEISHU_VERIFICATION_TOKEN` | - | str | Feishu/Lark Webhook Verification Token |
| `TRAINPILOT_FEISHU_ENCRYPT_KEY` | - | str | Feishu/Lark Webhook message encryption key (optional) |
| `TRAINPILOT_FEISHU_RECEIVE_ID_TYPE` | `chat_id` | str | Receiver ID type (`chat_id`, `open_id`, `user_id`, `email`) |
| `TRAINPILOT_FEISHU_RECEIVER_ID` | - | str | Receiver ID (group chat `oc_xxxx` or individual `ou_xxxx`) |
| `TRAINPILOT_ENABLE_MOCK_FEISHU` | `false` | bool | Force enable Feishu mock mode (for local testing/simulation) |
| `TRAINPILOT_API_TOKEN` | - | str | API authentication token (clients must provide this if set) |
| `TRAINPILOT_ALERT_DECISION_TIMEOUT_SECONDS` | `30` | int | Timeout (seconds) waiting for human decision before auto fallback |
| `TRAINPILOT_TASK_HEARTBEAT_TIMEOUT_SECONDS` | `300` | int | Threshold (seconds) to mark a task as disconnected |
| `TRAINPILOT_LONG_POLL_TIMEOUT_SECONDS` | `20.0` | float | Maximum server-side long-polling suspension duration (seconds) |
| `TRAINPILOT_ENABLE_SQLITE` | `true` | bool | Enable SQLite WAL persistence; `false` uses process-local in-memory storage |
| `TRAINPILOT_SQLITE_PATH` | `trainpilot.db` | str | SQLite database file path |
| `TRAINPILOT_MAX_EVENTS_PER_TASK` | `500` | int | Maximum historical events retained per task (ring buffer) |
| `TRAINPILOT_ENABLE_TASK_CLEANUP` | `true` | bool | Periodically delete expired `COMPLETED`/`FAILED` tasks and their events |
| `TRAINPILOT_TASK_RETENTION_HOURS` | `72` | float | Retention period for terminal tasks in hours |
| `TRAINPILOT_TASK_CLEANUP_INTERVAL_SECONDS` | `3600` | float | Interval between expired-task cleanup sweeps |
| `TRAINPILOT_ENABLE_WATCHDOG` | `true` | bool | Enable server-side background heartbeat watchdog |
| `TRAINPILOT_WATCHDOG_INTERVAL_SECONDS` | `15` | int | Watchdog background scan interval (seconds) |
| `TRAINPILOT_MCP_ENABLE_DNS_REBINDING_PROTECTION` | `false` | bool | Enable MCP DNS rebinding protection (set to false for public access) |

#### Module 2: Intranet GPU Client Configuration (Runs on GPU Nodes)

| Variable | Example Value | Description |
|:---|:---|:---|
| `TRAINPILOT_GATEWAY_URL` | `http://1.2.3.4:28780` | Complete public control plane URL (highest priority; **do not use 0.0.0.0**) |
| `TRAINPILOT_HOST` | `1.2.3.4` | Public control plane IP or domain (combined with `TRAINPILOT_PORT`) |
| `TRAINPILOT_PORT` | `28780` | Public control plane listening port |
| `TRAINPILOT_TASK_ID` | `llama3-8b-lora` | Unique identifier for current training workload |
| `TRAINPILOT_API_TOKEN` | - | API authentication token matching the server configuration |

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Issues and Pull Requests are warmly welcome!
