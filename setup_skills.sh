#!/usr/bin/env bash
# ==============================================================================
# TrainPilot GPU 端部署与 Agent Skills 安装脚本 (仅 GPU 训练端服务器使用)
#
# 角色与职责：
# 1. GPU 端运行环境准备：基于 uv 虚拟环境管理器检查并同步 Python 依赖 (.venv)
# 2. GPU 端专属环境变量管理：自动检查/初始化 GPU 端 .env，防呆校验 0.0.0.0 错误
# 3. Agent Skills 全局同步：为 OpenCode、Antigravity/Gemini、通用 Agent 全局安装技能
# 4. 可选连通性测试：通过 --test / --ping 测试与 Web 端控制面网关的网络连通性
#
# 运行环境：uv (Astral uv package manager)
# ==============================================================================

set -eo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

# 解析命令行参数
CHECK_ONLY=false
CLEAN_ONLY=false
TEST_CONN=false
SKIP_SYNC=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check|-c)
            CHECK_ONLY=true
            shift
            ;;
        --clean)
            CLEAN_ONLY=true
            shift
            ;;
        --test|--ping|-t)
            TEST_CONN=true
            shift
            ;;
        --skip-sync)
            SKIP_SYNC=true
            shift
            ;;
        --help|-h)
            echo "TrainPilot Agent Skills 生成脚本使用说明 (GPU 训练端专用, 运行环境: uv):"
            echo "  ./setup_skills.sh             一键配置 GPU 环境并生成精简版 Agent Skills (优化 Agent 上下文)"
            echo "  ./setup_skills.sh --test      测试与 Web 控制面网关的网络连通性"
            echo "  ./setup_skills.sh --check     仅检查技能目录与文件完整性，不重新生成"
            echo "  ./setup_skills.sh --clean     清理已安装的 Agent Skills"
            echo "  ./setup_skills.sh --skip-sync 跳过 uv sync 依赖同步"
            echo ""
            echo "全局安装位置："
            echo "  - \${XDG_CONFIG_HOME:-\$HOME/.config}/opencode/skills/trainpilot  (OpenCode 官方全局)"
            echo "  - \$HOME/.gemini/skills/trainpilot  (Antigravity/Gemini CLI 全局)"
            echo "  - \$HOME/.agents/skills/trainpilot  (通用 Agent 全局)"
            echo ""
            echo "GPU 端环境变量 (.env 填写说明):"
            echo "  TRAINPILOT_HOST        Web 服务器公网 IP 或域名 (严禁填写 0.0.0.0)"
            echo "  TRAINPILOT_PORT        Web 网关监听端口 (默认 28780)"
            echo "  TRAINPILOT_GATEWAY_URL 完整网关 URL (可选，优先级最高)"
            echo "  TRAINPILOT_TASK_ID     默认训练任务标识符"
            echo "  TRAINPILOT_API_TOKEN   网关 API 鉴权 Token (若 Web 端开启鉴权时填写)"
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            echo "使用 ./setup_skills.sh --help 查看可用参数"
            exit 1
            ;;
    esac
done

SOURCE_SKILL_DIR="${PROJECT_ROOT}/skills/trainpilot"

# 全局目标（尊重 XDG / HOME，便于多用户与容器环境）
CONFIG_BASE="${XDG_CONFIG_HOME:-${HOME}/.config}"
OPENCODE_GLOBAL_DIR="${CONFIG_BASE}/opencode/skills/trainpilot"
GEMINI_GLOBAL_DIR="${HOME}/.gemini/skills/trainpilot"
AGENTS_GLOBAL_DIR="${HOME}/.agents/skills/trainpilot"

if [ ! -d "${SOURCE_SKILL_DIR}" ]; then
    log_error "源 Skills 目录不存在: ${SOURCE_SKILL_DIR}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 1. 清理已安装目录 (--clean)
# ------------------------------------------------------------------------------
if [ "${CLEAN_ONLY}" = true ]; then
    log_info "正在清理已安装的 Agent Skills 目录..."
    rm -rf "${OPENCODE_GLOBAL_DIR}" "${GEMINI_GLOBAL_DIR}" "${AGENTS_GLOBAL_DIR}"
    log_success "Agent Skills 目录已清理完成。"
    exit 0
fi

# ------------------------------------------------------------------------------
# 2. 准备运行环境 (uv 优先)
# ------------------------------------------------------------------------------
if [ "${CHECK_ONLY}" = false ]; then
    log_info "正在检查 GPU 端 Python / uv 运行环境..."
    if command -v uv >/dev/null 2>&1; then
        if [ "${SKIP_SYNC}" = false ]; then
            log_info "检测到 uv 包管理器，正在同步项目虚拟环境与依赖 (uv sync)..."
            uv sync --quiet || uv sync
            log_success "uv 虚拟环境与依赖同步完成 (.venv)。"
        fi
    elif [ -f "${PROJECT_ROOT}/.venv/bin/python" ]; then
        log_info "检测到项目虚拟环境: ${PROJECT_ROOT}/.venv"
    elif command -v python3 >/dev/null 2>&1; then
        log_warn "未检测到 uv 包管理器，使用系统 python3。建议安装 uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    else
        log_error "未找到可用的 Python 运行环境，请先安装 uv 或 Python 3.10+。"
        exit 1
    fi
fi

# ------------------------------------------------------------------------------
# 3. GPU 端专属环境变量检查与初始化 (.env)
# ------------------------------------------------------------------------------
read_env_val() {
    local key="$1"
    local file="${PROJECT_ROOT}/.env"
    [ -f "${file}" ] || return 0
    grep -E "^[[:space:]]*(export[[:space:]]+)?${key}=" "${file}" 2>/dev/null \
        | tail -n 1 \
        | sed -E "s/^[[:space:]]*(export[[:space:]]+)?${key}=//" \
        | tr -d '\r' \
        | sed -E "s/^[[:space:]]+//;s/[[:space:]]+$//;s/^['\"]//;s/['\"]$//" || true
}

if [ "${CHECK_ONLY}" = false ]; then
    if [ ! -f "${PROJECT_ROOT}/.env" ]; then
        log_info "未检测到 .env 配置文件，正在为 GPU 端创建专属模板..."
        if [ -f "${PROJECT_ROOT}/.env.gpu.example" ]; then
            cp "${PROJECT_ROOT}/.env.gpu.example" "${PROJECT_ROOT}/.env"
        else
            cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
        fi
        log_success "已创建 GPU 端配置文件: ${PROJECT_ROOT}/.env (基于 .env.gpu.example)"
        log_warn "【重要提示】请编辑 .env 文件，将 TRAINPILOT_HOST 设置为 Web 服务器的公网 IP 或域名！"
    fi

    # 检查 GPU 端配置是否误填为 0.0.0.0
    GPU_HOST="$(read_env_val 'TRAINPILOT_HOST')"
    GPU_GATEWAY_URL="$(read_env_val 'TRAINPILOT_GATEWAY_URL')"
    GPU_PORT="$(read_env_val 'TRAINPILOT_PORT')"
    GPU_PORT="${GPU_PORT:-28780}"

    if [ "${GPU_HOST}" = "0.0.0.0" ]; then
        echo -e "${YELLOW}======================================================================${NC}"
        echo -e "${YELLOW}[配置警告] 检测到当前 .env 中的 TRAINPILOT_HOST=0.0.0.0！${NC}"
        echo -e "在 GPU 训练端，TRAINPILOT_HOST 表示要连接的 Web 控制面公网地址，不能填写 0.0.0.0。"
        echo -e "请将其修改为 Web 端云主机的公网 IP 或域名 (例如: TRAINPILOT_HOST=35.202.16.245)。"
        echo -e "${YELLOW}======================================================================${NC}"
    fi

    TARGET_GATEWAY=""
    if [ -n "${GPU_GATEWAY_URL}" ]; then
        TARGET_GATEWAY="${GPU_GATEWAY_URL}"
    elif [ -n "${GPU_HOST}" ] && [ "${GPU_HOST}" != "0.0.0.0" ]; then
        TARGET_GATEWAY="http://${GPU_HOST}:${GPU_PORT}"
    else
        TARGET_GATEWAY="http://127.0.0.1:${GPU_PORT}"
    fi
fi

# ------------------------------------------------------------------------------
# 4. 可选网关连通性测试 (--test)
# ------------------------------------------------------------------------------
if [ "${TEST_CONN}" = true ]; then
    log_info "正在测试与 Web 控制面网关的连通性: ${TARGET_GATEWAY}/health ..."
    if command -v curl >/dev/null 2>&1; then
        HTTP_CODE="$(curl -s -o /dev/null -w "%{http_code}" -m 5 "${TARGET_GATEWAY}/health" 2>/dev/null || echo "000")"
        if [ "${HTTP_CODE}" = "200" ]; then
            log_success "网关连通性测试成功！HTTP 响应码: 200 OK"
        else
            log_error "网关连接失败或响应异常 (HTTP Code: ${HTTP_CODE})，请检查 Web 服务是否已启动且公网端口可达。"
            exit 1
        fi
    else
        log_warn "未检测到 curl，跳过网络测试。"
    fi
    exit 0
fi

# ------------------------------------------------------------------------------
# 5. Agent Skills 生成与全局安装
# ------------------------------------------------------------------------------
TARGETS=(
    "${OPENCODE_GLOBAL_DIR}|# TrainPilot Agent Skill (OpenCode Global Edition)|__TOOL_PATH__"
    "${GEMINI_GLOBAL_DIR}|# TrainPilot Agent Skill (Antigravity/Gemini Edition)|__TOOL_PATH__"
    "${AGENTS_GLOBAL_DIR}|# TrainPilot Agent Skill (Agents Global Edition)|__TOOL_PATH__"
)

if [ "${CHECK_ONLY}" = true ]; then
    log_info "正在检查 Agent Skills 完整性..."
    MISSING=0
    for spec in "${TARGETS[@]}"; do
        dir="${spec%%|*}"
        for f in "${dir}/SKILL.md" "${dir}/scripts/trainpilot_tool.py"; do
            if [ ! -f "$f" ]; then
                log_warn "缺少文件: $f"
                MISSING=$((MISSING + 1))
            fi
        done
    done
    if [ "${MISSING}" -eq 0 ]; then
        log_success "所有 Agent Skills 文件夹及文件均完整就绪。"
        exit 0
    else
        log_error "共检测到 ${MISSING} 个缺失项，请运行 ./setup_skills.sh 重新生成。"
        exit 1
    fi
fi

log_info "正在为 GPU 端生成针对 Agent 上下文优化的精简版 Agent Skills (Compact)..."


generate_compact_skill() {
    local target_file="$1"
    local title="$2"
    local tool_path="$3"

    cat <<EOF > "${target_file}"
---
name: trainpilot
description: Supervise GPU training jobs, report milestones/alerts (NaN, OOM), poll human decisions (Feishu HITL), and execute recovery actions via TrainPilot gateway.
---

${title}

## Overview
TrainPilot bridges isolated GPU training jobs with the central gateway and Feishu HITL interactive cards.
**State Flow**: \`RUNNING\` ➔ (anomaly) ➔ \`WAITING\` ➔ (polled decision) ➔ \`RECOVERING\` ➔ (ack) ➔ \`RUNNING\`.

## Environment & Gateway Config
- **Gateway Resolution**: \`--gateway <URL>\` > \`TRAINPILOT_GATEWAY_URL\` > \`http://\${TRAINPILOT_HOST}:\${TRAINPILOT_PORT:-28780}\` (Default: \`http://127.0.0.1:28780\`).
- **GPU Node**: Set \`TRAINPILOT_HOST=<Web Public IP>\` in \`.env\` (never \`0.0.0.0\`). Set \`TRAINPILOT_TASK_ID=<ID>\` to omit \`--task-id\`.

## CLI Quick Reference (Agent Execution)
Tool path: \`${tool_path}\`

| Task | Command |
|---|---|
| **Report Milestone** | \`python3 ${tool_path} report-milestone --step <N> --epoch <E> --message "<Phase desc>" --metrics "loss=0.35,val_loss=0.42" --agent-note "<1-3 sentence autonomous analysis & recommendation>"\` |
| **Report Anomaly & Alert** | \`python3 ${tool_path} report-alert --step <N> --message "<Error desc, e.g. Loss NaN>" --metrics "loss=NaN"\` |
| **Poll Human Decision** | \`python3 ${tool_path} poll-instruction --wait --interval 2 --wait-timeout 300\` |
| **Acknowledge Recovery** | \`python3 ${tool_path} ack-instruction --action self_resolve --status success --message "Checkpoint reloaded and resumed"\` |
| **Send Heartbeat** | \`python3 ${tool_path} send-heartbeat --step <N> --metrics "gpu_mem=82%"\` |
| **Query Task Status** | \`python3 ${tool_path} get-status\` |
| **Mock Decision (Dev/Test)** | \`python3 ${tool_path} mock-decision --action self_resolve\` |

## Agent Guidelines & Rules
1. **Autonomous \`--agent-note\`**: On \`report-milestone\`, you MUST provide 1-3 sentences of autonomous analysis based on real logs/curves (trend, overfitting risk, advice). Do NOT duplicate \`--message\` or fabricate metrics.
2. **Poll & Pop**: \`poll-instruction\` pops the pending decision and transitions status to \`RECOVERING\`. To check status without popping, use \`get-status\`.
3. **Always ACK**: After executing the recovery strategy (e.g. \`self_resolve\` / reload checkpoint / adjust lr), call \`ack-instruction\` to transition back to \`RUNNING\`.

## Python In-Process Usage (TrainingGuardian)
\`\`\`python
from trainpilot.agent import TrainPilotClient, TrainingGuardian

client = TrainPilotClient(gateway_url="http://control-plane:28780", task_id="qwen2-7b-sft")
guardian = TrainingGuardian(client=client)
guardian.register_action_handler("self_resolve", lambda p: print("Resume"))

# Training loop: auto freeze, Feishu alert, poll decision & ACK on NaN/Inf
guardian.check_and_handle_loss(loss.item(), step=step)
# Report milestone with AI Agent note:
client.notify_milestone(message="Epoch done", step=step, metrics={"loss": 0.35}, agent_note="收敛平稳，建议继续")
\`\`\`
EOF
}

install_skill() {
    local target_dir="$1"
    local title="$2"
    local tool_path="$3"

    if [ "${tool_path}" = "__TOOL_PATH__" ]; then
        tool_path="${target_dir}/scripts/trainpilot_tool.py"
    fi

    mkdir -p "${target_dir}/scripts"
    cp -f "${SOURCE_SKILL_DIR}/scripts/trainpilot_tool.py" "${target_dir}/scripts/trainpilot_tool.py"
    chmod +x "${target_dir}/scripts/trainpilot_tool.py"

    generate_compact_skill "${target_dir}/SKILL.md" "${title}" "${tool_path}"

    echo -e "  - ${BOLD}Skill 目录:${NC} ${target_dir}"
    echo -e "    * 规范定义: ${target_dir}/SKILL.md (精简版, 降低 Agent 上下文开销)"
    echo -e "    * CLI 工具: ${target_dir}/scripts/trainpilot_tool.py"
}

chmod +x "${SOURCE_SKILL_DIR}/scripts/trainpilot_tool.py"

for spec in "${TARGETS[@]}"; do
    dir="${spec%%|*}"
    rest="${spec#*|}"
    title="${rest%%|*}"
    tool_path="${rest#*|}"
    install_skill "${dir}" "${title}" "${tool_path}"
done

echo ""
echo -e "${CYAN}${BOLD}======================================================================${NC}"
echo -e "${GREEN}${BOLD}Agent Skills 自动生成并同步完成！${NC}"
echo -e "${CYAN}----------------------------------------------------------------------${NC}"
echo -e "  - ${BOLD}当前运行环境:${NC}   uv (.venv 虚拟环境)"
echo -e "  - ${BOLD}目标网关地址:${NC}   ${TARGET_GATEWAY}"
echo -e "  - ${BOLD}Agent 常用命令示例 (在训练代码或终端调用):${NC}"
echo -e "    * 上报异常: ${CYAN}uv run python skills/trainpilot/scripts/trainpilot_tool.py report-alert --step 100 --message \"Loss NaN\"${NC}"
echo -e "    * 轮询决策: ${CYAN}uv run python skills/trainpilot/scripts/trainpilot_tool.py poll-instruction --wait${NC}"
echo -e "${CYAN}${BOLD}======================================================================${NC}"
