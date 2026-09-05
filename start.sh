#!/usr/bin/env bash
# ==============================================================================
# TrainPilot Control Plane Gateway Web 服务一键启动脚本
# 
# 功能特性：
# 1. 自动配置小众高位端口（默认 28780），进行智能端口防冲突检测与指引
# 2. 自动调用 ./setup_skills.sh 同步全局 Agent Skills（~/.config/opencode 与 ~/.agents）
# 3. 自动同步/检查 Python 虚拟环境与依赖 (优先使用 .venv / uv)
# 4. 支持前台交互运行、后台守护进程模式、状态检查与一键停止
# ==============================================================================

set -eo pipefail

# 终端颜色定义
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

# 1. 严格定位当前项目根目录（无论在何处调用本脚本）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

# 2. 默认小众端口与关键文件路径（使用 28780 规避 8000、8080 等高频冲突）
DEFAULT_PORT=28780
PID_FILE="${PROJECT_ROOT}/.trainpilot.pid"
LOG_FILE="${PROJECT_ROOT}/trainpilot.log"

# 3. 端口可用性检测函数 (跨平台 Python Socket 检测)
check_port_available() {
    local port="$1"
    python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', int(${port})))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

# 获取配置端口
resolve_port() {
    if [ -n "${CUSTOM_PORT}" ]; then
        echo "${CUSTOM_PORT}"
    elif [ -n "${TRAINPILOT_PORT}" ]; then
        echo "${TRAINPILOT_PORT}"
    elif [ -f "${PROJECT_ROOT}/.env" ]; then
        local env_port
        env_port="$(grep -E '^TRAINPILOT_PORT=' "${PROJECT_ROOT}/.env" | cut -d '=' -f2 | tr -d ' \r\n' || true)"
        if [ -n "${env_port}" ]; then
            echo "${env_port}"
        else
            echo "${DEFAULT_PORT}"
        fi
    else
        echo "${DEFAULT_PORT}"
    fi
}

# 解析命令行参数 (默认后台守护进程，避免 Ctrl+C 退出)
RUN_MODE="daemon"
CUSTOM_PORT=""
SKIP_SKILLS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --daemon|-d)
            RUN_MODE="daemon"
            shift
            ;;
        --foreground|--fg|-f)
            RUN_MODE="foreground"
            shift
            ;;
        --stop)
            RUN_MODE="stop"
            shift
            ;;
        --status)
            RUN_MODE="status"
            shift
            ;;
        --skip-skills)
            SKIP_SKILLS=true
            shift
            ;;
        --port|-p)
            CUSTOM_PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "TrainPilot Web 服务启动脚本使用说明:"
            echo "  ./start.sh                 后台守护进程模式启动 (默认，Ctrl+C 不退出，自动同步全局 skills)"
            echo "  ./start.sh --foreground, -f  前台交互式启动控制面网关 (按 Ctrl+C 退出)"
            echo "  ./start.sh --daemon, -d    后台守护进程模式启动 (同默认，保持兼容)"
            echo "  ./start.sh --stop          停止后台运行的网关进程"
            echo "  ./start.sh --status        查看网关运行状态及健康检查"
            echo "  ./start.sh --skip-skills   启动时跳过自动执行 ./setup_skills.sh"
            echo "  ./start.sh -p <PORT>       临时指定监听端口 (默认: 28780)"
            echo ""
            echo "独立 Skills 管理脚本:"
            echo "  ./setup_skills.sh          生成或同步全局 Agent Skills（默认，~/.config/opencode 与 ~/.agents）"
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            echo "使用 ./start.sh --help 查看可用参数"
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------------------------
# 停止后台服务处理 (--stop)
# ------------------------------------------------------------------------------
if [ "${RUN_MODE}" = "stop" ]; then
    CURRENT_PORT="$(resolve_port)"
    STOPPED=false

    if [ -f "${PID_FILE}" ]; then
        PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
        if [ -n "${PID}" ] && kill -0 "${PID}" 2>/dev/null; then
            log_info "正在停止 TrainPilot Gateway (PID: ${PID})..."
            kill "${PID}" 2>/dev/null || true
            for _ in {1..20}; do
                if ! kill -0 "${PID}" 2>/dev/null; then
                    break
                fi
                sleep 0.2
            done
            if kill -0 "${PID}" 2>/dev/null; then
                kill -9 "${PID}" 2>/dev/null || true
            fi
            rm -f "${PID_FILE}"
            log_success "TrainPilot Gateway (PID: ${PID}) 已成功停止。"
            STOPPED=true
        else
            rm -f "${PID_FILE}"
        fi
    fi

    if [ "${STOPPED}" = false ]; then
        if ! check_port_available "${CURRENT_PORT}"; then
            log_warn "端口 ${CURRENT_PORT} 仍在被占用，但未发现有效 PID 文件。"
            if command -v fuser >/dev/null 2>&1; then
                fuser -k "${CURRENT_PORT}/tcp" 2>/dev/null || true
                log_success "已释放端口 ${CURRENT_PORT} 上的残留进程。"
            fi
        else
            log_info "TrainPilot Gateway 当前未在运行。"
        fi
    fi
    exit 0
fi

# ------------------------------------------------------------------------------
# 查看状态处理 (--status)
# ------------------------------------------------------------------------------
if [ "${RUN_MODE}" = "status" ]; then
    CURRENT_PORT="$(resolve_port)"
    PORT_IN_USE=false
    if ! check_port_available "${CURRENT_PORT}"; then
        PORT_IN_USE=true
    fi

    PID=""
    if [ -f "${PID_FILE}" ]; then
        PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
    fi

    if [ "${PORT_IN_USE}" = true ]; then
        log_success "TrainPilot Gateway 正在运行！(监听端口: ${CURRENT_PORT}${PID:+, PID: $PID})"
        if command -v curl >/dev/null 2>&1; then
            HEALTH_JSON="$(curl -s -m 2 "http://127.0.0.1:${CURRENT_PORT}/health" 2>/dev/null || true)"
            if [ -n "${HEALTH_JSON}" ]; then
                echo -e "  - 健康检查: ${GREEN}${HEALTH_JSON}${NC}"
            else
                echo -e "  - 健康检查: ${YELLOW}端口响应中，等待应用初始化完成${NC}"
            fi
        fi
        exit 0
    else
        if [ -n "${PID}" ] && kill -0 "${PID}" 2>/dev/null; then
            log_warn "进程 (PID: ${PID}) 存活但端口 ${CURRENT_PORT} 暂未就绪。"
            exit 0
        fi
        log_info "TrainPilot Gateway 未运行 (端口 ${CURRENT_PORT} 空闲)。"
        exit 0
    fi
fi

# ------------------------------------------------------------------------------
# 步骤 1: 调用独立的 setup_skills.sh 生成/同步 Agent Skills
# ------------------------------------------------------------------------------
if [ "${SKIP_SKILLS}" = false ]; then
    if [ -f "${PROJECT_ROOT}/setup_skills.sh" ]; then
        log_info "正在通过 ./setup_skills.sh 同步 Agent Skills..."
        bash "${PROJECT_ROOT}/setup_skills.sh"
    else
        log_warn "未找到 ./setup_skills.sh，跳过 Agent Skills 自动同步。"
    fi
else
    log_info "已根据参数 --skip-skills 跳过 Agent Skills 同步。"
fi

# ------------------------------------------------------------------------------
# 步骤 2: 环境配置与端口冲突防范
# ------------------------------------------------------------------------------
# 自动清理失效的旧 PID 文件
if [ -f "${PID_FILE}" ]; then
    OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [ -n "${OLD_PID}" ] && ! kill -0 "${OLD_PID}" 2>/dev/null; then
        rm -f "${PID_FILE}"
    fi
fi

# 1. 检查并初始化 .env
if [ ! -f "${PROJECT_ROOT}/.env" ]; then
    log_info "未检测到 .env 文件，正在从 .env.example 自动生成并配置小众端口..."
    cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
    sed -i "s/^TRAINPILOT_PORT=.*/TRAINPILOT_PORT=${DEFAULT_PORT}/" "${PROJECT_ROOT}/.env"
    log_success ".env 文件已创建并设定 TRAINPILOT_PORT=${DEFAULT_PORT}。"
fi

# 2. 获取最终端口
TARGET_PORT="$(resolve_port)"

# 3. 检查端口是否被占用
if ! check_port_available "${TARGET_PORT}"; then
    log_error "端口 ${TARGET_PORT} 已被占用！"
    # 尝试查找占用者
    if command -v ss >/dev/null 2>&1; then
        OCCUPIER="$(ss -lptn "sport = :${TARGET_PORT}" 2>/dev/null | tail -n +2 || true)"
        if [ -n "${OCCUPIER}" ]; then
            echo -e "${YELLOW}占用端口的信息如下:${NC}\n${OCCUPIER}"
        fi
    elif command -v lsof >/dev/null 2>&1; then
        OCCUPIER="$(lsof -i ":${TARGET_PORT}" 2>/dev/null || true)"
        if [ -n "${OCCUPIER}" ]; then
            echo -e "${YELLOW}占用端口的信息如下:${NC}\n${OCCUPIER}"
        fi
    fi
    echo ""
    echo "解决方案："
    echo "  1. 停止占用该端口的旧进程（例如: ./start.sh --stop 或 kill <PID>）"
    echo "  2. 使用自定义端口启动: ./start.sh -p <其它端口号>"
    echo "  3. 修改 .env 文件中的 TRAINPILOT_PORT"
    exit 1
fi

# 导出环境变量供 Python 代码和配置读取
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TRAINPILOT_PORT="${TARGET_PORT}"
export TRAINPILOT_GATEWAY_URL="http://127.0.0.1:${TARGET_PORT}"

# ------------------------------------------------------------------------------
# 步骤 3: 依赖环境就绪检查 (优先使用 .venv 与 uv)
# ------------------------------------------------------------------------------
log_info "正在验证 Python 运行环境..."

export PYTHONUNBUFFERED=1

PYTHON_CMD=""
if [ -f "${PROJECT_ROOT}/.venv/bin/python" ]; then
    log_info "使用项目虚拟环境: ${PROJECT_ROOT}/.venv/bin/python"
    PYTHON_CMD="${PROJECT_ROOT}/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
    log_info "检测到 uv 包管理器，正在初始化虚拟环境..."
    uv sync
    PYTHON_CMD="${PROJECT_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    log_warn "未检测到 uv 或 .venv，回退使用系统 python3..."
    PYTHON_CMD="python3"
else
    log_error "未找到可用的 Python 运行环境，请先安装 Python 3.10+ 或 uv。"
    exit 1
fi

# ------------------------------------------------------------------------------
# 步骤 4: 启动服务 (前台 / 后台守护进程)
# ------------------------------------------------------------------------------
echo -e "${CYAN}${BOLD}"
cat << 'EOF'
======================================================================
  ______           _       _____  _ _       _   
 |_   _|         (_)     |  __ \(_) |     | |  
   | |_ __ __ _ _ _ __   | |__) |_| | ___ | |_ 
   | | '__/ _` | | '_ \  |  ___/| | |/ _ \| __|
   | | | | (_| | | | | | | |    | | | (_) | |_ 
   \_/_|  \__,_|_|_| |_| |_|    |_|_|\___/ \__|
======================================================================
EOF
echo -e "${NC}"
echo -e "${BOLD}TrainPilot Control Plane Gateway 启动信息:${NC}"
echo -e "  - ${CYAN}监听端口:${NC}      ${GREEN}${TARGET_PORT}${NC} (已采用小众高位端口规避冲突)"
echo -e "  - ${CYAN}网关服务地址:${NC}  http://0.0.0.0:${TARGET_PORT}"
echo -e "  - ${CYAN}API 交互文档:${NC}  http://127.0.0.1:${TARGET_PORT}/docs"
echo -e "  - ${CYAN}健康检查接口:${NC}  http://127.0.0.1:${TARGET_PORT}/health"
echo -e "  - ${CYAN}OpenCode Skill:${NC}   ~/.config/opencode/skills/trainpilot"
echo -e "  - ${CYAN}Agents Skill:${NC}    ~/.agents/skills/trainpilot"
echo -e "  - ${CYAN}运行模式:${NC}      ${RUN_MODE}"
echo "----------------------------------------------------------------------"

if [ "${RUN_MODE}" = "daemon" ]; then
    log_info "正在后台启动网关服务..."
    if command -v setsid >/dev/null 2>&1; then
        setsid ${PYTHON_CMD} -m uvicorn trainpilot.server.main:app --host 0.0.0.0 --port "${TARGET_PORT}" </dev/null >> "${LOG_FILE}" 2>&1 &
    else
        nohup ${PYTHON_CMD} -m uvicorn trainpilot.server.main:app --host 0.0.0.0 --port "${TARGET_PORT}" </dev/null >> "${LOG_FILE}" 2>&1 &
    fi
    NEW_PID=$!
    disown "${NEW_PID}" 2>/dev/null || true
    echo "${NEW_PID}" > "${PID_FILE}"
    
    # 稍作等待并检查进程是否存活
    sleep 1.5
    if kill -0 "${NEW_PID}" 2>/dev/null; then
        log_success "TrainPilot Gateway 已成功在后台启动 (PID: ${NEW_PID})！"
        echo -e "  - 日志文件: ${LOG_FILE}"
        echo -e "  - 停止命令: ./start.sh --stop"
        echo -e "  - 状态命令: ./start.sh --status"
    else
        log_error "服务在启动时退出，请查看日志: ${LOG_FILE}"
        rm -f "${PID_FILE}"
        exit 1
    fi
else
    log_info "正在前台启动网关服务 (按 Ctrl+C 退出)..."
    exec ${PYTHON_CMD} -m uvicorn trainpilot.server.main:app --host 0.0.0.0 --port "${TARGET_PORT}"
fi
