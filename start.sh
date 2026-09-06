#!/usr/bin/env bash
# ==============================================================================
# TrainPilot Control Plane Gateway Web 服务启动脚本 (仅 Web 侧使用)
#
# 职责 (v0.2+ 精简版):
# 1. 纯 Web 服务启停: 前台 / 后台守护进程 / 状态检查 / 一键停止, 不再触碰 Skills
# 2. 自动配置小众高位端口 (默认 28780), 进行智能端口防冲突检测与指引
# 3. Web 绑定地址解析: --host > $TRAINPILOT_BIND_HOST > $TRAINPILOT_HOST(仅本地值有效) > 0.0.0.0
# 4. 自动同步/检查 Python 虚拟环境与依赖 (优先使用 .venv / uv)
#
# 环境变量 Web/GPU 区分约定:
# - Web 服务器 (.env 在本机): TRAINPILOT_HOST=0.0.0.0 (绑定地址), TRAINPILOT_PORT=28780
#   如需覆盖绑定可单独设置 TRAINPILOT_BIND_HOST (优先级更高)。
# - GPU 服务器 (训练内网机): TRAINPILOT_HOST=<Web公网IP/域名> (如 35.202.16.245),
#   或直接设置完整 TRAINPILOT_GATEWAY_URL=http://<Web公网IP>:28780。
#   两台机器不可共用同一 .env, 必须分别配置 TRAINPILOT_HOST。
#
# Skills 说明: 本脚本不再创建/同步任何 Skill。
# GPU 侧如需 Agent Skill, 请在 GPU 机器上手动执行一次 ./setup_skills.sh (独立脚本)。
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
DEFAULT_BIND_HOST="0.0.0.0"
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

# 从 .env 文件读取单个键 (忽略注释与前后空格, 兼容 export 前缀与引号)
read_env_file_key() {
    local key="$1"
    local file="${PROJECT_ROOT}/.env"
    [ -f "${file}" ] || return 0
    grep -E "^[[:space:]]*(export[[:space:]]+)?${key}=" "${file}" 2>/dev/null \
        | tail -n 1 \
        | sed -E "s/^[[:space:]]*(export[[:space:]]+)?${key}=//" \
        | tr -d '\r' \
        | sed -E "s/^[[:space:]]+//;s/[[:space:]]+$//;s/^['\"]//;s/['\"]$//" || true
}

# 获取配置端口: --port > $TRAINPILOT_PORT > .env > 默认
resolve_port() {
    if [ -n "${CUSTOM_PORT}" ]; then
        echo "${CUSTOM_PORT}"
    elif [ -n "${TRAINPILOT_PORT}" ]; then
        echo "${TRAINPILOT_PORT}"
    else
        local env_port
        env_port="$(read_env_file_key 'TRAINPILOT_PORT')"
        if [ -n "${env_port}" ]; then
            echo "${env_port}"
        else
            echo "${DEFAULT_PORT}"
        fi
    fi
}

# 获取 Web 绑定地址: --host > $TRAINPILOT_BIND_HOST > $TRAINPILOT_HOST(仅本地值) > 0.0.0.0
# 即使误把 GPU 侧公网 IP 拷贝到 Web 侧, 也回退到 0.0.0.0 避免 bind 失败。
resolve_bind_host() {
    if [ -n "${CUSTOM_HOST}" ]; then
        echo "${CUSTOM_HOST}"
        return
    fi
    if [ -n "${TRAINPILOT_BIND_HOST}" ]; then
        echo "${TRAINPILOT_BIND_HOST}"
        return
    fi
    local host_candidate="${TRAINPILOT_HOST}"
    if [ -z "${host_candidate}" ]; then
        host_candidate="$(read_env_file_key 'TRAINPILOT_HOST')"
    fi
    if [ -z "${host_candidate}" ]; then
        host_candidate="$(read_env_file_key 'TRAINPILOT_BIND_HOST')"
    fi
    case "${host_candidate}" in
        ""|"0.0.0.0"|"::"|"127.0.0.1"|"localhost")
            if [ -n "${host_candidate}" ]; then
                echo "${host_candidate}"
            else
                echo "${DEFAULT_BIND_HOST}"
            fi
            ;;
        *)
            echo "${DEFAULT_BIND_HOST}"
            ;;
    esac
}

# 对外展示用 Host: 0.0.0.0/空 -> 127.0.0.1
display_host() {
    local h="$1"
    case "${h}" in
        ""|"0.0.0.0"|"::") echo "127.0.0.1" ;;
        *) echo "${h}" ;;
    esac
}

# 解析命令行参数
RUN_MODE="foreground"
CUSTOM_PORT=""
CUSTOM_HOST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --daemon|-d)
            RUN_MODE="daemon"
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
        --port|-p)
            CUSTOM_PORT="$2"
            shift 2
            ;;
        --host|--bind|-H)
            CUSTOM_HOST="$2"
            shift 2
            ;;
        --help|-h)
            echo "TrainPilot Web 服务启动脚本使用说明 (仅 Web 侧, 不创建 Skills):"
            echo "  ./start.sh                 前台交互式启动控制面网关 (默认)"
            echo "  ./start.sh --daemon, -d    后台守护进程模式启动"
            echo "  ./start.sh --stop          停止后台运行的网关进程"
            echo "  ./start.sh --status        查看网关运行状态及健康检查"
            echo "  ./start.sh -p <PORT>       临时指定监听端口 (默认: 28780)"
            echo "  ./start.sh --host <HOST>   临时指定绑定地址 (默认: 0.0.0.0)"
            echo ""
            echo "环境变量 (Web 侧 .env):"
            echo "  TRAINPILOT_BIND_HOST       Web 绑定地址 (最高优先级, 默认 0.0.0.0)"
            echo "  TRAINPILOT_HOST            Web 绑定地址 (兼容旧配置, 仅本地值有效; GPU 侧含义不同)"
            echo "  TRAINPILOT_PORT            监听端口 (默认 28780)"
            echo ""
            echo "GPU 侧请勿使用本脚本, 只需配置:"
            echo "  TRAINPILOT_HOST=<Web公网IP/域名> 或 TRAINPILOT_GATEWAY_URL=http://<Web公网IP>:28780"
            echo "  并按需手动执行 ./setup_skills.sh 安装 Agent Skill (独立脚本)。"
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
# 步骤 1: 环境配置与端口冲突防范 (纯 Web, 不触碰 Skills)
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
    log_info "未检测到 .env 文件，正在为 Web 控制面自动初始化配置..."
    if [ -f "${PROJECT_ROOT}/.env.web.example" ]; then
        cp "${PROJECT_ROOT}/.env.web.example" "${PROJECT_ROOT}/.env"
    else
        cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
    fi
    sed -i "s/^TRAINPILOT_PORT=.*/TRAINPILOT_PORT=${DEFAULT_PORT}/" "${PROJECT_ROOT}/.env" 2>/dev/null || true
    log_success "Web 控制面专属 .env 文件已创建并设定 TRAINPILOT_PORT=${DEFAULT_PORT}。"
fi

# 2. 获取最终端口与绑定地址
TARGET_PORT="$(resolve_port)"
TARGET_BIND_HOST="$(resolve_bind_host)"
TARGET_DISPLAY_HOST="$(display_host "${TARGET_BIND_HOST}")"

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

# 导出环境变量供 Python 代码和配置读取 (仅 PORT/BIND, 不再伪造 GATEWAY_URL)
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TRAINPILOT_PORT="${TARGET_PORT}"
export TRAINPILOT_BIND_HOST="${TARGET_BIND_HOST}"

# ------------------------------------------------------------------------------
# 步骤 2: 依赖环境就绪检查 (优先使用 .venv 与 uv)
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
# 步骤 3: 启动服务 (前台 / 后台守护进程, 纯 Web)
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
echo -e "${BOLD}TrainPilot Control Plane Gateway 启动信息 (Web 侧, 不创建 Skills):${NC}"
echo -e "  - ${CYAN}绑定地址:${NC}      ${GREEN}${TARGET_BIND_HOST}${NC}"
echo -e "  - ${CYAN}监听端口:${NC}      ${GREEN}${TARGET_PORT}${NC} (已采用小众高位端口规避冲突)"
echo -e "  - ${CYAN}网关服务地址:${NC}  http://${TARGET_BIND_HOST}:${TARGET_PORT}"
echo -e "  - ${CYAN}API 交互文档:${NC}  http://${TARGET_DISPLAY_HOST}:${TARGET_PORT}/docs"
echo -e "  - ${CYAN}健康检查接口:${NC}  http://${TARGET_DISPLAY_HOST}:${TARGET_PORT}/health"
echo -e "  - ${CYAN}运行模式:${NC}      ${RUN_MODE}"
echo -e "  - ${CYAN}Skills 说明:${NC}   本脚本不再同步 Skills；GPU 侧请手动执行 ./setup_skills.sh"
echo "----------------------------------------------------------------------"

if [ "${RUN_MODE}" = "daemon" ]; then
    log_info "正在后台启动网关服务..."
    if command -v setsid >/dev/null 2>&1; then
        setsid ${PYTHON_CMD} -m uvicorn trainpilot.server.main:app --host "${TARGET_BIND_HOST}" --port "${TARGET_PORT}" </dev/null >> "${LOG_FILE}" 2>&1 &
    else
        nohup ${PYTHON_CMD} -m uvicorn trainpilot.server.main:app --host "${TARGET_BIND_HOST}" --port "${TARGET_PORT}" </dev/null >> "${LOG_FILE}" 2>&1 &
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
    exec ${PYTHON_CMD} -m uvicorn trainpilot.server.main:app --host "${TARGET_BIND_HOST}" --port "${TARGET_PORT}"
fi
