#!/usr/bin/env bash
# ==============================================================================
# TrainPilot Agent Skills 专属生成与同步脚本
#
# 功能：
# 1. 默认在全局位置生成 skills（任意工作目录均可发现）：
#    - OpenCode 官方全局：${XDG_CONFIG_HOME:-$HOME/.config}/opencode/skills/trainpilot
#    - 通用 Agent 全局：$HOME/.agents/skills/trainpilot（OpenCode/Claude/Codex/Antigravity 均可发现）
# 2. 可选保留旧的项目级生成（--project / --all，仅用于兼容）
# 3. 自动配置工具执行路径与可执行权限 (chmod +x)
# 4. 可清理遗留的项目级生成目录 (--clean)
# ==============================================================================

set -eo pipefail

# 终端颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

# 定位当前项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

# 解析命令行参数
CHECK_ONLY=false
CLEAN_ONLY=false
SCOPE="global"  # global | project | all

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
        --global|-g)
            SCOPE="global"
            shift
            ;;
        --project|-p)
            SCOPE="project"
            shift
            ;;
        --all|-a)
            SCOPE="all"
            shift
            ;;
        --help|-h)
            echo "TrainPilot Agent Skills 生成脚本使用说明:"
            echo "  ./setup_skills.sh            生成或同步全局 Agent Skills (默认，推荐)"
            echo "  ./setup_skills.sh --global   同上：仅安装到全局位置"
            echo "  ./setup_skills.sh --project  仅安装到项目级 .opencode/.antigravity (旧行为，兼容用)"
            echo "  ./setup_skills.sh --all      全局 + 项目级同时安装"
            echo "  ./setup_skills.sh --check    仅检查技能目录与文件完整性，不重新生成"
            echo "  ./setup_skills.sh --clean    仅清理遗留的项目级生成目录，不重新生成"
            echo ""
            echo "全局安装位置："
            echo "  - \${XDG_CONFIG_HOME:-\$HOME/.config}/opencode/skills/trainpilot  (OpenCode 官方全局)"
            echo "  - \$HOME/.agents/skills/trainpilot  (通用 Agent 全局)"
            echo "可通过 XDG_CONFIG_HOME / HOME 环境变量覆盖目标根目录（便于测试隔离）。"
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

# 全局目标（尊重 XDG / HOME，便于测试隔离与多用户环境）
CONFIG_BASE="${XDG_CONFIG_HOME:-${HOME}/.config}"
OPENCODE_GLOBAL_DIR="${CONFIG_BASE}/opencode/skills/trainpilot"
AGENTS_GLOBAL_DIR="${HOME}/.agents/skills/trainpilot"

# 项目级目标（旧行为，仅 --project / --all 时使用）
OPENCODE_PROJECT_DIR="${PROJECT_ROOT}/.opencode/skills/trainpilot"
ANTIGRAVITY_PROJECT_DIR="${PROJECT_ROOT}/.antigravity/skills/trainpilot"

if [ ! -d "${SOURCE_SKILL_DIR}" ]; then
    log_error "源 Skills 目录不存在: ${SOURCE_SKILL_DIR}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 清理遗留项目级目录 (--clean)
# ------------------------------------------------------------------------------
if [ "${CLEAN_ONLY}" = true ]; then
    log_info "正在清理遗留的项目级 Agent Skills 目录..."
    rm -rf "${OPENCODE_PROJECT_DIR}" "${ANTIGRAVITY_PROJECT_DIR}"
    # .antigravity 下若只剩空 skills 目录则一并移除（该目录完全由本脚本生成）
    rmdir -p "${PROJECT_ROOT}/.antigravity/skills" 2>/dev/null || true
    log_success "项目级遗留目录已清理（全局安装不受影响）。"
    exit 0
fi

# 按 SCOPE 确定本次目标列表：每个元素为 "目标目录|版本标题|路径占位替换"
TARGETS=()
if [ "${SCOPE}" = "global" ] || [ "${SCOPE}" = "all" ]; then
    TARGETS+=("${OPENCODE_GLOBAL_DIR}|# TrainPilot Agent Skill (OpenCode Global Edition)|__TOOL_PATH__")
    TARGETS+=("${AGENTS_GLOBAL_DIR}|# TrainPilot Agent Skill (Agents Global Edition)|__TOOL_PATH__")
fi
if [ "${SCOPE}" = "project" ] || [ "${SCOPE}" = "all" ]; then
    TARGETS+=("${OPENCODE_PROJECT_DIR}|# TrainPilot Agent Skill (OpenCode Edition)|.opencode/skills/trainpilot/scripts/trainpilot_tool.py")
    TARGETS+=("${ANTIGRAVITY_PROJECT_DIR}|# TrainPilot Agent Skill (Antigravity Edition)|.antigravity/skills/trainpilot/scripts/trainpilot_tool.py")
fi

if [ "${CHECK_ONLY}" = true ]; then
    log_info "正在检查 Agent Skills 完整性 (scope: ${SCOPE})..."
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

log_info "正在生成 / 同步 Agent Skills (scope: ${SCOPE})..."

install_skill() {
    local target_dir="$1"
    local title="$2"
    local tool_path="$3"

    # 全局安装使用绝对路径（任意工作目录均可直接调用）；项目级保持相对路径占位
    if [ "${tool_path}" = "__TOOL_PATH__" ]; then
        tool_path="${target_dir}/scripts/trainpilot_tool.py"
    fi

    mkdir -p "${target_dir}/scripts"
    cp -f "${SOURCE_SKILL_DIR}/scripts/trainpilot_tool.py" "${target_dir}/scripts/trainpilot_tool.py"
    chmod +x "${target_dir}/scripts/trainpilot_tool.py"

    sed -e "s|skills/trainpilot/scripts/trainpilot_tool.py|${tool_path}|g" \
        -e "s|# TrainPilot Agent Skill|${title}|g" \
        "${SOURCE_SKILL_DIR}/SKILL.md" > "${target_dir}/SKILL.md"

    echo -e "  - ${BOLD}Skill 目录:${NC} ${target_dir}"
    echo -e "    * 定义文件: ${target_dir}/SKILL.md"
    echo -e "    * 可执行工具: ${target_dir}/scripts/trainpilot_tool.py"
}

chmod +x "${SOURCE_SKILL_DIR}/scripts/trainpilot_tool.py"

for spec in "${TARGETS[@]}"; do
    dir="${spec%%|*}"
    rest="${spec#*|}"
    title="${rest%%|*}"
    tool_path="${rest#*|}"
    install_skill "${dir}" "${title}" "${tool_path}"
done

log_success "Agent Skills 自动生成并同步完成！"
