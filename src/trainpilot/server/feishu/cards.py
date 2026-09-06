"""Feishu interactive and rich-text card builders."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


def _format_metrics(metrics: Optional[Dict[str, Any]]) -> str:
    if not metrics:
        return "*无额外指标*"
    items = []
    for k, v in metrics.items():
        if isinstance(v, float):
            items.append(f"**{k}**: `{v:.6f}`" if abs(v) < 1e-2 or abs(v) > 1e4 else f"**{k}**: `{v:.4f}`")
        else:
            items.append(f"**{k}**: `{v}`")
    return "  |  ".join(items)


def _resolve_agent_note(
    agent_note: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """统一解析 Agent 自主点评。

    优先级: 显式 ``agent_note`` 参数 > ``extra`` 中的
    ``agent_note`` / ``agent_analysis`` / ``agent_comment``。
    空字符串视为未提供, 返回 None 以保持向后兼容。
    """
    if agent_note is not None and str(agent_note).strip():
        return str(agent_note).strip()
    if extra:
        for key in ("agent_note", "agent_analysis", "agent_comment"):
            val = extra.get(key)
            if val is not None and str(val).strip():
                return str(val).strip()
    return None


def build_alert_card(
    task_id: str,
    message: str,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 30,
    agent_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a high-priority interactive alert card for Feishu with HITL action buttons.

    仅保留两个决策按钮：停止训练 (stop_training) 与自行解决 (self_resolve)。
    超过 timeout_seconds 未点击则视为“自行解决”。
    若提供 agent_note，将展示 🤖 Agent 异常研判 区块。
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    resolved_note = _resolve_agent_note(agent_note, extra)

    markdown_lines = [
        f"**🚨 任务标识**: `{task_id}`",
        f"**⏱ 触发进度**: Epoch `{epoch if epoch is not None else '-'}` | Step `{step if step is not None else '-'}`",
        f"**📊 实时指标**: {_format_metrics(metrics)}",
        f"**⚠️ 异常详情**: \n> {message}",
    ]
    if resolved_note:
        markdown_lines.extend([
            f"**🤖 Agent 异常研判**: \n> {resolved_note}",
        ])
    markdown_lines.extend([
        "",
        "---",
        f"**请在下方选择（{timeout_seconds} 秒内未决策将自动视为“自行解决”，训练自行继续）：**",
    ])

    elements: List[Dict[str, Any]] = [
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": "\n".join(markdown_lines),
            },
        },
        {
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "🛑 停止训练"},
                    "type": "danger",
                    "value": {
                        "task_id": task_id,
                        "action": "stop_training",
                    },
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "✅ 自行解决"},
                    "type": "primary",
                    "value": {
                        "task_id": task_id,
                        "action": "self_resolve",
                    },
                },
            ],
        },
        {
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": f"TrainPilot Control Plane • 上报时间: {now_str} • {timeout_seconds}s无决策自动视为自行解决",
                }
            ],
        },
    ]

    return {
        "config": {"wide_screen_mode": True, "enable_forward": True},
        "header": {
            "template": "red",
            "title": {
                "tag": "plain_text",
                "content": f"⚠️【训练异常告警】任务: {task_id}",
            },
        },
        "elements": elements,
    }


def build_milestone_card(
    task_id: str,
    message: str,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    agent_note: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct an informative milestone card (read-only) without buttons.

    ``agent_note`` 为外部 AI 智能体 (agy / opencode / claude-code) 针对当前
    实际情况自主生成的 1-3 句点评 (如收敛趋势、指标解读、风险提示、下一步建议),
    渲染为独立的 ``🤖 Agent 智能点评`` 区块, 与原始 ``📝 阶段描述`` 分开展示。
    为向后兼容, 未提供时也会尝试从 ``extra`` 中解析
    ``agent_note`` / ``agent_analysis`` / ``agent_comment``。
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    resolved_note = _resolve_agent_note(agent_note, extra)

    markdown_lines = [
        f"**🎯 任务标识**: `{task_id}`",
        f"**📈 当前进度**: Epoch `{epoch if epoch is not None else '-'}` | Step `{step if step is not None else '-'}`",
        f"**📊 关键指标**: {_format_metrics(metrics)}",
        f"**📝 阶段描述**: \n> {message}",
    ]
    if resolved_note:
        markdown_lines.extend([
            f"**🤖 Agent 智能点评**: \n> {resolved_note}",
        ])

    elements: List[Dict[str, Any]] = [
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": "\n".join(markdown_lines),
            },
        },
        {
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": f"TrainPilot 静默巡航 • 里程碑记录: {now_str}",
                }
            ],
        },
    ]

    return {
        "config": {"wide_screen_mode": True, "enable_forward": True},
        "header": {
            "template": "green",
            "title": {
                "tag": "plain_text",
                "content": f"🚀【训练里程碑达成】任务: {task_id}",
            },
        },
        "elements": elements,
    }


def build_resolved_card(
    task_id: str,
    action_name: str,
    operator: str = "专家工程师",
    original_message: Optional[str] = None,
    resolved_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a 'Resolved' card replacing the alert card to prevent duplicate clicks."""
    time_str = resolved_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    action_label_map = {
        "stop_training": "🛑 立即终止训练并安全存档",
        "self_resolve": "✅ 自行解决（训练自行继续，无须干预）",
    }
    action_desc = action_label_map.get(action_name, f"⚙️ 自定义动作: `{action_name}`")

    is_auto = "auto" in operator.lower() or "timeout" in operator.lower() or "系统" in operator

    markdown_lines = [
        f"**🎯 任务标识**: `{task_id}`",
        f"**👤 决策执行人**: {operator}",
        f"**⚡️ 选定策略**: **{action_desc}**",
        f"**🕒 闭环时间**: `{time_str}`",
    ]
    if is_auto:
        markdown_lines.append("**⏳ 触发原因**: 超过 30 秒无人工决策，系统自动视为“自行解决”")
    if original_message:
        markdown_lines.extend([
            "",
            f"**📋 原告警说明**: *{original_message}*",
        ])
    if action_name == "self_resolve":
        markdown_lines.append("\n> ✅ 指令已派发至任务信箱，GPU 边缘节点将自动拉取执行并回传解决方法卡片。按钮已停用，防呆机制生效。")
    else:
        markdown_lines.append("\n> ✅ 指令已派发至任务信箱，GPU 边缘节点将自动拉取执行。按钮已停用，防呆机制生效。")

    elements: List[Dict[str, Any]] = [
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": "\n".join(markdown_lines),
            },
        },
        {
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": f"TrainPilot HITL 闭环决策已确认 • {time_str}",
                }
            ],
        },
    ]

    return {
        "config": {"wide_screen_mode": True, "enable_forward": True},
        "header": {
            "template": "turquoise",
            "title": {
                "tag": "plain_text",
                "content": f"✅【决策已闭环】任务: {task_id}",
            },
        },
        "elements": elements,
    }


def build_recovery_card(
    task_id: str,
    solution: str,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct an informative recovery card sent by Agent when self_resolve is executed.

    告知问题解决方法与恢复状态，简单清晰，由 Agent 概括一句解决方法发送回来，
    让用户知道问题已经解决、训练任务恢复正常。
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    markdown_lines = [
        f"**🎯 任务标识**: `{task_id}`",
        f"**⏱ 恢复进度**: Epoch `{epoch if epoch is not None else '-'}` | Step `{step if step is not None else '-'}`",
    ]
    if metrics:
        markdown_lines.append(f"**📊 恢复指标**: {_format_metrics(metrics)}")
    markdown_lines.extend([
        f"**💡 解决方法**: \n> {solution}",
        "",
        "**🟢 运行状态**: ✅ 问题已解决，训练任务已恢复正常运行",
    ])

    elements: List[Dict[str, Any]] = [
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": "\n".join(markdown_lines),
            },
        },
        {
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": f"TrainPilot Agent 智能自愈 • 解决时间: {now_str}",
                }
            ],
        },
    ]

    return {
        "config": {"wide_screen_mode": True, "enable_forward": True},
        "header": {
            "template": "green",
            "title": {
                "tag": "plain_text",
                "content": f"🛠️【异常已自行解决】任务: {task_id}",
            },
        },
        "elements": elements,
    }


def build_stale_alert_card(
    task_id: str,
    silent_seconds: float,
    last_heartbeat_at: Optional[str] = None,
    latest_step: Optional[int] = None,
    latest_epoch: Optional[int] = None,
    latest_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a warning notification card dispatched by server Watchdog on heartbeat timeout."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    markdown_lines = [
        f"**🚨 任务标识**: `{task_id}`",
        f"**⏳ 失联时长**: 已超过 `{int(silent_seconds)}` 秒未收到心跳",
        f"**⏱ 最后进度**: Epoch `{latest_epoch if latest_epoch is not None else '-'}` | Step `{latest_step if latest_step is not None else '-'}`",
        f"**🕒 最后心跳**: `{last_heartbeat_at or '未知'}`",
    ]
    if latest_message:
        markdown_lines.append(f"**📋 最后已知状态**: {latest_message}")

    markdown_lines.extend([
        "",
        "> ⚠️ **服务端看门狗检测到该任务可能已崩溃、硬件断电、或陷入 NCCL 通信死锁**。请运维/算法工程师检查 GPU 服务器日志。",
    ])

    elements: List[Dict[str, Any]] = [
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": "\n".join(markdown_lines),
            },
        },
        {
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": f"TrainPilot 看门狗失联预警 • 检测时间: {now_str}",
                }
            ],
        },
    ]

    return {
        "config": {"wide_screen_mode": True, "enable_forward": True},
        "header": {
            "template": "orange",
            "title": {
                "tag": "plain_text",
                "content": f"⚠️【训练任务失联预警】任务: {task_id}",
            },
        },
        "elements": elements,
    }


