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


def build_alert_card(
    task_id: str,
    message: str,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct a high-priority interactive alert card for Feishu with HITL action buttons."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    markdown_lines = [
        f"**🚨 任务标识**: `{task_id}`",
        f"**⏱ 触发进度**: Epoch `{epoch if epoch is not None else '-'}` | Step `{step if step is not None else '-'}`",
        f"**📊 实时指标**: {_format_metrics(metrics)}",
        f"**⚠️ 异常详情**: \n> {message}",
        "",
        "---",
        "**请在下方选择恢复策略（点击后将立即下发至 GPU 节点并冻结重复操作）：**",
    ]

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
                    "text": {"tag": "plain_text", "content": "📉 降LR并回滚"},
                    "type": "danger",
                    "value": {
                        "task_id": task_id,
                        "action": "reduce_lr_rollback",
                    },
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "⏭ 跳过Batch"},
                    "type": "primary",
                    "value": {
                        "task_id": task_id,
                        "action": "skip_batch",
                    },
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "▶ 忽略并继续"},
                    "type": "default",
                    "value": {
                        "task_id": task_id,
                        "action": "resume",
                    },
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "🛑 终止训练"},
                    "type": "danger",
                    "value": {
                        "task_id": task_id,
                        "action": "stop_training",
                    },
                },
            ],
        },
        {
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": f"TrainPilot Control Plane • 上报时间: {now_str}",
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
) -> Dict[str, Any]:
    """Construct an informative milestone card (read-only) without buttons."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    markdown_lines = [
        f"**🎯 任务标识**: `{task_id}`",
        f"**📈 当前进度**: Epoch `{epoch if epoch is not None else '-'}` | Step `{step if step is not None else '-'}`",
        f"**📊 关键指标**: {_format_metrics(metrics)}",
        f"**📝 阶段描述**: \n> {message}",
    ]

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
        "reduce_lr_rollback": "📉 降低学习率并回滚上一检查点",
        "skip_batch": "⏭ 跳过当前异常 Batch 并继续",
        "resume": "▶ 忽略告警继续迭代",
        "stop_training": "🛑 立即终止训练并安全存档",
    }
    action_desc = action_label_map.get(action_name, f"⚙️ 自定义动作: `{action_name}`")

    markdown_lines = [
        f"**🎯 任务标识**: `{task_id}`",
        f"**👤 决策执行人**: {operator}",
        f"**⚡️ 选定策略**: **{action_desc}**",
        f"**🕒 闭环时间**: `{time_str}`",
    ]
    if original_message:
        markdown_lines.extend([
            "",
            f"**📋 原告警说明**: *{original_message}*",
        ])
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
