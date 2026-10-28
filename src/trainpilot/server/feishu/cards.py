"""Feishu interactive and rich-text card builders."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from trainpilot.common.states import EventType, WAIT_FOR_ARRANGEMENT


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
    """Construct a high-priority interactive alert card with HITL action buttons.

    共三个按钮：取消任务并关机 (cancel)、自行解决 (self_resolve)、等待用户
    (wait_for_arrangement)。「等待用户」不产生指令，也不再有兜底定时器：任务保持
    WAITING，Agent 停止轮询并把控制权交回对话，由工程师直接告知 Agent 如何处理。
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
        "**🛑 取消任务并关机**：由 Agent 取消本次运行并验证关机；共用机器上会回报失败，避免误关。",
        "**需要直接与 Agent 讨论？**点「⏳ 等待用户」暂停倒计时并结束轮询，"
        "随后在对话中直接告诉 Agent 你的处理方案；该选择没有自动兜底。",
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
                    "text": {"tag": "plain_text", "content": "🛑 取消任务并关机"},
                    "type": "danger",
                    "value": {
                        "task_id": task_id,
                        "action": "cancel",
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
                {
                    # Not a decision: hands the incident to a direct conversation and
                    # stops the agent's poll loop. Never delivered to the mailbox.
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "⏳ 等待用户"},
                    "type": "default",
                    "value": {
                        "task_id": task_id,
                        "action": WAIT_FOR_ARRANGEMENT,
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

    ``agent_note`` 为外部 AI 智能体针对当前
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
    trigger_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a 'Resolved' card replacing the alert card to prevent duplicate clicks.

    Both `cancel` and the legacy `stop_training` wire name render the cancel wording.
    """
    time_str = resolved_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    action_label_map = {
        "cancel": "🛑 取消任务并关机（由 Agent 执行并验证）",
        "stop_training": "🛑 取消任务并关机（由 Agent 执行并验证）",
        "self_resolve": "✅ 自行解决（训练自行继续，无须干预）",
        "custom": "🛠️ 按人工指定的方案处理",
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
        markdown_lines.append(
            f"**⏳ 触发原因**: {trigger_note or '超过 30 秒无人工决策，系统自动视为“自行解决”'}"
        )
    if original_message:
        markdown_lines.extend([
            "",
            f"**📋 原告警说明**: *{original_message}*",
        ])
    if action_name in ("cancel", "stop_training"):
        markdown_lines.append(
            "\n> ✅ 指令已派发至任务信箱。取消与关机只由 Agent 执行并验证；若机器被多个任务"
            "共享而无法安全关机，Agent 将以 failed ACK 回报。按钮已停用。"
        )
    elif action_name == "self_resolve":
        markdown_lines.append(
            "\n> ✅ 指令已派发至任务信箱，GPU 边缘节点将自动拉取执行并以 ACK 回传唯一的修复汇总。"
        )
    else:
        markdown_lines.append("\n> ✅ 指令已派发至任务信箱，GPU 边缘节点将自动拉取执行。按钮已停用。")

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
    """The single post-repair report for a successful self_resolve/custom ACK.

    告知问题解决方法与恢复状态；Agent 不需要再补发一条里程碑来做“恢复汇报”。
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


def build_outcome_card(
    task_id: str,
    event_type: EventType,
    message: str,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    agent_note: Optional[str] = None,
) -> Dict[str, Any]:
    """One terminal notification for completed/failed (and a verified cancel ACK).

    ``report(completed)`` and ``report(failed)`` each send exactly this card once, so
    no follow-up report is needed. Cancellation reuses the failed outcome and must
    never be rendered as "training recovered".
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    is_completed = event_type == EventType.COMPLETED

    header_title = (
        f"🏁【训练任务已完成】任务: {task_id}"
        if is_completed
        else f"⛔【训练任务已失败/取消】任务: {task_id}"
    )
    template = "green" if is_completed else "red"
    resolved_note = _resolve_agent_note(agent_note)

    markdown_lines = [
        f"**🎯 任务标识**: `{task_id}`",
        f"**📈 结束进度**: Epoch `{epoch if epoch is not None else '-'}` | Step `{step if step is not None else '-'}`",
        f"**📊 最终指标**: {_format_metrics(metrics)}",
        f"**📝 结束说明**: \n> {message}",
    ]
    if resolved_note:
        markdown_lines.extend([
            f"**🤖 Agent 点评**: \n> {resolved_note}",
        ])
    markdown_lines.extend([
        "",
        f"**{'🟢' if is_completed else '🔴'} 终态**: {'训练已正常结束' if is_completed else '任务已结束，不再继续训练'}",
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
                    "content": f"TrainPilot 终态通知（仅此一条） • {now_str}",
                }
            ],
        },
    ]

    return {
        "config": {"wide_screen_mode": True, "enable_forward": True},
        "header": {
            "template": template,
            "title": {"tag": "plain_text", "content": header_title},
        },
        "elements": elements,
    }


def build_awaiting_user_card(
    task_id: str,
    operator: Optional[str] = None,
    requested_at: Optional[str] = None,
    original_message: Optional[str] = None,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
) -> Dict[str, Any]:
    """Status card replacing the alert card after 「⏳ 等待用户」.

    The click replaces the alert card in place, so it repeats the anomaly context.
    There is no fallback timer and no group-message ingestion: the engineer talks to
    the agent directly, and the agent returns control when its next poll reports the
    wait. The card must not claim the agent already observed the click.
    """
    time_str = requested_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    markdown_lines = [
        f"**🎯 任务标识**: `{task_id}`",
        f"**👤 操作人**: {operator or '飞书工程师'}",
        f"**🕒 暂停时间**: `{time_str}`",
        f"**⏭ 触发进度**: Epoch `{epoch if epoch is not None else '-'}` | Step `{step if step is not None else '-'}`",
    ]
    if original_message:
        markdown_lines.extend([
            "",
            f"**📋 原告警说明**: *{original_message}*",
        ])
    markdown_lines.extend([
        "",
        "---",
        "**⏸ 自动决策已暂停，且没有自动兜底。**",
        "**👉 请回到与 Agent 的直接对话提供处理方案；Agent 下次轮询将收到等待状态并交还控制权。**",
        "**✅ 处理完成后，Agent 会以 ACK 回传唯一的修复汇总。**",
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
                    "content": f"TrainPilot 等待直接沟通 • {time_str}",
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
                "content": f"⏳【等待与 Agent 直接沟通】任务: {task_id}",
            },
        },
        "elements": elements,
    }
