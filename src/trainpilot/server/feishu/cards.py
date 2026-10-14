"""Feishu interactive and rich-text card builders."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from trainpilot.common.states import WAIT_FOR_ARRANGEMENT


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

    共三个按钮：停止训练 (stop_training)、自行解决 (self_resolve)、等待安排
    (wait_for_arrangement)。前两个是下发给 Agent 的决策；「等待安排」只暂停自动决策，
    等工程师在群里用 `@机器人 #fix(任务名) <方案>` 给出处理方案。
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
        f"**需要时间排查？**点「⏳ 等待安排」暂停倒计时，然后在群里发"
        f"`@机器人 #fix({task_id}) 你的处理方案`。",
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
                {
                    # Not a decision: pauses the countdown so a human can write a fix.
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "⏳ 等待安排"},
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
    trigger_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a 'Resolved' card replacing the alert card to prevent duplicate clicks.

    trigger_note replaces the built-in "超过 30 秒无人工决策" line, which would be wrong
    for the arrangement fallback (a much longer window).
    """
    time_str = resolved_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    action_label_map = {
        "stop_training": "🛑 立即终止训练并安全存档",
        "self_resolve": "✅ 自行解决（训练自行继续，无须干预）",
        "custom": "🛠️ 按群里下发的方案处理（人工指定）",
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


def _receipt_card(
    template: str,
    title: str,
    markdown_lines: List[str],
    note_text: str,
) -> Dict[str, Any]:
    """Shared frame for the group-message receipts (next-task plan / dispatched fix).

    Both are the only feedback channel for a group message, so they carry the same
    anatomy: header + one markdown block echoing the payload + a footer note.
    """
    return {
        "config": {"wide_screen_mode": True, "enable_forward": True},
        "header": {
            "template": template,
            "title": {"tag": "plain_text", "content": title},
        },
        "elements": [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": "\n".join(markdown_lines),
                },
            },
            {
                "tag": "note",
                "elements": [{"tag": "plain_text", "content": note_text}],
            },
        ],
    }


def build_plan_card(
    plan_id: str,
    text: str,
    task_id: Optional[str] = None,
    created_by: Optional[str] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct the receipt card for a next-task plan registered from the group.

    这是登记成功唯一的反馈通道（群里 @机器人 即登记），因此必须回显计划编号、
    完整正文与绑定对象；同时告知撤销方式。
    """
    time_str = created_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    binding = f"`{task_id}`" if task_id else "全局（适用所有任务）"

    markdown_lines = [
        f"**🆔 计划编号**: `{plan_id}`",
        f"**🎯 绑定任务**: {binding}",
        f"**📝 计划内容**: \n> {text}",
        f"**👤 提交人**: {created_by or '飞书工程师'}",
        f"**🕒 登记时间**: `{time_str}`",
        "",
        "> 📌 Agent 将在下次 report 汇报时读取并按顺序执行；执行完成后以 `ack_plan` 确认。"
        "计划在被确认或清除之前会一直保留。",
    ]

    return _receipt_card(
        template="blue",
        title="📌【下一步计划已登记】",
        markdown_lines=markdown_lines,
        note_text="TrainPilot 下一步计划 • 撤销：@机器人 发送「清除」或「清除 #任务ID」",
    )


def build_fix_card(
    task_id: str,
    solution: str,
    operator: Optional[str] = None,
    instruction_id: Optional[str] = None,
    dispatched_at: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Receipt for a `#fix(任务名) <方案>` group message.

    Confirms which task the solution was dispatched to; with `reason` set it becomes
    the rejection form explaining why nothing was dispatched.
    """
    time_str = dispatched_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rejected = reason is not None

    if rejected:
        title = "⚠️【修复方案未下发】"
        template = "red"
        head = f"**⚠️ 未下发原因**: {reason}"
        body = f"**📝 方案内容**（未下发）: \n> {solution or '（空）'}"
        note = "TrainPilot 修复方案 • 仅「等待中(WAITING)」的告警任务可接收方案；排下一步请用 #feat(任务名)"
        tail = (
            "> 🛠 该消息没有产生任何决策。若任务当时正在等待，请重新 @机器人 发送；"
            "若只是要排下一步的工作，请改用 `#feat(任务名) <计划>`（或 `#任务名`）。"
        )
    else:
        title = f"🛠️【修复方案已下发】任务: {task_id}"
        template = "turquoise"
        head = f"**🎯 下发任务**: `{task_id}`"
        if instruction_id:
            head += f"\n**🆔 指令编号**: `{instruction_id}`"
        body = f"**📝 方案内容**: \n> {solution}"
        note = "TrainPilot 群内修复方案 • 已作为该任务的人工决策下发"
        tail = "> ⚡️ 已写入任务信箱并唤醒 Agent 长轮询；执行结果将以自愈卡片回报。"

    markdown_lines = [
        head,
        body,
        f"**👤 提交人**: {operator or '飞书工程师'}",
        f"**🕒 {'拦截' if rejected else '下发'}时间**: `{time_str}`",
        "",
        tail,
    ]

    return _receipt_card(
        template=template,
        title=title,
        markdown_lines=markdown_lines,
        note_text=note,
    )


def build_deferred_card(
    task_id: str,
    operator: Optional[str] = None,
    timeout_seconds: int = 1800,
    requested_at: Optional[str] = None,
    original_message: Optional[str] = None,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
) -> Dict[str, Any]:
    """Status card replacing the alert card after 「⏳ 等待安排」.

    The click replaces the alert card in place, so it repeats the anomaly context --
    the engineer is being asked to think about exactly that for the next while.
    """
    time_str = requested_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    window = f"{timeout_seconds // 60} 分钟" if timeout_seconds >= 60 else f"{timeout_seconds} 秒"

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
        "**⏸ 自动决策已暂停**，该任务保持等待，Agent 仍在长轮询。",
        f"**👉 请在群里发**：`@机器人 #fix({task_id}) 你的处理方案`",
        f"**⚠️ 兜底**：{window}内仍无方案，将自动按“自行解决”处理，避免任务永久卡死。",
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
                    "content": f"TrainPilot 等待人工安排 • {time_str}",
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
                "content": f"⏳【等待人工安排】任务: {task_id}",
            },
        },
        "elements": elements,
    }


def build_plan_cleared_card(
    cleared_count: int,
    scope_task_id: Optional[str] = None,
    operator: Optional[str] = None,
    cleared_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct the receipt card for a clear command (sent even when nothing matched)."""
    time_str = cleared_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    scope = f"任务 `{scope_task_id}`" if scope_task_id else "全局（所有未确认计划）"

    markdown_lines = [
        f"**🎯 清除范围**: {scope}",
        f"**🗑 已清除条数**: `{cleared_count}`",
        f"**👤 操作人**: {operator or '飞书工程师'}",
        f"**🕒 操作时间**: `{time_str}`",
        "",
        "> 🧹 如仍需该计划，请重新 @机器人 发送。此处的清除只影响尚未被 Agent 确认的计划。",
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
                    "content": "TrainPilot 下一步计划 • 已清除的计划不会再出现在 report 响应中",
                }
            ],
        },
    ]

    return {
        "config": {"wide_screen_mode": True, "enable_forward": True},
        "header": {
            "template": "yellow",
            "title": {
                "tag": "plain_text",
                "content": "🧹【待办计划已清除】",
            },
        },
        "elements": elements,
    }


