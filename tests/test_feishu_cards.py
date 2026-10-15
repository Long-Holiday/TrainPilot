"""Unit tests for Feishu card structures."""

import pytest

from trainpilot.server.feishu.cards import (
    build_alert_card,
    build_deferred_card,
    build_fix_card,
    build_milestone_card,
    build_plan_card,
    build_plan_cleared_card,
    build_recovery_card,
    build_resolved_card,
)


def _div_text(card) -> str:
    """The human-readable part of a card: every div/text element joined."""
    return " ".join(
        elem.get("text", {}).get("content", "")
        for elem in card["elements"]
        if elem.get("tag") == "div"
    )


def _action_values(card):
    """The payloads carried by the card's buttons, or None when it has none."""
    for elem in card["elements"]:
        if elem.get("tag") == "action":
            return [btn["value"] for btn in elem["actions"]]
    return None


@pytest.mark.parametrize("timeout_seconds,shown", [(None, "30"), (60, "60")])
def test_build_alert_card(timeout_seconds, shown):
    kwargs = {} if timeout_seconds is None else {"timeout_seconds": timeout_seconds}
    card = build_alert_card(
        task_id="llm-task-1",
        message="CUDA Out Of Memory",
        step=500,
        epoch=2,
        metrics={"allocated_mb": 79000, "loss": 1.25},
        **kwargs,
    )
    assert card["header"]["template"] == "red"
    assert "llm-task-1" in card["header"]["title"]["content"]

    # Verify action buttons: 停止训练、自行解决，以及暂停倒计时的「等待安排」
    actions = _action_values(card)
    assert [a["action"] for a in actions] == [
        "stop_training",
        "self_resolve",
        "wait_for_arrangement",
    ]
    for action in actions:
        assert action["task_id"] == "llm-task-1"

    # 超时提示必须出现在卡片文案中
    contents = _div_text(card)
    assert shown in contents and "自行解决" in contents
    # 「等待安排」的用法必须写在卡片上，否则没人知道下一步该做什么
    assert "#fix(llm-task-1)" in contents


def test_build_milestone_card():
    card = build_milestone_card(
        task_id="llm-task-2",
        message="Epoch 2 completed with validation loss drop",
        step=2000,
        epoch=2,
        metrics={"val_loss": 0.32},
        agent_note="val_loss 新低，收敛平稳，未见明显过拟合，建议保持当前超参继续观察。",
    )
    assert card["header"]["template"] == "green"
    # Milestones are read-only, should not have action buttons
    assert _action_values(card) is None

    # 阶段描述 + Agent 智能点评 两个区块必须同时出现
    contents = _div_text(card)
    assert "📝 阶段描述" in contents
    assert "🤖 Agent 智能点评" in contents
    assert "Epoch 2 completed" in contents
    assert "收敛平稳" in contents


@pytest.mark.parametrize("kwargs,expected_note", [
    ({"agent_note": "该 Batch 梯度异常，已跳过。"}, "梯度异常"),
    # 兼容旧调用方: 仅通过 extra 携带 agent_note 也能渲染
    ({"extra": {"agent_comment": "loss 稳定下降，无风险。"}}, "loss 稳定下降"),
    # 不提供 agent_note 时不应出现点评区块 (向后兼容)
    ({}, None),
])
def test_build_milestone_card_agent_note_section(kwargs, expected_note):
    card = build_milestone_card(
        task_id="llm-task-2", message="Epoch 2 done", step=2000, epoch=2, **kwargs
    )
    contents = _div_text(card)
    if expected_note is None:
        assert "🤖 Agent 智能点评" not in contents
    else:
        assert "🤖 Agent 智能点评" in contents
        assert expected_note in contents


@pytest.mark.parametrize("operator,expected_reason", [
    ("Alice", None),
    ("系统自动决策（30s超时未决策）", "⏳ 触发原因"),
])
def test_build_resolved_card(operator, expected_reason):
    card = build_resolved_card(
        task_id="llm-task-1",
        action_name="self_resolve",
        operator=operator,
    )
    assert card["header"]["template"] == "turquoise"
    assert "【决策已闭环】" in card["header"]["title"]["content"]
    # Buttons must be stripped for anti-duplicate click protection
    assert _action_values(card) is None

    contents = _div_text(card)
    assert operator in contents
    if expected_reason is None:
        # A human decision must not claim the auto-timeout fallback fired
        assert "⏳ 触发原因" not in contents
    else:
        assert expected_reason in contents


def test_build_resolved_card_labels_a_custom_fix_and_its_trigger():
    card = build_resolved_card(
        task_id="llm-task-1",
        action_name="custom",
        operator="ou_engineer",
        trigger_note="超过等待安排窗口仍无方案",
    )
    contents = _div_text(card)
    assert "🛠️ 按群里下发的方案处理" in contents
    # The 30-second wording would be a lie for the arrangement fallback
    assert "超过 30 秒" not in contents


def test_build_deferred_card_pauses_the_countdown():
    card = build_deferred_card(
        task_id="llm-task-1",
        operator="ou_engineer",
        timeout_seconds=1800,
        requested_at="2026-10-14T08:00:00+00:00",
        original_message="CUDA Out Of Memory",
        step=500,
        epoch=2,
    )
    assert card["header"]["template"] == "orange"
    assert "【等待人工安排】" in card["header"]["title"]["content"]
    # No buttons: the only way out is the group message (or the fallback)
    assert _action_values(card) is None

    contents = _div_text(card)
    # The exact command is spelled out, with the real task id, ready to copy
    assert "`@机器人 #fix(llm-task-1) 你的处理方案`" in contents
    assert "30 分钟" in contents
    assert "自动决策已暂停" in contents
    # The card replaces the alert in place, so the anomaly context must survive
    assert "CUDA Out Of Memory" in contents
    assert "ou_engineer" in contents


def test_build_fix_card_confirms_the_task_it_reached():
    card = build_fix_card(
        task_id="llm-task-1",
        solution="回滚到 step4 检查点并把 lr 减半",
        operator="ou_engineer",
        instruction_id="inst_abc123",
        dispatched_at="2026-10-14T08:00:00+00:00",
    )
    assert card["header"]["template"] == "turquoise"
    assert "【修复方案已下发】" in card["header"]["title"]["content"]
    assert "llm-task-1" in card["header"]["title"]["content"]
    assert _action_values(card) is None

    contents = _div_text(card)
    assert "`llm-task-1`" in contents
    assert "`inst_abc123`" in contents
    assert "回滚到 step4 检查点并把 lr 减半" in contents
    assert "ou_engineer" in contents


def test_build_fix_card_with_a_reason_is_the_rejection_form():
    card = build_fix_card(
        task_id="llm-task-1",
        solution="回滚到 step4",
        operator="ou_engineer",
        reason="任务当前状态为 RECOVERING，没有等待中的决策",
    )
    assert card["header"]["template"] == "red"
    assert "【修复方案未下发】" in card["header"]["title"]["content"]

    contents = _div_text(card)
    assert "任务当前状态为 RECOVERING" in contents
    # Points at the right escape hatch for follow-up work
    assert "#feat(任务名)" in contents


def test_build_recovery_card():
    card = build_recovery_card(
        task_id="llm-task-9",
        solution="已跳过异常 Batch 并重置优化器梯度状态，Loss 恢复正常，训练继续进行。",
        step=150,
        epoch=2,
        metrics={"loss": 1.45, "lr": 1e-4},
    )
    assert card["header"]["template"] == "green"
    assert "【异常已自行解决】" in card["header"]["title"]["content"]
    assert "llm-task-9" in card["header"]["title"]["content"]

    contents = _div_text(card)
    assert "💡 解决方法" in contents
    assert "已跳过异常 Batch" in contents
    assert "问题已解决，训练任务已恢复正常运行" in contents
    assert "150" in contents
    assert "Epoch `2`" in contents


def test_build_plan_card_echoes_id_text_and_binding():
    card = build_plan_card(
        plan_id="plan_abc123",
        text="用 5e-5 重跑 10 个 epoch",
        task_id="demo-llm-pretrain",
        created_by="ou_engineer",
        created_at="2026-10-14T08:00:00+00:00",
    )
    assert card["header"]["template"] == "blue"
    assert "【下一步计划已登记】" in card["header"]["title"]["content"]
    assert _action_values(card) is None

    contents = _div_text(card)
    assert "`plan_abc123`" in contents
    assert "用 5e-5 重跑 10 个 epoch" in contents
    assert "`demo-llm-pretrain`" in contents
    assert "ou_engineer" in contents
    assert "ack_plan" in contents


def test_build_plan_card_with_a_reason_is_the_rejection_form():
    card = build_plan_card(
        text="用 5e-5 重跑 10 个 epoch",
        task_id="demo-llm-pretrain",
        created_by="ou_engineer",
        created_at="2026-10-14T08:00:00+00:00",
        reason="任务 `demo-llm-pretrain` 不存在",
        known_task_ids=["llm-task-1", "llm-task-2"],
    )
    assert card["header"]["template"] == "red"
    assert "【下一步计划未登记】" in card["header"]["title"]["content"]

    contents = _div_text(card)
    assert "任务 `demo-llm-pretrain` 不存在" in contents
    # The body is echoed back so the sender can fix the tag and resend it
    assert "用 5e-5 重跑 10 个 epoch" in contents
    # The tasks that do exist are listed so the right name is one copy-paste away
    assert "`llm-task-1`" in contents and "`llm-task-2`" in contents
    # Nothing was stored, so there is no plan id to acknowledge
    assert "计划编号" not in contents


def test_build_plan_card_rejection_without_a_task_named_lists_no_tasks():
    card = build_plan_card(text="跑评测", reason="缺少任务名：请写成 `#任务名 <计划>`")
    contents = _div_text(card)
    assert "缺少任务名" in contents
    assert "你绑定的任务" not in contents
    assert "当前已有任务" not in contents


def test_build_plan_cleared_card_reports_scope_count_and_operator():
    card = build_plan_cleared_card(
        cleared_count=2,
        scope_task_id="demo-llm-pretrain",
        operator="ou_engineer",
        cleared_at="2026-10-14T08:05:00+00:00",
    )
    assert card["header"]["template"] == "yellow"
    assert "【待办计划已清除】" in card["header"]["title"]["content"]

    contents = _div_text(card)
    assert "`2`" in contents
    assert "`demo-llm-pretrain`" in contents
    assert "ou_engineer" in contents

    # An unscoped clear still renders as global, and a zero count still sends the card.
    global_card = build_plan_cleared_card(cleared_count=0)
    assert "全局（所有未确认计划）" in _div_text(global_card)
    assert "`0`" in _div_text(global_card)
