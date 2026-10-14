"""Unit tests for Feishu card structures."""

import pytest

from trainpilot.server.feishu.cards import (
    build_alert_card,
    build_milestone_card,
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

    # Verify action buttons: only 停止训练 & 自行解决
    actions = _action_values(card)
    assert [a["action"] for a in actions] == ["stop_training", "self_resolve"]
    for action in actions:
        assert action["task_id"] == "llm-task-1"

    # 超时提示必须出现在卡片文案中
    contents = _div_text(card)
    assert shown in contents and "自行解决" in contents


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
