"""Unit tests for Feishu card structures."""

from trainpilot.server.feishu.cards import (
    build_alert_card,
    build_milestone_card,
    build_resolved_card,
)


def test_build_alert_card():
    card = build_alert_card(
        task_id="llm-task-1",
        message="CUDA Out Of Memory",
        step=500,
        epoch=2,
        metrics={"allocated_mb": 79000, "loss": 1.25},
    )
    assert card["header"]["template"] == "red"
    assert "llm-task-1" in card["header"]["title"]["content"]

    # Verify action buttons: only 停止训练 & 自行解决
    action_elem = None
    for elem in card["elements"]:
        if elem.get("tag") == "action":
            action_elem = elem
            break

    assert action_elem is not None
    actions = [btn["value"]["action"] for btn in action_elem["actions"]]
    assert actions == ["stop_training", "self_resolve"]

    for btn in action_elem["actions"]:
        assert btn["value"]["task_id"] == "llm-task-1"

    # 30s 超时提示必须出现在卡片文案中
    contents = " ".join(
        elem.get("text", {}).get("content", "")
        for elem in card["elements"]
        if elem.get("tag") == "div"
    )
    assert "30" in contents and "自行解决" in contents


def test_build_alert_card_custom_timeout():
    card = build_alert_card(task_id="t1", message="boom", timeout_seconds=60)
    contents = " ".join(
        elem.get("text", {}).get("content", "")
        for elem in card["elements"]
        if elem.get("tag") == "div"
    )
    assert "60" in contents


def test_build_milestone_card():
    card = build_milestone_card(
        task_id="llm-task-2",
        message="Epoch 2 completed with validation loss drop",
        step=2000,
        epoch=2,
        metrics={"val_loss": 0.32},
    )
    assert card["header"]["template"] == "green"
    # Milestones are read-only, should not have action buttons
    tags = [elem.get("tag") for elem in card["elements"]]
    assert "action" not in tags


def test_build_resolved_card():
    card = build_resolved_card(
        task_id="llm-task-1",
        action_name="self_resolve",
        operator="Alice",
    )
    assert card["header"]["template"] == "turquoise"
    assert "【决策已闭环】" in card["header"]["title"]["content"]
    # Buttons must be stripped for anti-duplicate click protection
    tags = [elem.get("tag") for elem in card["elements"]]
    assert "action" not in tags


def test_build_resolved_card_auto_timeout():
    card = build_resolved_card(
        task_id="llm-task-1",
        action_name="self_resolve",
        operator="系统自动决策（30s超时未决策）",
    )
    contents = " ".join(
        elem.get("text", {}).get("content", "")
        for elem in card["elements"]
        if elem.get("tag") == "div"
    )
    assert "自行解决" in contents
    assert "30" in contents
