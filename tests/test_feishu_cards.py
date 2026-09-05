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

    # Verify action buttons
    action_elem = None
    for elem in card["elements"]:
        if elem.get("tag") == "action":
            action_elem = elem
            break

    assert action_elem is not None
    actions = [btn["value"]["action"] for btn in action_elem["actions"]]
    assert "reduce_lr_rollback" in actions
    assert "skip_batch" in actions
    assert "stop_training" in actions
    assert "resume" in actions

    for btn in action_elem["actions"]:
        assert btn["value"]["task_id"] == "llm-task-1"


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
        action_name="reduce_lr_rollback",
        operator="Alice",
    )
    assert card["header"]["template"] == "turquoise"
    assert "【决策已闭环】" in card["header"]["title"]["content"]
    # Buttons must be stripped for anti-duplicate click protection
    tags = [elem.get("tag") for elem in card["elements"]]
    assert "action" not in tags
