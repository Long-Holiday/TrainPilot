"""Tests for the next-task plan registry (Feishu queue -> report -> ack)."""

import pytest

from trainpilot.common.states import PlanStatus
from trainpilot.server.mailbox import default_mailbox
from trainpilot.server.plans import MAX_PLAN_TEXT_CHARS, PlanRegistry, default_plans


@pytest.fixture
def registry():
    """Bind to the redirected test database through the default singleton."""
    return PlanRegistry()


def test_register_and_scope_filtering(registry):
    global_plan, created = registry.register("重跑评测", created_by="ou_human")
    assert created is True
    assert global_plan.plan_id.startswith("plan_")
    assert global_plan.status == PlanStatus.PENDING
    assert global_plan.task_id is None

    bound, _ = registry.register("换学习率", task_id="task-a")
    other, _ = registry.register("别的任务", task_id="task-b")

    assert [p.plan_id for p in registry.list_pending("task-a")] == [
        global_plan.plan_id,
        bound.plan_id,
    ]
    # task-b does not see task-a's plan, and a global-only query returns just the global one.
    assert other.plan_id not in [p.plan_id for p in registry.list_pending("task-a")]
    assert [p.plan_id for p in registry.list_pending(None)] == [global_plan.plan_id]


def test_register_is_idempotent_per_feishu_message(registry):
    first, created = registry.register("计划", source_message_id="om_1")
    assert created is True

    second, created_again = registry.register("计划", source_message_id="om_1")
    assert created_again is False
    assert second.plan_id == first.plan_id
    assert len(registry.list_pending("task-a")) == 1


def test_register_rejects_empty_text(registry):
    with pytest.raises(ValueError):
        registry.register("   ")


def test_register_truncates_overlong_text(registry):
    plan, _ = registry.register("x" * (MAX_PLAN_TEXT_CHARS + 50))
    assert len(plan.text) == MAX_PLAN_TEXT_CHARS


def test_registering_a_bound_plan_does_not_create_a_task_row(registry):
    """A plan may precede its run: the plan path must never fabricate task state."""
    registry.register("先排队再开跑", task_id="not-yet-started")
    assert default_mailbox.get_task("not-yet-started") is None
    assert default_mailbox.list_tasks() == []


def test_ack_moves_the_plan_out_of_pending(registry):
    plan, _ = registry.register("跑评测", task_id="task-a")

    acked = registry.ack(plan.plan_id, status="done", note="已跑完")
    assert acked.status == PlanStatus.DONE
    assert acked.acked_at
    assert acked.ack_note == "已跑完"
    assert registry.list_pending("task-a") == []


@pytest.mark.parametrize("status", ["failed", "skipped"])
def test_ack_accepts_the_other_terminal_statuses(registry, status):
    plan, _ = registry.register("跑评测")
    assert registry.ack(plan.plan_id, status=status).status == PlanStatus(status)


def test_ack_rejects_bad_status_unknown_and_duplicate(registry):
    with pytest.raises(ValueError, match="Invalid status 'pending'"):
        registry.ack("plan_whatever", status="pending")
    with pytest.raises(ValueError, match="not found"):
        registry.ack("plan_missing")

    plan, _ = registry.register("一次性")
    registry.ack(plan.plan_id)
    with pytest.raises(ValueError, match="already acknowledged"):
        registry.ack(plan.plan_id)


def test_clear_scopes_do_not_cross(registry):
    global_plan, _ = registry.register("全局计划")
    bound_a, _ = registry.register("A 的计划", task_id="task-a")
    bound_b, _ = registry.register("B 的计划", task_id="task-b")

    assert registry.clear_pending_for_task("task-a") == 1
    remaining = [p.plan_id for p in registry.list_pending("task-b")]
    assert bound_a.plan_id not in remaining
    assert global_plan.plan_id in remaining and bound_b.plan_id in remaining

    assert registry.clear_all_pending() == 2
    assert registry.list_pending("task-b") == []


def test_delete_expired_spares_pending_plans(registry):
    acked, _ = registry.register("旧计划")
    registry.ack(acked.plan_id)
    pending, _ = registry.register("还没做")

    removed = registry.delete_expired(max_age_seconds=-1)  # every acknowledged plan is "old"
    assert removed == 1
    assert [p.plan_id for p in registry.list_pending(None)] == [pending.plan_id]


def test_default_singleton_shares_storage_with_the_mailbox():
    plan, _ = default_plans.register("共享存储", task_id="task-a")
    assert default_mailbox._storage.get_plan(plan.plan_id) is not None
