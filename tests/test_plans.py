"""Tests for the next-task plan registry (Feishu queue -> report -> ack)."""

import pytest

from trainpilot.common.schemas import EventNotifyRequest, NextTaskPlan
from trainpilot.common.states import EventType, PlanStatus
from trainpilot.server.mailbox import default_mailbox
from trainpilot.server.plans import (
    MAX_PLAN_TEXT_CHARS,
    PlanBindingError,
    PlanRegistry,
    default_plans,
)


@pytest.fixture
def registry():
    """Bind to the redirected test database through the default singleton."""
    return PlanRegistry()


def _seed_task(task_id: str) -> None:
    """Create a task row the supported way -- a report. Every plan binds to a live task."""
    default_mailbox.record_event(
        EventNotifyRequest(task_id=task_id, event_type=EventType.MILESTONE, message="step 1 done")
    )


def test_register_and_scope_filtering(registry):
    _seed_task("task-a")
    _seed_task("task-b")

    first, created = registry.register("重跑评测", task_id="task-a", created_by="ou_human")
    assert created is True
    assert first.plan_id.startswith("plan_")
    assert first.status == PlanStatus.PENDING
    assert first.task_id == "task-a"

    second, _ = registry.register("换学习率", task_id="task-a")
    other, _ = registry.register("别的任务", task_id="task-b")

    # FIFO within a task: plans run in the order the humans queued them.
    assert [p.plan_id for p in registry.list_pending("task-a")] == [first.plan_id, second.plan_id]
    # A plan is carried by its own task's reports only, and a caller with no task sees none.
    assert other.plan_id not in [p.plan_id for p in registry.list_pending("task-a")]
    assert registry.list_pending(None) == []


def test_register_is_idempotent_per_feishu_message(registry):
    _seed_task("task-a")

    first, created = registry.register("计划", task_id="task-a", source_message_id="om_1")
    assert created is True

    second, created_again = registry.register("计划", task_id="task-a", source_message_id="om_1")
    assert created_again is False
    assert second.plan_id == first.plan_id
    assert len(registry.list_pending("task-a")) == 1


def test_register_rejects_empty_text(registry):
    with pytest.raises(ValueError):
        registry.register("   ")


def test_register_rejects_a_plan_that_names_no_task(registry):
    """Every plan is bound: a nameless one would be carried by no report and never expire."""
    with pytest.raises(PlanBindingError) as excinfo:
        registry.register("没人认领的计划")

    assert excinfo.value.task_id is None
    assert registry.list_pending(None) == []


def test_registering_a_bound_plan_does_not_create_a_task_row(registry):
    """The plan path may read the task table, never write it: an unknown task is refused
    rather than fabricated."""
    with pytest.raises(PlanBindingError) as excinfo:
        registry.register("先排队再开跑", task_id="not-yet-started")

    assert excinfo.value.task_id == "not-yet-started"
    assert registry.list_pending("not-yet-started") == []
    assert default_mailbox.get_task("not-yet-started") is None
    assert default_mailbox.list_tasks() == []


def test_a_task_created_after_the_rejection_accepts_the_plan(registry):
    """The refusal is about this moment, not the name: retrying once the run reports works."""
    with pytest.raises(PlanBindingError):
        registry.register("先排队再开跑", task_id="slow-start")

    _seed_task("slow-start")
    plan, created = registry.register("先排队再开跑", task_id="slow-start")
    assert created is True
    assert [p.plan_id for p in registry.list_pending("slow-start")] == [plan.plan_id]


def test_a_pre_binding_global_plan_is_never_returned(registry):
    """Rows written before plans required a binding must not resurface on any task."""
    _seed_task("task-a")
    default_mailbox._storage.add_plan(
        NextTaskPlan(plan_id="plan_legacy", text="旧的全局计划", task_id=None)
    )

    assert registry.list_pending("task-a") == []
    assert registry.list_pending(None) == []
    # 清除 without a task scope is the way out for them.
    assert registry.clear_all_pending() == 1


def test_register_truncates_overlong_text(registry):
    _seed_task("task-a")
    plan, _ = registry.register("x" * (MAX_PLAN_TEXT_CHARS + 50), task_id="task-a")
    assert len(plan.text) == MAX_PLAN_TEXT_CHARS


def test_ack_moves_the_plan_out_of_pending(registry):
    _seed_task("task-a")
    plan, _ = registry.register("跑评测", task_id="task-a")

    acked = registry.ack(plan.plan_id, status="done", note="已跑完")
    assert acked.status == PlanStatus.DONE
    assert acked.acked_at
    assert acked.ack_note == "已跑完"
    assert registry.list_pending("task-a") == []


@pytest.mark.parametrize("status", ["failed", "skipped"])
def test_ack_accepts_the_other_terminal_statuses(registry, status):
    _seed_task("task-a")
    plan, _ = registry.register("跑评测", task_id="task-a")
    assert registry.ack(plan.plan_id, status=status).status == PlanStatus(status)


def test_ack_rejects_bad_status_unknown_and_duplicate(registry):
    _seed_task("task-a")
    with pytest.raises(ValueError, match="Invalid status 'pending'"):
        registry.ack("plan_whatever", status="pending")
    with pytest.raises(ValueError, match="not found"):
        registry.ack("plan_missing")

    plan, _ = registry.register("一次性", task_id="task-a")
    registry.ack(plan.plan_id)
    with pytest.raises(ValueError, match="already acknowledged"):
        registry.ack(plan.plan_id)


def test_clear_scopes_do_not_cross(registry):
    _seed_task("task-a")
    _seed_task("task-b")
    bound_a, _ = registry.register("A 的计划", task_id="task-a")
    bound_b, _ = registry.register("B 的计划", task_id="task-b")

    assert registry.clear_pending_for_task("task-a") == 1
    remaining = [p.plan_id for p in registry.list_pending("task-b")]
    assert bound_a.plan_id not in remaining
    assert bound_b.plan_id in remaining

    assert registry.clear_all_pending() == 1
    assert registry.list_pending("task-b") == []


def test_delete_expired_spares_pending_plans(registry):
    _seed_task("task-a")
    acked, _ = registry.register("旧计划", task_id="task-a")
    registry.ack(acked.plan_id)
    pending, _ = registry.register("还没做", task_id="task-a")

    removed = registry.delete_expired(max_age_seconds=-1)  # every acknowledged plan is "old"
    assert removed == 1
    assert [p.plan_id for p in registry.list_pending("task-a")] == [pending.plan_id]


def test_default_singleton_shares_storage_with_the_mailbox():
    _seed_task("task-a")
    plan, _ = default_plans.register("共享存储", task_id="task-a")
    assert default_mailbox._storage.get_plan(plan.plan_id) is not None
