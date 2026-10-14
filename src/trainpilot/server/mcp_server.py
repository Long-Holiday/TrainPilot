"""Model Context Protocol (MCP) Server implementation for TrainPilot Control Plane.

Exposes the training-supervision tools over MCP Streamable HTTP, allowing a
GPU-side AI Agent (Claude Code, Cursor, OpenCode, ...) to drive the Feishu
human-in-the-loop decision loop against this control plane.
"""

import logging
import threading
from typing import Any, Dict, List, Optional
from mcp.server.mcpserver import MCPServer

from trainpilot.common.schemas import (
    EventNotifyRequest,
)
from trainpilot.common.states import EventType
from trainpilot.server.config import settings
from trainpilot.server.background import feishu_dispatcher
from trainpilot.server.feishu.client import default_feishu_client
from trainpilot.server.mailbox import default_mailbox
from trainpilot.server.plans import ACK_STATUSES, default_plans

logger = logging.getLogger("trainpilot.server.mcp")

_PLANS_GUIDANCE = (
    "The human queued the next-task plans above in Feishu. They stay pending until "
    "acknowledged: carry each one out at the next safe point (never mid-step, never before an "
    "open alert is resolved), then call ack_plan with its plan_id, status, and a note describing "
    "what you did. An unacknowledged plan reappears on every report -- that is not a repeat "
    "instruction."
)

# Event types accepted by the unified `report` tool.
_REPORTABLE_STATES = {
    "milestone": EventType.MILESTONE,
    "alert": EventType.ALERT,
    "completed": EventType.COMPLETED,
    "failed": EventType.FAILED,
}

# The authoritative behaviour contract for the supervising agent. README and tool
# docstrings must not restate these rules -- they live here, once.
_INSTRUCTIONS = (
    "TrainPilot supervises a training run through Feishu approval cards. Connecting MCP does not "
    "monitor or pause GPU processes: you must watch training output, logs, and process exit status "
    "yourself.\n"
    "On NaN/Inf, loss explosion, CUDA OOM, unexpected exit, or any error blocking training, call "
    "report(event_type='alert') BEFORE repair, restart, or abandonment. Never report an active "
    "anomaly as a milestone, and never use failed in place of an alert needing a human. Safely pause "
    "the GPU job when you can; if it already exited, say so. WAITING means only that the control "
    "plane is waiting, not that a process stopped.\n"
    "Then poll_instruction on the same task until has_instruction=true. A poll timeout is not "
    "permission to resume, and do not resend the alert. Stop polling on COMPLETED or FAILED.\n"
    "Execute the returned action, verify the result, then ack_instruction with its instruction_id and "
    "action: stop_training means stop the job, self_resolve permits a recovery attempt. ACK success "
    "only for a verified result, otherwise ACK failed, and describe only what you actually did. After "
    "ACKing stop_training, report(event_type='failed').\n"
    "Every report response also carries next_task_plans: follow-up work a human queued in the Feishu "
    "group, to do after the current run. A plan stays pending until you ack it, so the same plan_id "
    "reappears on every report -- it is the same plan, not a repeat instruction, and never redo work "
    "you already finished. At the next safe point, carry out each plan, then call ack_plan with its "
    "plan_id, a status (done, failed, or skipped), and a note saying what you actually did. A plan "
    "with task_id null applies to every task; one bound to a task_id only appears on that task's "
    "reports. While an alert is open, the alert protocol above comes first."
)

# Initialize central MCP Server instance
mcp_server = MCPServer(
    name="TrainPilot",
    instructions=_INSTRUCTIONS,
)


def _dispatch_feishu_async(req: EventNotifyRequest) -> None:
    """Queue Feishu delivery so MCP tools return immediately."""
    def _run():
        try:
            if req.event_type == EventType.ALERT:
                default_feishu_client.send_alert(
                    task_id=req.task_id,
                    message=req.message,
                    step=req.step,
                    epoch=req.epoch,
                    metrics=req.metrics,
                    extra=req.extra,
                    timeout_seconds=settings.alert_decision_timeout_seconds,
                    agent_note=req.agent_note,
                )
            elif req.event_type == EventType.MILESTONE:
                default_feishu_client.send_milestone(
                    task_id=req.task_id,
                    message=req.message,
                    step=req.step,
                    epoch=req.epoch,
                    metrics=req.metrics,
                    agent_note=req.agent_note,
                    extra=req.extra,
                )
        except Exception as exc:
            logger.error("Failed to forward event %s to Feishu: %s", req.event_type, exc)

    feishu_dispatcher.submit(_run)


def _auto_self_resolve_callback(task_id: str, timeout_seconds: int) -> None:
    """Timer callback: auto self-resolve if no human decision arrived within timeout."""
    try:
        instruction = default_mailbox.try_auto_resolve(
            task_id=task_id,
            action="self_resolve",
            operator=f"系统自动决策（{timeout_seconds}s超时未决策）",
            timeout_seconds=timeout_seconds,
            payload={"timeout_seconds": timeout_seconds},
        )
        if instruction is None:
            return
        try:
            default_feishu_client.update_card_to_resolved(
                task_id=task_id,
                action="self_resolve",
                operator=instruction.decision_by or "系统自动决策",
                resolved_at=instruction.decided_at,
            )
        except Exception as exc:
            logger.warning("Failed to patch card after auto self-resolve for %s: %s", task_id, exc)
    except Exception as exc:
        logger.error("Auto self-resolve callback failed for task %s: %s", task_id, exc)


def _schedule_auto_self_resolve(task_id: str) -> None:
    """Schedule daemon timer to auto self-resolve alert upon timeout."""
    try:
        timeout_seconds = int(settings.alert_decision_timeout_seconds)
    except Exception:
        timeout_seconds = 30
    if timeout_seconds <= 0:
        return
    timer = threading.Timer(
        timeout_seconds,
        _auto_self_resolve_callback,
        args=(task_id, timeout_seconds),
    )
    timer.daemon = True
    timer.start()
    logger.info("Scheduled auto self-resolve for task %s in %ss", task_id, timeout_seconds)


# ---------------------------------------------------------------------------
# MCP Tools Registration
# ---------------------------------------------------------------------------


def _resolve_task_id(task_id: Optional[str]) -> str:
    """Fall back to the server-configured default task id when omitted."""
    return task_id or settings.task_id


def _pending_plans_for(task_id: str) -> List[Dict[str, Any]]:
    """Next-task plans visible to this task. A read failure must never fail a report."""
    try:
        return [plan.brief() for plan in default_plans.list_pending(task_id)]
    except Exception as exc:
        logger.error("Failed to read next-task plans for task %s: %s", task_id, exc)
        return []


@mcp_server.tool()
def report(
    message: str,
    event_type: str = "milestone",
    task_id: Optional[str] = None,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    agent_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Report a training event to the control plane.

    Which event_type to use, and what to do next, is specified in the server instructions.

    Args:
        message: Human-readable event description. Required.
        event_type: 'milestone' (default), 'alert', 'completed', or 'failed'.
        task_id: Defaults to the server-configured task.
        step: Current training step.
        epoch: Current training epoch.
        metrics: e.g. {'loss': 0.35, 'val_loss': 0.40}.
        agent_note: Your own 1-3 sentence analysis; rendered on the Feishu card.
    """
    key = (event_type or "milestone").strip().lower()
    resolved_type = _REPORTABLE_STATES.get(key)
    if resolved_type is None:
        return {
            "success": False,
            "error": f"Invalid event_type '{event_type}'. Use one of: {', '.join(_REPORTABLE_STATES)}",
        }

    tid = _resolve_task_id(task_id)
    req = EventNotifyRequest(
        task_id=tid,
        event_type=resolved_type,
        message=message,
        step=step,
        epoch=epoch,
        metrics=metrics,
        agent_note=agent_note,
    )
    current_state = default_mailbox.record_event(req)
    _dispatch_feishu_async(req)
    if resolved_type == EventType.ALERT:
        _schedule_auto_self_resolve(tid)

    response = {
        "success": True,
        "task_id": tid,
        "state": current_state.value,
        "message": f"Event '{key}' recorded. Current state: {current_state.value}",
        # Always present, on every event type: [] is the explicit "I checked, nothing queued".
        "next_task_plans": _pending_plans_for(tid),
    }
    if response["next_task_plans"]:
        response["next_task_plans_guidance"] = _PLANS_GUIDANCE
    if resolved_type == EventType.ALERT:
        response["next_action"] = {
            "tool": "poll_instruction",
            "arguments": {"task_id": tid},
        }
        response["guidance"] = (
            "Alert recorded. Feishu delivery is asynchronous and not confirmed by this response. "
            "Now poll_instruction on this task until has_instruction=true; execute the returned "
            "action, verify the result, then ack_instruction. See the server instructions for the "
            "full protocol."
        )
    return response


@mcp_server.tool()
async def poll_instruction(
    task_id: Optional[str] = None,
    wait_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Wait for a human decision on a task, consuming it on arrival.

    Long-polls up to wait_timeout seconds; has_instruction tells you whether one
    arrived. See the server instructions for the polling protocol.

    Args:
        task_id: Defaults to the server-configured task.
        wait_timeout: Seconds (0 returns immediately). Defaults to the server's
            configured long-poll timeout.
    """
    tid = _resolve_task_id(task_id)
    if wait_timeout is None:
        wait_timeout = settings.long_poll_timeout_seconds
    try:
        instruction = await default_mailbox.get_instruction_async(
            tid, pop=True, wait_timeout=wait_timeout
        )
    except ValueError as exc:
        return {
            "success": False,
            "task_id": tid,
            "has_instruction": False,
            "error": str(exc),
        }
    return instruction.model_dump()


@mcp_server.tool()
def ack_instruction(
    task_id: Optional[str] = None,
    instruction_id: Optional[str] = None,
    action: str = "self_resolve",
    status: str = "success",
    solution: str = "",
) -> Dict[str, Any]:
    """Acknowledge a polled instruction after you have executed it.

    Pass the instruction_id and action returned by poll_instruction. See the server
    instructions for what counts as a verified result. A successful self_resolve ACK
    returns the task to RUNNING and queues a Feishu recovery card built from the
    task's last reported step/metrics, so you need not re-send them.

    Args:
        task_id: Defaults to the server-configured task.
        instruction_id: ID of the instruction you executed.
        action: The action you performed ('self_resolve' or 'stop_training').
        status: 'success' or 'failed'.
        solution: What you actually did, in human-readable form.
    """
    tid = _resolve_task_id(task_id)
    try:
        new_state = default_mailbox.ack_instruction(
            task_id=tid,
            instruction_id=instruction_id,
            action=action,
            status=status,
            message=solution or None,
        )
    except ValueError as exc:
        return {"success": False, "task_id": tid, "error": str(exc)}

    # Send recovery card to Feishu if successfully resolved. The step/epoch/metrics
    # come from the task's last report; the agent is not asked to re-send them.
    if status.lower() == "success" and (action == "self_resolve" or bool(solution)):
        solution_text = solution or "已完成现场自愈检查并恢复正常训练。"
        summary = default_mailbox.get_task(tid)

        def _send_recovery():
            try:
                default_feishu_client.send_recovery(
                    task_id=tid,
                    solution=solution_text,
                    step=getattr(summary, "latest_step", None),
                    epoch=getattr(summary, "latest_epoch", None),
                    metrics=getattr(summary, "latest_metrics", None),
                )
            except Exception as exc:
                logger.error("Failed to send Feishu recovery card for task %s: %s", tid, exc)

        feishu_dispatcher.submit(_send_recovery)

    return {
        "success": True,
        "task_id": tid,
        "state": new_state.value,
        "message": f"Instruction acknowledged with status '{status}'",
    }


@mcp_server.tool()
def ack_plan(plan_id: str, status: str = "done", note: str = "") -> Dict[str, Any]:
    """Acknowledge a next-task plan you have carried out.

    Plans arrive in the next_task_plans field of every report response; how to execute
    and acknowledge them is specified in the server instructions. Until you acknowledge
    one, it keeps reappearing on every report.

    Args:
        plan_id: The plan_id from a report response's next_task_plans entry. Required.
        status: 'done' (default), 'failed', or 'skipped'.
        note: One or two sentences on what you actually did.
    """
    normalized = (status or "").strip().lower()
    if normalized not in ACK_STATUSES:
        return {
            "success": False,
            "plan_id": plan_id,
            "error": f"Invalid status '{status}'. Use one of: {', '.join(ACK_STATUSES)}",
        }
    try:
        plan = default_plans.ack(plan_id, status=normalized, note=note)
    except ValueError as exc:
        return {"success": False, "plan_id": plan_id, "error": str(exc)}

    logger.info("Next-task plan %s acknowledged as '%s'", plan_id, normalized)
    return {
        "success": True,
        "plan_id": plan.plan_id,
        "status": normalized,
        "acked_at": plan.acked_at,
        "message": f"Plan '{plan.plan_id}' acknowledged as '{normalized}'",
    }


@mcp_server.tool()
def get_task_status(task_id: Optional[str] = None) -> Dict[str, Any]:
    """Read the current state and instruction-mailbox summary of one task.

    Args:
        task_id: Defaults to the server-configured task.
    """
    tid = _resolve_task_id(task_id)
    summary = default_mailbox.get_task(tid)
    if not summary:
        return {"success": False, "task_id": tid, "error": f"Task '{tid}' not found"}
    return {"success": True, "task": summary.model_dump()}
