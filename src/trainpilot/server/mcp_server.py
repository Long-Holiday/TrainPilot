"""Model Context Protocol (MCP) Server implementation for TrainPilot Control Plane.

Exposes the training-supervision tools over MCP Streamable HTTP, allowing a
GPU-side AI Agent (Claude Code, Cursor, OpenCode, ...) to drive the Feishu
human-in-the-loop decision loop against this control plane.
"""

import logging
import threading
from typing import Any, Dict, Optional
from mcp.server.mcpserver import MCPServer

from trainpilot.common.schemas import (
    EventNotifyRequest,
)
from trainpilot.common.states import EventType
from trainpilot.server.config import settings
from trainpilot.server.background import feishu_dispatcher
from trainpilot.server.feishu.client import default_feishu_client
from trainpilot.server.mailbox import default_mailbox

logger = logging.getLogger("trainpilot.server.mcp")

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
    "permission to resume, and do not resend the alert. Stop polling on COMPLETED or FAILED. If the "
    "returned action is 'wait_for_user' (the engineer chose ⏳ 等待用户), stop polling immediately, "
    "return control to the conversation, and resolve the incident with them directly. Do not wait "
    "for a mailbox instruction and do not resume on your own; there is no automatic fallback. After "
    "the engineer tells you what to do and you have verified the result, close the wait with "
    "ack_instruction(task_id='<id>', action='self_resolve', status='success', "
    "solution='<what you did>', step=<n>, epoch=<n>, metrics={...}) -- omit instruction_id. "
    "In that direct path self_resolve is the usual action; cancel and custom are also "
    "accepted because the conversation may have decided otherwise. "
    "get_task_status still returns WAITING until that ACK, which is expected.\n"
    "Execute the returned action, verify the result, then ack_instruction with its instruction_id. "
    "self_resolve permits one recovery attempt; cancel means cancel only this run and shut the "
    "machine down yourself, then verify both before ACK success; custom means carry out the "
    "human-written solution in payload['solution'] exactly as specified. The control plane never "
    "stops a process or shuts a machine down. ACK success only for a verified result, otherwise ACK "
    "failed, and describe only what you actually did. On a machine shared by other runs, do not "
    "power it off: ACK cancel with status='failed' so the incident stays open. Example: "
    "ack_instruction(task_id='<id>', instruction_id='<id>', action='cancel', status='success', "
    "solution='cancelled this run and verified shutdown'). A successful self_resolve/custom ACK "
    "returns the task to RUNNING and sends the one recovery summary; pass real post-recovery "
    "step/epoch/metrics, which are stored on the task. Do not also report a milestone just to "
    "announce recovery. A successful cancel ACK is terminal FAILED and sends one cancellation "
    "notification (never a recovery card); do not send a separate failed report.\n"
    "report(event_type='completed') and report(event_type='failed') each notify the engineer once, "
    "directly. Send exactly one of them when the run is definitively over; no follow-up report is "
    "needed."
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
            elif req.event_type in (EventType.COMPLETED, EventType.FAILED):
                # Terminal outcome: notify exactly once, directly from the report path.
                default_feishu_client.send_outcome(
                    task_id=req.task_id,
                    event_type=req.event_type,
                    message=req.message,
                    step=req.step,
                    epoch=req.epoch,
                    metrics=req.metrics,
                    agent_note=req.agent_note,
                )
        except Exception as exc:
            logger.error("Failed to forward event %s to Feishu: %s", req.event_type, exc)

    feishu_dispatcher.submit(_run)


def _auto_self_resolve_callback(task_id: str, timeout_seconds: int) -> None:
    """Timer callback: auto self-resolve if no human decision arrived within timeout.

    The mailbox refuses this while the engineer is in a direct conversation with the
    agent, so 「等待用户」 needs no separate timer and has no fallback.
    """
    operator = f"系统自动决策（{timeout_seconds}s超时未决策）"
    try:
        instruction = default_mailbox.try_auto_resolve(
            task_id=task_id,
            action="self_resolve",
            operator=operator,
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


def schedule_auto_resolve(task_id: str) -> None:
    """Schedule the daemon timer for the ordinary alert decision window.

    A non-positive timeout disables auto-resolution entirely.
    """
    try:
        timeout_seconds = int(settings.alert_decision_timeout_seconds)
    except Exception:
        timeout_seconds = 30
    if timeout_seconds <= 0:
        logger.warning("Auto self-resolve disabled by a non-positive timeout")
        return
    timer = threading.Timer(
        timeout_seconds,
        _auto_self_resolve_callback,
        args=(task_id, timeout_seconds),
    )
    timer.daemon = True
    timer.start()
    logger.info("Scheduled auto self-resolve for task %s in %ss", task_id, timeout_seconds)


def _schedule_auto_self_resolve(task_id: str) -> None:
    """Schedule daemon timer to auto self-resolve alert upon timeout."""
    schedule_auto_resolve(task_id)


# ---------------------------------------------------------------------------
# MCP Tools Registration
# ---------------------------------------------------------------------------


def _resolve_task_id(task_id: Optional[str]) -> str:
    """Fall back to the server-configured default task id when omitted."""
    return task_id or settings.task_id


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
    }
    if resolved_type == EventType.ALERT:
        response["next_action"] = {
            "tool": "poll_instruction",
            "arguments": {"task_id": tid},
        }
        response["guidance"] = (
            "Alert recorded. Feishu delivery is asynchronous and not confirmed by this response. "
            "Now poll_instruction on this task until has_instruction=true or "
            "awaiting_direct_conversation=true. Execute the returned action, verify the result, "
            "then ack_instruction. See the server instructions for the full protocol."
        )
    return response


@mcp_server.tool()
async def poll_instruction(
    task_id: Optional[str] = None,
    wait_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Wait for a human decision on a task, consuming it on arrival.

    Long-polls up to wait_timeout seconds; has_instruction tells you whether one
    arrived. If awaiting_direct_conversation is true, the engineer chose
    ⏳ 等待用户: this returns immediately with the 'wait_for_user' control action,
    no instruction will arrive through the
    mailbox, and you must stop polling and return control to the conversation.
    See the server instructions for the polling protocol.

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


# `stop_training` is the old wire name for the same action. It is accepted so a
# decision queued by an older card still cancels instead of silently doing something
# else, but every label and notification describes cancel + shutdown.
_CANCEL_ACTIONS = frozenset({"cancel", "stop_training"})


@mcp_server.tool()
def ack_instruction(
    task_id: Optional[str] = None,
    instruction_id: Optional[str] = None,
    action: str = "self_resolve",
    status: str = "success",
    solution: str = "",
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Acknowledge an executed decision; this ACK is the one recovery report.

    Pass the instruction_id and action returned by poll_instruction. After
    ⏳ 等待用户 the engineer resolves the incident with you directly: omit
    instruction_id and ACK with action self_resolve (cancel only after a verified
    shutdown; custom when the conversation specified a custom action). See the server instructions for
    what counts as a verified result.

    A successful self_resolve/custom ACK returns the task to RUNNING and sends one
    recovery card. Pass real post-recovery step/epoch/metrics when you have them;
    otherwise the last reported values are used. A successful cancel ACK (cancel, or
    the legacy stop_training alias) is terminal FAILED and sends one cancellation
    card -- never a recovery card. The control plane does not stop or shut down
    anything; only ACK success after you performed and verified that yourself, and
    ACK failed if the machine is shared or shutdown cannot be verified.

    Args:
        task_id: Defaults to the server-configured task.
        instruction_id: ID of the instruction you executed.
        action: 'self_resolve', 'cancel' (alias 'stop_training'), or 'custom'.
        status: 'success' or 'failed'.
        solution: What you actually did, in human-readable form.
        step: Real post-recovery step, when known.
        epoch: Real post-recovery epoch, when known.
        metrics: Real post-recovery metrics, when known.
    """
    tid = _resolve_task_id(task_id)
    normalized_action = (action or "").strip().lower()
    normalized_status = (status or "").strip().lower()
    is_cancel = normalized_action in _CANCEL_ACTIONS
    mailbox_action = "cancel" if is_cancel else normalized_action
    try:
        new_state = default_mailbox.ack_instruction(
            task_id=tid,
            instruction_id=instruction_id,
            action=mailbox_action,
            status=status,
            message=solution or None,
            step=step,
            epoch=epoch,
            metrics=metrics,
        )
    except ValueError as exc:
        return {"success": False, "task_id": tid, "error": str(exc)}

    # Normalize exactly like the mailbox, or whitespace could commit a success with
    # no notification.
    succeeded = normalized_status == "success"
    if succeeded and is_cancel:
        # Terminal cancellation, verifiably performed by the agent. This must never
        # be rendered as a recovery-training card.
        message = solution or "已由 Agent 取消本次训练并验证机器关机。"
        feishu_dispatcher.submit(
            default_feishu_client.send_outcome,
            task_id=tid,
            event_type=EventType.FAILED,
            message=message,
            step=step,
            epoch=epoch,
            metrics=metrics,
        )
    elif succeeded and (normalized_action in ("self_resolve", "custom") or bool(solution)):
        # One recovery summary, built from the ACK's real values when supplied and the
        # task's last report otherwise. No second milestone report is expected.
        solution_text = solution or "已完成现场自愈检查并恢复正常训练。"
        summary = default_mailbox.get_task(tid)
        feishu_dispatcher.submit(
            default_feishu_client.send_recovery,
            task_id=tid,
            solution=solution_text,
            # The mailbox stored any values supplied above, so notification and
            # get_task_status can no longer disagree.
            step=getattr(summary, "latest_step", None),
            epoch=getattr(summary, "latest_epoch", None),
            metrics=getattr(summary, "latest_metrics", None),
        )

    return {
        "success": True,
        "task_id": tid,
        "state": new_state.value,
        "message": f"Instruction acknowledged with status '{normalized_status}'",
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
