"""Model Context Protocol (MCP) Server implementation for TrainPilot Control Plane.

Exposes deep learning training supervisor tools and resources over MCP protocol,
allowing remote GPU nodes or AI Agents (e.g. Claude Code, Cursor, OpenCode) to
interact with the central control plane, Feishu cards, and HITL decision loop.
"""

import json
import logging
import threading
from typing import Any, Dict, List, Optional
from mcp.server.mcpserver import MCPServer

from trainpilot.common.schemas import (
    EventNotifyRequest,
)
from trainpilot.common.states import EventType, TaskState
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

# Initialize central MCP Server instance
mcp_server = MCPServer(
    name="TrainPilot",
    instructions=(
        "TrainPilot Control Plane MCP Server: Provides tools to supervise deep learning training jobs, "
        "report milestone metrics with AI notes, intercept NaN/Inf anomalies, poll human-in-the-loop decisions, "
        "and acknowledge self-healing recovery actions connected to Feishu interactive cards."
    ),
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


@mcp_server.tool()
def report(
    message: str,
    event_type: str = "milestone",
    task_id: Optional[str] = None,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    agent_note: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    gpu_host: Optional[str] = None,
) -> Dict[str, Any]:
    """Report any training event in one unified tool (milestone / alert / completed / failed).

    - `milestone`: normal progress; pushes a green Feishu card with the AI agent note.
    - `alert`: anomaly (NaN/Inf, loss spike, OOM); sets state to WAITING, pushes a red
      interactive Feishu card, and arms the fallback auto-decision timer.
    - `completed` / `failed`: terminal state of the run.

    Args:
        message: Human-readable event description (the only required field).
        event_type: One of 'milestone' (default), 'alert', 'completed', 'failed'.
        task_id: Task identifier; defaults to the server-configured task when omitted.
        step: Current training step.
        epoch: Current training epoch (optional).
        metrics: Training metrics, e.g. {'loss': 0.35, 'val_loss': 0.40}.
        agent_note: Autonomous 1-3 sentence analysis/recommendation written by the AI agent.
        extra: Additional contextual metadata.
        gpu_host: GPU server IP/hostname. Reported on the first request; the watchdog pings it for liveness.
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
        extra=extra,
        gpu_host=gpu_host,
    )
    current_state = default_mailbox.record_event(req)
    _dispatch_feishu_async(req)
    if resolved_type == EventType.ALERT:
        _schedule_auto_self_resolve(tid)

    return {
        "success": True,
        "task_id": tid,
        "state": current_state.value,
        "message": f"Event '{key}' recorded. Current state: {current_state.value}",
    }


@mcp_server.tool()
async def poll_instruction(
    task_id: Optional[str] = None,
    wait_timeout: Optional[float] = None,
    pop: bool = True,
) -> Dict[str, Any]:
    """Poll for pending human-in-the-loop instructions for a training task.

    Supports native AsyncIO long-polling: holds connection up to wait_timeout seconds
    without consuming thread pool workers until a decision arrives or timeout expires.

    Args:
        task_id: Task identifier; defaults to the server-configured task when omitted.
        wait_timeout: Long-polling timeout in seconds (0 for immediate non-blocking return).
        pop: If true, consumes the instruction and transitions state to RECOVERING.
    """
    tid = _resolve_task_id(task_id)
    if wait_timeout is None:
        wait_timeout = settings.long_poll_timeout_seconds
    try:
        instruction = await default_mailbox.get_instruction_async(
            tid, pop=pop, wait_timeout=wait_timeout
        )
    except ValueError as exc:
        return {
            "success": False,
            "task_id": tid,
            "has_instruction": False,
            "error": str(exc),
        }
    res = instruction.model_dump()
    res["has_instruction"] = res.get("ready", False)
    return res


@mcp_server.tool()
def ack_instruction(
    task_id: Optional[str] = None,
    instruction_id: Optional[str] = None,
    action: str = "self_resolve",
    status: str = "success",
    solution: str = "",
    message: str = "",
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Acknowledge execution of an instruction with solution details.

    On successful recovery, automatically pushes a Feishu recovery card and resets
    the task state to RUNNING.

    Args:
        task_id: Task identifier; defaults to the server-configured task when omitted.
        instruction_id: ID of the instruction executed (optional).
        action: The action performed (e.g. 'self_resolve', 'stop_training').
        status: Result status ('success' or 'failed').
        solution: Human-readable description of how the issue was fixed.
        message: Additional execution log or message.
        step: Current training step after recovery.
        epoch: Current epoch after recovery.
        metrics: Metrics after recovery.
    """
    tid = _resolve_task_id(task_id)
    try:
        new_state = default_mailbox.ack_instruction(
            task_id=tid,
            instruction_id=instruction_id,
            action=action,
            status=status,
            message=message,
        )
    except ValueError as exc:
        return {"success": False, "task_id": tid, "error": str(exc)}

    # Send recovery card to Feishu if successfully resolved
    if status.lower() == "success" and (action == "self_resolve" or bool(solution)):
        solution_text = solution or message or "已完成现场自愈检查并恢复正常训练。"

        def _send_recovery():
            try:
                default_feishu_client.send_recovery(
                    task_id=tid,
                    solution=solution_text,
                    step=step,
                    epoch=epoch,
                    metrics=metrics,
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
def get_task_status(task_id: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve current state, GPU host, ping result, and instruction mailbox summary for a task.

    Args:
        task_id: Task identifier; defaults to the server-configured task when omitted.
    """
    tid = _resolve_task_id(task_id)
    summary = default_mailbox.get_task(tid)
    if not summary:
        return {"success": False, "task_id": tid, "error": f"Task '{tid}' not found"}
    return {"success": True, "task": summary.model_dump()}


@mcp_server.tool()
def list_tasks(
    limit: int = 100,
    offset: int = 0,
    state: Optional[str] = None,
    stale_only: bool = False,
) -> Dict[str, Any]:
    """List tracked training tasks with optional state filtering and pagination.

    Args:
        limit: Maximum number of tasks to return.
        offset: Number of tasks to skip.
        state: Optional filter by TaskState (e.g. 'RUNNING', 'WAITING').
        stale_only: Only return ping-unreachable tasks.
    """
    parsed_state = None
    if state:
        try:
            parsed_state = TaskState(state.upper())
        except ValueError:
            return {"success": False, "error": f"Invalid state '{state}'"}
    tasks = default_mailbox.list_tasks(limit=limit, offset=offset, state=parsed_state, stale_only=stale_only)
    return {
        "success": True,
        "count": len(tasks),
        "tasks": [t.model_dump() for t in tasks],
    }


@mcp_server.tool()
def submit_decision(
    action: str,
    task_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    operator: str = "mcp_operator",
) -> Dict[str, Any]:
    """Directly submit an operator decision for a frozen task (for administration or tests).

    Args:
        action: Decision action ('self_resolve', 'stop_training', or custom action).
        task_id: Task identifier; defaults to the server-configured task when omitted.
        payload: Optional parameters for the action.
        operator: Name or role of the operator making the decision.
    """
    tid = _resolve_task_id(task_id)
    try:
        instruction = default_mailbox.submit_decision(
            task_id=tid,
            action=action,
            payload=payload,
            operator=operator,
        )
    except ValueError as exc:
        return {"success": False, "task_id": tid, "error": str(exc)}

    def _patch_card():
        try:
            default_feishu_client.update_card_to_resolved(
                task_id=tid,
                action=action,
                operator=operator,
            )
        except Exception as exc:
            logger.warning("Failed to patch card after decision for %s: %s", tid, exc)

    feishu_dispatcher.submit(_patch_card)

    return {
        "success": True,
        "task_id": tid,
        "instruction": instruction.model_dump(),
    }


# ---------------------------------------------------------------------------
# MCP Resources Registration
# ---------------------------------------------------------------------------


@mcp_server.resource("tasks://status/{task_id}")
def resource_task_status(task_id: str) -> str:
    """Read full status of a training task as JSON text."""
    summary = default_mailbox.get_task(task_id)
    if not summary:
        return json.dumps({"error": f"Task '{task_id}' not found"})
    return json.dumps(summary.model_dump(), default=str)


@mcp_server.resource("tasks://list")
def resource_task_list() -> str:
    """Read list of all active training tasks as JSON text."""
    tasks = default_mailbox.list_tasks(limit=1000)
    return json.dumps([t.model_dump() for t in tasks], default=str)
