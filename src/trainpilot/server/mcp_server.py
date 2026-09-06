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
    HeartbeatRequest,
)
from trainpilot.common.states import EventType, TaskState
from trainpilot.server.config import settings
from trainpilot.server.feishu.client import default_feishu_client
from trainpilot.server.mailbox import default_mailbox

logger = logging.getLogger("trainpilot.server.mcp")

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
    """Run Feishu card delivery in a background thread so MCP tool returns immediately."""
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

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


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


@mcp_server.tool()
def report_milestone(
    task_id: str,
    message: str,
    step: int = 0,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    agent_note: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Report a training milestone with metrics and an AI agent commentary note.

    Pushes a rich green milestone card to Feishu and updates the task state.

    Args:
        task_id: Unique identifier for the training task.
        message: Description of the milestone (e.g. 'Epoch 1 completed').
        step: Current training step.
        epoch: Current training epoch (optional).
        metrics: Dictionary of training metrics (e.g. {'loss': 0.35, 'val_loss': 0.40}).
        agent_note: Autonomous 1-3 sentence analysis/recommendation written by the AI agent.
        extra: Additional contextual metadata.
    """
    req = EventNotifyRequest(
        task_id=task_id,
        event_type=EventType.MILESTONE,
        message=message,
        step=step,
        epoch=epoch,
        metrics=metrics,
        agent_note=agent_note,
        extra=extra,
    )
    current_state = default_mailbox.record_event(req)
    _dispatch_feishu_async(req)

    return {
        "success": True,
        "task_id": task_id,
        "state": current_state.value,
        "message": f"Milestone recorded. Current state: {current_state.value}",
    }


@mcp_server.tool()
def report_alert(
    task_id: str,
    message: str,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    agent_note: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Report a training anomaly (NaN/Inf loss, loss spike, OOM) and freeze training.

    Transitions task to WAITING, pushes a red interactive alert card with action buttons
    to Feishu, and arms a fallback auto-decision timer (default 30s).

    Args:
        task_id: Unique identifier for the training task.
        message: Anomaly description (e.g. 'Loss NaN detected at step 1450').
        step: Step number where the anomaly occurred.
        epoch: Current epoch number.
        metrics: Anomaly metrics (e.g. {'loss': 'NaN'}).
        agent_note: Optional diagnosis note from the AI agent.
        extra: Additional error traceback or diagnostic details.
    """
    req = EventNotifyRequest(
        task_id=task_id,
        event_type=EventType.ALERT,
        message=message,
        step=step,
        epoch=epoch,
        metrics=metrics,
        agent_note=agent_note,
        extra=extra,
    )
    current_state = default_mailbox.record_event(req)
    _dispatch_feishu_async(req)
    _schedule_auto_self_resolve(task_id)

    return {
        "success": True,
        "task_id": task_id,
        "state": current_state.value,
        "message": f"Alert recorded and training frozen. Current state: {current_state.value}",
    }


@mcp_server.tool()
def poll_instruction(
    task_id: str,
    wait_timeout: float = 20.0,
    pop: bool = True,
) -> Dict[str, Any]:
    """Poll for pending human-in-the-loop instructions for a training task.

    Supports long-polling: holds connection up to wait_timeout seconds until a
    decision arrives or timeout expires.

    Args:
        task_id: Unique identifier for the training task.
        wait_timeout: Long-polling timeout in seconds (0 for immediate non-blocking return).
        pop: If true, consumes the instruction and transitions state to RECOVERING.
    """
    instruction = default_mailbox.get_instruction(task_id, pop=pop, wait_timeout=wait_timeout)
    res = instruction.model_dump()
    res["has_instruction"] = res.get("ready", False)
    return res


@mcp_server.tool()
def ack_instruction(
    task_id: str,
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
        task_id: Unique identifier for the training task.
        instruction_id: ID of the instruction executed (optional).
        action: The action performed (e.g. 'self_resolve', 'stop_training').
        status: Result status ('success' or 'failed').
        solution: Human-readable description of how the issue was fixed.
        message: Additional execution log or message.
        step: Current training step after recovery.
        epoch: Current epoch after recovery.
        metrics: Metrics after recovery.
    """
    try:
        new_state = default_mailbox.ack_instruction(
            task_id=task_id,
            instruction_id=instruction_id,
            action=action,
            status=status,
            message=message,
        )
    except ValueError as exc:
        return {"success": False, "task_id": task_id, "error": str(exc)}

    # Send recovery card to Feishu if successfully resolved
    if status.lower() == "success" and (action == "self_resolve" or bool(solution)):
        solution_text = solution or message or "已完成现场自愈检查并恢复正常训练。"

        def _send_recovery():
            try:
                default_feishu_client.send_recovery(
                    task_id=task_id,
                    solution=solution_text,
                    step=step,
                    epoch=epoch,
                    metrics=metrics,
                )
            except Exception as exc:
                logger.error("Failed to send Feishu recovery card for task %s: %s", task_id, exc)

        t = threading.Thread(target=_send_recovery, daemon=True)
        t.start()

    return {
        "success": True,
        "task_id": task_id,
        "state": new_state.value,
        "message": f"Instruction acknowledged with status '{status}'",
    }


@mcp_server.tool()
def send_heartbeat(
    task_id: str,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    status: str = "running",
) -> Dict[str, Any]:
    """Send GPU node heartbeat to maintain active status and report resource usage.

    Args:
        task_id: Unique identifier for the training task.
        step: Current training step.
        epoch: Current training epoch.
        metrics: GPU utilization, memory usage, or other telemetry.
        status: Current health status of the worker.
    """
    req = HeartbeatRequest(
        task_id=task_id,
        step=step,
        epoch=epoch,
        metrics=metrics,
        status=status,
    )
    default_mailbox.record_heartbeat(req)
    return {"success": True, "task_id": task_id}


@mcp_server.tool()
def get_task_status(task_id: str) -> Dict[str, Any]:
    """Retrieve current state, heartbeat, and instruction mailbox summary for a task.

    Args:
        task_id: Unique identifier for the training task.
    """
    summary = default_mailbox.get_task(task_id)
    if not summary:
        return {"success": False, "task_id": task_id, "error": f"Task '{task_id}' not found"}
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
        stale_only: Only return tasks whose heartbeat has timed out.
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
    task_id: str,
    action: str,
    payload: Optional[Dict[str, Any]] = None,
    operator: str = "mcp_operator",
) -> Dict[str, Any]:
    """Directly submit an operator decision for a frozen task (for administration or tests).

    Args:
        task_id: Unique identifier for the training task.
        action: Decision action ('self_resolve', 'stop_training', or custom action).
        payload: Optional parameters for the action.
        operator: Name or role of the operator making the decision.
    """
    try:
        instruction = default_mailbox.submit_decision(
            task_id=task_id,
            action=action,
            payload=payload,
            operator=operator,
        )
    except ValueError as exc:
        return {"success": False, "task_id": task_id, "error": str(exc)}

    def _patch_card():
        try:
            default_feishu_client.update_card_to_resolved(
                task_id=task_id,
                action=action,
                operator=operator,
            )
        except Exception as exc:
            logger.warning("Failed to patch card after decision for %s: %s", task_id, exc)

    t = threading.Thread(target=_patch_card, daemon=True)
    t.start()

    return {
        "success": True,
        "task_id": task_id,
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
