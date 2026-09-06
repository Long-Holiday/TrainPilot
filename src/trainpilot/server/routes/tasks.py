"""FastAPI router for Agent-facing task APIs."""

import logging
import threading
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status

from trainpilot.common.schemas import (
    EventNotifyRequest,
    EventNotifyResponse,
    HeartbeatRequest,
    InstructionAckRequest,
    InstructionResponse,
    TaskDecisionRequest,
    TaskSummary,
)
from trainpilot.common.states import EventType, TaskState
from trainpilot.server.auth import verify_api_token
from trainpilot.server.config import settings
from trainpilot.server.feishu.client import default_feishu_client
from trainpilot.server.mailbox import default_mailbox

logger = logging.getLogger("trainpilot.api.tasks")

router = APIRouter(
    prefix="/api/tasks",
    tags=["Tasks"],
    dependencies=[Depends(verify_api_token)],
)


def _dispatch_feishu(req: EventNotifyRequest) -> None:
    """Background delivery of Feishu cards (never blocks GPU training loop)."""
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
        elif req.event_type == EventType.RECOVERY:
            default_feishu_client.send_recovery(
                task_id=req.task_id,
                solution=req.message,
                step=req.step,
                epoch=req.epoch,
                metrics=req.metrics,
                extra=req.extra,
            )
    except Exception as exc:
        logger.error("Failed to forward event %s to Feishu: %s", req.event_type, exc)


def _auto_self_resolve_callback(task_id: str, timeout_seconds: int) -> None:
    """Timer callback: auto self-resolve if no human decision arrived in time."""
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
    """Schedule a daemon timer that auto self-resolves the alert after timeout."""
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


@router.post("/notify", response_model=EventNotifyResponse, status_code=status.HTTP_200_OK)
def notify_event(req: EventNotifyRequest, background_tasks: BackgroundTasks) -> EventNotifyResponse:
    """Receive training events (alerts, milestones, completion) from GPU Agent."""
    current_state = default_mailbox.record_event(req)

    # Async dispatch: do not block the training loop on Feishu latency.
    background_tasks.add_task(_dispatch_feishu, req)

    # 告警卡片：超过阈值无人工点击则服务端自动视为“自行解决”。
    if req.event_type == EventType.ALERT:
        _schedule_auto_self_resolve(req.task_id)

    return EventNotifyResponse(
        success=True,
        task_id=req.task_id,
        state=current_state,
        message=f"Event {req.event_type.value} recorded. Current state: {current_state.value}",
    )


@router.get("/{task_id}/instruction", response_model=InstructionResponse)
def poll_instruction(
    task_id: str,
    pop: bool = Query(default=True, description="Whether to consume and transition state to RECOVERING"),
    wait_timeout: float = Query(
        default=0.0,
        ge=0.0,
        le=60.0,
        description="Long-polling timeout in seconds. Blocks until decision is ready or timeout occurs. 0 for immediate response.",
    ),
) -> InstructionResponse:
    """Poll for pending human instructions for a given task.

    Supports Long Polling: specify wait_timeout > 0 to hold connection until
    human decision arrives or timeout expires.
    """
    instruction = default_mailbox.get_instruction(task_id, pop=pop, wait_timeout=wait_timeout)
    return instruction


@router.post("/{task_id}/ack", status_code=status.HTTP_200_OK)
def ack_instruction(
    task_id: str,
    req: InstructionAckRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """Acknowledge execution of an instruction by the GPU Agent."""
    if req.task_id != task_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task ID mismatch: path has {task_id}, body has {req.task_id}",
        )
    try:
        new_state = default_mailbox.ack_instruction(
            task_id=task_id,
            instruction_id=req.instruction_id,
            action=req.action,
            status=req.status,
            message=req.message,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    # If successfully resolved, dispatch Feishu recovery card directly from ACK (eliminating duplicate notify)
    if req.status.lower() == "success" and (req.action == "self_resolve" or req.solution):
        solution_text = req.solution or req.message or "已完成现场自愈检查并恢复正常训练。"
        background_tasks.add_task(
            default_feishu_client.send_recovery,
            task_id=task_id,
            solution=solution_text,
            step=req.step,
            epoch=req.epoch,
            metrics=req.metrics,
        )

    return {
        "success": True,
        "task_id": task_id,
        "state": new_state.value,
        "message": f"Instruction acknowledged with status {req.status}",
    }


@router.post("/{task_id}/heartbeat", status_code=status.HTTP_200_OK)
def heartbeat(task_id: str, req: HeartbeatRequest) -> dict:
    """Record heartbeat from GPU Agent."""
    if req.task_id != task_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task ID mismatch in heartbeat: {task_id} vs {req.task_id}",
        )
    default_mailbox.record_heartbeat(req)
    return {"success": True, "task_id": task_id}


@router.post("/{task_id}/decision", response_model=InstructionResponse)
def submit_decision(
    task_id: str, req: TaskDecisionRequest, background_tasks: BackgroundTasks
) -> InstructionResponse:
    """Directly submit a human decision for a task (via Web UI, CLI, or test automation)."""
    if req.task_id != task_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task ID mismatch: {task_id} vs {req.task_id}",
        )
    try:
        instruction = default_mailbox.submit_decision(
            task_id=task_id,
            action=req.action,
            payload=req.payload,
            operator=req.operator or "api_operator",
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    # Update Feishu card asynchronously in background to avoid blocking API latency
    background_tasks.add_task(
        default_feishu_client.update_card_to_resolved,
        task_id=task_id,
        action=req.action,
        operator=req.operator or "API Operator",
    )

    return instruction


@router.get("/{task_id}/status", response_model=TaskSummary)
def get_task_status(task_id: str) -> TaskSummary:
    """Retrieve full status for a specific training task."""
    summary = default_mailbox.get_task(task_id)
    if not summary:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task '{task_id}' not found in mailbox")
    return summary


@router.get("/{task_id}/events")
def get_task_events(
    task_id: str,
    limit: int = Query(default=100, ge=1, le=500, description="Max events to return"),
    offset: int = Query(default=0, ge=0, description="Skip first N events (chronological)"),
) -> Dict[str, Any]:
    """Paginated event history for audit (chronological order)."""
    total = default_mailbox.get_events_count(task_id)
    if total is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task '{task_id}' not found in mailbox")
    events = default_mailbox.get_events(task_id, limit=limit, offset=offset)
    return {"task_id": task_id, "total": total, "limit": limit, "offset": offset, "events": events}


@router.get("", response_model=List[TaskSummary])
def list_tasks(
    limit: int = Query(default=100, ge=1, le=1000, description="Max tasks to return"),
    offset: int = Query(default=0, ge=0, description="Skip first N tasks"),
    state: Optional[TaskState] = Query(default=None, description="Filter by task state"),
    stale_only: bool = Query(default=False, description="Only return heartbeat-stale tasks"),
) -> List[TaskSummary]:
    """List tracked training tasks with pagination and optional filters."""
    return default_mailbox.list_tasks(limit=limit, offset=offset, state=state, stale_only=stale_only)
