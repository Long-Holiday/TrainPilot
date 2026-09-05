"""FastAPI router for Agent-facing task APIs."""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status

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
from trainpilot.server.feishu.client import default_feishu_client
from trainpilot.server.mailbox import default_mailbox

logger = logging.getLogger("trainpilot.api.tasks")

router = APIRouter(prefix="/api/tasks", tags=["Tasks"])


@router.post("/notify", response_model=EventNotifyResponse, status_code=status.HTTP_200_OK)
def notify_event(req: EventNotifyRequest) -> EventNotifyResponse:
    """Receive training events (alerts, milestones, completion) from GPU Agent."""
    current_state = default_mailbox.record_event(req)

    # Dispatch to Feishu
    try:
        if req.event_type == EventType.ALERT:
            default_feishu_client.send_alert(
                task_id=req.task_id,
                message=req.message,
                step=req.step,
                epoch=req.epoch,
                metrics=req.metrics,
                extra=req.extra,
            )
        elif req.event_type == EventType.MILESTONE:
            default_feishu_client.send_milestone(
                task_id=req.task_id,
                message=req.message,
                step=req.step,
                epoch=req.epoch,
                metrics=req.metrics,
            )
    except Exception as exc:
        logger.error("Failed to forward event %s to Feishu: %s", req.event_type, exc)

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
) -> InstructionResponse:
    """Poll for pending human instructions for a given task."""
    instruction = default_mailbox.get_instruction(task_id, pop=pop)
    return instruction


@router.post("/{task_id}/ack", status_code=status.HTTP_200_OK)
def ack_instruction(task_id: str, req: InstructionAckRequest) -> dict:
    """Acknowledge execution of an instruction by the GPU Agent."""
    if req.task_id != task_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task ID mismatch: path has {task_id}, body has {req.task_id}",
        )
    new_state = default_mailbox.ack_instruction(
        task_id=task_id,
        instruction_id=req.instruction_id,
        action=req.action,
        status=req.status,
        message=req.message,
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
def submit_decision(task_id: str, req: TaskDecisionRequest) -> InstructionResponse:
    """Directly submit a human decision for a task (via Web UI, CLI, or test automation)."""
    if req.task_id != task_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task ID mismatch: {task_id} vs {req.task_id}",
        )
    instruction = default_mailbox.submit_decision(
        task_id=task_id,
        action=req.action,
        payload=req.payload,
        operator=req.operator or "api_operator",
    )
    # Update Feishu card if active
    try:
        default_feishu_client.update_card_to_resolved(
            task_id=task_id,
            action=req.action,
            operator=req.operator or "API Operator",
        )
    except Exception as exc:
        logger.warning("Failed to patch card after direct decision: %s", exc)

    return instruction


@router.get("/{task_id}/status", response_model=TaskSummary)
def get_task_status(task_id: str) -> TaskSummary:
    """Retrieve full status for a specific training task."""
    summary = default_mailbox.get_task(task_id)
    if not summary:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task '{task_id}' not found in mailbox")
    return summary


@router.get("", response_model=List[TaskSummary])
def list_tasks() -> List[TaskSummary]:
    """List all tracked training tasks and their current states."""
    return default_mailbox.list_tasks()
