"""FastAPI router exposing the operator-facing task APIs.

Agents talk to the control plane over MCP (`/mcp`); these REST endpoints exist for
operations, dashboards, and decision injection (Web UI, Feishu webhook, tests).
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from trainpilot.common.schemas import (
    InstructionResponse,
    TaskDecisionRequest,
    TaskSummary,
)
from trainpilot.common.states import TaskState
from trainpilot.server.auth import verify_api_token
from trainpilot.server.background import feishu_dispatcher
from trainpilot.server.feishu.client import default_feishu_client
from trainpilot.server.mailbox import TaskNotFoundError, default_mailbox

logger = logging.getLogger("trainpilot.api.tasks")

router = APIRouter(
    prefix="/api/tasks",
    tags=["Tasks"],
    dependencies=[Depends(verify_api_token)],
)


@router.post("/{task_id}/decision", response_model=InstructionResponse)
def submit_decision(
    task_id: str, req: TaskDecisionRequest
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
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    # Update Feishu card asynchronously in background to avoid blocking API latency
    feishu_dispatcher.submit(
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
) -> List[TaskSummary]:
    """List tracked training tasks with pagination and optional filters."""
    return default_mailbox.list_tasks(limit=limit, offset=offset, state=state)
