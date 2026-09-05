"""FastAPI router for Feishu interactive card webhook callbacks."""

import json
import logging
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Request, Response, status

from trainpilot.server.config import settings
from trainpilot.server.feishu.cards import build_resolved_card
from trainpilot.server.feishu.client import default_feishu_client
from trainpilot.server.mailbox import default_mailbox

logger = logging.getLogger("trainpilot.webhook.feishu")

router = APIRouter(prefix="/webhook", tags=["Feishu Webhook"])


@router.post("/feishu")
async def feishu_webhook(request: Request) -> Any:
    """Handle Feishu URL verification challenge and interactive card button clicks."""
    try:
        raw_body = await request.body()
        data: Dict[str, Any] = json.loads(raw_body.decode("utf-8"))
    except Exception as exc:
        logger.error("Failed to parse incoming webhook payload: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload")

    # 1. URL Verification Challenge (for initial Feishu app event subscription)
    if data.get("type") == "url_verification":
        challenge = data.get("challenge")
        token = data.get("token")
        if settings.feishu_verification_token and token != settings.feishu_verification_token:
            logger.warning("Feishu verification token mismatch during url_verification")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Verification token mismatch")
        logger.info("Responded to Feishu url_verification challenge")
        return {"challenge": challenge}

    # 2. Token validation if configured (strict: missing token also rejected)
    token = data.get("token") or (data.get("header", {}).get("token") if isinstance(data.get("header"), dict) else None)
    if settings.feishu_verification_token and token != settings.feishu_verification_token:
        logger.warning("Received invalid or missing token on Feishu webhook")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized token")

    # 3. Handle interactive card action trigger (card.action.trigger)
    # Extract action value and operator across v1 and v2 payload shapes
    action_value = {}
    operator_name = "飞书工程师"

    # Shape A: Direct card.action.trigger
    if "action" in data and isinstance(data["action"], dict):
        action_value = data["action"].get("value", {})
        operator_name = data.get("open_id", "feishu_user")
    # Shape B: Event schema 2.0
    elif "event" in data and isinstance(data["event"], dict):
        event_obj = data["event"]
        action_obj = event_obj.get("action", {})
        if isinstance(action_obj, dict):
            action_value = action_obj.get("value", {})
        operator_info = event_obj.get("operator", {})
        if isinstance(operator_info, dict):
            operator_name = operator_info.get("open_id", "feishu_user")

    task_id = action_value.get("task_id")
    action = action_value.get("action")
    payload = action_value.get("payload")

    if not task_id or not action:
        logger.warning("Received webhook event without task_id or action in value: %s", data)
        return {"code": 0, "msg": "Ignored non-card-action event"}

    logger.info("Received Feishu card action: task_id=%s, action=%s, operator=%s",
                task_id, action, operator_name)

    # 4. Save decision into Task Mailbox
    try:
        instruction = default_mailbox.submit_decision(
            task_id=task_id,
            action=action,
            payload=payload,
            operator=operator_name,
        )
    except ValueError as exc:
        # Terminal-state task: acknowledge to Feishu but do not resurrect.
        logger.warning("Webhook decision rejected for task %s: %s", task_id, exc)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    # 5. Build resolved card for in-place card replacement (anti-duplicate click)
    resolved_card = build_resolved_card(
        task_id=task_id,
        action_name=action,
        operator=operator_name,
        resolved_at=instruction.decided_at,
    )

    # Also notify client tracker
    default_feishu_client.update_card_to_resolved(
        task_id=task_id,
        action=action,
        operator=operator_name,
        resolved_at=instruction.decided_at,
    )

    # Feishu Interactive Card response format to update card on the fly:
    # Returning `card` updates the message in place.
    return {
        "toast": {
            "type": "success",
            "content": f"策略 [{action}] 已下发，训练将自动恢复",
        },
        "card": resolved_card,
    }
