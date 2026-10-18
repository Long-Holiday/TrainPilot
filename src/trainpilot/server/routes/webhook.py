"""FastAPI router for Feishu interactive card webhook callbacks."""

import json
import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status

from trainpilot.common.states import WAIT_FOR_ARRANGEMENT
from trainpilot.server.background import feishu_dispatcher
from trainpilot.server.config import settings
from trainpilot.server.feishu.cards import (
    build_awaiting_user_card,
    build_resolved_card,
)
from trainpilot.server.feishu.client import default_feishu_client
from trainpilot.server.mailbox import TaskNotFoundError, default_mailbox

logger = logging.getLogger("trainpilot.webhook.feishu")

router = APIRouter(prefix="/webhook", tags=["Feishu Webhook"])

# Feishu shows the clicker an opaque code for ANY non-200 answer from a card callback
# (200671) and for a malformed body (200672/200673), so a refused click still has to be
# answered 200 -- the toast in the body is the only channel that reaches the engineer.
CARD_ACTION_EVENT_TYPE = "card.action.trigger"


def _card_action_shape(data: Dict[str, Any]) -> Optional[str]:
    """Classify a card callback payload: ``"v2"``, the legacy ``"v1"``, or None.

    The two shapes differ in more than nesting: only the v2 ``card.action.trigger``
    payload carries the Verification Token (in ``header.token``). A v1 payload's
    top-level ``token`` is the ``c-xxxx`` card-update credential, so it can never
    match the verification token -- comparing it is what turned every v1 click into
    a 403 and therefore into a 200671 in the client.
    """
    for container in (data.get("header"), data.get("event")):
        if isinstance(container, dict) and container.get("event_type") == CARD_ACTION_EVENT_TYPE:
            return "v2"
    action = data.get("action")
    if isinstance(action, dict) and isinstance(action.get("value"), dict):
        return "v1"
    return None


def _toast(toast_type: str, content: str) -> Dict[str, Any]:
    """Answer a card callback with HTTP 200 and a toast, leaving the card untouched."""
    return {"toast": {"type": toast_type, "content": content}}


@router.post("/feishu")
# Same handler under the trailing-slash form: the console URL is pasted by hand, and a
# stray "/" would otherwise be swallowed by the MCP mount at "/" and answered 404 --
# which the client reports as 200671 on every click.
@router.post("/feishu/", include_in_schema=False)
async def feishu_webhook(request: Request, background_tasks: BackgroundTasks) -> Any:
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

    # 2. Token validation if configured (strict: missing token also rejected).
    # A rejected *card action* is answered 200 with an explanatory toast instead of 403:
    # the click is refused either way, but 403 only ever surfaces as the client-side
    # 200671, which tells the engineer nothing.
    shape = _card_action_shape(data)
    token = data.get("token") or (data.get("header", {}).get("token") if isinstance(data.get("header"), dict) else None)
    if settings.feishu_verification_token and token != settings.feishu_verification_token:
        if shape == "v1":
            logger.warning("Rejected legacy card callback: this payload shape carries no "
                           "verification token, only the c-xxxx card credential")
            return _toast(
                "error",
                "该按钮回调是旧版卡片回调，没有可校验的 Verification Token。请到开发者后台"
                "「事件与回调 → 回调配置」改用新版「卡片回传交互」回调（card.action.trigger）。",
            )
        if shape == "v2":
            logger.warning("Rejected card action with invalid verification token")
            return _toast(
                "error",
                "验证令牌不匹配，操作未生效。请核对 TRAINPILOT_FEISHU_VERIFICATION_TOKEN "
                "与开发者后台的 Verification Token 是否一致。",
            )
        logger.warning("Received invalid or missing token on Feishu webhook")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized token")

    # 3. Only interactive card actions are consumed. Group message events are
    # deliberately ignored: there is no plan/#fix/#feat ingestion and the control
    # plane never pulls a decision out of chat text.
    # Handle interactive card action trigger (card.action.trigger)
    if shape is None:
        logger.warning("Received webhook event without task_id or action in value: %s", data)
        return {"code": 0, "msg": "Ignored non-card-action event"}

    action_value: Dict[str, Any] = {}
    operator_name = "飞书工程师"
    if shape == "v2":
        event_obj = data["event"] if isinstance(data.get("event"), dict) else {}
        action_obj = event_obj.get("action")
        if isinstance(action_obj, dict):
            action_value = action_obj.get("value") or {}
        operator_info = event_obj.get("operator")
        if isinstance(operator_info, dict):
            operator_name = operator_info.get("open_id", "feishu_user")
    else:
        action_value = data["action"].get("value") or {}
        operator_name = data.get("open_id", "feishu_user")

    task_id = action_value.get("task_id")
    action = action_value.get("action")
    payload = action_value.get("payload")

    if not task_id or not action:
        logger.warning("Received card action without task_id or action in value: %s", data)
        return {"code": 0, "msg": "Ignored card action without a decision"}

    logger.info("Received Feishu card action (%s): task_id=%s, action=%s, operator=%s",
                shape, task_id, action, operator_name)

    # 4b. 「等待用户」 is not a decision and queues nothing. It keeps the task WAITING,
    # suppresses the alert timer with no fallback, wakes the agent's long poll, and
    # hands the incident to a direct conversation between the engineer and the agent.
    if action == WAIT_FOR_ARRANGEMENT:
        try:
            summary = default_mailbox.wait_for_direct_conversation(task_id, operator_name)
        except TaskNotFoundError:
            logger.warning("Direct-conversation request for unknown task %s ignored", task_id)
            return _toast("error", f"任务 `{task_id}` 不存在或已被清理，无法暂停自动决策")
        except ValueError as exc:
            # Only a still-WAITING alert can be handed to a direct conversation.
            logger.warning("Direct-conversation request rejected for task %s: %s", task_id, exc)
            return _toast("error", f"当前状态无法进入等待沟通（{exc}）")

        awaiting_card = build_awaiting_user_card(
            task_id=task_id,
            operator=operator_name,
            requested_at=summary.direct_conversation_requested_at,
            original_message=summary.latest_message,
            step=summary.latest_step,
            epoch=summary.latest_epoch,
        )
        background_tasks.add_task(
            default_feishu_client.update_card_to_awaiting_user,
            task_id=task_id,
            operator=operator_name,
            requested_at=summary.direct_conversation_requested_at,
            original_message=summary.latest_message,
            step=summary.latest_step,
            epoch=summary.latest_epoch,
        )
        return {
            "toast": {
                "type": "success",
                "content": "已暂停自动决策，请直接在对话中告诉 Agent 你的处理方案",
            },
            "card": {
                "type": "raw",
                "data": awaiting_card,
            },
        }

    # 5. Save decision into Task Mailbox
    try:
        instruction = default_mailbox.submit_decision(
            task_id=task_id,
            action=action,
            payload=payload,
            operator=operator_name,
        )
    except TaskNotFoundError:
        # No task row is ever created from a card click: a button on a card whose task is
        # gone (retention cleanup, replaced database) is stale, not a new task.
        logger.warning("Card action for unknown task %s ignored", task_id)
        return _toast("error", f"任务 `{task_id}` 不存在或已被清理，该决策未生效")
    except ValueError as exc:
        logger.warning("Webhook decision rejected for task %s: %s", task_id, exc)
        return _toast("error", f"决策未生效：{exc}")
    except Exception:
        # Feishu's callback protocol has no way to show a server error: anything other
        # than HTTP 200 reaches the engineer as an opaque 200671, so a failure here is
        # reported as a toast and left to the logs.
        logger.exception("Failed to record decision for task %s", task_id)
        return _toast("error", "服务器内部错误，决策可能未生效，请查看服务端日志")

    # 6. Build resolved card for in-place card replacement (anti-duplicate click)
    resolved_card = build_resolved_card(
        task_id=task_id,
        action_name=action,
        operator=operator_name,
        resolved_at=instruction.decided_at,
    )

    # Record resolution in card history for monitoring, returning card in response updates it in-place
    try:
        record = {
            "task_id": task_id,
            "message_id": default_mailbox.get_task_feishu_message_id(task_id),
            "action": action,
            "operator": operator_name,
            "card": resolved_card,
            "timestamp": instruction.decided_at,
        }
        default_feishu_client._sent_cards_history.append({"type": "resolved_update", "data": record})
    except Exception:
        # The decision is already queued; only the in-place card refresh degrades.
        logger.exception("Failed to record resolved card for task %s", task_id)

    # Asynchronously patch Feishu message as double insurance (in case of client sync issues)
    background_tasks.add_task(
        default_feishu_client.update_card_to_resolved,
        task_id=task_id,
        action=action,
        operator=operator_name,
        resolved_at=instruction.decided_at,
    )

    # Feishu Interactive Card response format to update card on the fly:
    # According to Feishu card.action.trigger callback protocol:
    # `card` must be an object with `type: "raw"` and `data: <card_json>`
    # Returning plain card dict without `type` causes Feishu error 200672 (invalid response body format).
    _toast_labels = {
        "cancel": "取消任务并关机",
        "stop_training": "取消任务并关机",
        "self_resolve": "自行解决",
    }
    toast_label = _toast_labels.get(action, action)
    if action in ("cancel", "stop_training"):
        toast_content = (
            f"策略 [{toast_label}] 已下发；关机必须由 Agent 执行并验证，"
            "共享机器上无法安全关机时 Agent 会回报 failed"
        )
    else:
        toast_content = f"策略 [{toast_label}] 已下发，训练将自动恢复"
    return {
        "toast": {
            "type": "success",
            "content": toast_content,
        },
        "card": {
            "type": "raw",
            "data": resolved_card,
        },
    }
