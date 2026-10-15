"""FastAPI router for Feishu interactive card webhook callbacks."""

import json
import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status

from trainpilot.common.states import TaskState, WAIT_FOR_ARRANGEMENT
from trainpilot.server.background import feishu_dispatcher
from trainpilot.server.config import settings
from trainpilot.server.feishu.cards import (
    build_deferred_card,
    build_fix_card,
    build_plan_card,
    build_plan_cleared_card,
    build_resolved_card,
)
from trainpilot.server.feishu.client import default_feishu_client
from trainpilot.server.feishu.inbound import (
    InboundPlanMessage,
    is_message_event,
    parse_plan_message,
)
from trainpilot.server.mailbox import TaskNotFoundError, default_mailbox
from trainpilot.server.plans import MAX_PLAN_TEXT_CHARS, PlanBindingError, default_plans

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


def _handle_message_event(data: Dict[str, Any]) -> Dict[str, Any]:
    """Register (or clear) a next-task plan from an @-bot group message.

    Feishu retries a callback it does not see answered quickly, so this path does one
    INSERT and one dispatcher submit; the receipt card is sent out of band.
    """
    if not settings.enable_next_task_plans:
        return {"code": 0, "msg": "Next-task plans are disabled"}

    msg = parse_plan_message(data)
    if msg is None:
        return {"code": 0, "msg": "Ignored Feishu message event"}

    # `#fix(task) <solution>` comes first: it addresses a task explicitly, so a bare
    # `清除` in its body is part of the solution, not a clear command.
    if msg.fix_task_id or msg.fix_error:
        return _handle_fix_message(msg)

    if msg.is_clear:
        if msg.task_id:
            cleared = default_plans.clear_pending_for_task(msg.task_id)
        else:
            cleared = default_plans.clear_all_pending()
        logger.info(
            "Feishu clear command from %s cleared %d pending plan(s) (scope=%s)",
            msg.sender_open_id, cleared, msg.task_id or "global",
        )
        feishu_dispatcher.submit(
            default_feishu_client.send_card_to_chat,
            chat_id=msg.chat_id,
            card_dict=build_plan_cleared_card(
                cleared_count=cleared,
                scope_task_id=msg.task_id,
                operator=msg.sender_open_id,
            ),
            card_type="plan_cleared",
        )
        return {"code": 0, "msg": "Cleared pending next-task plans"}

    try:
        plan, created = default_plans.register(
            text=msg.text,
            task_id=msg.task_id,
            created_by=msg.sender_open_id,
            chat_id=msg.chat_id,
            source_message_id=msg.message_id,
        )
    except PlanBindingError as exc:
        # Every plan is bound to an existing task: a plan no report ever carries is
        # invisible for good, so a wrong task name is refused instead of stored.
        return _reject_plan(msg, exc.task_id)

    if not created:
        logger.info("Duplicate Feishu message %s; plan %s already registered",
                    msg.message_id, plan.plan_id)
        return {"code": 0, "msg": "Duplicate plan message ignored"}

    feishu_dispatcher.submit(
        default_feishu_client.send_card_to_chat,
        chat_id=msg.chat_id,
        card_dict=build_plan_card(
            plan_id=plan.plan_id,
            text=plan.text,
            task_id=plan.task_id,
            created_by=plan.created_by,
            created_at=plan.created_at,
        ),
        card_type="plan_registered",
        plan_id=plan.plan_id,
    )
    return {"code": 0, "msg": "Plan registered"}


def _reject_plan(msg: InboundPlanMessage, task_id: Optional[str]) -> Dict[str, Any]:
    """Refuse a plan whose binding is unusable and say why in the group.

    `task_id` is the one the sender named, or None when the message named no task. The
    receipt lists the tasks that do exist so the sender can correct the name in place.
    """
    reason = (
        f"任务 `{task_id}` 不存在" if task_id else "缺少任务名：请写成 `#任务名 <计划>`"
    )
    logger.warning("Refusing next-task plan from %s (task=%s)",
                   msg.sender_open_id, task_id or "none")
    feishu_dispatcher.submit(
        default_feishu_client.send_card_to_chat,
        chat_id=msg.chat_id,
        card_dict=build_plan_card(
            text=msg.text,
            task_id=task_id,
            created_by=msg.sender_open_id,
            reason=reason,
            known_task_ids=[t.task_id for t in default_mailbox.list_tasks(limit=5)],
        ),
        card_type="plan_rejected",
    )
    return {"code": 0, "msg": "Plan not registered"}


def _handle_fix_message(msg: InboundPlanMessage) -> Dict[str, Any]:
    """Dispatch a human-written fix to the task that is waiting for one.

    Reuses the plan path's conventions: the receipt card is the only feedback channel,
    it is sent only when something was actually dispatched, and a redelivered message
    is a silent no-op. A fix NEVER creates a task row -- an unknown task is refused.
    """
    def _reject(reason: str, task_id: str) -> Dict[str, Any]:
        feishu_dispatcher.submit(
            default_feishu_client.send_card_to_chat,
            chat_id=msg.chat_id,
            card_dict=build_fix_card(
                task_id=task_id,
                solution=msg.text,
                operator=msg.sender_open_id,
                reason=reason,
            ),
            card_type="fix_rejected",
        )
        return {"code": 0, "msg": "Fix not dispatched"}

    if msg.fix_error or not msg.fix_task_id:
        return _reject(msg.fix_error or "缺少任务名：请写成 `#fix(任务名) <方案>`", "-")

    task_id = msg.fix_task_id
    solution = msg.text
    if len(solution) > MAX_PLAN_TEXT_CHARS:
        logger.warning("Truncating fix solution for %s from %d to %d chars",
                       task_id, len(solution), MAX_PLAN_TEXT_CHARS)
        solution = solution[:MAX_PLAN_TEXT_CHARS]

    try:
        instruction, created = default_mailbox.submit_fix_decision(
            task_id=task_id,
            solution=solution,
            operator=msg.sender_open_id,
            source_message_id=msg.message_id,
        )
    except TaskNotFoundError:
        logger.warning("Refusing fix for unknown task %s (no task row is created)", task_id)
        return _reject(f"任务 `{task_id}` 不存在（#fix 不会为它创建任务行）", task_id)
    except ValueError as exc:
        logger.warning("Refusing fix for %s: %s", task_id, exc)
        return _reject(str(exc), task_id)

    if not created:
        logger.info("Duplicate fix message %s for task %s; no card sent", msg.message_id, task_id)
        return {"code": 0, "msg": "Duplicate fix message ignored"}

    logger.warning("Fix dispatched to task %s by %s via group message",
                   task_id, msg.sender_open_id)
    feishu_dispatcher.submit(
        default_feishu_client.send_card_to_chat,
        chat_id=msg.chat_id,
        card_dict=build_fix_card(
            task_id=task_id,
            solution=solution,
            operator=msg.sender_open_id,
            instruction_id=instruction.instruction_id if instruction else None,
            dispatched_at=instruction.decided_at if instruction else None,
        ),
        card_type="fix_dispatched",
    )
    # Replace the alert / waiting card in place, so the group sees one closed decision.
    feishu_dispatcher.submit(
        default_feishu_client.update_card_to_resolved,
        task_id=task_id,
        action="custom",
        operator=msg.sender_open_id,
        resolved_at=instruction.decided_at if instruction else None,
    )
    return {"code": 0, "msg": "Fix dispatched"}


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

    # 3. Handle group messages (@-mentioning the bot) carrying a next-task plan.
    # Must run before the card-action extraction below, whose empty value short-circuits.
    if is_message_event(data):
        return _handle_message_event(data)

    # 4. Handle interactive card action trigger (card.action.trigger)
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

    # 4b. 「等待安排」 is not a decision: it only pauses the auto-decision countdown and
    # asks the human to send `#fix(task) <solution>` in the group. The task stays WAITING
    # with nothing queued, so the agent keeps long-polling.
    if action == WAIT_FOR_ARRANGEMENT:
        try:
            summary = default_mailbox.defer_alert(task_id, operator_name)
        except TaskNotFoundError:
            logger.warning("Arrangement request for unknown task %s ignored", task_id)
            return _toast("error", f"任务 `{task_id}` 不存在或已被清理，无法暂停自动决策")
        except ValueError as exc:
            # Only a still-WAITING alert has a countdown to pause.
            logger.warning("Arrangement request rejected for task %s: %s", task_id, exc)
            return _toast("error", f"当前状态无需暂停自动决策（{exc}）")

        # Imported lazily so tests can patch `mcp_server.schedule_auto_resolve` by name.
        from trainpilot.server.mcp_server import schedule_auto_resolve

        schedule_auto_resolve(task_id, stage="arrangement")
        deferred_card = build_deferred_card(
            task_id=task_id,
            operator=operator_name,
            timeout_seconds=settings.arrangement_timeout_seconds,
            requested_at=summary.arrangement_requested_at,
            original_message=summary.latest_message,
            step=summary.latest_step,
            epoch=summary.latest_epoch,
        )
        background_tasks.add_task(
            default_feishu_client.update_card_to_deferred,
            task_id=task_id,
            operator=operator_name,
            timeout_seconds=settings.arrangement_timeout_seconds,
            requested_at=summary.arrangement_requested_at,
            original_message=summary.latest_message,
            step=summary.latest_step,
            epoch=summary.latest_epoch,
        )
        return {
            "toast": {
                "type": "success",
                "content": f"已暂停自动决策，请在群里 @机器人 发送 #fix({task_id}) <方案>",
            },
            "card": {
                "type": "raw",
                "data": deferred_card,
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
        "stop_training": "停止训练",
        "self_resolve": "自行解决",
    }
    toast_label = _toast_labels.get(action, action)
    return {
        "toast": {
            "type": "success",
            "content": f"策略 [{toast_label}] 已下发，训练将自动恢复",
        },
        "card": {
            "type": "raw",
            "data": resolved_card,
        },
    }
