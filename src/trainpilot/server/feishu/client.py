"""Feishu API client wrapper with lark-oapi and mock support."""

import json
import logging
from collections import deque
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from trainpilot.common.states import WAIT_FOR_ARRANGEMENT
from trainpilot.server.config import ServerSettings, settings
from trainpilot.server.feishu.cards import (
    build_alert_card,
    build_deferred_card,
    build_milestone_card,
    build_recovery_card,
    build_resolved_card,
)

logger = logging.getLogger("trainpilot.feishu")


class FeishuCardClient:
    """Client for dispatching and updating Feishu cards with fallback mock support."""

    def __init__(self, server_settings: Optional[ServerSettings] = None):
        self.settings = server_settings or settings
        # Use bounded deque (maxlen=200) to prevent unbounded memory growth on long-running servers
        self._sent_cards_history: deque = deque(maxlen=200)
        self._task_message_map: Dict[str, str] = {}  # task_id -> latest message_id
        self._lark_client = None

        if self.settings.is_feishu_configured:
            try:
                import lark_oapi as lark
                self._lark_client = lark.Client.builder() \
                    .app_id(self.settings.feishu_app_id) \
                    .app_secret(self.settings.feishu_app_secret) \
                    .log_level(lark.LogLevel.INFO if not self.settings.debug else lark.LogLevel.DEBUG) \
                    .build()
                logger.info("Initialized real Feishu lark_oapi client for app_id: %s", self.settings.feishu_app_id)
            except Exception as e:
                logger.warning("Failed to initialize lark_oapi client (%s), falling back to mock mode", e)
        else:
            logger.info("Feishu credentials not configured or mock enabled; running Feishu client in Mock/Log mode")

    @property
    def sent_cards_history(self) -> List[Dict[str, Any]]:
        """Return history of sent cards (useful for test assertions and dev logs)."""
        return list(self._sent_cards_history)

    def send_alert(
        self,
        task_id: str,
        message: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
        agent_note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send an interactive alert card to the configured receiver."""
        if timeout_seconds is None:
            try:
                timeout_seconds = int(self.settings.alert_decision_timeout_seconds)
            except Exception:
                timeout_seconds = 30
        card_content = build_alert_card(
            task_id=task_id,
            message=message,
            step=step,
            epoch=epoch,
            metrics=metrics,
            extra=extra,
            timeout_seconds=timeout_seconds,
            agent_note=agent_note,
        )
        return self._send_card(
            task_id=task_id,
            card_dict=card_content,
            card_type="alert",
        )

    def send_milestone(
        self,
        task_id: str,
        message: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        agent_note: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a milestone notification card to the configured receiver."""
        card_content = build_milestone_card(
            task_id=task_id,
            message=message,
            step=step,
            epoch=epoch,
            metrics=metrics,
            agent_note=agent_note,
            extra=extra,
        )
        return self._send_card(
            task_id=task_id,
            card_dict=card_content,
            card_type="milestone",
        )

    def send_recovery(
        self,
        task_id: str,
        solution: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a recovery notification card to the configured receiver."""
        card_content = build_recovery_card(
            task_id=task_id,
            solution=solution,
            step=step,
            epoch=epoch,
            metrics=metrics,
            extra=extra,
        )
        return self._send_card(
            task_id=task_id,
            card_dict=card_content,
            card_type="recovery",
        )

    def update_card_to_resolved(
        self,
        task_id: str,
        action: str,
        operator: str = "专家工程师",
        original_message: Optional[str] = None,
        resolved_at: Optional[str] = None,
        trigger_note: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update existing card to 'Resolved' status on Feishu to prevent multiple clicks."""
        resolved_card = build_resolved_card(
            task_id=task_id,
            action_name=action,
            operator=operator,
            original_message=original_message,
            resolved_at=resolved_at,
            trigger_note=trigger_note,
        )
        return self._patch_task_card(
            task_id, resolved_card, "resolved_update",
            {"action": action, "operator": operator},
        )

    def update_card_to_deferred(
        self,
        task_id: str,
        operator: str = "专家工程师",
        timeout_seconds: int = 1800,
        requested_at: Optional[str] = None,
        original_message: Optional[str] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Replace the alert card with the 'waiting for arrangement' status card."""
        deferred_card = build_deferred_card(
            task_id=task_id,
            operator=operator,
            timeout_seconds=timeout_seconds,
            requested_at=requested_at,
            original_message=original_message,
            step=step,
            epoch=epoch,
        )
        return self._patch_task_card(
            task_id, deferred_card, "arrangement_update",
            {"action": WAIT_FOR_ARRANGEMENT, "operator": operator},
        )

    def _patch_task_card(
        self,
        task_id: str,
        card_dict: Dict[str, Any],
        card_type: str,
        extra: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """PATCH the task's latest Feishu message in place, recording history either way."""
        # Retrieve message_id from SQLite/mailbox first (survives restarts), fallback to local map
        message_id = None
        try:
            from trainpilot.server.mailbox import default_mailbox
            message_id = default_mailbox.get_task_feishu_message_id(task_id)
        except Exception:
            pass
        if not message_id:
            message_id = self._task_message_map.get(task_id)

        record = {
            "task_id": task_id,
            "message_id": message_id,
            **extra,
            "card": card_dict,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._sent_cards_history.append({"type": card_type, "data": record})

        if self._lark_client and message_id:
            try:
                from lark_oapi.api.im.v1 import PatchMessageRequest, PatchMessageRequestBody

                request = PatchMessageRequest.builder() \
                    .message_id(message_id) \
                    .request_body(PatchMessageRequestBody.builder()
                                  .content(json.dumps(card_dict))
                                  .build()) \
                    .build()

                response = self._lark_client.im.v1.message.patch(request)
                if not response.success():
                    logger.error("Failed to patch Feishu message %s: code=%s, msg=%s",
                                 message_id, response.code, response.msg)
                else:
                    logger.info("Successfully patched Feishu message %s (%s) for task %s",
                                message_id, card_type, task_id)
            except Exception as e:
                logger.error("Exception occurred while patching Feishu card: %s", e)

        return card_dict

    def _send_card(
        self,
        task_id: str,
        card_dict: Dict[str, Any],
        card_type: str,
    ) -> Dict[str, Any]:
        """Internal helper to dispatch card via real lark_oapi or mock record."""
        simulated_msg_id = f"mock_msg_{task_id}_{len(self._sent_cards_history) + 1}"
        self._task_message_map[task_id] = simulated_msg_id

        entry = {
            "type": card_type,
            "task_id": task_id,
            "message_id": simulated_msg_id,
            "receiver_id": self.settings.feishu_receiver_id or "mock_receiver",
            "receive_id_type": self.settings.feishu_receive_id_type,
            "card": card_dict,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._sent_cards_history.append(entry)

        final_msg_id = simulated_msg_id

        if self._lark_client and self.settings.feishu_receiver_id:
            try:
                from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

                request = CreateMessageRequest.builder() \
                    .receive_id_type(self.settings.feishu_receive_id_type) \
                    .request_body(CreateMessageRequestBody.builder()
                                  .receive_id(self.settings.feishu_receiver_id)
                                  .msg_type("interactive")
                                  .content(json.dumps(card_dict))
                                  .build()) \
                    .build()

                response = self._lark_client.im.v1.message.create(request)
                if response.success() and response.data:
                    real_msg_id = getattr(response.data, "message_id", None)
                    if real_msg_id:
                        entry["message_id"] = real_msg_id
                        self._task_message_map[task_id] = real_msg_id
                        final_msg_id = real_msg_id
                    logger.info("Dispatched Feishu %s card for task %s, msg_id=%s",
                                card_type, task_id, real_msg_id)
                else:
                    logger.error("Feishu API error sending %s card: code=%s, msg=%s",
                                 card_type, response.code, response.msg)
            except Exception as e:
                logger.error("Exception during Feishu send_card: %s", e)
        else:
            logger.info("[Mock Feishu] Dispatched %s card for task %s: %s",
                        card_type, task_id, json.dumps(card_dict, ensure_ascii=False))

        # For interactive alert cards, persist message_id to SQLite so restart won't break PATCH
        if card_type == "alert":
            try:
                from trainpilot.server.mailbox import default_mailbox
                default_mailbox.set_task_feishu_message_id(task_id, final_msg_id)
            except Exception:
                pass

        return entry

    def send_card_to_chat(
        self,
        chat_id: str,
        card_dict: Dict[str, Any],
        card_type: str,
        plan_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a card to the chat a message came from.

        Deliberately separate from `_send_card`: that one targets the configured
        alert receiver and maintains the task->message map, while receipts for
        next-task plans must go back to the originating group and are not tied to
        any task row.
        """
        target_chat = chat_id or self.settings.feishu_receiver_id or "mock_chat"
        simulated_msg_id = f"mock_msg_{plan_id or card_type}_{len(self._sent_cards_history) + 1}"

        entry = {
            "type": card_type,
            "plan_id": plan_id,
            "chat_id": target_chat,
            "message_id": simulated_msg_id,
            "receiver_id": target_chat,
            "receive_id_type": "chat_id",
            "card": card_dict,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._sent_cards_history.append(entry)

        if self._lark_client:
            try:
                from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

                request = CreateMessageRequest.builder() \
                    .receive_id_type("chat_id") \
                    .request_body(CreateMessageRequestBody.builder()
                                  .receive_id(target_chat)
                                  .msg_type("interactive")
                                  .content(json.dumps(card_dict, ensure_ascii=False))
                                  .build()) \
                    .build()

                response = self._lark_client.im.v1.message.create(request)
                if response.success() and response.data:
                    real_msg_id = getattr(response.data, "message_id", None)
                    if real_msg_id:
                        entry["message_id"] = real_msg_id
                    logger.info("Dispatched Feishu %s card to chat %s, msg_id=%s",
                                card_type, target_chat, real_msg_id)
                else:
                    logger.error("Feishu API error sending %s card: code=%s, msg=%s",
                                 card_type, response.code, response.msg)
            except Exception as e:
                logger.error("Exception during Feishu send_card_to_chat: %s", e)
        else:
            logger.info("[Mock Feishu] Dispatched %s card for chat %s: %s",
                        card_type, target_chat, json.dumps(card_dict, ensure_ascii=False))

        return entry


# Singleton default client
default_feishu_client = FeishuCardClient()
