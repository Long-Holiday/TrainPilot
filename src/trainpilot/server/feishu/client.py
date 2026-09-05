"""Feishu API client wrapper with lark-oapi and mock support."""

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from trainpilot.server.config import ServerSettings, settings
from trainpilot.server.feishu.cards import (
    build_alert_card,
    build_milestone_card,
    build_recovery_card,
    build_resolved_card,
)

logger = logging.getLogger("trainpilot.feishu")


class FeishuCardClient:
    """Client for dispatching and updating Feishu cards with fallback mock support."""

    def __init__(self, server_settings: Optional[ServerSettings] = None):
        self.settings = server_settings or settings
        self._sent_cards_history: List[Dict[str, Any]] = []
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
        return self._sent_cards_history

    def send_alert(
        self,
        task_id: str,
        message: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
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
    ) -> Optional[Dict[str, Any]]:
        """Update existing card to 'Resolved' status on Feishu to prevent multiple clicks."""
        resolved_card = build_resolved_card(
            task_id=task_id,
            action_name=action,
            operator=operator,
            original_message=original_message,
            resolved_at=resolved_at,
        )

        message_id = self._task_message_map.get(task_id)

        record = {
            "task_id": task_id,
            "message_id": message_id,
            "action": action,
            "operator": operator,
            "card": resolved_card,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._sent_cards_history.append({"type": "resolved_update", "data": record})

        if self._lark_client and message_id:
            try:
                import lark_oapi as lark
                from lark_oapi.api.im.v1 import PatchMessageRequest, PatchMessageRequestBody

                request = PatchMessageRequest.builder() \
                    .message_id(message_id) \
                    .request_body(PatchMessageRequestBody.builder()
                                  .content(json.dumps(resolved_card))
                                  .build()) \
                    .build()

                response = self._lark_client.im.v1.message.patch(request)
                if not response.success():
                    logger.error("Failed to patch Feishu message %s: code=%s, msg=%s",
                                 message_id, response.code, response.msg)
                else:
                    logger.info("Successfully patched Feishu message %s to resolved for task %s", message_id, task_id)
            except Exception as e:
                logger.error("Exception occurred while patching Feishu card: %s", e)

        return resolved_card

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

        if self._lark_client and self.settings.feishu_receiver_id:
            try:
                import lark_oapi as lark
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

        return entry


# Singleton default client
default_feishu_client = FeishuCardClient()
