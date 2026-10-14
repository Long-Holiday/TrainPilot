"""Parsing of inbound Feishu `im.message.receive_v1` events into next-task plans.

Pure functions only (no I/O), mirroring how `cards.py` isolates card construction:
the webhook router stays a thin dispatch layer and every rule below is unit-testable.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("trainpilot.feishu.inbound")

MESSAGE_EVENT_TYPE = "im.message.receive_v1"

# Exact-match control words that clear queued plans instead of registering one.
CLEAR_COMMANDS = frozenset({"清除", "取消", "clear"})

# Feishu renders an @-mention in the text as `@_user_N` and also lists it in
# `message.mentions`; strip both.
_MENTION_TOKEN_RE = re.compile(r"@_user_\d+")
_AT_TAG_RE = re.compile(r"<at[^>]*>.*?</at>", re.DOTALL)

# `#task-id` binds the plan to one task. The first character may not be `/` or `#`
# so `#/detail` and `C#` are left alone; the tag is removed from the plan body.
_TASK_TAG_RE = re.compile(r"#([A-Za-z0-9][A-Za-z0-9_.:/-]{0,63})")


@dataclass(frozen=True)
class InboundPlanMessage:
    """A Feishu group message that carries a next-task plan (or a clear command)."""

    chat_id: str
    chat_type: str
    message_id: str
    sender_open_id: str
    text: str
    task_id: Optional[str]
    is_clear: bool


def is_message_event(data: Dict[str, Any]) -> bool:
    """True when the webhook payload is a Feishu message event rather than a card action."""
    header = data.get("header")
    if isinstance(header, dict) and header.get("event_type") == MESSAGE_EVENT_TYPE:
        return True
    event = data.get("event")
    if isinstance(event, dict) and event.get("type") == MESSAGE_EVENT_TYPE:
        return True
    return False


def _extract_text(message: Dict[str, Any]) -> str:
    """Decode the JSON-string `content` field; anything unusable becomes empty."""
    content = message.get("content")
    if isinstance(content, dict):
        payload = content
    else:
        try:
            payload = json.loads(content or "")
        except Exception:
            return ""
    if not isinstance(payload, dict):
        return ""
    text = payload.get("text")
    return text if isinstance(text, str) else ""


def _strip_mentions(raw: str, message: Dict[str, Any]) -> str:
    """Remove @-mention markup so the plan body is what the human actually typed."""
    text = _AT_TAG_RE.sub("", raw)
    text = _MENTION_TOKEN_RE.sub("", text)
    mentions = message.get("mentions")
    if isinstance(mentions, list):
        for mention in mentions:
            if isinstance(mention, dict):
                key = mention.get("key")
                if isinstance(key, str) and key:
                    text = text.replace(key, "")
    return text


def _has_mention(raw: str, message: Dict[str, Any]) -> bool:
    """Whether the message @-mentioned anyone.

    Feishu only delivers group messages that @ the bot under the documented
    `im:message.group_at_msg` permission, and this needs no bot open_id. It cannot
    prove the mentioned entity is *our* bot, so the confirmation card always echoes
    what was stored.
    """
    if _MENTION_TOKEN_RE.search(raw):
        return True
    mentions = message.get("mentions")
    return isinstance(mentions, list) and len(mentions) > 0


def parse_plan_message(data: Dict[str, Any]) -> Optional[InboundPlanMessage]:
    """Turn a message event into a plan, or None when it must be ignored."""
    event = data.get("event")
    if not isinstance(event, dict):
        return None
    message = event.get("message")
    if not isinstance(message, dict):
        return None

    if (message.get("message_type") or "").lower() != "text":
        logger.info("Ignoring non-text Feishu message event")
        return None

    sender = event.get("sender")
    if isinstance(sender, dict):
        sender_type = sender.get("sender_type")
        if sender_type and sender_type != "user":
            logger.info("Ignoring Feishu message from sender_type '%s'", sender_type)
            return None
        sender_id = sender.get("sender_id")
        sender_open_id = (
            sender_id.get("open_id") if isinstance(sender_id, dict) else None
        )
    else:
        sender_open_id = None

    message_id = message.get("message_id")
    chat_id = message.get("chat_id")
    chat_type = (message.get("chat_type") or "").lower()
    if not message_id or not chat_id:
        logger.warning("Feishu message event missing message_id or chat_id")
        return None

    raw_text = _extract_text(message)
    if chat_type == "group" and not _has_mention(raw_text, message):
        logger.info("Ignoring group message that did not @ the bot")
        return None

    text = _strip_mentions(raw_text, message)

    task_id: Optional[str] = None
    tag = _TASK_TAG_RE.search(text)
    if tag:
        task_id = tag.group(1)
        text = text[: tag.start()] + text[tag.end() :]

    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    if not text:
        logger.info("Ignoring Feishu message with no plan text after cleanup")
        return None

    is_clear = text.strip("。.!！?？").strip().casefold() in CLEAR_COMMANDS

    return InboundPlanMessage(
        chat_id=chat_id,
        chat_type=chat_type,
        message_id=message_id,
        sender_open_id=sender_open_id or "feishu_user",
        text=text,
        task_id=task_id,
        is_clear=is_clear,
    )
