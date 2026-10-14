"""Tests for the pure Feishu message-event parser (no webhook, no I/O)."""

import json

from trainpilot.server.feishu.inbound import is_message_event, parse_plan_message


def _message_event(
    text: str,
    *,
    chat_type: str = "group",
    message_type: str = "text",
    message_id: str = "om_1",
    chat_id: str = "oc_group",
    sender_type: str = "user",
    mentions: bool = True,
    content: object = None,
    event_key: str = "header",
):
    """Build an `im.message.receive_v1` payload; mirrors what Feishu actually posts."""
    message = {
        "message_id": message_id,
        "chat_id": chat_id,
        "chat_type": chat_type,
        "message_type": message_type,
        "content": content if content is not None else json.dumps({"text": text}),
    }
    if mentions:
        message["mentions"] = [{"key": "@_user_1", "id": {"open_id": "ou_bot"}, "name": "TrainPilot"}]
    event = {"sender": {"sender_type": sender_type, "sender_id": {"open_id": "ou_human"}}, "message": message}
    if event_key == "header":
        return {"schema": "2.0", "header": {"event_type": "im.message.receive_v1"}, "event": event}
    event["type"] = "im.message.receive_v1"
    return {"event": event}


def test_is_message_event_detects_both_payload_shapes():
    assert is_message_event(_message_event("hi")) is True
    assert is_message_event(_message_event("hi", event_key="event")) is True
    assert is_message_event({"action": {"value": {"task_id": "t", "action": "x"}}}) is False
    assert is_message_event({"type": "url_verification", "challenge": "c"}) is False


def test_mention_placeholder_is_stripped_from_the_plan_body():
    msg = parse_plan_message(_message_event("@_user_1 用 5e-5 重跑 10 个 epoch"))
    assert msg is not None
    assert msg.text == "用 5e-5 重跑 10 个 epoch"
    assert msg.task_id is None
    assert msg.is_clear is False
    assert msg.chat_id == "oc_group"
    assert msg.sender_open_id == "ou_human"


def test_task_tag_binds_and_is_removed_from_the_body():
    msg = parse_plan_message(_message_event("@_user_1 用 5e-5 重跑 10 个 epoch #demo-llm-pretrain"))
    assert msg.task_id == "demo-llm-pretrain"
    assert msg.text == "用 5e-5 重跑 10 个 epoch"


def test_task_tag_matches_mid_sentence():
    msg = parse_plan_message(_message_event("@_user_1 重跑 #demo-llm-pretrain 并评测"))
    assert msg.task_id == "demo-llm-pretrain"
    assert msg.text == "重跑 并评测"


def test_slash_and_csharp_are_not_task_tags():
    msg = parse_plan_message(_message_event("@_user_1 打开 #/detail 页面 复盘 C# 版本"))
    assert msg.task_id is None
    assert "#/detail" in msg.text and "C#" in msg.text


def test_group_message_without_a_mention_is_ignored():
    assert parse_plan_message(_message_event("下一步跑评测", mentions=False)) is None


def test_p2p_message_needs_no_mention():
    msg = parse_plan_message(_message_event("跑评测", chat_type="p2p", mentions=False))
    assert msg is not None
    assert msg.text == "跑评测"


def test_non_text_and_bot_messages_are_ignored():
    assert parse_plan_message(_message_event("img", message_type="image")) is None
    assert parse_plan_message(_message_event("跑评测", sender_type="app")) is None


def test_missing_ids_and_unparseable_content_are_ignored():
    assert parse_plan_message(_message_event("跑评测", message_id="")) is None
    assert parse_plan_message(_message_event("跑评测", chat_id="")) is None
    assert parse_plan_message(_message_event("跑评测", content="not json")) is None


def test_whitespace_only_body_is_ignored():
    assert parse_plan_message(_message_event("@_user_1   ")) is None


def test_clear_commands_are_recognised_but_other_text_is_a_plan():
    for text in ("清除", "取消", "clear", "Clear", "清除。", " @_user_1 取消！"):
        msg = parse_plan_message(_message_event(text))
        assert msg is not None and msg.is_clear is True, text

    for text in ("取消昨天的实验", "清除后重跑", "别取消"):
        msg = parse_plan_message(_message_event(text))
        assert msg is not None and msg.is_clear is False, text


def test_clear_command_can_be_scoped_to_one_task():
    msg = parse_plan_message(_message_event("@_user_1 清除 #demo-llm-pretrain"))
    assert msg.is_clear is True
    assert msg.task_id == "demo-llm-pretrain"


def test_multiline_plan_keeps_its_newlines():
    msg = parse_plan_message(_message_event("@_user_1 第一步：重跑\n第二步：评测"))
    assert msg.text == "第一步：重跑\n第二步：评测"
