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


def test_a_chinese_task_name_binds_like_an_ascii_one():
    """#fix(任务名) always accepted CJK; the bare tag must not silently degrade instead."""
    msg = parse_plan_message(_message_event("@_user_1 用 5e-5 重跑 #训练任务A"))
    assert msg.task_id == "训练任务A"
    assert msg.text == "用 5e-5 重跑"


def test_a_mixed_script_task_name_binds_whole():
    msg = parse_plan_message(_message_event("@_user_1 继续 #第3次实验"))
    assert msg.task_id == "第3次实验"
    assert msg.text == "继续"


def test_cjk_prose_after_a_hash_is_not_a_tag():
    """Only a word directly after `#` binds; punctuation keeps the text as ordinary prose."""
    msg = parse_plan_message(_message_event("@_user_1 参考 #【重要】那条 再跑一遍"))
    assert msg.task_id is None
    assert "#【重要】" in msg.text


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


# --------------------------------------------------------------------------
# `#fix(task)` / `#feat(task)` constructs
# --------------------------------------------------------------------------


def test_fix_tag_binds_the_task_and_keeps_the_solution_body():
    msg = parse_plan_message(
        _message_event("@_user_1 #fix(demo-llm-pretrain) 回滚到 step4 检查点并把 lr 减半")
    )
    assert msg.fix_task_id == "demo-llm-pretrain"
    assert msg.fix_error is None
    assert msg.text == "回滚到 step4 检查点并把 lr 减半"
    # A fix is not a plan: the plan binding must stay empty.
    assert msg.task_id is None
    assert msg.is_clear is False


def test_fix_tag_is_case_insensitive_and_may_follow_prose():
    msg = parse_plan_message(_message_event("@_user_1 先看日志，然后 #FIX(demo-llm) 回滚"))
    assert msg.fix_task_id == "demo-llm"
    assert msg.text == "先看日志，然后 回滚"


def test_fix_solution_keeps_hash_tags_verbatim():
    """`#3` in a solution is a checkpoint/issue reference, not a task binding."""
    msg = parse_plan_message(_message_event("@_user_1 #fix(demo-llm) 回滚到 #3 检查点"))
    assert msg.fix_task_id == "demo-llm"
    assert msg.text == "回滚到 #3 检查点"


def test_feat_tag_is_an_alias_for_the_bare_tag():
    explicit = parse_plan_message(_message_event("@_user_1 #feat(demo-llm) 用 5e-5 重跑"))
    implicit = parse_plan_message(_message_event("@_user_1 用 5e-5 重跑 #demo-llm"))
    assert explicit.task_id == implicit.task_id == "demo-llm"
    assert explicit.text == implicit.text == "用 5e-5 重跑"
    assert explicit.fix_task_id is None


def test_fix_wins_over_feat_and_bare_tags():
    msg = parse_plan_message(_message_event("@_user_1 #fix(a) 按 #feat(b) 的思路重跑 #c"))
    assert msg.fix_task_id == "a"
    assert msg.task_id is None
    assert msg.fix_error is None


def test_a_construct_suppresses_the_bare_tag_scan():
    """`#feat(a) 重跑 #b` must not silently rebind the plan to b."""
    msg = parse_plan_message(_message_event("@_user_1 #feat(a) 重跑 #b"))
    assert msg.task_id == "a"


def test_parenthesis_less_fix_is_a_syntax_error_not_a_task_named_fix():
    """The words fix/feat are reserved: binding a plan to a task named "fix" hides it."""
    msg = parse_plan_message(_message_event("@_user_1 #fix 回滚到 step4"))
    assert msg.fix_task_id is None
    assert msg.task_id is None
    assert "缺少任务名" in msg.fix_error

    msg = parse_plan_message(_message_event("@_user_1 #feat 重跑"))
    assert msg.task_id is None
    assert "缺少任务名" in msg.fix_error


def test_empty_fix_body_is_surfaced_as_an_error():
    msg = parse_plan_message(_message_event("@_user_1 #fix(demo-llm)"))
    assert msg is not None
    assert msg.fix_task_id is None
    assert "方案正文为空" in msg.fix_error


def test_clear_words_inside_a_fix_body_are_part_of_the_solution():
    msg = parse_plan_message(_message_event("@_user_1 #fix(demo-llm) 清除"))
    assert msg.fix_task_id == "demo-llm"
    assert msg.is_clear is False
    assert msg.text == "清除"
