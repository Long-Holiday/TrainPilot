#!/usr/bin/env python3
"""TrainPilot CLI Tool for AI Agents (agy, opencode, claude-code).

Provides command-line actions to interact with the TrainPilot Gateway:
- report-milestone
- report-alert
- poll-instruction
- ack-instruction
- send-heartbeat
- get-status
- mock-decision
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional
import requests

# Ensure src is on python path
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from trainpilot.agent.client import _sanitize_for_json
from trainpilot.common.gateway import resolve_gateway_url


def get_default_task_id() -> str:
    return os.environ.get("TRAINPILOT_TASK_ID", "train-task-default")


def _auth_headers(args) -> Dict[str, str]:
    token = getattr(args, "api_token", None) or os.environ.get("TRAINPILOT_API_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _parse_json_dict(arg_val: Optional[str]) -> Optional[Dict[str, Any]]:
    if not arg_val:
        return None
    try:
        data = json.loads(arg_val)
        return _sanitize_for_json(data)
    except json.JSONDecodeError:
        # Support simple comma-separated key=value pairs: "loss=NaN,acc=0.91"
        res = {}
        for pair in arg_val.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                val_str = v.strip()
                if val_str.lower() in ("nan", "none", "null"):
                    res[k.strip()] = "NaN"
                elif val_str.lower() in ("inf", "infinity"):
                    res[k.strip()] = "Infinity"
                elif val_str.lower() in ("-inf", "-infinity"):
                    res[k.strip()] = "-Infinity"
                else:
                    try:
                        res[k.strip()] = float(val_str)
                    except ValueError:
                        res[k.strip()] = val_str
        return _sanitize_for_json(res)


def cmd_report_milestone(args) -> int:
    """Report training milestone to gateway."""
    url = f"{args.gateway}/api/tasks/notify"
    metrics = _parse_json_dict(args.metrics)
    payload = {
        "task_id": args.task_id,
        "event_type": "milestone",
        "message": args.message,
        "step": args.step,
        "epoch": args.epoch,
        "metrics": metrics,
        "agent_note": getattr(args, "agent_note", None),
    }
    try:
        resp = requests.post(url, json=payload, timeout=args.timeout, headers=_auth_headers(args))
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"error": str(e), "task_id": args.task_id, "status": "failed"}), file=sys.stderr)
        return 1


def cmd_report_alert(args) -> int:
    """Report anomaly/alert to gateway (Loss NaN, OOM, etc.)."""
    url = f"{args.gateway}/api/tasks/notify"
    metrics = _parse_json_dict(args.metrics)
    payload = {
        "task_id": args.task_id,
        "event_type": "alert",
        "message": args.message,
        "step": args.step,
        "epoch": args.epoch,
        "metrics": metrics,
        "agent_note": getattr(args, "agent_note", None),
    }
    try:
        resp = requests.post(url, json=payload, timeout=args.timeout, headers=_auth_headers(args))
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"error": str(e), "task_id": args.task_id, "status": "failed"}), file=sys.stderr)
        return 1


def cmd_poll_instruction(args) -> int:
    """Poll gateway mailbox for human decision (with server Long Polling support)."""
    url = f"{args.gateway}/api/tasks/{args.task_id}/instruction"
    start_time = time.time()
    pop_param = "true" if not args.no_pop else "false"

    while True:
        elapsed = time.time() - start_time
        if args.wait and args.wait_timeout and elapsed >= args.wait_timeout:
            print(json.dumps({
                "ready": False,
                "error": f"Polling timed out after {args.wait_timeout}s",
                "task_id": args.task_id,
            }), file=sys.stderr)
            return 2

        params = {"pop": pop_param}
        req_timeout = args.timeout

        # Enable long polling on server if --wait is set
        if args.wait:
            remaining = (args.wait_timeout - elapsed) if args.wait_timeout else 20.0
            lp_wait = max(1.0, min(20.0, remaining))
            params["wait_timeout"] = str(round(lp_wait, 1))
            req_timeout = lp_wait + args.timeout

        try:
            resp = requests.get(url, params=params, timeout=req_timeout, headers=_auth_headers(args))
            resp.raise_for_status()
            data = resp.json()
            if data.get("ready"):
                print(json.dumps(data, indent=2, ensure_ascii=False))
                return 0
            if not args.wait:
                print(json.dumps(data, indent=2, ensure_ascii=False))
                return 0
            # Next cycle immediately for long polling, or sleep if short-polling
            time.sleep(0.1 if args.wait else args.interval)
        except Exception as e:
            if not args.wait:
                print(json.dumps({"error": str(e), "task_id": args.task_id}), file=sys.stderr)
                return 1
            time.sleep(args.interval)


def cmd_ack_instruction(args) -> int:
    """Acknowledge completed instruction."""
    url = f"{args.gateway}/api/tasks/{args.task_id}/ack"
    payload = {
        "task_id": args.task_id,
        "instruction_id": args.instruction_id,
        "action": args.action,
        "status": args.status,
        "message": args.message,
        "solution": getattr(args, "solution", None),
        "step": getattr(args, "step", None),
        "epoch": getattr(args, "epoch", None),
        "metrics": _parse_json_dict(getattr(args, "metrics", None)),
    }
    try:
        resp = requests.post(url, json=payload, timeout=args.timeout, headers=_auth_headers(args))
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"error": str(e), "task_id": args.task_id}), file=sys.stderr)
        return 1


def cmd_send_heartbeat(args) -> int:
    """Send liveness heartbeat."""
    url = f"{args.gateway}/api/tasks/{args.task_id}/heartbeat"
    metrics = _parse_json_dict(args.metrics)
    payload = {
        "task_id": args.task_id,
        "step": args.step,
        "epoch": args.epoch,
        "metrics": metrics,
    }
    try:
        resp = requests.post(url, json=payload, timeout=args.timeout, headers=_auth_headers(args))
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"error": str(e), "task_id": args.task_id}), file=sys.stderr)
        return 1


def cmd_get_status(args) -> int:
    """Query task status from gateway."""
    url = f"{args.gateway}/api/tasks/{args.task_id}/status"
    try:
        resp = requests.get(url, timeout=args.timeout, headers=_auth_headers(args))
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"error": str(e), "task_id": args.task_id}), file=sys.stderr)
        return 1


def cmd_mock_decision(args) -> int:
    """Directly submit human decision (useful for dev/test without Feishu)."""
    url = f"{args.gateway}/api/tasks/{args.task_id}/decision"
    payload = {
        "task_id": args.task_id,
        "action": args.action,
        "operator": args.operator,
        "payload": _parse_json_dict(args.payload),
    }
    try:
        resp = requests.post(url, json=payload, timeout=args.timeout, headers=_auth_headers(args))
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"error": str(e), "task_id": args.task_id}), file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="TrainPilot Agent Tool - Bridge GPU Training with Control Plane Gateway",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gateway", default=resolve_gateway_url(), help="Control Plane Gateway URL")
    parser.add_argument("--task-id", default=get_default_task_id(), help="Unique Training Task ID")
    parser.add_argument("--timeout", type=int, default=10, help="HTTP request timeout in seconds")
    parser.add_argument("--api-token", default=os.environ.get("TRAINPILOT_API_TOKEN"), help="API token when gateway enforces auth")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # report-milestone
    p_milestone = subparsers.add_parser("report-milestone", help="Report regular progress milestone")
    p_milestone.add_argument("--message", required=True, help="Milestone description")
    p_milestone.add_argument("--step", type=int, default=None, help="Training step counter")
    p_milestone.add_argument("--epoch", type=int, default=None, help="Training epoch counter")
    p_milestone.add_argument("--metrics", default=None, help='Metrics JSON or "loss=0.2,acc=0.9"')
    p_milestone.add_argument(
        "--agent-note",
        default=None,
        help="外部 AI 智能体针对当前实际情况自主生成的 1-3 句点评, 将渲染到飞书卡片 🤖 Agent 智能点评 区块",
    )
    p_milestone.set_defaults(func=cmd_report_milestone)

    # report-alert
    p_alert = subparsers.add_parser("report-alert", help="Report training anomaly/alert and pause")
    p_alert.add_argument("--message", required=True, help="Alert description (e.g. Loss NaN)")
    p_alert.add_argument("--step", type=int, default=None, help="Training step counter")
    p_alert.add_argument("--epoch", type=int, default=None, help="Training epoch counter")
    p_alert.add_argument("--metrics", default=None, help='Metrics JSON or "loss=NaN"')
    p_alert.add_argument(
        "--agent-note",
        default=None,
        help="可选: AI 智能体对异常的初步研判 (里程碑卡片会独立展示, 告警卡片暂透传存储)",
    )
    p_alert.set_defaults(func=cmd_report_alert)

    # poll-instruction
    p_poll = subparsers.add_parser("poll-instruction", help="Poll for pending human instruction")
    p_poll.add_argument("--wait", action="store_true", help="Keep polling until instruction is ready")
    p_poll.add_argument("--wait-timeout", type=float, default=300.0, help="Max wait duration in seconds")
    p_poll.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds")
    p_poll.add_argument("--no-pop", action="store_true", help="Peek instruction without consuming it")
    p_poll.set_defaults(func=cmd_poll_instruction)

    # ack-instruction
    p_ack = subparsers.add_parser("ack-instruction", help="Acknowledge executed instruction")
    p_ack.add_argument("--action", required=True, help="Action that was executed")
    p_ack.add_argument("--instruction-id", default=None, help="Instruction ID")
    p_ack.add_argument("--status", choices=["success", "failed"], default="success", help="Execution result")
    p_ack.add_argument("--message", default=None, help="Optional details or error message")
    p_ack.add_argument("--solution", default=None, help="Optional solution description for recovery")
    p_ack.add_argument("--step", type=int, default=None, help="Current training step")
    p_ack.add_argument("--epoch", type=int, default=None, help="Current training epoch")
    p_ack.add_argument("--metrics", default=None, help="Current metrics JSON string")
    p_ack.set_defaults(func=cmd_ack_instruction)

    # send-heartbeat
    p_hb = subparsers.add_parser("send-heartbeat", help="Send liveness heartbeat")
    p_hb.add_argument("--step", type=int, default=None, help="Current step")
    p_hb.add_argument("--epoch", type=int, default=None, help="Current epoch")
    p_hb.add_argument("--metrics", default=None, help="Current metrics")
    p_hb.set_defaults(func=cmd_send_heartbeat)

    # get-status
    p_status = subparsers.add_parser("get-status", help="Get current task status")
    p_status.set_defaults(func=cmd_get_status)

    # mock-decision
    p_dec = subparsers.add_parser("mock-decision", help="Directly submit a human decision (testing/debug)")
    p_dec.add_argument("--action", required=True, help="Action (stop_training, self_resolve)")
    p_dec.add_argument("--operator", default="test_operator", help="Operator name")
    p_dec.add_argument("--payload", default=None, help="Optional payload JSON")
    p_dec.set_defaults(func=cmd_mock_decision)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
