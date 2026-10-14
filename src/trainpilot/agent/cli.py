"""Command Line Interface for TrainPilot GPU Nodes (MCP Client).

Connects from GPU worker nodes to the central TrainPilot MCP Server so engineers
and scripts can report events, poll human decisions, and acknowledge recovery
actions without writing code.
"""

import argparse
import json
import sys
from typing import Any, Dict, Optional

from trainpilot.agent.mcp_client import TrainPilotMCPClient


def _parse_key_value_or_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse string input as JSON or comma-separated key=value pairs."""
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        try:
            return json.loads(raw)
        except Exception:
            pass
    # Parse key1=val1,key2=val2
    result: Dict[str, Any] = {}
    for part in raw.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            k, v = k.strip(), v.strip()
            # Try float/int conversion
            if v.lower() == "nan":
                result[k] = "NaN"
            elif v.lower() in ("inf", "+inf"):
                result[k] = "Infinity"
            elif v.lower() == "-inf":
                result[k] = "-Infinity"
            else:
                try:
                    if "." in v:
                        result[k] = float(v)
                    else:
                        result[k] = int(v)
                except ValueError:
                    result[k] = v
        elif part.strip():
            result[part.strip()] = True
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="trainpilot-cli",
        description="TrainPilot GPU Client - Model Context Protocol (MCP) tool caller",
    )
    parser.add_argument(
        "--server-url",
        "-s",
        help="TrainPilot MCP Server URL (default: $TRAINPILOT_GATEWAY_URL or http://$TRAINPILOT_HOST:$TRAINPILOT_PORT)",
    )
    parser.add_argument(
        "--task-id",
        "-t",
        help="Task ID (default: $TRAINPILOT_TASK_ID)",
    )
    parser.add_argument(
        "--api-token",
        help="API token for Bearer authorization (default: $TRAINPILOT_API_TOKEN)",
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True, help="Available subcommands")

    # 1. report
    p_report = subparsers.add_parser("report", help="Report a training event")
    p_report.add_argument("--message", "-m", required=True, help="Event description")
    p_report.add_argument(
        "--type",
        default="milestone",
        choices=["milestone", "alert", "completed", "failed"],
        help="Event type (default: milestone)",
    )
    p_report.add_argument("--step", type=int, default=None, help="Current step number")
    p_report.add_argument("--epoch", type=int, default=None, help="Current epoch number")
    p_report.add_argument("--metrics", help="JSON or comma-separated key=val metrics (e.g. loss=0.45)")
    p_report.add_argument("--agent-note", help="AI agent autonomous analysis/commentary note")

    # 2. poll
    p_poll = subparsers.add_parser("poll", help="Wait for a human decision")
    p_poll.add_argument(
        "--wait-timeout",
        type=float,
        default=None,
        help="Long-poll timeout in seconds (default: server-configured; 0 returns immediately)",
    )

    # 3. ack
    p_ack = subparsers.add_parser("ack", help="Acknowledge execution of a decision")
    p_ack.add_argument("--action", default="self_resolve", help="Executed action (default: self_resolve)")
    p_ack.add_argument("--status", default="success", choices=["success", "failed"], help="Execution status")
    p_ack.add_argument("--solution", default="", help="What you actually did")
    p_ack.add_argument("--instruction-id", default=None, help="Instruction ID from poll")

    # 4. ack-plan
    p_ack_plan = subparsers.add_parser("ack-plan", help="Acknowledge a next-task plan from a report")
    p_ack_plan.add_argument("--plan-id", required=True, help="plan_id from a report's next_task_plans")
    p_ack_plan.add_argument(
        "--status", default="done", choices=["done", "failed", "skipped"], help="What happened"
    )
    p_ack_plan.add_argument("--note", default="", help="What you actually did")

    # 5. status
    subparsers.add_parser("status", help="Get current task status")

    args = parser.parse_args(argv)

    client = TrainPilotMCPClient(
        server_url=args.server_url,
        task_id=args.task_id,
        api_token=args.api_token,
    )

    try:
        if args.subcommand == "report":
            res = client.report(
                message=args.message,
                event_type=args.type,
                step=args.step,
                epoch=args.epoch,
                metrics=_parse_key_value_or_json(args.metrics),
                agent_note=args.agent_note,
                fail_silently=False,
            )

        elif args.subcommand == "poll":
            res = client.poll_instruction(wait_timeout=args.wait_timeout)

        elif args.subcommand == "ack":
            res = client.ack_instruction(
                action=args.action,
                status=args.status,
                solution=args.solution,
                instruction_id=args.instruction_id,
            )

        elif args.subcommand == "ack-plan":
            res = client.ack_plan(
                plan_id=args.plan_id,
                status=args.status,
                note=args.note,
            )

        else:  # status
            res = client.get_task_status()

        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
