"""Command Line Interface for TrainPilot GPU Nodes (MCP Client).

Connects directly from GPU worker nodes to the central TrainPilot MCP Server,
allowing engineers and automated scripts to report milestones, alert on anomalies,
poll human decisions, and acknowledge self-healing recovery actions.
"""

import argparse
import json
import os
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

    # 1. report-milestone
    p_milestone = subparsers.add_parser("report-milestone", help="Report a training milestone")
    p_milestone.add_argument("--message", "-m", required=True, help="Milestone description")
    p_milestone.add_argument("--step", type=int, default=0, help="Current step number")
    p_milestone.add_argument("--epoch", type=int, default=None, help="Current epoch number")
    p_milestone.add_argument("--metrics", help="JSON or comma-separated key=val metrics (e.g. loss=0.45,val_loss=0.48)")
    p_milestone.add_argument("--agent-note", help="AI agent autonomous analysis/commentary note")

    # 2. report-alert
    p_alert = subparsers.add_parser("report-alert", help="Report training anomaly and freeze training")
    p_alert.add_argument("--message", "-m", required=True, help="Alert description (e.g. 'Loss NaN at step 1200')")
    p_alert.add_argument("--step", type=int, default=None, help="Current step number")
    p_alert.add_argument("--epoch", type=int, default=None, help="Current epoch number")
    p_alert.add_argument("--metrics", help="Metrics at anomaly (e.g. loss=NaN)")
    p_alert.add_argument("--agent-note", help="AI agent diagnostic note")

    # 3. poll-instruction
    p_poll = subparsers.add_parser("poll-instruction", help="Poll for human decisions")
    p_poll.add_argument("--wait", action="store_true", help="Enable long-polling wait")
    p_poll.add_argument("--wait-timeout", type=float, default=20.0, help="Wait timeout in seconds")
    p_poll.add_argument("--no-pop", action="store_true", help="Do not pop/consume the instruction")

    # 4. ack-instruction
    p_ack = subparsers.add_parser("ack-instruction", help="Acknowledge recovery execution")
    p_ack.add_argument("--action", default="self_resolve", help="Executed action (default: self_resolve)")
    p_ack.add_argument("--status", default="success", choices=["success", "failed"], help="Execution status")
    p_ack.add_argument("--solution", default="", help="Description of the self-healing solution")
    p_ack.add_argument("--instruction-id", default=None, help="Instruction ID")
    p_ack.add_argument("--message", default="", help="Additional execution message")

    # 5. send-heartbeat
    p_hb = subparsers.add_parser("send-heartbeat", help="Send heartbeat telemetry")
    p_hb.add_argument("--step", type=int, default=None, help="Current step number")
    p_hb.add_argument("--epoch", type=int, default=None, help="Current epoch number")
    p_hb.add_argument("--metrics", help="GPU telemetry metrics (e.g. gpu_mem=85%)")

    # 6. get-status
    subparsers.add_parser("get-status", help="Get current task status and mailbox")

    # 7. list-tasks
    p_list = subparsers.add_parser("list-tasks", help="List all tracked tasks on server")
    p_list.add_argument("--limit", type=int, default=100, help="Max tasks to return")
    p_list.add_argument("--offset", type=int, default=0, help="Offset")
    p_list.add_argument("--state", help="Filter by state (e.g. RUNNING, WAITING)")
    p_list.add_argument("--stale-only", action="store_true", help="Only show stale tasks")

    # 8. submit-decision
    p_dec = subparsers.add_parser("submit-decision", help="Directly submit an operator decision")
    p_dec.add_argument("--action", required=True, help="Action name (e.g. self_resolve, stop_training)")
    p_dec.add_argument("--operator", default="cli_operator", help="Operator name")

    # 9. list-tools
    subparsers.add_parser("list-tools", help="List all tools exposed by the MCP Server")

    args = parser.parse_args(argv)

    client = TrainPilotMCPClient(
        server_url=args.server_url,
        task_id=args.task_id,
        api_token=args.api_token,
    )

    try:
        if args.subcommand == "report-milestone":
            metrics = _parse_key_value_or_json(args.metrics)
            res = client.report_milestone(
                message=args.message,
                step=args.step,
                epoch=args.epoch,
                metrics=metrics,
                agent_note=args.agent_note,
            )
            print(json.dumps(res, indent=2, ensure_ascii=False))

        elif args.subcommand == "report-alert":
            metrics = _parse_key_value_or_json(args.metrics)
            res = client.report_alert(
                message=args.message,
                step=args.step,
                epoch=args.epoch,
                metrics=metrics,
                agent_note=args.agent_note,
            )
            print(json.dumps(res, indent=2, ensure_ascii=False))

        elif args.subcommand == "poll-instruction":
            wait_timeout = args.wait_timeout if args.wait else 0.0
            res = client.poll_instruction(
                wait_timeout=wait_timeout,
                pop=not args.no_pop,
            )
            print(json.dumps(res, indent=2, ensure_ascii=False))

        elif args.subcommand == "ack-instruction":
            res = client.ack_instruction(
                action=args.action,
                status=args.status,
                solution=args.solution,
                instruction_id=args.instruction_id,
                message=args.message,
            )
            print(json.dumps(res, indent=2, ensure_ascii=False))

        elif args.subcommand == "send-heartbeat":
            metrics = _parse_key_value_or_json(args.metrics)
            res = client.send_heartbeat(
                step=args.step,
                epoch=args.epoch,
                metrics=metrics,
            )
            print(json.dumps(res, indent=2, ensure_ascii=False))

        elif args.subcommand == "get-status":
            res = client.get_task_status()
            print(json.dumps(res, indent=2, ensure_ascii=False))

        elif args.subcommand == "list-tasks":
            res = client.list_tasks(
                limit=args.limit,
                offset=args.offset,
                state=args.state,
                stale_only=args.stale_only,
            )
            print(json.dumps(res, indent=2, ensure_ascii=False))

        elif args.subcommand == "submit-decision":
            res = client.submit_decision(
                action=args.action,
                operator=args.operator,
            )
            print(json.dumps(res, indent=2, ensure_ascii=False))

        elif args.subcommand == "list-tools":
            tools = client.list_tools()
            print(json.dumps({"tools": tools}, indent=2))

        return 0

    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
