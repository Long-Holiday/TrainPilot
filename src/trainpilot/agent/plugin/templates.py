"""The one copy of every word the GPU-side plugin ships.

Five agents get five file layouts, but they get identical prose. Keeping the prose here
means a fix lands everywhere at once -- the alternative (a checked-in directory per agent)
is five copies that drift the first time one is edited.

What is deliberately *not* here is the behaviour protocol. Per AGENTS.md the agent-facing
contract lives in exactly one place, ``mcp_server._INSTRUCTIONS`` plus the tool docstrings,
and the text below points at it rather than restating it. The pointer is executable --
``trainpilot-cli plugin protocol`` reads the ``instructions`` field out of a live MCP
handshake -- so a plugin installed today cannot go stale against a server upgraded later.
"""

from dataclasses import dataclass
from typing import Tuple

SKILL_NAME = "trainpilot"
PLUGIN_NAME = "trainpilot"

# One line, single line, <= 500 chars: Codex enforces both limits on skill frontmatter.
SKILL_DESCRIPTION = (
    "Supervise a GPU training run through the TrainPilot control plane: report progress "
    "and anomalies, wait for the engineer's Feishu decision, and acknowledge the recovery. "
    "Use when asked to watch, babysit, or monitor a training job, or when a run crashes, "
    "diverges, or OOMs and a human needs to decide what happens next."
)

SKILL_BODY = """\
# TrainPilot

TrainPilot is the human-in-the-loop control plane for a training run. It does not watch the
GPU and it cannot pause a job: you watch the run and control the process, and TrainPilot
carries an anomaly to the engineer's Feishu card and brings their decision back to you.

## Load the protocol before you report anything

The authoritative rules -- when to alert, what counts as a verified recovery, how to poll,
how to acknowledge a queued next-task plan -- are served by the control plane itself in the
MCP `instructions` field. Read them at the start of every supervised run:

```bash
trainpilot-cli plugin protocol
```

Work from that output, not from memory and not from this file. The server owns the text, and
it changes when the server is upgraded; this skill only tells you how to reach it.

## Bind the run to a task id

Every call takes a `task_id`. Omitted, it falls back to `TRAINPILOT_TASK_ID` on this node.
Pass it explicitly whenever you supervise more than one run, or two jobs will share one
mailbox and one engineer decision will wake the wrong process.

## Check the wiring

```bash
trainpilot-cli plugin doctor
```

Confirms the gateway is reachable, the token is accepted, the protocol is readable, and the
task id resolves -- before anything goes wrong at step 12,000.

## If the MCP tools are absent

A session can start without the TrainPilot MCP server (the agent was not restarted after
install, the connection failed, the token rotated). The same protocol is reachable over the
CLI, which calls the same endpoint:

```bash
trainpilot-cli report --message "loss spiked to 9.3 at step 1200" --type alert --step 1200
trainpilot-cli poll --wait-timeout 20
trainpilot-cli ack --instruction-id <id> --action self_resolve --status success --solution "..."
trainpilot-cli status
trainpilot-cli ack-plan --plan-id <plan_id> --status done --note "..."
```

The behaviour rules do not change with the transport. Load them with
`trainpilot-cli plugin protocol` either way.
"""


@dataclass(frozen=True)
class Command:
    """A slash command, rendered into each agent's own command format.

    Attributes:
        name: Base name, before any per-agent namespacing.
        description: One line shown in the agent's command list.
        argument_hint: Hint shown after the name, where the agent supports one.
        prompt: The instruction body.
    """

    name: str
    description: str
    argument_hint: str
    prompt: str


COMMANDS: Tuple[Command, ...] = (
    Command(
        name="protocol",
        description="Print TrainPilot's authoritative supervision protocol from the server",
        argument_hint="",
        prompt=(
            "Run `trainpilot-cli plugin protocol` and treat its output as the authoritative "
            "TrainPilot supervision protocol for this session. Summarise the key obligations "
            "in three bullets, then stop."
        ),
    ),
    Command(
        name="status",
        description="Show the current TrainPilot task state, metrics and pending instruction",
        argument_hint="[task_id]",
        prompt=(
            "Run `trainpilot-cli status` (add `--task-id $ARGUMENTS` when a task id was given) "
            "and report the task's state, last metrics, and whether an instruction is pending."
        ),
    ),
    Command(
        name="report",
        description="Report a training event to the TrainPilot control plane",
        argument_hint="<message>",
        prompt=(
            "Report a training event with `trainpilot-cli report --message \"$ARGUMENTS\"`. "
            "Choose `--type` from milestone, alert, completed, or failed according to the "
            "protocol returned by `trainpilot-cli plugin protocol` -- read it if you have not "
            "already. Include `--step`, `--epoch` and `--metrics` when you know them."
        ),
    ),
)

# Appended to an agent's instruction file. Short on purpose: it exists to get the agent to
# the skill, not to teach it the protocol a second time.
RULES_BLOCK = """\
## TrainPilot

This machine supervises training runs through a TrainPilot control plane. Before reporting
anything about a run, load the live protocol with `trainpilot-cli plugin protocol` and read
the `trainpilot` skill, which says how to reach it and how to bind a run to a task id.
"""
