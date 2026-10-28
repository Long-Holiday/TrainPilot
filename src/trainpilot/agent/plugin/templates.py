"""The one copy of every word the GPU-side plugin ships.

One set of prose for every agent layout: the skill, the rules pointer and the commands.
Keeping the prose here means the published assets and the one-shot installer cannot drift
apart when support for another agent is added.

What is deliberately *not* here is the behaviour protocol. Per AGENTS.md the agent-facing
contract lives in exactly one place, ``mcp_server._INSTRUCTIONS`` plus the tool
docstrings, and the text below points at it rather than restating it. The pointer is the
agent's own MCP connection: a remote client receives those instructions in the
initialize handshake, so a plugin installed today cannot go stale against a server
upgraded later. The templates therefore assume the agent's native MCP tools, and name no
TrainPilot command-line tool at all -- the GPU node installs no Python code.
"""

from dataclasses import dataclass
from typing import Tuple

SKILL_NAME = "trainpilot"
PLUGIN_NAME = "trainpilot"

# One line, <= 500 chars: Codex enforces both limits on skill frontmatter.
SKILL_DESCRIPTION = (
    "Supervise a GPU training run through the TrainPilot control plane: report progress "
    "and anomalies, wait for the engineer's Feishu decision, and acknowledge the recovery. "
    "Use when asked to watch, babysit, or monitor a training job, or when a run crashes, "
    "diverges, or OOMs and a human needs to decide what happens next."
)

SKILL_BODY = """\
# TrainPilot

Use your built-in MCP client to connect to the remote TrainPilot control plane. The
GPU node needs only this skill and the remote MCP configuration: do not install or
start the TrainPilot Python service, an MCP proxy, or a local training watcher.
You watch the training process; the control plane connects you to the engineer.

## Read the live protocol

Before supervising a run, read the TrainPilot server's MCP initialization
`instructions` and the tool descriptions supplied by the live connection. They are
the authoritative supervision protocol, not this skill or a remembered procedure.
If your client has not exposed those instructions, stop and ask to reconnect or fix
the MCP integration; do not infer the protocol from this file.

## Bind the run

Agree on a distinct `task_id` for each run and pass it explicitly to every TrainPilot
MCP tool call. If omitted, native MCP calls use the default configured on the
**server**, not `TRAINPILOT_TASK_ID` in the GPU node's environment. Do not let two
runs accidentally share a mailbox.

## Check the connection

Locate the TrainPilot tools in your client's MCP tool list (plugin clients may
prefix their names). Use `get_task_status` with the agreed task id to check the
connection without creating a task. An unknown task is normal before its first
report. Available tools are `report`, `poll_instruction`, `ack_instruction`, and
`get_task_status`; use them according to the live server protocol.

If the tools are missing, check the remote `/mcp` URL and credentials, then restart
or reconnect the agent. Do not silently substitute a different server or install
Python packages to work around a missing connection.
"""


@dataclass(frozen=True)
class Command:
    """A shared command prompt, rendered as commands or Codex command skills.

    Antigravity's plugin page documents skills, agents, rules, MCP config and hooks -- not
    a ``commands/`` directory. ``agy plugin install`` processes one anyway ("commands: N
    processed (converted to skills)"), for a plugin in either shape, so the commands are
    shipped rather than dropped on the strength of the docs. Argument substitution is
    undocumented, so each prompt names its argument in words as well as with ``$ARGUMENTS``.

    Attributes:
        name: Base name; Antigravity namespaces the converted skill as ``<plugin>-cmd-<name>``.
        description: One line shown in the agent's command list.
        argument_hint: Hint shown after the name.
        prompt: The instruction body.
    """

    name: str
    description: str
    argument_hint: str
    prompt: str


COMMANDS: Tuple[Command, ...] = (
    Command(
        name="protocol",
        description="Read TrainPilot's authoritative protocol from the live MCP connection",
        argument_hint="",
        prompt=(
            "Read the TrainPilot MCP initialization instructions and tool descriptions from "
            "the live connection. Summarise the key obligations in three bullets, then stop. "
            "If the instructions are not visible, ask to reconnect or fix the MCP integration; "
            "do not invent the protocol. No local TrainPilot CLI is required."
        ),
    ),
    Command(
        name="status",
        description="Show the current TrainPilot task state, metrics and pending instruction",
        argument_hint="[task_id]",
        prompt=(
            "Use the TrainPilot get_task_status MCP tool for $ARGUMENTS, or the task id "
            "already agreed for this run. Ask for a task id if none is bound. Report the "
            "state, last metrics, and whether an instruction is pending. Do not create a "
            "task just to inspect it."
        ),
    ),
    Command(
        name="report",
        description="Report a training event to the TrainPilot control plane",
        argument_hint="<message>",
        prompt=(
            "Read the live TrainPilot MCP initialization instructions and tool descriptions "
            "if you have not already. Use the report MCP tool to report $ARGUMENTS, with "
            "the task_id agreed for this run (ask if unbound). Choose event_type according "
            "to the live protocol, and include step, epoch and metrics when known. "
            "Do not run a local CLI or start a server."
        ),
    ),
)

RULES_BLOCK = """\
## TrainPilot

Read the `trainpilot` skill and the remote TrainPilot MCP initialization
`instructions` before supervising a run. Use the agent's built-in MCP tools and an
explicit task id. The live control plane owns the protocol; no local Python service
or TrainPilot CLI is required.
"""
