---
name: trainpilot
description: "Supervise a GPU training run through the TrainPilot control plane: report progress and anomalies, wait for the engineer's Feishu decision, and acknowledge the recovery. Use when asked to watch, babysit, or monitor a training job, or when a run crashes, diverges, or OOMs and a human needs to decide what happens next."
---

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
report. Available tools are `report`, `poll_instruction`, `ack_instruction`,
`get_task_status`, and `ack_plan`; use them according to the live server protocol.

If the tools are missing, check the remote `/mcp` URL and credentials, then restart
or reconnect the agent. Do not silently substitute a different server or install
Python packages to work around a missing connection.
