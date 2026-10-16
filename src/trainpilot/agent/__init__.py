"""GPU-side integration support.

There is no TrainPilot Python client and no GPU-side CLI: a GPU node installs a skill and a
remote MCP entry, and the agent's own MCP client talks to the control plane. What lives here
is ``plugin`` -- the standard-library renderer and installer that produces those assets for
each supported agent.
"""
