"""GPU-side agent plugin: install TrainPilot into a coding agent's native format.

The GPU node needs no TrainPilot code to *use* the control plane -- every agent here
carries its own remote MCP client and talks straight to the public ``/mcp`` endpoint.
What it cannot guess is each vendor's spelling of "here is a remote MCP server", which
differs in the one place that matters: Antigravity wants ``serverUrl``, Gemini CLI wants
``httpUrl``, Claude Code wants ``type: http`` beside ``url``, OpenCode wants the servers
under ``mcp`` with ``type: remote``, and Codex wants a TOML table. Getting it wrong is
usually a silent failure to connect, not a loud error.

This package renders one canonical body of text into each of those native formats:

``catalog``   declarative per-agent facts (paths, key spellings, the registering command)
``templates`` the single source of the skill, command, and rules text
``render``    canonical text + a :class:`PluginContext` -> a :class:`Bundle` (pure)
``install``   write a bundle to disk safely -- backs up, merges, never clobbers
"""

from trainpilot.agent.plugin.catalog import TARGETS, AgentTarget, target_keys
from trainpilot.agent.plugin.render import Bundle, PluginContext, render_bundle

__all__ = [
    "TARGETS",
    "AgentTarget",
    "Bundle",
    "PluginContext",
    "render_bundle",
    "target_keys",
]
