"""GPU-side native plugins for Antigravity, Claude Code and Codex.

The installed assets use each agent's own MCP client. This package contains only the
standard-library catalog, shared prose, renderer and installer; no GPU-side Python client.
"""

from trainpilot.agent.plugin.catalog import ANTIGRAVITY, CLAUDE_CODE, CODEX, TARGETS, AgentTarget
from trainpilot.agent.plugin.render import Bundle, PluginContext, render_bundle

__all__ = [
    "ANTIGRAVITY",
    "CLAUDE_CODE",
    "CODEX",
    "TARGETS",
    "AgentTarget",
    "Bundle",
    "PluginContext",
    "render_bundle",
]
