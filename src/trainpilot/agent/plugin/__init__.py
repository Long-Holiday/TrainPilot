"""GPU-side agent plugin: install TrainPilot into Antigravity CLI's native format.

The GPU node needs no TrainPilot code to *use* the control plane -- Antigravity carries its
own remote MCP client and talks straight to the public ``/mcp`` endpoint. What it cannot
guess is the vendor's spelling of "here is a remote MCP server", which is ``serverUrl``
(``url`` and ``httpUrl`` are rejected outright). Getting it wrong is usually a silent
failure to connect, not a loud error.

Antigravity is the only supported agent. This package renders the canonical text into its
native layout:

``catalog``   the vendor facts (docs page, key spelling, the registering command)
``templates`` the single source of the skill and rules text
``render``    canonical text + a :class:`PluginContext` -> a :class:`Bundle` (pure)
``install``   write a bundle to disk, and nothing that belongs to the user
"""

from trainpilot.agent.plugin.catalog import ANTIGRAVITY, AgentTarget
from trainpilot.agent.plugin.render import Bundle, PluginContext, render_bundle

__all__ = [
    "ANTIGRAVITY",
    "AgentTarget",
    "Bundle",
    "PluginContext",
    "render_bundle",
]
