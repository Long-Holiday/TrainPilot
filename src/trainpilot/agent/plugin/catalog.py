"""The GPU-side agent this plugin targets, as data.

Everything here was read off Antigravity's own documentation (``docs_url``) or its CLI
``--help``; nothing is inferred from another agent's format, because the field most often
got wrong is the one that fails silently -- a wrong remote-server key gives you an agent
that starts up fine and never sees the tools.

Antigravity is the only target. Claude Code, Gemini CLI, OpenCode and Codex each spell a
remote MCP server differently, but none of them is a supported GPU-side agent here, and
Antigravity's own CLI converts a Claude-shaped plugin on install while *dropping* the
remote URL (see ``note``). So the plugin is rendered and published in Antigravity's shape
only, and there is no table to keep in step: one agent, one set of facts.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AgentTarget:
    """The one supported GPU-side agent.

    Attributes:
        key: Short key for messages and log lines, e.g. ``antigravity``.
        display_name: Human label for listings.
        docs_url: Vendor page the paths and key spellings below were verified against.
        transport: The vendor's own spelling for a remote Streamable HTTP server. Kept as
            data because it is the field most often got wrong, and tests assert on it.
        register: argv templates run after staging the bundle. ``{bundle}`` is replaced with
            the staged directory.
        unregister: argv templates that take the plugin back out, run by ``uninstall``.
        agent_cli: Binary that must exist for ``register`` to run.
        note: One line of caveat shown by ``plugin install``.
    """

    key: str
    display_name: str
    docs_url: str
    transport: str
    register: Tuple[Tuple[str, ...], ...]
    unregister: Tuple[Tuple[str, ...], ...]
    agent_cli: str
    note: str = ""


ANTIGRAVITY = AgentTarget(
    key="antigravity",
    display_name="Antigravity CLI",
    docs_url="https://antigravity.google/docs/plugins",
    transport="serverUrl + headers",
    register=(("agy", "plugin", "install", "{bundle}"),),
    unregister=(("agy", "plugin", "uninstall", "trainpilot"),),
    agent_cli="agy",
    note=(
        "Antigravity rejects url and httpUrl: the remote server key must be serverUrl. "
        "Its CLI also converts a Claude Code plugin on install but drops the remote URL, "
        "so only this shape connects."
    ),
)
