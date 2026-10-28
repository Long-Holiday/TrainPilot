"""Vendor facts for the supported GPU-side agents.

Each target uses its own native plugin layout and remote MCP field names. In particular,
Antigravity's CLI drops the remote URL when converting a Claude-shaped plugin, so it must
continue to receive its own format.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AgentTarget:
    """Native plugin commands and documentation for one agent.

    ``{bundle}`` in argv templates is replaced with the staged directory. Marketplace
    names are distinct from the published repository marketplace to allow a configured
    local install without changing the repository installation's source.
    """

    key: str
    display_name: str
    docs_url: str
    transport: str
    register: Tuple[Tuple[str, ...], ...]
    unregister: Tuple[Tuple[str, ...], ...]
    agent_cli: str
    note: str = ""
    marketplace_name: str = ""


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

CLAUDE_CODE = AgentTarget(
    key="claude-code",
    display_name="Claude Code",
    docs_url="https://code.claude.com/docs/en/plugins-reference",
    transport="type=http + url + headers",
    register=(
        ("claude", "plugin", "marketplace", "add", "{bundle}"),
        ("claude", "plugin", "install", "trainpilot@trainpilot-local-claude-code", "--scope", "user"),
    ),
    unregister=(
        ("claude", "plugin", "uninstall", "trainpilot@trainpilot-local-claude-code", "--scope", "user"),
        ("claude", "plugin", "marketplace", "remove", "trainpilot-local-claude-code"),
    ),
    agent_cli="claude",
    marketplace_name="trainpilot-local-claude-code",
)

CODEX = AgentTarget(
    key="codex",
    display_name="Codex",
    docs_url="https://developers.openai.com/plugins/build/plugins",
    transport="type=http + url + http_headers",
    register=(
        ("codex", "plugin", "marketplace", "add", "{bundle}"),
        ("codex", "plugin", "add", "trainpilot@trainpilot-local-codex"),
    ),
    unregister=(
        ("codex", "plugin", "remove", "trainpilot@trainpilot-local-codex"),
        ("codex", "plugin", "marketplace", "remove", "trainpilot-local-codex"),
    ),
    agent_cli="codex",
    marketplace_name="trainpilot-local-codex",
    note="Requires a Codex version with `codex plugin` support; restart after installation.",
)

TARGETS = {target.key: target for target in (ANTIGRAVITY, CLAUDE_CODE, CODEX)}
