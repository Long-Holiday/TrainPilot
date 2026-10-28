"""Vendor facts for the supported GPU-side agents.

Each target uses its own native plugin layout and remote MCP field names. In particular,
Antigravity's CLI drops the remote URL when converting a Claude-shaped plugin, so it must
continue to receive its own format.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentTarget:
    """Native plugin documentation and transport spelling for one agent."""

    key: str
    display_name: str
    docs_url: str
    transport: str
    agent_cli: str


ANTIGRAVITY = AgentTarget(
    key="antigravity",
    display_name="Antigravity CLI",
    docs_url="https://antigravity.google/docs/plugins",
    transport="serverUrl + headers",
    agent_cli="agy",
)

CLAUDE_CODE = AgentTarget(
    key="claude-code",
    display_name="Claude Code",
    docs_url="https://code.claude.com/docs/en/plugins-reference",
    transport="type=http + url + headers",
    agent_cli="claude",
)

CODEX = AgentTarget(
    key="codex",
    display_name="Codex",
    docs_url="https://developers.openai.com/plugins/build/plugins",
    transport="type=http + url + http_headers",
    agent_cli="codex",
)

TARGETS = {target.key: target for target in (ANTIGRAVITY, CLAUDE_CODE, CODEX)}
