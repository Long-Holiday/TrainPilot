"""Per-agent facts, in one declarative table.

Everything here was read off the vendor's own documentation (``docs_url``) or its CLI
``--help``; nothing is inferred from another agent's format. The spellings differ enough
that guessing is the main way a TrainPilot install fails silently -- a wrong remote-server
key usually yields an agent that starts up fine and simply never sees the tools.

Adding an agent is meant to be a data change here plus a renderer in ``render.py``; the
two are pinned together by ``tests/test_plugin_catalog.py``.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

# How a target gets its files onto the machine.
#
# ``bundle``   -- render a directory in the vendor's own plugin/extension layout and hand
#                 it to that agent's CLI, which owns the install location.
# ``in-place`` -- the agent has no plugin package; write into its real config tree so the
#                 files are live where it already reads them.
BUNDLE = "bundle"
IN_PLACE = "in-place"

# Where a target installs to. `user` is the default because a GPU node runs many jobs from
# many directories and should be wired up once, not once per project.
USER = "user"
PROJECT = "project"
SCOPES = (USER, PROJECT)


@dataclass(frozen=True)
class AgentTarget:
    """One supported GPU-side agent.

    Attributes:
        key: Value the user types, e.g. ``antigravity``.
        display_name: Human label for listings.
        docs_url: Vendor page the paths and key spellings below were verified against.
        kind: :data:`BUNDLE` or :data:`IN_PLACE`.
        transport: The vendor's own spelling for a remote Streamable HTTP server. Kept as
            data because it is the field most often got wrong, and tests assert on it.
        register: argv templates run after staging a ``bundle``. ``{bundle}`` is replaced with
            the staged directory and ``{scope}`` with the install scope. Empty for
            ``in-place`` targets, which are live as soon as the files are written.
        unregister: argv templates that take the plugin back out, run by ``uninstall``.
        agent_cli: Binary that must exist for ``register`` to run, when there is one.
        detect: Paths whose existence suggests the agent is installed here, used by
            ``plugin install all`` to skip agents this node does not have. Empty means the
            target is never auto-selected.
        manual_hint: Command to print when the agent's CLI is missing or absent.
        note: One line of caveat shown by ``plugin install``.
    """

    key: str
    display_name: str
    docs_url: str
    kind: str
    transport: str
    register: Tuple[Tuple[str, ...], ...]
    unregister: Tuple[Tuple[str, ...], ...]
    agent_cli: str
    manual_hint: str
    detect: Tuple[str, ...] = ()
    note: str = ""


TARGETS: Dict[str, AgentTarget] = {
    "claude-code": AgentTarget(
        key="claude-code",
        display_name="Claude Code",
        docs_url="https://code.claude.com/docs/en/plugins-reference",
        kind=BUNDLE,
        transport="url + headers, with type: http",
        register=(
            (
                "claude",
                "plugin",
                "install",
                "trainpilot",
                "--marketplace",
                "{bundle}",
                "--scope",
                "{scope}",
            ),
        ),
        unregister=(("claude", "plugin", "uninstall", "trainpilot"),),
        agent_cli="claude",
        manual_hint="claude plugin marketplace add {bundle}",
        note=(
            "Adding the marketplace only registers it; install the plugin itself with "
            "`claude plugin install trainpilot@trainpilot`."
        ),
    ),
    "antigravity": AgentTarget(
        key="antigravity",
        display_name="Antigravity (CLI, IDE and app)",
        docs_url="https://antigravity.google/docs/plugins",
        kind=BUNDLE,
        transport="serverUrl + headers",
        register=(("agy", "plugin", "install", "{bundle}"),),
        unregister=(("agy", "plugin", "uninstall", "trainpilot"),),
        agent_cli="agy",
        manual_hint="agy plugin install {bundle}",
        note="Antigravity rejects url and httpUrl: the remote server key must be serverUrl.",
    ),
    "gemini-cli": AgentTarget(
        key="gemini-cli",
        display_name="Gemini CLI",
        docs_url="https://geminicli.com/docs/extensions/reference/",
        kind=BUNDLE,
        transport="httpUrl + headers",
        register=(("gemini", "extensions", "link", "{bundle}"),),
        unregister=(("gemini", "extensions", "uninstall", "trainpilot"),),
        agent_cli="gemini",
        manual_hint="gemini extensions link {bundle}",
        note=(
            "httpUrl is the Streamable HTTP key for Gemini CLI; url there means SSE and will "
            "not connect to /mcp."
        ),
    ),
    "opencode": AgentTarget(
        key="opencode",
        display_name="OpenCode",
        docs_url="https://opencode.ai/docs/mcp-servers/",
        kind=IN_PLACE,
        transport="mcp.servers.<name> (remote)",
        register=(),
        unregister=(),
        agent_cli="",
        manual_hint="opencode mcp add trainpilot --url {mcp_url}",
        detect=("~/.config/opencode", ".opencode"),
        note=(
            "Servers live at mcp.servers.<name>, taken from opencode's own config schema; the "
            "flat mcp.<name> form its documentation still shows is marked deprecated."
        ),
    ),
    "codex": AgentTarget(
        key="codex",
        display_name="Codex CLI",
        docs_url="https://learn.chatgpt.com/docs/extend/mcp",
        kind=IN_PLACE,
        transport="[mcp_servers.<name>] url",
        register=(),
        unregister=(),
        agent_cli="",
        manual_hint=(
            "codex mcp add trainpilot --url {mcp_url} "
            "--bearer-token-env-var TRAINPILOT_API_TOKEN"
        ),
        detect=("~/.codex", ".codex"),
        note=(
            "The bearer token is recorded by environment-variable name, so the token itself "
            "is never written into config.toml."
        ),
    ),
    "generic": AgentTarget(
        key="generic",
        display_name="Any other MCP client",
        docs_url="https://modelcontextprotocol.io/docs/develop/connect-remote-servers",
        kind=IN_PLACE,
        transport="mcpServers.<name> url + headers",
        register=(),
        unregister=(),
        agent_cli="",
        manual_hint="copy the printed mcpServers entry into your agent's MCP configuration",
        note=(
            "Also writes an AGENTS.md pointer and a .agents/skills/ skill, the layout Codex, "
            "OpenCode and other AGENTS.md-convention agents read."
        ),
    ),
}


def target_keys() -> Tuple[str, ...]:
    """Supported target keys, in catalogue order, for help text and error messages."""
    return tuple(TARGETS)


def get_target(key: str) -> AgentTarget:
    """Look up a target, or raise ``KeyError`` naming the ones that do exist."""
    try:
        return TARGETS[key]
    except KeyError:
        raise KeyError(
            f"Unknown agent '{key}'. Supported: {', '.join(target_keys())}"
        ) from None
