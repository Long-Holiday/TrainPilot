"""Render the canonical plugin text into each agent's native layout.

Pure functions: a :class:`PluginContext` goes in, a :class:`Bundle` comes out, nothing
touches the disk. ``install`` does the writing.

A renderer only produces the *content* -- files, merges, and its own caveats. The paths,
the registering command and the manual hint all come from :mod:`catalog`, so a fact is
written down once and a renderer cannot disagree with the table.

Two shapes of bundle, matching :data:`catalog.BUNDLE` and :data:`catalog.IN_PLACE`:

* **bundle** -- ``File.path`` is relative to a staging directory that is then handed to the
  agent's own CLI, which owns where it finally lives.
* **in-place** -- ``File.path`` and ``Merge.path`` are real destinations, either ``~``-anchored
  (user scope) or relative to the working directory (project scope).
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from trainpilot import __version__
from trainpilot.agent.plugin import catalog, templates

JSON = "json"
TOML = "toml"
MARKDOWN = "markdown"


@dataclass(frozen=True)
class PluginContext:
    """Everything the rendered text depends on.

    Attributes:
        gateway_url: Control plane base URL, no ``/mcp`` suffix, no trailing slash.
        api_token: Bearer token, or ``None`` when the server runs open.
        task_id: Default task id handed to the agent.
        scope: :data:`catalog.USER` or :data:`catalog.PROJECT`.
        version: Version stamped into the manifests.
    """

    gateway_url: str
    api_token: Optional[str]
    task_id: str
    scope: str = catalog.USER
    version: str = __version__

    @property
    def mcp_url(self) -> str:
        return f"{self.gateway_url.rstrip('/')}/mcp"

    @property
    def headers(self) -> Dict[str, str]:
        """Auth header for a remote server, empty when the control plane is open."""
        return {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}


@dataclass(frozen=True)
class File:
    """A file the plugin owns outright. ``path`` may start with ``~``."""

    path: str
    content: str


@dataclass(frozen=True)
class Merge:
    """A file the plugin shares with the user, so it must be edited, not replaced.

    Attributes:
        path: Destination; may start with ``~``.
        fmt: :data:`JSON`, :data:`TOML` or :data:`MARKDOWN`.
        payload: For JSON, the nested mapping to deep-merge in. For the text formats, the
            block body written between trainpilot markers, replacing any previous block.
        remove_keys: For JSON, the nested key path ``uninstall`` deletes.
    """

    path: str
    fmt: str
    payload: Any
    remove_keys: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Bundle:
    """A rendered plugin, ready to install.

    ``register`` and ``unregister`` hold argv templates rather than final argv: ``{bundle}``
    is only known once the staging directory exists, and ``{scope}`` only at install time.
    """

    target: str
    kind: str
    files: Tuple[File, ...]
    merges: Tuple[Merge, ...]
    register: Tuple[Tuple[str, ...], ...]
    unregister: Tuple[Tuple[str, ...], ...]
    manual_hint: str
    mcp_url: str
    notes: Tuple[str, ...] = field(default=())


# --------------------------------------------------------------------------- helpers


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def _toml_string(value: str) -> str:
    """A TOML basic string. JSON escaping is a subset of TOML's, minus backspace/formfeed."""
    escaped = json.dumps(value, ensure_ascii=False)
    return escaped.replace("\\b", "\\u0008").replace("\\f", "\\u000c")


def _skill_md(ctx: PluginContext) -> str:
    """The SKILL.md shared by every agent that reads skills.

    The description is emitted through ``json.dumps`` because it is a single YAML scalar
    that contains a colon: unquoted, that is not valid YAML. Codex caps it at 500
    characters, which ``templates.SKILL_DESCRIPTION`` is written to respect.
    """
    description = json.dumps(templates.SKILL_DESCRIPTION, ensure_ascii=False)
    binding = (
        "\n## Installed task binding\n\n"
        f"The installer selected task id `{ctx.task_id}`. Pass this value explicitly "
        "unless the user binds this session to a different run.\n"
        if ctx.task_id else ""
    )
    return (
        f"---\nname: {templates.SKILL_NAME}\ndescription: {description}\n---\n\n"
        f"{templates.SKILL_BODY}{binding}"
    )


def _markdown_command(description: str, argument_hint: str, prompt: str) -> str:
    """A Claude Code / OpenCode style command: YAML frontmatter then the prompt body."""
    front = f"description: {json.dumps(description, ensure_ascii=False)}\n"
    if argument_hint:
        front += f"argument-hint: {json.dumps(argument_hint, ensure_ascii=False)}\n"
    return f"---\n{front}---\n\n{prompt}\n"


def _description() -> str:
    return (
        "Supervise a GPU training run through a TrainPilot control plane: report progress "
        "and anomalies, wait for the engineer's decision, and acknowledge the recovery."
    )


def _pick(user_path: str, project_path: str, scope: str) -> str:
    return project_path if scope == catalog.PROJECT else user_path


# --------------------------------------------------------------------------- renderers
#
# Each returns (files, merges, extra notes); render_bundle adds the catalogue's register
# commands, manual hint and standard caveat.


def _claude_code(ctx: PluginContext) -> Tuple[List[File], List[Merge], List[str]]:
    """A Claude Code plugin plus the marketplace that makes it installable by path.

    One directory is both the marketplace root and the plugin root (``source: "./"``), so a
    single ``claude plugin install --marketplace <dir>`` covers registration and install.
    """
    manifest = {
        "name": templates.PLUGIN_NAME,
        "displayName": "TrainPilot",
        "version": ctx.version,
        "description": _description(),
        "author": {"name": "TrainPilot"},
        "license": "MIT",
        "keywords": ["training", "gpu", "hitl", "mcp", "feishu"],
    }
    marketplace = {
        "name": templates.PLUGIN_NAME,
        "owner": {"name": "TrainPilot"},
        "description": "TrainPilot: human-in-the-loop supervision for GPU training runs.",
        "plugins": [
            {
                "name": templates.PLUGIN_NAME,
                "source": "./",
                "description": _description(),
                "category": "monitoring",
                "tags": ["training", "gpu", "hitl"],
            }
        ],
    }
    mcp = {
        "mcpServers": {
            templates.PLUGIN_NAME: {
                # `type` is not optional: Claude Code reads a url-with-no-type entry as a
                # stdio server and reports the whole entry as a config error.
                "type": "http",
                "url": ctx.mcp_url,
                **({"headers": ctx.headers} if ctx.headers else {}),
            }
        }
    }

    files: List[File] = [
        File(".claude-plugin/plugin.json", _json(manifest)),
        File(".claude-plugin/marketplace.json", _json(marketplace)),
        File(".mcp.json", _json(mcp)),
        File(f"skills/{templates.SKILL_NAME}/SKILL.md", _skill_md(ctx)),
    ]
    for command in templates.COMMANDS:
        files.append(
            File(
                f"commands/{command.name}.md",
                _markdown_command(command.description, command.argument_hint, command.prompt),
            )
        )
    return files, [], []


def _antigravity(ctx: PluginContext) -> Tuple[List[File], List[Merge], List[str]]:
    """An Antigravity plugin. ``serverUrl`` is mandatory -- url and httpUrl are rejected."""
    manifest = {
        "$schema": "https://antigravity.google/schemas/v1/plugin.json",
        "name": templates.PLUGIN_NAME,
        "description": _description(),
    }
    mcp = {
        "mcpServers": {
            templates.PLUGIN_NAME: {
                "serverUrl": ctx.mcp_url,
                **({"headers": ctx.headers} if ctx.headers else {}),
            }
        }
    }
    files = [
        File("plugin.json", _json(manifest)),
        File("mcp_config.json", _json(mcp)),
        File(f"skills/{templates.SKILL_NAME}/SKILL.md", _skill_md(ctx)),
        File("rules/trainpilot.md", templates.RULES_BLOCK),
    ]
    return files, [], []


def _gemini_cli(ctx: PluginContext) -> Tuple[List[File], List[Merge], List[str]]:
    """A Gemini CLI extension. ``httpUrl`` is Streamable HTTP; ``url`` there means SSE."""
    manifest = {
        "name": templates.PLUGIN_NAME,
        "version": ctx.version,
        "description": _description(),
        "contextFileName": "GEMINI.md",
        "mcpServers": {
            templates.PLUGIN_NAME: {
                "httpUrl": ctx.mcp_url,
                **({"headers": ctx.headers} if ctx.headers else {}),
            }
        },
    }
    files: List[File] = [
        File("gemini-extension.json", _json(manifest)),
        File("GEMINI.md", templates.RULES_BLOCK),
        File(f"skills/{templates.SKILL_NAME}/SKILL.md", _skill_md(ctx)),
    ]
    for command in templates.COMMANDS:
        # Gemini CLI custom commands are TOML and take their argument as {{args}}.
        prompt = command.prompt.replace("$ARGUMENTS", "{{args}}")
        files.append(
            File(
                f"commands/{templates.PLUGIN_NAME}-{command.name}.toml",
                f"description = {_toml_string(command.description)}\n"
                f"prompt = {_toml_string(prompt)}\n",
            )
        )
    return files, [], []


def _opencode(ctx: PluginContext) -> Tuple[List[File], List[Merge], List[str]]:
    """In-place: OpenCode has no plugin package, so config and assets are written live."""
    user = ctx.scope == catalog.USER
    config_path = _pick("~/.config/opencode/opencode.json", "opencode.json", ctx.scope)
    # Both directories are plural in both scopes, verbatim from the OpenCode docs: skills at
    # <config>/skills/<name>/SKILL.md, commands at <config>/commands/<name>.md, where <config>
    # is ~/.config/opencode globally and .opencode/ per project.
    command_dir = "~/.config/opencode/commands" if user else ".opencode/commands"
    skill_dir = "~/.config/opencode/skills" if user else ".opencode/skills"

    entry: Dict[str, Any] = {"type": "remote", "url": ctx.mcp_url}
    if ctx.headers:
        entry["headers"] = ctx.headers

    files = [File(f"{skill_dir}/{templates.SKILL_NAME}/SKILL.md", _skill_md(ctx))]
    for command in templates.COMMANDS:
        files.append(
            File(
                f"{command_dir}/trainpilot-{command.name}.md",
                _markdown_command(command.description, command.argument_hint, command.prompt),
            )
        )

    # `mcp.servers.<name>` is the current shape and the one `opencode mcp add` writes; the
    # flat `mcp.<name>` the docs still show is marked deprecated in opencode's own
    # config.json schema. The encoded remote entry has no `enabled` field, and a server is
    # on by default, so the entry is left minimal. No `$schema` is added either: it belongs
    # to the user's document, and uninstall could not tell theirs from ours.
    merge = Merge(
        config_path,
        JSON,
        {"mcp": {"servers": {templates.PLUGIN_NAME: entry}}},
        remove_keys=("mcp", "servers", templates.PLUGIN_NAME),
    )
    return files, [merge], []


def _codex(ctx: PluginContext) -> Tuple[List[File], List[Merge], List[str]]:
    """In-place: config.toml, an AGENTS.md pointer, and the shared .agents/ skill.

    The token is recorded by environment-variable name rather than value, so it stays out of
    ``config.toml``; ``bearer_token_env_var`` is exactly what ``codex mcp add`` writes.
    """
    user = ctx.scope == catalog.USER
    config_path = _pick("~/.codex/config.toml", ".codex/config.toml", ctx.scope)
    instructions_path = _pick("~/.codex/AGENTS.md", "AGENTS.md", ctx.scope)
    skill_root = "~/.agents/skills" if user else ".agents/skills"

    entry: Dict[str, Any] = {"url": ctx.mcp_url, "enabled": True}
    if ctx.api_token:
        entry["bearer_token_env_var"] = "TRAINPILOT_API_TOKEN"

    table = "\n".join(
        [f"[mcp_servers.{templates.PLUGIN_NAME}]"]
        + [
            f"{key} = {_toml_string(value) if isinstance(value, str) else json.dumps(value)}"
            for key, value in entry.items()
        ]
    )

    notes: List[str] = []
    if ctx.api_token:
        notes.append(
            "Export TRAINPILOT_API_TOKEN on this node, or the Codex MCP client will call "
            "/mcp unauthenticated."
        )
    if ctx.scope == catalog.PROJECT:
        notes.append(
            "Project-scope Codex config is only read in a project you have marked trusted."
        )

    return (
        [File(f"{skill_root}/{templates.SKILL_NAME}/SKILL.md", _skill_md(ctx))],
        [
            Merge(config_path, TOML, table),
            Merge(instructions_path, MARKDOWN, templates.RULES_BLOCK),
        ],
        notes,
    )


def _generic(ctx: PluginContext) -> Tuple[List[File], List[Merge], List[str]]:
    """The fallback: the cross-vendor skill, an AGENTS.md pointer, and a printed snippet.

    The MCP entry is only printed -- guessing an unknown agent's config file is exactly how
    this goes wrong -- but ``.agents/skills/`` is a convention Codex, OpenCode and others
    already read, so the skill half genuinely lands.
    """
    user = ctx.scope == catalog.USER
    skill_root = "~/.agents/skills" if user else ".agents/skills"
    merges: List[Merge] = [] if user else [Merge("AGENTS.md", MARKDOWN, templates.RULES_BLOCK)]
    return [File(f"{skill_root}/{templates.SKILL_NAME}/SKILL.md", _skill_md(ctx))], merges, []


_RENDERERS = {
    "claude-code": _claude_code,
    "antigravity": _antigravity,
    "gemini-cli": _gemini_cli,
    "opencode": _opencode,
    "codex": _codex,
    "generic": _generic,
}


def render_bundle(target: str, ctx: PluginContext) -> Bundle:
    """Render ``target`` for ``ctx``, or raise ``KeyError`` listing the supported agents."""
    entry = catalog.get_target(target)
    if ctx.scope not in catalog.SCOPES:
        raise ValueError(f"scope must be one of {', '.join(catalog.SCOPES)}, got '{ctx.scope}'")
    files, merges, notes = _RENDERERS[target](ctx)
    if entry.note:
        notes = [entry.note, *notes]
    return Bundle(
        target=target,
        kind=entry.kind,
        files=tuple(files),
        merges=tuple(merges),
        register=entry.register,
        unregister=entry.unregister,
        manual_hint=entry.manual_hint,
        mcp_url=ctx.mcp_url,
        notes=tuple(notes),
    )
