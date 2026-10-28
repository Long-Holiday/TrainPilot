"""Render the canonical plugin text into each agent's native layout.

Pure functions: a :class:`PluginContext` goes in, a :class:`Bundle` comes out, nothing
touches the disk. ``install`` does the writing.

A bundle is a directory of files whose paths are relative to a staging directory that is
then handed to the agent's plugin manager, which owns where the plugin finally lives. The
commands, the caveats and the vendor facts all come from :mod:`catalog`, so a fact is
written down once and this module cannot disagree with the table.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from trainpilot.agent.plugin import catalog, templates


@dataclass(frozen=True)
class PluginContext:
    """Everything the rendered text depends on.

    Attributes:
        gateway_url: Control plane base URL, no ``/mcp`` suffix, no trailing slash.
        api_token: Bearer token, or ``None`` when the server runs open.
        task_id: Default task id handed to the agent.
    """

    gateway_url: str
    api_token: str | None
    task_id: str

    @property
    def mcp_url(self) -> str:
        return f"{self.gateway_url.rstrip('/')}/mcp"

    @property
    def headers(self) -> Dict[str, str]:
        """Auth header for a remote server, empty when the control plane is open."""
        return {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}


@dataclass(frozen=True)
class File:
    """A file the plugin owns outright. ``path`` is relative to the bundle root."""

    path: str
    content: str


@dataclass(frozen=True)
class Bundle:
    """A rendered plugin, ready to install.

    ``register`` and ``unregister`` hold argv templates rather than final argv: ``{bundle}``
    is only known once the staging directory exists.
    """

    files: Tuple[File, ...]
    register: Tuple[Tuple[str, ...], ...]
    unregister: Tuple[Tuple[str, ...], ...]
    mcp_url: str
    notes: Tuple[str, ...] = field(default=())


# --------------------------------------------------------------------------- helpers


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def _skill_md(ctx: PluginContext) -> str:
    """The SKILL.md.

    The description is emitted through ``json.dumps`` because it is a single YAML scalar
    that contains a colon: unquoted, that is not valid YAML.
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


def _description() -> str:
    return (
        "Supervise a GPU training run through a TrainPilot control plane: report progress "
        "and anomalies, wait for the engineer's decision, and acknowledge the recovery."
    )


def _markdown_command(description: str, argument_hint: str, prompt: str) -> str:
    """A plugin command: YAML frontmatter then the prompt body.

    ``agy`` keeps both keys on the converted skill, so the hint survives to the command
    list. The description goes through ``json.dumps`` for the same reason the skill's does:
    it is a YAML scalar containing a colon.
    """
    front = f"description: {json.dumps(description, ensure_ascii=False)}\n"
    if argument_hint:
        front += f"argument-hint: {json.dumps(argument_hint, ensure_ascii=False)}\n"
    return f"---\n{front}---\n\n{prompt}\n"


# --------------------------------------------------------------------------- renderer


def _antigravity_files(ctx: PluginContext) -> List[File]:
    """Render the Antigravity plugin: manifest, remote MCP config, skill, rules, commands.

    ``serverUrl`` is mandatory -- Antigravity rejects ``url`` and ``httpUrl`` outright --
    and the manifest carries only the two fields its schema declares.
    """
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
    for command in templates.COMMANDS:
        files.append(
            File(
                f"commands/{command.name}.md",
                _markdown_command(command.description, command.argument_hint, command.prompt),
            )
        )
    return files


def marketplace_file(target: catalog.AgentTarget, name: str, source: str) -> File:
    """A marketplace index; source is relative to the marketplace repository root."""
    if target == catalog.CLAUDE_CODE:
        path = ".claude-plugin/marketplace.json"
        marketplace = {
            "name": name,
            "owner": {"name": "TrainPilot"},
            "plugins": [{"name": templates.PLUGIN_NAME, "source": source}],
        }
    elif target == catalog.CODEX:
        path = ".agents/plugins/marketplace.json"
        marketplace = {
            "name": name,
            "interface": {"displayName": "TrainPilot"},
            "plugins": [{
                "name": templates.PLUGIN_NAME,
                "source": {"source": "local", "path": source},
                "policy": {"installation": "AVAILABLE", "authentication": "ON_INSTALL"},
            }],
        }
    else:
        raise ValueError(f"{target.key} does not use a marketplace")
    return File(path, _json(marketplace))


def _native_files(ctx: PluginContext, target: catalog.AgentTarget) -> List[File]:
    """Render native manifests, remote MCP configuration and commands or command skills."""
    manifest = {"name": templates.PLUGIN_NAME, "version": "0.2.0", "description": _description()}
    if target == catalog.CLAUDE_CODE:
        manifest_path = ".claude-plugin/plugin.json"
    elif target == catalog.CODEX:
        manifest_path = ".codex-plugin/plugin.json"
        manifest["skills"] = "./skills/"
    else:
        raise ValueError(f"Unsupported agent: {target.key}")
    entry = {"type": "http", "url": ctx.mcp_url}
    if ctx.headers:
        entry["http_headers" if target == catalog.CODEX else "headers"] = ctx.headers
    files = [
        File(manifest_path, _json(manifest)),
        File(".mcp.json", _json({"mcpServers": {templates.PLUGIN_NAME: entry}})),
        File(f"skills/{templates.SKILL_NAME}/SKILL.md", _skill_md(ctx)),
        marketplace_file(target, target.marketplace_name, "./"),
    ]
    for command in templates.COMMANDS:
        if target == catalog.CLAUDE_CODE:
            files.append(File(
                f"commands/{command.name}.md",
                _markdown_command(command.description, command.argument_hint, command.prompt),
            ))
        else:
            name = f"trainpilot-{command.name}"
            # Codex skills take the user's request as context, not Claude's $ARGUMENTS.
            prompt = command.prompt.replace("$ARGUMENTS", "the arguments supplied by the user")
            files.append(File(
                f"skills/{name}/SKILL.md",
                f"---\nname: {name}\ndescription: {json.dumps(command.description)}\n---\n\n{prompt}\n",
            ))
    return files


def render_bundle(
    ctx: PluginContext, target: catalog.AgentTarget = catalog.ANTIGRAVITY,
) -> Bundle:
    """Render the selected native plugin; preserve Antigravity as the default."""
    files = _antigravity_files(ctx) if target == catalog.ANTIGRAVITY else _native_files(ctx, target)
    return Bundle(
        files=tuple(files),
        register=target.register,
        unregister=target.unregister,
        mcp_url=ctx.mcp_url,
        notes=(target.note,) if target.note else (),
    )
