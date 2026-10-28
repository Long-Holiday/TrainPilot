"""Native transport spellings and the shared live-protocol pointer.

This is the whole point of the plugin, and the failure mode is silent: Antigravity given
``url`` or ``httpUrl`` (or a Claude-shaped plugin, which its CLI converts into a stdio entry
with the URL dropped) produces an agent that starts normally and simply never sees the
TrainPilot tools. So the transport key, the manifest shape and the file set are pinned by
name below, from the vendor's own documentation and CLI.

The other contract pinned here is a negative one: the shipped text must not restate the
supervision protocol. AGENTS.md puts that text in exactly one place -- the server's MCP
``instructions`` -- and the skill's job is to make the agent go and read it.
"""

import json
from typing import Dict

import pytest

from trainpilot.agent.plugin import catalog, render, templates

GATEWAY = "http://gpu-gateway.example.com:28780"
MCP_URL = f"{GATEWAY}/mcp"
TOKEN = "tok-abc123"

MANIFEST = "plugin.json"
MCP_CONFIG = "mcp_config.json"


def ctx(token=TOKEN) -> render.PluginContext:
    return render.PluginContext(gateway_url=GATEWAY, api_token=token)


def bundle(**kwargs) -> render.Bundle:
    return render.render_bundle(ctx(**kwargs))


def file_map(b: render.Bundle) -> Dict[str, str]:
    return {item.path: item.content for item in b.files}


def skill_body(b: render.Bundle) -> str:
    for path, content in file_map(b).items():
        if path.endswith("SKILL.md"):
            return content
    raise AssertionError("the bundle ships no SKILL.md")


def rendered(b: render.Bundle) -> str:
    return "\n".join(file_map(b).values())


# --------------------------------------------------------------------------- transport


def test_antigravity_uses_serverUrl_and_nothing_else():
    """Antigravity rejects url and httpUrl outright."""
    entry = json.loads(file_map(bundle())[MCP_CONFIG])["mcpServers"]["trainpilot"]
    assert entry["serverUrl"] == MCP_URL
    assert "url" not in entry and "httpUrl" not in entry


def test_no_transport_key_of_another_agent_leaks_in():
    """A converted Claude plugin is exactly how a remote server loses its URL."""
    text = rendered(bundle())
    assert "httpUrl" not in text
    assert '"type": "http"' not in text
    assert "[mcp_servers" not in text


def test_the_manifest_carries_only_what_the_vendor_schema_declares():
    """Antigravity's plugin.json sets additionalProperties: false with two properties."""
    manifest = json.loads(file_map(bundle())[MANIFEST])
    assert manifest["name"] == templates.PLUGIN_NAME
    assert manifest["description"]
    assert set(manifest) == {"$schema", "name", "description"}
    assert manifest["$schema"].startswith("https://antigravity.google/")


def test_the_bundle_is_exactly_the_plugin_assets():
    """Pointing Antigravity at a directory hands it *everything* in that directory, so the
    bundle must not pick up the Python project."""
    assert set(file_map(bundle())) == {
        MANIFEST,
        MCP_CONFIG,
        f"skills/{templates.SKILL_NAME}/SKILL.md",
        "rules/trainpilot.md",
    } | {f"commands/{command.name}.md" for command in templates.COMMANDS}


# --------------------------------------------------------------------------- auth


def test_open_control_plane_writes_no_empty_bearer_header():
    """An empty `Authorization: Bearer ` reads as a failed credential, not as no auth."""
    text = rendered(bundle(token=None))
    assert "Bearer" not in text
    assert "headers" not in text


def test_a_configured_token_becomes_a_bearer_header():
    entry = json.loads(file_map(bundle())[MCP_CONFIG])["mcpServers"]["trainpilot"]
    assert entry["headers"] == {"Authorization": f"Bearer {TOKEN}"}


# --------------------------------------------------------------------------- the text


def test_skill_sends_the_agent_to_the_live_protocol():
    body = skill_body(bundle())
    assert "MCP initialization" in body
    assert "instructions" in body
    assert "**server**, not `TRAINPILOT_TASK_ID`" in body


def test_no_shipped_text_names_a_trainpilot_command():
    """The GPU node installs no Python, so there is no TrainPilot command to run."""
    text = rendered(bundle())
    assert "trainpilot-cli" not in text
    assert "uv tool install" not in text


def test_skill_does_not_ship_an_installer_task_binding():
    body = skill_body(bundle())
    assert "Installed task binding" not in body
    assert "train-task-default" not in body


def test_skill_does_not_restate_the_server_protocol():
    """AGENTS.md: the behaviour contract lives in mcp_server._INSTRUCTIONS alone. The
    skill may say where to read it, never what it says."""
    from trainpilot.server.mcp_server import _INSTRUCTIONS

    body = skill_body(bundle())
    for sentence in ("self_resolve", "ack_instruction with its instruction_id"):
        assert sentence in _INSTRUCTIONS  # guard: the rule text really is in the server
    assert "ack_instruction with its instruction_id" not in body
    assert "next_task_plans" not in body


def test_rules_block_stays_a_pointer():
    assert "`instructions`" in templates.RULES_BLOCK
    assert "trainpilot-cli" not in templates.RULES_BLOCK
    assert len(templates.RULES_BLOCK.splitlines()) < 10


def test_skill_frontmatter_survives_a_yaml_parser():
    """The colon in the description means it has to be a quoted YAML scalar."""
    header = skill_body(bundle()).split("---")[1]
    lines = [line for line in header.strip().splitlines() if line.strip()]
    assert len(lines) == 2, "frontmatter is name and description only"
    name_line, description_line = lines
    assert name_line == f"name: {templates.SKILL_NAME}"
    raw = description_line[len("description: "):]
    assert raw.startswith('"') and raw.endswith('"'), "must be quoted to be valid YAML"
    description = json.loads(raw)
    assert description and "\n" not in description


def test_skill_name_matches_its_directory():
    for path in file_map(bundle()):
        if path.endswith("SKILL.md"):
            assert path.split("/")[-2] == templates.SKILL_NAME


def test_commands_use_native_mcp_without_a_cli():
    """`agy` reports "commands: N processed (converted to skills)" for this directory, so
    the commands ship even though Antigravity's docs never mention one."""
    files = file_map(bundle())
    commands = {path: content for path, content in files.items() if path.startswith("commands/")}
    assert len(commands) == len(templates.COMMANDS) == 3

    for command in templates.COMMANDS:
        content = commands[f"commands/{command.name}.md"]
        header = content.split("---")[1]
        lines = [line for line in header.strip().splitlines() if line.strip()]
        assert lines[0].startswith("description: ")
        assert json.loads(lines[0][len("description: "):]) == command.description
        assert content.strip().endswith(command.prompt)
        assert "trainpilot-cli" not in content


def test_a_command_names_its_argument_in_words_as_well_as_with_a_placeholder():
    """Argument substitution is undocumented in Antigravity, so a prompt that only said
    `$ARGUMENTS` would read oddly if the token is never expanded."""
    status = next(c for c in templates.COMMANDS if c.name == "status")
    assert "already agreed for this run" in status.prompt


# --------------------------------------------------------------------------- artifacts


def test_bundle_artifacts_are_parseable():
    for path, content in file_map(bundle()).items():
        if path.endswith(".json"):
            json.loads(content)


def test_the_target_names_the_documentation_it_was_verified_against():
    assert catalog.ANTIGRAVITY.docs_url.startswith("https://")
    assert "serverUrl" in catalog.ANTIGRAVITY.transport


@pytest.mark.parametrize("key", ["gemini-cli", "opencode", "generic", "cursor"])
def test_no_other_agent_is_supported(key):
    assert key not in catalog.TARGETS


def test_all_three_agents_are_catalogued():
    assert set(catalog.TARGETS) == {"antigravity", "claude-code", "codex"}
    for key, target in catalog.TARGETS.items():
        assert target.key == key
        assert target.docs_url.startswith("https://")
        assert target.agent_cli


@pytest.mark.parametrize("agent", ["antigravity", "claude-code", "codex"])
def test_each_agent_uses_the_same_live_protocol_skill(agent):
    native = render.render_bundle(ctx(), catalog.TARGETS[agent])
    assert file_map(native)["skills/trainpilot/SKILL.md"] == skill_body(bundle())
    assert "trainpilot-cli" not in rendered(native)
    for item in native.files:
        if item.path.endswith(".json"):
            json.loads(item.content)
