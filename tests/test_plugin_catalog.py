"""Every agent must get its own vendor's spelling of "here is a remote MCP server".

This is the whole point of the plugin, and the failure mode is silent: Antigravity given
``url`` or Gemini CLI given ``url`` (which means SSE there) produces an agent that starts
normally and simply never sees the TrainPilot tools. So each target's transport key is
pinned by name below, from the vendor's own documentation or CLI.

The other contract pinned here is a negative one: the shipped text must not restate the
supervision protocol. AGENTS.md puts that text in exactly one place -- the server's MCP
``instructions`` -- and the skill's job is to make the agent go and read it.
"""

import json
import sys
from typing import Any, Dict

import pytest

from trainpilot.agent.plugin import catalog, render, templates

if sys.version_info < (3, 11):
    pytest.skip("TOML command validation needs tomllib", allow_module_level=True)

import tomllib  # noqa: E402

GATEWAY = "http://gpu-gateway.example.com:28780"
MCP_URL = f"{GATEWAY}/mcp"
TOKEN = "tok-abc123"

ALL_TARGETS = list(catalog.target_keys())
BUNDLE_TARGETS = [k for k, t in catalog.TARGETS.items() if t.kind == catalog.BUNDLE]


def ctx(token=TOKEN, scope=catalog.USER) -> render.PluginContext:
    return render.PluginContext(
        gateway_url=GATEWAY, api_token=token, task_id="train-task-default", scope=scope
    )


def bundle_for(target: str, **kwargs) -> render.Bundle:
    return render.render_bundle(target, ctx(**kwargs))


def file_map(bundle: render.Bundle) -> Dict[str, str]:
    return {item.path: item.content for item in bundle.files}


def merged(bundle: render.Bundle, suffix: str) -> render.Merge:
    for item in bundle.merges:
        if item.path.endswith(suffix):
            return item
    raise AssertionError(f"no merge targeting {suffix} in {bundle.target}")


def skill_body(bundle: render.Bundle) -> str:
    for path, content in file_map(bundle).items():
        if path.endswith("SKILL.md"):
            return content
    raise AssertionError(f"{bundle.target} ships no SKILL.md")


# --------------------------------------------------------------------------- transport


def test_claude_code_uses_type_http_beside_url():
    """A url with no `type` is read as a stdio server and rejects the whole entry."""
    entry = json.loads(file_map(bundle_for("claude-code"))[".mcp.json"])["mcpServers"]["trainpilot"]
    assert entry["type"] == "http"
    assert entry["url"] == MCP_URL
    assert entry["headers"] == {"Authorization": f"Bearer {TOKEN}"}


def test_antigravity_uses_serverUrl_and_nothing_else():
    """Antigravity rejects url and httpUrl outright."""
    entry = json.loads(file_map(bundle_for("antigravity"))["mcp_config.json"])["mcpServers"]["trainpilot"]
    assert entry["serverUrl"] == MCP_URL
    assert "url" not in entry and "httpUrl" not in entry


def test_gemini_cli_uses_httpUrl_not_url():
    """In Gemini CLI `url` means SSE, so getting this wrong connects to nothing."""
    manifest = json.loads(file_map(bundle_for("gemini-cli"))["gemini-extension.json"])
    entry = manifest["mcpServers"]["trainpilot"]
    assert entry["httpUrl"] == MCP_URL
    assert "url" not in entry
    assert manifest["contextFileName"] == "GEMINI.md"


def test_opencode_uses_the_current_nested_shape_only():
    """`mcp.<name>` is deprecated in opencode's config schema, and the encoded remote
    config sets additionalProperties: false -- so an `enabled` key there is rejected."""
    payload = merged(bundle_for("opencode"), "opencode.json").payload
    entry = payload["mcp"]["servers"]["trainpilot"]
    assert entry["type"] == "remote"
    assert entry["url"] == MCP_URL
    assert set(entry) <= {"type", "url", "headers"}, "opencode rejects unknown keys here"
    assert "trainpilot" not in payload["mcp"], "the deprecated flat form must not be written"


def test_codex_writes_a_parsable_mcp_servers_table():
    block = merged(bundle_for("codex"), "config.toml").payload
    parsed = tomllib.loads(block)
    assert parsed["mcp_servers"]["trainpilot"]["url"] == MCP_URL


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_no_agent_is_given_another_agents_transport_key(target):
    """The remote-server spellings are mutually exclusive, and leaking one into the wrong
    target is the exact mistake this installer exists to prevent."""
    bundle = bundle_for(target)
    rendered = "\n".join(file_map(bundle).values())
    for item in bundle.merges:
        rendered += "\n" + (
            json.dumps(item.payload) if isinstance(item.payload, dict) else str(item.payload)
        )

    if target != "antigravity":
        assert "serverUrl" not in rendered
    if target != "gemini-cli":
        assert "httpUrl" not in rendered
    if target != "claude-code":
        assert '"type": "http"' not in rendered


# --------------------------------------------------------------------------- auth


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_mcp_url_is_the_gateway_plus_mcp(target):
    assert bundle_for(target).mcp_url == MCP_URL


def test_codex_never_writes_the_token_itself():
    """Codex takes the bearer token by env-var name; the secret stays out of config.toml."""
    block = merged(bundle_for("codex"), "config.toml").payload
    assert "bearer_token_env_var = \"TRAINPILOT_API_TOKEN\"" in block
    assert TOKEN not in block


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_open_control_plane_writes_no_empty_bearer_header(target):
    """An empty `Authorization: Bearer ` reads as a failed credential, not as no auth."""
    bundle = bundle_for(target, token=None)
    rendered = "\n".join(file_map(bundle).values())
    for item in bundle.merges:
        rendered += "\n" + (json.dumps(item.payload) if isinstance(item.payload, dict) else str(item.payload))
    assert "Bearer" not in rendered
    assert "headers" not in rendered


# --------------------------------------------------------------------------- the text


def test_every_agent_ships_the_same_skill_text():
    """One source, five layouts. A per-agent copy is what drifts."""
    bodies = {target: skill_body(bundle_for(target)) for target in ALL_TARGETS}
    assert len(set(bodies.values())) == 1


def test_skill_sends_the_agent_to_the_live_protocol():
    body = skill_body(bundle_for("claude-code"))
    assert "MCP initialization" in body
    assert "instructions" in body
    assert "**server**, not `TRAINPILOT_TASK_ID`" in body


def test_no_shipped_text_names_a_trainpilot_command():
    """The GPU node installs no Python, so there is no TrainPilot command to run."""
    for target in ALL_TARGETS:
        bundle = bundle_for(target)
        rendered = "\n".join(file_map(bundle).values())
        for item in bundle.merges:
            rendered += "\n" + (
                json.dumps(item.payload) if isinstance(item.payload, dict) else str(item.payload)
            )
        assert "trainpilot-cli" not in rendered, f"{target} still points at a CLI"
        assert "uv tool install" not in rendered


def test_commands_use_native_mcp_without_a_cli():
    for command in templates.COMMANDS:
        assert "trainpilot-cli" not in command.prompt
    assert "get_task_status MCP tool" in templates.COMMANDS[1].prompt
    assert "report MCP tool" in templates.COMMANDS[2].prompt


def test_an_installed_task_id_becomes_a_binding_hint():
    """The binding is an install-time hint; the rules themselves stay on the server."""
    bound = skill_body(bundle_for("claude-code"))
    assert "Installed task binding" in bound
    assert "train-task-default" in bound

    unbound = render.render_bundle(
        "claude-code", render.PluginContext(GATEWAY, TOKEN, task_id="")
    )
    assert "Installed task binding" not in skill_body(unbound)


def test_skill_does_not_restate_the_server_protocol():
    """AGENTS.md: the behaviour contract lives in mcp_server._INSTRUCTIONS alone. The
    skill may say where to read it, never what it says."""
    from trainpilot.server.mcp_server import _INSTRUCTIONS

    body = skill_body(bundle_for("claude-code"))
    for sentence in ("self_resolve", "ack_instruction with its instruction_id"):
        assert sentence in _INSTRUCTIONS  # guard: the rule text really is in the server
    assert "ack_instruction with its instruction_id" not in body
    assert "next_task_plans" not in body


def test_rules_block_stays_a_pointer():
    assert "`instructions`" in templates.RULES_BLOCK
    assert "trainpilot-cli" not in templates.RULES_BLOCK
    assert len(templates.RULES_BLOCK.splitlines()) < 10


def test_skill_frontmatter_survives_a_yaml_parser():
    """Codex caps the description at 500 chars and requires it on one line; the colon in
    the description means it has to be a quoted scalar."""
    body = skill_body(bundle_for("codex"))
    header = body.split("---")[1]
    lines = [line for line in header.strip().splitlines() if line.strip()]
    assert len(lines) == 2, "frontmatter is name and description only"
    name_line, description_line = lines
    assert name_line == f"name: {templates.SKILL_NAME}"
    assert description_line.startswith("description: ")
    raw = description_line[len("description: "):]
    assert raw.startswith('"') and raw.endswith('"'), "must be quoted to be valid YAML"
    description = json.loads(raw)
    assert len(description) <= 500
    assert "\n" not in description


def test_skill_name_matches_its_directory():
    """Both Codex and OpenCode require the skill name to equal its folder name."""
    for target in ALL_TARGETS:
        for path in file_map(bundle_for(target)):
            if path.endswith("SKILL.md"):
                assert path.split("/")[-2] == templates.SKILL_NAME


# --------------------------------------------------------------------------- artifacts


@pytest.mark.parametrize("target", ["claude-code", "antigravity", "gemini-cli"])
def test_bundle_artifacts_are_parseable(target):
    for path, content in file_map(bundle_for(target)).items():
        if path.endswith(".json"):
            json.loads(content)


def test_gemini_commands_are_valid_toml_with_translated_arguments():
    files = file_map(bundle_for("gemini-cli"))
    commands = {path: content for path, content in files.items() if path.startswith("commands/")}
    assert len(commands) == len(templates.COMMANDS)
    for content in commands.values():
        parsed = tomllib.loads(content)
        assert parsed["description"]
        assert "{{args}}" not in parsed["prompt"] or "$ARGUMENTS" not in parsed["prompt"]
    report = tomllib.loads(files["commands/trainpilot-report.toml"])
    assert "{{args}}" in report["prompt"]


def test_claude_commands_are_namespaced_by_the_plugin():
    """Claude Code namespaces as /<plugin>:<command>, so a bare name is unambiguous."""
    paths = set(file_map(bundle_for("claude-code")))
    assert {f"commands/{c.name}.md" for c in templates.COMMANDS} <= paths


@pytest.mark.parametrize("target", BUNDLE_TARGETS)
def test_bundle_targets_can_be_registered_by_their_own_cli(target):
    target_entry = catalog.TARGETS[target]
    assert target_entry.register, f"{target} has no way to register its bundle"
    assert target_entry.agent_cli, f"{target} registers via no CLI"
    assert any("{bundle}" in part for argv in target_entry.register for part in argv)


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_every_target_names_the_documentation_it_was_verified_against(target):
    url = catalog.TARGETS[target].docs_url
    assert url.startswith("https://")


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_renderer_exists_for_every_catalog_entry(target):
    """A catalogue entry with no renderer would fail only at install time."""
    assert bundle_for(target).target == target


def test_unknown_target_is_rejected_by_name():
    with pytest.raises(KeyError) as excinfo:
        render.render_bundle("cursor", ctx())
    assert "claude-code" in str(excinfo.value) and "antigravity" in str(excinfo.value)


def test_invalid_scope_is_rejected():
    with pytest.raises(ValueError):
        render.render_bundle("codex", render.PluginContext(GATEWAY, TOKEN, "t", scope="galaxy"))


def test_generic_is_never_auto_selected():
    """`install all` must not scatter a fallback config onto a node that named no agent."""
    generic: Any = catalog.TARGETS["generic"]
    assert generic.detect == ()
    assert generic.agent_cli == ""
