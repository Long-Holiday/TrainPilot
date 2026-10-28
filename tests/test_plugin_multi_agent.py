"""Each native plugin keeps its own transport and command layout."""

import json

import pytest

from trainpilot.agent.plugin import catalog, render

GATEWAY = "https://gateway.example.com"


@pytest.mark.parametrize("agent", ["claude-code", "codex"])
@pytest.mark.parametrize("token", [None, "test-token"])
def test_native_mcp_transport_and_auth(agent, token):
    ctx = render.PluginContext(GATEWAY, token)
    bundle = render.render_bundle(ctx, catalog.TARGETS[agent])
    files = {item.path: item.content for item in bundle.files}
    entry = json.loads(files[".mcp.json"])["mcpServers"]["trainpilot"]
    assert entry["url"] == f"{GATEWAY}/mcp"
    assert "serverUrl" not in entry
    assert "httpUrl" not in entry
    auth_key = "http_headers" if agent == "codex" else "headers"
    other_key = "headers" if agent == "codex" else "http_headers"
    assert other_key not in entry
    if token:
        assert entry[auth_key] == {"Authorization": f"Bearer {token}"}
    else:
        assert auth_key not in entry


@pytest.mark.parametrize("agent", ["claude-code", "codex"])
def test_native_manifest_and_command_layout(agent):
    target = catalog.TARGETS[agent]
    files = {item.path: item.content for item in render.render_bundle(
        render.PluginContext(GATEWAY, None), target,
    ).files}
    if agent == "claude-code":
        manifest_path = ".claude-plugin/plugin.json"
        assert {path for path in files if path.startswith("commands/")} == {
            "commands/protocol.md", "commands/status.md", "commands/report.md",
        }
    else:
        manifest_path = ".codex-plugin/plugin.json"
        for name in ("protocol", "status", "report"):
            skill_name = f"trainpilot-{name}"
            skill = files[f"skills/{skill_name}/SKILL.md"]
            assert f"name: {skill_name}\n" in skill
            assert "$ARGUMENTS" not in skill
        assert not any(path.startswith("commands/") for path in files)
    assert json.loads(files[manifest_path])["name"] == "trainpilot"
    assert not any(path.endswith("marketplace.json") for path in files)
