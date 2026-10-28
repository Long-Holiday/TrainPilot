"""Native plugin installs coexist and run only inside a throwaway HOME."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from trainpilot.agent.plugin import catalog, commands, render

GATEWAY = "https://gateway.example.com"


def run_installer(*argv):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="group", required=True)
    commands.add_parser(sub)
    args = parser.parse_args(["plugin", *argv])
    return args.func(args)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    assert Path(os.path.expanduser("~")).resolve() == home.resolve()
    calls = []
    monkeypatch.setattr(commands.shutil, "which", lambda name: f"/stub/{name}")

    def run(argv, check=False):
        calls.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(commands.subprocess, "run", run)
    return SimpleNamespace(home=home, work=work, calls=calls)


@pytest.mark.parametrize("agent", ["claude-code", "codex"])
def test_native_install_and_uninstall(sandbox, agent):
    target = catalog.TARGETS[agent]
    staging = commands._staging_dir(target)
    assert run_installer("install", "--agent", agent, "--server-url", GATEWAY) == 0
    assert staging.is_dir()
    assert (staging / "skills/trainpilot/SKILL.md").is_file()
    assert sandbox.calls == [
        [part.replace("{bundle}", str(staging)) for part in template]
        for template in target.register
    ]
    assert run_installer("uninstall", "--agent", agent) == 0
    assert not staging.exists()
    assert sandbox.calls[-len(target.unregister):] == [
        [part.replace("{bundle}", str(staging)) for part in template]
        for template in target.unregister
    ]
    assert not list(sandbox.work.iterdir())


def test_three_installs_coexist_and_uninstall_is_scoped(sandbox):
    for agent in catalog.TARGETS:
        assert run_installer("install", "--agent", agent) == 0
    roots = {key: commands._staging_dir(target) for key, target in catalog.TARGETS.items()}
    assert len(set(roots.values())) == 3
    before = {
        key: {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        for key, root in roots.items()
    }
    assert run_installer("uninstall", "--agent", "codex") == 0
    for agent in ("antigravity", "claude-code"):
        root = roots[agent]
        assert {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before[agent]


@pytest.mark.parametrize("agent", ["antigravity", "claude-code", "codex"])
def test_native_dry_run_does_not_write_or_register(sandbox, agent):
    assert run_installer("install", "--agent", agent, "--dry-run") == 0
    assert not list(sandbox.home.iterdir())
    assert sandbox.calls == []


@pytest.mark.parametrize("agent", ["claude-code", "codex"])
def test_missing_native_cli_reports_failure(sandbox, monkeypatch, capsys, agent):
    monkeypatch.setattr(commands.shutil, "which", lambda name: None)
    assert run_installer("install", "--agent", agent) == 1
    assert "not on PATH" in capsys.readouterr().out
    assert commands._staging_dir(catalog.TARGETS[agent]).is_dir()
    assert sandbox.calls == []


@pytest.mark.parametrize("agent", ["claude-code", "codex"])
def test_reinstall_updates_gateway_auth_and_binding(sandbox, agent):
    assert run_installer(
        "install", "--agent", agent, "--server-url", GATEWAY,
        "--api-token", "old-token", "--task-id", "old-run",
    ) == 0
    assert run_installer(
        "install", "--agent", agent, "--server-url", "https://new.example.com",
        "--api-token", "new-token", "--task-id", "new-run",
    ) == 0
    target = catalog.TARGETS[agent]
    staging = commands._staging_dir(target)
    entry = json.loads((staging / ".mcp.json").read_text())["mcpServers"]["trainpilot"]
    assert entry["url"] == "https://new.example.com/mcp"
    auth_key = "http_headers" if agent == "codex" else "headers"
    assert entry[auth_key] == {"Authorization": "Bearer new-token"}
    assert "new-run" in (staging / "skills/trainpilot/SKILL.md").read_text()
    assert len(sandbox.calls) == len(target.register) * 2


def test_registration_stops_on_failure(sandbox, monkeypatch):
    calls = []

    def fail(argv, check=False):
        calls.append(argv)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(commands.subprocess, "run", fail)
    assert run_installer("install", "--agent", "claude-code") == 1
    assert len(calls) == 1
    assert commands._staging_dir(catalog.CLAUDE_CODE).is_dir()


def test_failed_uninstall_keeps_staged_files(sandbox, monkeypatch):
    assert run_installer("install", "--agent", "claude-code") == 0
    monkeypatch.setattr(commands.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    assert run_installer("uninstall", "--agent", "claude-code") == 1
    assert commands._staging_dir(catalog.CLAUDE_CODE).is_dir()


@pytest.mark.parametrize("agent", ["claude-code", "codex"])
def test_native_render_only_writes_to_requested_directory(sandbox, agent):
    out = sandbox.work / "bundle"
    assert run_installer("render", "--agent", agent, "--out", str(out)) == 0
    assert (out / "skills/trainpilot/SKILL.md").is_file()
    assert not list(sandbox.home.iterdir())
    assert sandbox.calls == []


def test_agent_flag_is_accepted_before_subcommand(sandbox):
    assert run_installer("--agent", "codex", "install") == 0
    assert sandbox.calls[0][0] == "codex"


@pytest.mark.parametrize("agent", ["claude-code", "codex"])
@pytest.mark.parametrize("token", [None, "test-token"])
def test_native_mcp_transport_and_auth(agent, token):
    ctx = render.PluginContext(GATEWAY, token, "run-1")
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
def test_native_manifest_marketplace_and_command_layout(agent):
    target = catalog.TARGETS[agent]
    files = {item.path: item.content for item in render.render_bundle(
        render.PluginContext(GATEWAY, None, ""), target,
    ).files}
    if agent == "claude-code":
        manifest_path = ".claude-plugin/plugin.json"
        index = json.loads(files[".claude-plugin/marketplace.json"])
        assert index["plugins"][0]["source"] == "./"
        assert {path for path in files if path.startswith("commands/")} == {
            "commands/protocol.md", "commands/status.md", "commands/report.md",
        }
    else:
        manifest_path = ".codex-plugin/plugin.json"
        index = json.loads(files[".agents/plugins/marketplace.json"])
        assert index["plugins"][0]["source"] == {"source": "local", "path": "./"}
        for name in ("protocol", "status", "report"):
            skill_name = f"trainpilot-{name}"
            skill = files[f"skills/{skill_name}/SKILL.md"]
            assert f"name: {skill_name}\n" in skill
            assert "$ARGUMENTS" not in skill
        assert not any(path.startswith("commands/") for path in files)
    assert json.loads(files[manifest_path])["name"] == "trainpilot"
    assert index["name"] == target.marketplace_name
