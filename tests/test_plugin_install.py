"""Pin the original Antigravity install behavior inside a throwaway HOME.

The plugin bundle is TrainPilot's own, but what ``agy plugin install`` does with it is not:
it copies the staged directory into the agent's plugin tree. So every test here runs inside
a throwaway ``$HOME`` with ``agy`` stubbed out, and asserts on what was *not* touched as
much as on what was.

What is pinned: the staging path, a re-install being a no-op, ``--dry-run`` writing
nothing, the argv handed to ``agy``, and ``uninstall`` leaving the tree as it found it.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from trainpilot.agent.plugin import catalog, commands, install as installer, render

GATEWAY = "http://gpu-gateway.example.com:28780"

STAGING = Path(".trainpilot/plugins/trainpilot")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A throwaway HOME and working directory.

    The assertion is the point: if ``~`` ever resolves outside the temp tree, every test
    below would be writing into the developer's real agent configuration.
    """
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    assert Path(os.path.expanduser("~")).resolve() == home.resolve()
    return SimpleNamespace(home=home, project=project, root=tmp_path)


@pytest.fixture
def agy(sandbox, monkeypatch):
    """A stub `agy` on PATH that records its argv instead of touching a real agent."""
    bin_dir = sandbox.home / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "agy"
    stub.write_text('#!/bin/sh\necho "agy $@" >> "$HOME/calls.txt"\n')
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))

    def calls() -> str:
        return (sandbox.home / "calls.txt").read_text()

    return calls


def make_bundle(gateway: str = GATEWAY, token: str = "tok-abc123") -> render.Bundle:
    return render.render_bundle(
        render.PluginContext(gateway_url=gateway, api_token=token, task_id="train-task-default")
    )


def staging(home: Path) -> Path:
    return home / STAGING


def install(gateway: str = GATEWAY, token: str = "tok-abc123"):
    """Render, plan and apply the way ``plugin install`` does."""
    bundle = make_bundle(gateway, token)
    changes = installer.plan(bundle, staging(Path(os.path.expanduser("~"))))
    return bundle, changes, installer.apply(changes)


def run_installer(*argv):
    """Drive the plugin command group the way ``scripts/install_agent.py`` does."""
    import argparse

    parser = argparse.ArgumentParser(prog="install_agent.py")
    sub = parser.add_subparsers(dest="group", required=True)
    commands.add_parser(sub, prog="install_agent.py")
    args = parser.parse_args(["plugin", *argv])
    return args.func(args)


def sandbox_files(root: Path):
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}


# --------------------------------------------------------------------------- happy path


def test_install_stages_the_plugin_under_the_home_directory(sandbox):
    install()
    root = staging(sandbox.home)
    assert (root / "plugin.json").is_file()
    assert (root / "mcp_config.json").is_file()
    assert (root / "skills" / "trainpilot" / "SKILL.md").is_file()
    assert (root / "rules" / "trainpilot.md").is_file()


def test_install_writes_only_inside_the_throwaway_home(sandbox):
    install()
    outside = [p for p in sandbox_files(sandbox.root) if not p.parts[0] == "home"]
    assert not outside, f"the install escaped $HOME: {outside}"


def test_reinstall_is_a_no_op(sandbox):
    """Idempotence is what keeps a second install from rewriting a hand-edited bundle."""
    install()
    _, changes, log = install()
    assert {c.action for c in changes} == {installer.UNCHANGED}
    assert all(line.startswith("unchanged") for line in log)


def test_a_changed_gateway_updates_the_staged_config(sandbox):
    install()
    install(gateway="http://other-gateway:9")
    entry = json.loads((staging(sandbox.home) / "mcp_config.json").read_text())
    assert entry["mcpServers"]["trainpilot"]["serverUrl"] == "http://other-gateway:9/mcp"


# --------------------------------------------------------------------------- dry run


def test_dry_run_writes_nothing_at_all(sandbox):
    before = sandbox_files(sandbox.root)
    installer.apply(installer.plan(make_bundle(), staging(sandbox.home)), dry_run=True)
    assert sandbox_files(sandbox.root) == before


def test_dry_run_still_reports_what_it_would_do(sandbox):
    log = installer.apply(installer.plan(make_bundle(), staging(sandbox.home)), dry_run=True)
    assert any("plugin.json" in line for line in log)


# --------------------------------------------------------------------------- argv


def test_register_argv_fills_in_the_bundle_path():
    bundle = make_bundle()
    argv = installer.expand_argv(
        bundle.register[0], bundle=Path("/tmp/staged"), mcp_url=bundle.mcp_url
    )
    assert argv == ["agy", "plugin", "install", "/tmp/staged"]


def test_unregister_argv_names_the_plugin():
    argv = installer.expand_argv(
        catalog.ANTIGRAVITY.unregister[0], bundle=Path("/tmp/staged"), mcp_url=""
    )
    assert argv == ["agy", "plugin", "uninstall", "trainpilot"]


def test_display_path_shortens_the_home_directory(sandbox):
    assert installer.display_path(staging(sandbox.home)) == "~/.trainpilot/plugins/trainpilot"


# --------------------------------------------------------------------------- the command


def test_install_command_registers_through_agy(sandbox, agy):
    assert run_installer("install", "--server-url", GATEWAY) == 0
    assert f"agy plugin install {staging(sandbox.home)}" in agy()


def test_install_command_reports_a_missing_agy_instead_of_claiming_success(sandbox, monkeypatch, capsys):
    """No `agy` on PATH: the bundle is staged, and the failure is reported as a failure."""
    monkeypatch.setenv("PATH", str(sandbox.home / "empty-bin"))
    assert run_installer("install", "--server-url", GATEWAY) == 1
    assert "not on PATH" in capsys.readouterr().out
    assert (staging(sandbox.home) / "mcp_config.json").is_file()


def test_install_command_dry_run_leaves_the_tree_alone(sandbox, capsys):
    before = sandbox_files(sandbox.root)
    assert run_installer("install", "--dry-run") == 0
    assert "dry run" in capsys.readouterr().out
    assert sandbox_files(sandbox.root) == before


def test_uninstall_command_takes_the_plugin_back_out(sandbox, agy):
    assert run_installer("install", "--server-url", GATEWAY) == 0
    assert run_installer("uninstall") == 0
    assert "agy plugin uninstall trainpilot" in agy()
    assert not staging(sandbox.home).exists(), "the staged bundle must not be left behind"


def test_render_writes_the_bundle_without_registering(sandbox, capsys):
    out = sandbox.project / "bundle"
    assert run_installer("render", "--out", str(out)) == 0
    assert (out / "plugin.json").is_file()
    assert not staging(sandbox.home).exists(), "render must not stage or install anything"
    assert "agy plugin install" in capsys.readouterr().out


def test_list_reports_the_target_and_the_staging_state(sandbox, capsys):
    assert run_installer("list") == 0
    printed = capsys.readouterr().out
    assert catalog.ANTIGRAVITY.display_name in printed
    assert catalog.ANTIGRAVITY.transport in printed
    assert GATEWAY not in printed, "list must not invent a gateway the user never configured"
