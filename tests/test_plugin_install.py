"""Installing must be survivable: these files belong to the user.

The installer edits ``config.toml``, ``opencode.json`` and ``AGENTS.md`` -- files a person
already owns and may have edited. So every test here runs inside a throwaway ``$HOME`` with
the working directory moved into a temp tree, and asserts on what was *not* touched as much
as on what was.

What is pinned: a re-install is a no-op, a merge keeps unrelated keys and unrelated
sections, a file that will not parse is refused rather than overwritten, ``--dry-run``
writes nothing, and ``uninstall`` gives the file back.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from trainpilot.agent.plugin import catalog, install as installer, render

GATEWAY = "http://gpu-gateway.example.com:28780"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A throwaway HOME and working directory.

    The assertion is the point: if ``~`` ever resolves outside the temp tree, every test
    below would be editing the developer's real agent configuration.
    """
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    assert Path(os.path.expanduser("~")).resolve() == home.resolve()
    return SimpleNamespace(home=home, project=project, root=tmp_path)


def make_bundle(target: str, gateway: str = GATEWAY, token: str = "tok-abc123", scope: str = catalog.USER):
    ctx = render.PluginContext(
        gateway_url=gateway, api_token=token, task_id="train-task-default", scope=scope
    )
    return render.render_bundle(target, ctx)


def install(target: str, **kwargs):
    """Render, plan and apply one target the way ``plugin install`` does."""
    bundle = make_bundle(target, **{k: v for k, v in kwargs.items() if k in ("gateway", "token", "scope")})
    staging = None
    if bundle.kind == catalog.BUNDLE:
        staging = Path(os.path.expanduser(f"~/.trainpilot/plugins/{target}"))
    changes = installplan(bundle, staging)
    log = installer.apply(changes)
    return bundle, changes, log


def installplan(bundle, staging):
    return installer.plan(bundle, staging)


def sandbox_files(root: Path):
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}


# --------------------------------------------------------------------------- happy path


def test_install_writes_only_inside_the_throwaway_home(sandbox):
    for target in (target for target, t in catalog.TARGETS.items() if t.kind == catalog.IN_PLACE):
        install(target)
    outside = [p for p in sandbox_files(sandbox.root) if not p.parts[0] == "home"]
    assert not outside, f"an in-place install escaped $HOME: {outside}"


def test_every_target_installs_cleanly(sandbox):
    for target in catalog.target_keys():
        _, changes, log = install(target)
        assert changes, f"{target} produced nothing to install"
        assert all(line.startswith(("create", "update", "unchanged")) for line in log)


def test_reinstall_is_a_no_op(sandbox):
    """Idempotence is what keeps a second install from churning the user's files."""
    install("codex")
    bundle, changes, log = install("codex")
    assert {c.action for c in changes} == {installer.UNCHANGED}
    assert all(line.startswith("unchanged") for line in log)


def test_reinstall_adds_no_second_backup(sandbox):
    """A no-op install must not overwrite the backup, or the original bytes are lost."""
    install("codex")
    config = sandbox.home / ".codex" / "config.toml"
    backup = config.with_name(config.name + installer.BACKUP_SUFFIX)
    assert not backup.exists(), "the first install creates the file, so there is nothing to back up"

    install("codex", gateway="http://other:1")
    after_first_change = backup.read_text()
    assert GATEWAY in after_first_change, "the backup must hold the bytes from before the change"

    install("codex", gateway="http://other:1")
    assert backup.read_text() == after_first_change


# --------------------------------------------------------------------------- merging


def test_json_merge_keeps_unrelated_keys_and_servers(sandbox):
    config = sandbox.home / ".config" / "opencode" / "opencode.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"theme": "tokyonight", "mcp": {"servers": {"other": {"type": "remote", "url": "http://x"}}}}))

    install("opencode")

    merged = json.loads(config.read_text())
    assert merged["theme"] == "tokyonight"
    assert merged["mcp"]["servers"]["other"] == {"type": "remote", "url": "http://x"}
    assert merged["mcp"]["servers"]["trainpilot"]["url"].endswith("/mcp")


def test_json_merge_refuses_a_file_it_cannot_parse(sandbox):
    """Overwriting would silently delete whatever the comments were documenting."""
    config = sandbox.home / ".config" / "opencode" / "opencode.json"
    config.parent.mkdir(parents=True)
    original = '{\n  // opencode accepts jsonc\n  "theme": "x"\n}\n'
    config.write_text(original)

    with pytest.raises(installer.ConfigError) as excinfo:
        install("opencode")

    assert "not valid JSON" in str(excinfo.value)
    assert config.read_text() == original


def test_block_merge_keeps_content_above_and_below(sandbox):
    config = sandbox.home / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text('model = "gpt-5"\n')

    install("codex")

    text = config.read_text()
    assert 'model = "gpt-5"' in text
    assert text.count("# >>> trainpilot >>>") == 1
    assert text.count("# <<< trainpilot <<<") == 1
    # TOML has to still parse, and the user's key has to still be there.
    import tomllib

    parsed = tomllib.loads(text)
    assert parsed["model"] == "gpt-5"
    assert parsed["mcp_servers"]["trainpilot"]["url"].endswith("/mcp")


def test_a_second_block_install_replaces_rather_than_appends(sandbox):
    install("codex")
    install("codex", gateway="http://changed:9")
    text = (sandbox.home / ".codex" / "config.toml").read_text()
    assert text.count("# >>> trainpilot >>>") == 1
    assert "changed:9" in text
    assert GATEWAY not in text


def test_markers_lower_below_a_later_section(sandbox):
    """The block is appended, so anything the user adds afterwards must survive."""
    install("codex")
    config = sandbox.home / ".codex" / "config.toml"
    config.write_text(config.read_text() + '\n[profiles.work]\nmodel = "o3"\n')

    install("codex", gateway="http://changed:9")

    text = config.read_text()
    assert '[profiles.work]\nmodel = "o3"' in text
    assert text.count("# >>> trainpilot >>>") == 1
    import tomllib

    assert tomllib.loads(text)["profiles"]["work"]["model"] == "o3"


# --------------------------------------------------------------------------- dry run


def test_dry_run_writes_nothing_at_all(sandbox):
    before = sandbox_files(sandbox.root)
    for target in catalog.target_keys():
        bundle = make_bundle(target)
        staging = (
            Path(os.path.expanduser(f"~/.trainpilot/plugins/{target}"))
            if bundle.kind == catalog.BUNDLE
            else None
        )
        installer.apply(installplan(bundle, staging), dry_run=True)
    assert sandbox_files(sandbox.root) == before


def test_dry_run_still_reports_what_it_would_do(sandbox):
    bundle = make_bundle("antigravity")
    staging = Path(os.path.expanduser("~/.trainpilot/plugins/antigravity"))
    log = installer.apply(installplan(bundle, staging), dry_run=True)
    assert any("plugin.json" in line for line in log)


# --------------------------------------------------------------------------- uninstall


def test_uninstall_gives_back_a_pre_existing_config(sandbox):
    config = sandbox.home / ".config" / "opencode" / "opencode.json"
    config.parent.mkdir(parents=True)
    before = {"theme": "tokyonight", "mcp": {"servers": {"other": {"type": "remote", "url": "http://x"}}}}
    config.write_text(json.dumps(before))

    install("opencode")
    assert "trainpilot" in config.read_text()

    installer.apply(installer.plan_revert(make_bundle("opencode")))

    assert json.loads(config.read_text()) == before, "uninstall must leave the document as it was"


def test_uninstall_removes_what_it_created_and_prunes_directories(sandbox):
    install("codex")
    assert (sandbox.home / ".agents" / "skills" / "trainpilot" / "SKILL.md").exists()

    installer.apply(installer.plan_revert(make_bundle("codex")))

    remaining = sandbox_files(sandbox.home)
    assert not any("SKILL.md" in str(p) for p in remaining)
    assert not (sandbox.home / ".agents").exists(), "an emptied directory is left behind as litter"
    assert not (sandbox.home / ".codex").exists()


def test_uninstall_keeps_a_config_the_user_still_uses(sandbox):
    config = sandbox.home / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text('model = "gpt-5"\n')

    install("codex")
    installer.apply(installer.plan_revert(make_bundle("codex")))

    assert config.exists()
    assert config.read_text().strip() == 'model = "gpt-5"'


def test_uninstall_is_safe_to_repeat(sandbox):
    install("codex")
    installer.apply(installer.plan_revert(make_bundle("codex")))
    second = installer.plan_revert(make_bundle("codex"))
    assert second == []


def test_uninstall_does_not_touch_a_config_it_never_wrote_to(sandbox):
    config = sandbox.home / ".config" / "opencode" / "opencode.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"theme": "tokyonight"}))
    installer.apply(installer.plan_revert(make_bundle("opencode")))
    assert json.loads(config.read_text()) == {"theme": "tokyonight"}


# --------------------------------------------------------------------------- argv


def test_register_argv_fills_in_the_bundle_path(sandbox):
    bundle = make_bundle("antigravity")
    argv = installer.expand_argv(
        bundle.register[0], bundle=Path("/tmp/staged"), scope="user", mcp_url=bundle.mcp_url
    )
    assert argv == ["agy", "plugin", "install", "/tmp/staged"]


def test_claude_code_registration_carries_the_scope(sandbox):
    bundle = make_bundle("claude-code", scope=catalog.PROJECT)
    argv = installer.expand_argv(
        bundle.register[0], bundle=Path("/tmp/staged"), scope="project", mcp_url=bundle.mcp_url
    )
    assert argv[:4] == ["claude", "plugin", "install", "trainpilot"]
    assert "--scope" in argv and argv[argv.index("--scope") + 1] == "project"


# --------------------------------------------------------------------------- the command


def run_cli(*argv):
    from trainpilot.agent.cli import main

    return main(list(argv))


def test_cli_refuses_an_unknown_agent_without_writing(sandbox, capsys):
    """The installer must never guess at a config file for an agent it does not know."""
    before = sandbox_files(sandbox.root)
    assert run_cli("plugin", "install", "cursor") == 1
    assert "cursor" in capsys.readouterr().err
    assert sandbox_files(sandbox.root) == before


def test_cli_installs_generic_user_scope_under_the_sandbox_home(sandbox):
    assert run_cli("plugin", "install", "generic", "--server-url", GATEWAY) == 0
    skill = sandbox.home / ".agents" / "skills" / "trainpilot" / "SKILL.md"
    assert skill.exists()
    # User scope has no single AGENTS.md to edit, so the config is left alone.
    assert not (sandbox.home / "AGENTS.md").exists()


def test_cli_project_scope_writes_into_the_working_directory(sandbox):
    assert run_cli("plugin", "install", "generic", "--scope", "project") == 0
    assert (sandbox.project / ".agents" / "skills" / "trainpilot" / "SKILL.md").exists()
    assert "trainpilot" in (sandbox.project / "AGENTS.md").read_text()


def test_cli_dry_run_leaves_the_tree_alone(sandbox, capsys):
    before = sandbox_files(sandbox.root)
    assert run_cli("plugin", "install", "codex", "--dry-run") == 0
    assert "dry run" in capsys.readouterr().out
    assert sandbox_files(sandbox.root) == before


def test_cli_render_prints_in_place_targets_instead_of_writing(sandbox, capsys):
    assert run_cli("plugin", "render", "opencode") == 0
    printed = capsys.readouterr().out
    assert "mcp.servers" in printed or '"servers"' in printed
    assert sandbox_files(sandbox.root) == set(), "render must not install anything"


def test_cli_uninstall_round_trips(sandbox):
    assert run_cli("plugin", "install", "codex") == 0
    assert run_cli("plugin", "uninstall", "codex") == 0
    assert sandbox_files(sandbox.root) == set()


def test_cli_list_reports_this_machines_agents(sandbox, capsys):
    assert run_cli("plugin", "list") == 0
    printed = capsys.readouterr().out
    for key in catalog.target_keys():
        assert key in printed
