"""Published plugins install through native Agent CLIs, without GPU-side Python."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trainpilot.agent.plugin import release

ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "scripts" / "build_plugins.py"


def tree(root: Path):
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


def test_checked_in_plugin_matches_the_generator():
    for relative, content in release.published_files().items():
        path = ROOT / relative
        assert path.is_file(), f"{relative} is missing; run scripts/build_plugins.py"
        assert path.read_text(encoding="utf-8") == content, (
            f"{relative} is stale; regenerate with scripts/build_plugins.py"
        )


def test_the_repository_publishes_plugin_assets_and_nothing_else():
    published = tree(ROOT / "plugins")
    assert published == {
        Path(relative).relative_to("plugins") for relative in release.published_files()
        if relative.startswith("plugins/")
    }
    assert (ROOT / ".claude-plugin" / "marketplace.json").is_file()
    assert (ROOT / ".agents" / "plugins" / "marketplace.json").is_file()


def test_published_plugin_carries_no_credentials_and_a_literal_default_address():
    files = release.published_files()
    entry = json.loads(files["plugins/trainpilot/mcp_config.json"])["mcpServers"]["trainpilot"]
    # Antigravity does not expand variables in serverUrl.
    assert entry["serverUrl"] == "http://127.0.0.1:28780/mcp"
    assert "headers" not in entry
    for content in files.values():
        assert "Bearer" not in content
        assert "${" not in content


@pytest.mark.parametrize("agent", ["claude-code", "codex"])
def test_published_native_marketplace_points_to_its_own_plugin(agent):
    files = release.published_files()
    directory = f"plugins/{agent}"
    if agent == "claude-code":
        index_path = ".claude-plugin/marketplace.json"
        manifest_path = f"{directory}/.claude-plugin/plugin.json"
        index = json.loads(files[index_path])
        assert index["plugins"][0]["source"] == f"./{directory}"
    else:
        index_path = ".agents/plugins/marketplace.json"
        manifest_path = f"{directory}/.codex-plugin/plugin.json"
        index = json.loads(files[index_path])
        assert index["plugins"][0]["source"] == {"source": "local", "path": f"./{directory}"}
    assert index["name"] == "trainpilot"
    assert json.loads(files[manifest_path])["name"] == "trainpilot"
    entry = json.loads(files[f"{directory}/.mcp.json"])["mcpServers"]["trainpilot"]
    assert entry == {"type": "http", "url": "http://127.0.0.1:28780/mcp"}


def test_published_manifest_is_the_vendor_shape():
    manifest = json.loads(release.published_files()["plugins/trainpilot/plugin.json"])
    assert manifest["name"] == "trainpilot"
    assert set(manifest) == {"$schema", "name", "description"}


def test_published_skill_uses_the_native_connection_not_a_cli():
    skill = release.published_files()["plugins/trainpilot/skills/trainpilot/SKILL.md"]
    assert "MCP initialization" in skill
    assert "trainpilot-cli" not in skill


def test_no_standalone_installer_is_shipped():
    assert not (ROOT / "scripts" / "install_agent.py").exists()
    plugin = ROOT / "src" / "trainpilot" / "agent" / "plugin"
    assert not (plugin / "commands.py").exists()
    assert not (plugin / "install.py").exists()
    for name in ("README.md", "README_zh.md", "AGENTS.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "install_agent.py" not in text
        assert "--agent" not in text


def test_plugin_generator_imports_without_site_packages():
    probe = (
        "import sys; sys.path.insert(0, %r);"
        "import trainpilot.agent.plugin.release;"
        "print(sorted(m for m in ('mcp', 'httpx', 'pydantic', 'fastapi') if m in sys.modules))"
        % str(ROOT / "src")
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"generator pulled in {result.stdout.strip()}"


def test_the_generator_writes_only_the_published_files(tmp_path):
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(GENERATOR)],
        cwd=work, env={**os.environ, "HOME": str(home)},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert set(result.stdout.split()) == set(release.published_files())
    assert tree(work) == set(), "the generator must write nothing outside the repository"
    assert tree(home) == set(), "the generator must not configure an agent"
