"""The GPU integration installs from the repository without installing Python packages.

Two things are pinned here:

* the checked-in Claude marketplace assets are exactly what the canonical renderer
  produces, and carry no real credentials;
* ``scripts/install_agent.py`` works with the standard library only (``python -I -S``),
  extracts only its allowlisted sources from a downloaded archive, and still writes
  nothing outside the throwaway ``$HOME`` it is given.
"""

import ast
import importlib.util
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from trainpilot.agent.plugin import catalog, release

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "install_agent.py"
GENERATOR = ROOT / "scripts" / "build_plugins.py"

STDLIB_IMPORTS = {
    "argparse",
    "importlib",
    "io",
    "json",
    "os",
    "pathlib",
    "sys",
    "tempfile",
    "urllib",
    "zipfile",
}


def load_installer():
    """Import the installer script itself, so its bootstrap can be unit tested."""
    spec = importlib.util.spec_from_file_location("trainpilot_install_agent", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def throwaway_home(tmp_path, monkeypatch):
    """A temp HOME and cwd with no agent CLI reachable, so nothing real is touched."""
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    assert Path(os.path.expanduser("~")).resolve() == home.resolve()
    return home, work


def run_installer(home: Path, work: Path, *argv: str):
    """Run the installer as a user would: isolated interpreter, throwaway HOME."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("TRAINPILOT_")
    }
    env["HOME"] = str(home)
    env["PATH"] = str(home / "empty-bin")
    return subprocess.run(
        [sys.executable, "-I", "-S", str(SCRIPT), *argv],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def tree(root: Path):
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


# --------------------------------------------------------------------------- release


def test_checked_in_plugin_matches_the_generator():
    """A hand-edited copy of the prose is what drifts; the generator is the source."""
    for relative, content in release.marketplace_files().items():
        path = ROOT / relative
        assert path.is_file(), f"{relative} is missing; run scripts/build_plugins.py"
        assert path.read_text(encoding="utf-8") == content, (
            f"{relative} is stale; regenerate with scripts/build_plugins.py"
        )


def test_marketplace_points_at_a_directory_of_plugin_assets_only():
    """Pointing at the repository root would hand the whole Python project to the agent."""
    marketplace = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    assert marketplace["plugins"][0]["source"] == "./plugins/claude-code"
    assert not (ROOT / "plugins" / "claude-code" / "src").exists()


def test_published_plugin_carries_no_credentials_and_no_default_address():
    files = release.marketplace_files()
    entry = json.loads(files["plugins/claude-code/.mcp.json"])["mcpServers"]["trainpilot"]
    assert entry["type"] == "http"
    assert entry["url"] == "${TRAINPILOT_GATEWAY_URL}/mcp"
    # An unset token must not produce a malformed `Bearer ` header.
    assert entry["headers"] == {"X-API-Token": "${TRAINPILOT_API_TOKEN:-}"}
    for content in files.values():
        assert "Bearer" not in content


def test_published_skill_uses_the_native_connection_not_a_cli():
    skill = release.marketplace_files()["plugins/claude-code/skills/trainpilot/SKILL.md"]
    assert "MCP initialization" in skill
    assert "trainpilot-cli" not in skill


# --------------------------------------------------------------------------- the script


def test_installer_imports_only_the_standard_library():
    """Anything else would defeat the point: the GPU node installs no packages.

    ``trainpilot`` is allowed because it is the allowlisted source tree the script
    loads itself, not a package the user has to install.
    """
    tree_ = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree_):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    allowed = STDLIB_IMPORTS | {"trainpilot"}
    assert imported <= allowed, f"non-stdlib imports: {imported - allowed}"


def test_plugin_imports_without_site_packages():
    """`-S` drops site-packages, so any third-party import would fail loudly here."""
    probe = (
        "import sys; sys.path.insert(0, %r);"
        "import trainpilot.agent.plugin.commands as c;"
        "import trainpilot.agent.plugin.release;"
        "print(sorted(m for m in ('mcp', 'httpx', 'pydantic', 'fastapi') if m in sys.modules))"
        % str(ROOT / "src")
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"installation pulled in {result.stdout.strip()}"


def test_installer_lists_every_supported_agent(throwaway_home):
    home, work = throwaway_home
    result = run_installer(home, work, "list")
    assert result.returncode == 0, result.stderr
    for key in catalog.target_keys():
        assert key in result.stdout


def test_installer_install_and_uninstall_stay_inside_the_throwaway_home(throwaway_home):
    home, work = throwaway_home
    install = run_installer(home, work, "install", "codex", "--server-url", "http://gw.example:1")
    assert install.returncode == 0, install.stderr
    assert (home / ".agents" / "skills" / "trainpilot" / "SKILL.md").is_file()
    assert (home / ".codex" / "config.toml").is_file()

    uninstall = run_installer(home, work, "uninstall", "codex")
    assert uninstall.returncode == 0, uninstall.stderr
    assert tree(home) == set(), "uninstall must give the documents back"
    assert tree(work) == set(), "nothing may be written outside $HOME"


def test_installer_reports_a_missing_agent_cli_instead_of_claiming_success(throwaway_home):
    """No `agy` on PATH: the bundle is staged, and the failure is reported as a failure."""
    home, work = throwaway_home
    result = run_installer(home, work, "install", "antigravity", "--server-url", "http://gw.example:1")
    assert result.returncode == 1
    assert "not on PATH" in result.stdout + result.stderr
    assert (home / ".trainpilot" / "plugins" / "antigravity" / "mcp_config.json").is_file()


@pytest.mark.parametrize(
    ("target", "agent_cli", "expected"),
    [
        ("claude-code", "claude", "plugin install trainpilot --marketplace"),
        ("antigravity", "agy", "plugin install"),
        ("gemini-cli", "gemini", "extensions link"),
    ],
)
def test_installer_registers_a_bundle_through_the_agents_own_cli(
    throwaway_home, target, agent_cli, expected
):
    """Registration stays with the vendor CLI; the installer only supplies the argv."""
    home, work = throwaway_home
    bin_dir = home / "bin"
    bin_dir.mkdir()
    stub = bin_dir / agent_cli
    stub.write_text(f'#!/bin/sh\necho "{agent_cli} $@" >> "$HOME/calls.txt"\n')
    stub.chmod(0o755)

    env = {k: v for k, v in os.environ.items() if not k.startswith("TRAINPILOT_")}
    env["HOME"] = str(home)
    env["PATH"] = str(bin_dir)
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(SCRIPT), "install", target, "--server-url", "http://gw.example:1"],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = (home / "calls.txt").read_text()
    assert calls.startswith(agent_cli)
    assert expected in calls
    assert str(home / ".trainpilot" / "plugins" / target) in calls


def test_generic_install_prints_a_pasteable_entry_without_a_real_url_leak(throwaway_home):
    home, work = throwaway_home
    result = run_installer(home, work, "install", "generic", "--server-url", "http://gw.example:1")
    assert result.returncode == 0, result.stderr
    assert '"mcpServers"' in result.stdout
    assert "http://gw.example:1/mcp" in result.stdout


# --------------------------------------------------------------------------- bootstrap


class _Response:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self, size: int = -1) -> bytes:
        return self._payload[:size] if size and size > 0 else self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def archive(entries) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, payload in entries:
            bundle.writestr(name, payload)
    return buffer.getvalue()


def fake_urlopen(monkeypatch, module, payload: bytes):
    class _Request:
        @staticmethod
        def urlopen(url, timeout=None):
            return _Response(payload)

    monkeypatch.setattr(module.urllib, "request", _Request)


def test_bootstrap_extracts_only_allowlisted_sources(throwaway_home, monkeypatch):
    home, _ = throwaway_home
    module = load_installer()
    entries = [(f"repo-abc/{source}", "x = 1\n") for source in module.SOURCES]
    entries += [("repo-abc/setup.py", "raise SystemExit\n"), ("repo-abc/.github/workflows/ci.yml", "")]
    fake_urlopen(monkeypatch, module, archive(entries))

    source = module.fetch_sources(home, "main")

    assert (source / "trainpilot" / "agent" / "plugin" / "render.py").is_file()
    assert not (home / "setup.py").exists(), "an archive file outside the allowlist was written"
    assert not (home / ".github").exists()


def test_bootstrap_rejects_an_archive_with_a_missing_source(throwaway_home, monkeypatch):
    home, _ = throwaway_home
    module = load_installer()
    fake_urlopen(monkeypatch, module, archive([("repo-abc/src/trainpilot/__init__.py", "x = 1\n")]))

    with pytest.raises(ValueError) as excinfo:
        module.fetch_sources(home, "main")

    assert "Incomplete" in str(excinfo.value)


def test_bootstrap_rejects_a_symlinked_source(throwaway_home, monkeypatch):
    """A symlink in the archive is how an allowlist gets bypassed."""
    home, _ = throwaway_home
    module = load_installer()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for source in module.SOURCES:
            info = zipfile.ZipInfo(f"repo-abc/{source}")
            info.external_attr = (0o120777 << 16) if source.endswith("render.py") else 0
            bundle.writestr(info, "x = 1\n")
    fake_urlopen(monkeypatch, module, buffer.getvalue())

    with pytest.raises(ValueError) as excinfo:
        module.fetch_sources(home, "main")

    assert "symbolic link" in str(excinfo.value)


def test_bootstrap_rejects_an_oversized_source(throwaway_home, monkeypatch):
    home, _ = throwaway_home
    module = load_installer()
    entries = [
        (f"repo-abc/{source}", "x" * (module.MAX_SOURCE + 1) if source.endswith("render.py") else "x = 1\n")
        for source in module.SOURCES
    ]
    fake_urlopen(monkeypatch, module, archive(entries))

    with pytest.raises(ValueError) as excinfo:
        module.fetch_sources(home, "main")

    assert "size limit" in str(excinfo.value)


def test_bootstrap_refuses_a_duplicated_source_path(throwaway_home, monkeypatch):
    home, _ = throwaway_home
    module = load_installer()
    entries = [(f"repo-abc/{source}", "x = 1\n") for source in module.SOURCES]
    entries.append(("repo-abc/src/trainpilot/agent/plugin/render.py", "x = 2\n"))
    fake_urlopen(monkeypatch, module, archive(entries))

    with pytest.raises(ValueError) as excinfo:
        module.fetch_sources(home, "main")

    assert "Duplicate" in str(excinfo.value)


def test_bootstrap_uses_a_ref_specific_archive_url(throwaway_home, monkeypatch):
    home, _ = throwaway_home
    module = load_installer()
    seen = {}

    class _Request:
        @staticmethod
        def urlopen(url, timeout=None):
            seen["url"] = url
            raise OSError("stop here")

    monkeypatch.setattr(module.urllib, "request", _Request)
    with pytest.raises(OSError):
        module.fetch_sources(home, "v0.2.0")
    assert seen["url"].endswith("/zip/v0.2.0")
