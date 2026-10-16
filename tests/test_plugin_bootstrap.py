"""The GPU integration installs from the repository without installing Python packages.

Two things are pinned here:

* the checked-in Antigravity plugin is exactly what the canonical renderer produces, is
  free of credentials, and is the *only* thing the repository publishes under ``plugins/``;
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


# --------------------------------------------------------------------------- published


def test_checked_in_plugin_matches_the_generator():
    """A hand-edited copy of the prose is what drifts; the generator is the source."""
    for relative, content in release.published_files().items():
        path = ROOT / relative
        assert path.is_file(), f"{relative} is missing; run scripts/build_plugins.py"
        assert path.read_text(encoding="utf-8") == content, (
            f"{relative} is stale; regenerate with scripts/build_plugins.py"
        )


def test_the_repository_publishes_plugin_assets_and_nothing_else():
    """Antigravity copies *everything* it finds into its plugin directory, so the published
    tree must hold the plugin and nothing else -- no stale directory from an older shape,
    no Python project."""
    published = tree(ROOT / "plugins")
    assert published == {
        Path(relative).relative_to("plugins") for relative in release.published_files()
    }
    assert not (ROOT / ".claude-plugin").exists(), "a Claude marketplace is no longer published"
    assert not (ROOT / "plugins" / "claude-code").exists()


def test_published_plugin_carries_no_credentials_and_a_literal_default_address():
    files = release.published_files()
    entry = json.loads(files["plugins/trainpilot/mcp_config.json"])["mcpServers"]["trainpilot"]
    # A literal address, not `${TRAINPILOT_GATEWAY_URL}`: Antigravity resolves variables in
    # neither `serverUrl` nor its own stdio sniffing, so a placeholder would silently
    # install a server that cannot connect.
    assert entry["serverUrl"] == "http://127.0.0.1:28780/mcp"
    assert "headers" not in entry
    for content in files.values():
        assert "Bearer" not in content
        assert "${" not in content


def test_published_manifest_is_the_vendor_shape():
    manifest = json.loads(release.published_files()["plugins/trainpilot/plugin.json"])
    assert manifest["name"] == "trainpilot"
    assert set(manifest) == {"$schema", "name", "description"}


def test_published_skill_uses_the_native_connection_not_a_cli():
    skill = release.published_files()["plugins/trainpilot/skills/trainpilot/SKILL.md"]
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


def test_the_generator_writes_only_the_published_files(throwaway_home):
    """`scripts/build_plugins.py` is what a contributor runs; it needs no venv."""
    home, work = throwaway_home
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(GENERATOR)],
        cwd=work,
        env={**os.environ, "HOME": str(home)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert set(result.stdout.split()) == set(release.published_files())
    assert tree(work) == set(), "the generator must write nothing outside the repository"


def test_installer_lists_the_supported_target(throwaway_home):
    home, work = throwaway_home
    result = run_installer(home, work, "list")
    assert result.returncode == 0, result.stderr
    assert catalog.ANTIGRAVITY.display_name in result.stdout
    assert catalog.ANTIGRAVITY.transport in result.stdout


def test_installer_stages_the_bundle_and_reports_a_missing_agy(throwaway_home):
    """No `agy` on PATH: the bundle is staged, and the failure is reported as a failure."""
    home, work = throwaway_home
    result = run_installer(home, work, "install", "--server-url", "http://gw.example:1")
    assert result.returncode == 1
    assert "not on PATH" in result.stdout + result.stderr
    assert (home / ".trainpilot" / "plugins" / "trainpilot" / "mcp_config.json").is_file()
    assert tree(work) == set(), "nothing may be written outside $HOME"


def test_installer_render_writes_only_where_it_is_told(throwaway_home):
    home, work = throwaway_home
    out = work / "bundle"
    result = run_installer(home, work, "render", "--out", str(out))
    assert result.returncode == 0, result.stderr
    assert tree(out) == {
        Path("plugin.json"),
        Path("mcp_config.json"),
        Path("rules/trainpilot.md"),
        Path("skills/trainpilot/SKILL.md"),
    }
    assert tree(home) == set(), "render must not stage or install anything"


def test_installer_uninstall_gives_the_tree_back(throwaway_home):
    home, work = throwaway_home
    assert run_installer(home, work, "install", "--server-url", "http://gw.example:1").returncode == 1
    assert home.joinpath(".trainpilot").exists()

    uninstall = run_installer(home, work, "uninstall")
    assert uninstall.returncode == 0, uninstall.stderr
    assert tree(home) == set(), "uninstall must give the tree back"
    assert tree(work) == set(), "nothing may be written outside $HOME"


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
