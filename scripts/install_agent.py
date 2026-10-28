#!/usr/bin/env python3
"""One-shot GPU integration installer. Python >= 3.10, standard library only.

Run from a checkout or download this file on its own. No pip/uv installation, server
process, or Python runtime is needed by the installed skill and remote MCP config.
"""

import argparse
import io
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

REPOSITORY = "Long-Holiday/TrainPilot"
MAX_DOWNLOAD = 16 * 1024 * 1024
MAX_SOURCE = 512 * 1024
SOURCES = (
    "src/trainpilot/__init__.py",
    "src/trainpilot/common/__init__.py",
    "src/trainpilot/common/states.py",
    "src/trainpilot/common/gateway.py",
    "src/trainpilot/agent/__init__.py",
    "src/trainpilot/agent/plugin/__init__.py",
    "src/trainpilot/agent/plugin/catalog.py",
    "src/trainpilot/agent/plugin/templates.py",
    "src/trainpilot/agent/plugin/render.py",
    "src/trainpilot/agent/plugin/install.py",
    "src/trainpilot/agent/plugin/commands.py",
)


def fetch_sources(destination: Path, ref: str) -> Path:
    """Copy only the installer allowlist, never execute or extract other archive files."""
    url = f"https://codeload.github.com/{REPOSITORY}/zip/{urllib.parse.quote(ref, safe='')}"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(MAX_DOWNLOAD + 1)
    if len(data) > MAX_DOWNLOAD:
        raise ValueError("Repository download exceeds the size limit")

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        selected = {}
        roots = set()
        for info in archive.infolist():
            # GitHub wraps all entries in one repository directory. Match exact
            # allowlisted suffixes; no extractall, symlinks, or archive-owned paths.
            root, separator, relative = info.filename.partition("/")
            if not separator or relative not in SOURCES:
                continue
            if root in ("", ".", "..") or "\\" in root:
                raise ValueError("Invalid repository archive path")
            if relative in selected:
                raise ValueError(f"Duplicate source in archive: {relative}")
            if info.file_size > MAX_SOURCE:
                raise ValueError(f"Source exceeds the size limit: {relative}")
            mode = info.external_attr >> 16
            if mode & 0o170000 == 0o120000:
                raise ValueError(f"Source is a symbolic link: {relative}")
            roots.add(root)
            selected[relative] = info
        missing = set(SOURCES) - selected.keys()
        if missing or len(roots) != 1:
            raise ValueError(f"Incomplete repository archive; missing sources: {sorted(missing)}")
        for relative, info in selected.items():
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(info))
    return destination / "src"


def run(source: Path, argv) -> int:
    # -I excludes cwd and the downloaded script directory. Only this selected
    # source tree is added, and the caller's cwd remains the install destination.
    sys.path.insert(0, str(source))
    from trainpilot.agent.plugin import commands

    parser = argparse.ArgumentParser(
        description="Install an agent-native skill and remote MCP config only; no Python packages.",
    )
    parser.add_argument("--server-url", "-s")
    parser.add_argument("--api-token")
    parser.add_argument("--task-id", "-t")
    sub = parser.add_subparsers(dest="group", required=True)
    commands.add_parser(sub, prog="install_agent.py")
    args = parser.parse_args(["plugin", *argv])
    try:
        return args.func(args)
    except (OSError, ValueError) as exc:
        print(f"Installation failed: {exc}", file=sys.stderr)
        return 1


def main(argv=None) -> int:
    if sys.version_info < (3, 10):
        print("Python >= 3.10 is needed for this one-shot installer", file=sys.stderr)
        return 1
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument(
        "--ref",
        default="main",
        help="Repository branch, tag or commit to download (ignored in a checkout)",
    )
    options, remaining = bootstrap.parse_known_args(argv)
    checkout = Path(__file__).resolve().parent.parent
    if all((checkout / source).is_file() for source in SOURCES):
        return run(checkout / "src", remaining)
    try:
        with tempfile.TemporaryDirectory(prefix="trainpilot-install-") as directory:
            source = fetch_sources(Path(directory), options.ref)
            return run(source, remaining)
    except (OSError, urllib.error.URLError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Could not load installer sources: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
