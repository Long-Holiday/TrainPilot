"""Write a rendered bundle to disk, carefully.

The bundle is TrainPilot's own: it is staged under ``~/.trainpilot/plugins/`` and then
handed to the selected agent's native plugin manager. Nothing here edits the agent's
configuration directly, and the rules are simple:

* the final content of every file is worked out before anything is written (``plan``), so a
  ``--dry-run`` can report the same thing a real run does;
* a file whose bytes already match is left alone rather than rewritten, so a re-install is
  a no-op.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from trainpilot.agent.plugin.render import Bundle

CREATE = "create"
UPDATE = "update"
UNCHANGED = "unchanged"


@dataclass(frozen=True)
class Change:
    """One file's intended final state.

    Attributes:
        path: Destination, ``~``-anchored or absolute.
        action: :data:`CREATE`, :data:`UPDATE` or :data:`UNCHANGED`.
        content: The bytes the file should end up with.
    """

    path: str
    action: str
    content: str = ""


def expand(path: str) -> Path:
    """``~``-expand a bundle path. Relative paths stay relative to the working directory."""
    return Path(os.path.expanduser(path))


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _classify(path: Path, content: str) -> str:
    existing = _read(path)
    if existing is None:
        return CREATE
    return UNCHANGED if existing == content else UPDATE


def plan(bundle: Bundle, staging_dir: Path) -> List[Change]:
    """Work out every file's final content without writing anything."""
    changes: List[Change] = []
    for item in bundle.files:
        path = staging_dir / item.path
        changes.append(Change(str(path), _classify(path, item.content), item.content))
    return changes


def apply(changes: Iterable[Change], dry_run: bool = False) -> List[str]:
    """Write the planned changes. Returns one human-readable line per file acted on."""
    log: List[str] = []
    for change in changes:
        path = Path(change.path)
        if change.action == UNCHANGED:
            log.append(f"unchanged  {display_path(path)}")
            continue
        log.append(f"{change.action:<9}  {display_path(path)}")
        if dry_run:
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(change.content, encoding="utf-8")
    return log


def display_path(path: Path) -> str:
    home = os.path.expanduser("~")
    text = str(path)
    return "~" + text[len(home):] if text.startswith(home) else text


def expand_argv(template: Sequence[str], *, bundle: Path, mcp_url: str) -> List[str]:
    """Fill in an argv template's ``{bundle}`` and ``{mcp_url}`` placeholders."""
    return [
        part.replace("{bundle}", str(bundle)).replace("{mcp_url}", mcp_url)
        for part in template
    ]
