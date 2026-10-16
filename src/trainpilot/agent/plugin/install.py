"""Write a rendered bundle to disk, carefully.

These files belong to the user -- ``config.toml``, ``opencode.json``, ``AGENTS.md`` -- so the
rules here are stricter than for anything TrainPilot owns outright:

* a file TrainPilot created is replaced wholesale; a file it shares is *merged*, and a JSON
  file that will not parse is refused rather than overwritten;
* text formats are edited between markers, so a re-install replaces only its own block;
* every modification leaves a ``.trainpilot.bak`` of the previous bytes, and the merge is
  idempotent, so a second install changes nothing and therefore adds no second backup.
"""

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from trainpilot.agent.plugin import catalog
from trainpilot.agent.plugin.render import JSON, MARKDOWN, TOML, Bundle, File, Merge

BACKUP_SUFFIX = ".trainpilot.bak"

# The pair of lines a merged text file is edited between. Chosen to be a comment in both
# TOML and Markdown, so the block is inert in either.
MARKERS = {
    TOML: ("# >>> trainpilot >>>", "# <<< trainpilot <<<"),
    MARKDOWN: ("<!-- >>> trainpilot >>> -->", "<!-- <<< trainpilot <<< -->"),
}

CREATE = "create"
UPDATE = "update"
UNCHANGED = "unchanged"
DELETE = "delete"


@dataclass(frozen=True)
class Change:
    """One file's intended final state.

    Attributes:
        path: Destination, ``~``-anchored or relative to the working directory.
        action: :data:`CREATE`, :data:`UPDATE`, :data:`UNCHANGED` or :data:`DELETE`.
        content: The bytes the file should end up with. Ignored for :data:`DELETE`.
    """

    path: str
    action: str
    content: str = ""


class ConfigError(RuntimeError):
    """A shared config file could not be edited safely. Never overwrite it instead."""


def expand(path: str) -> Path:
    """``~``-expand a bundle path. Relative paths stay relative to the working directory."""
    return Path(os.path.expanduser(path))


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _classify(path: Path, content: str) -> str:
    existing = _read(path)
    if existing is None:
        return CREATE
    return UNCHANGED if existing == content else UPDATE


# --------------------------------------------------------------------------- merges


def merge_json(path: Path, payload: Any) -> str:
    """Deep-merge ``payload`` into the JSON document at ``path`` and return the result."""
    raw = _read(path)
    if raw is None or not raw.strip():
        document: Any = {}
    else:
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"{path} is not valid JSON ({exc.msg} at line {exc.lineno}), so TrainPilot "
                "will not touch it. Add the entry by hand, or move the comments out of the "
                "file first: this installer only edits strict JSON."
            ) from exc
        if not isinstance(document, dict):
            raise ConfigError(f"{path} does not hold a JSON object at the top level.")

    _deep_merge(document, payload)
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def _deep_merge(target: Dict[str, Any], patch: Dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def unmerge_json(path: Path, keys: Sequence[str]) -> Optional[str]:
    """Remove the nested key path from a JSON file; return the new text, or ``None``.

    ``None`` means there was nothing to remove or the file is not one we can edit, and the
    caller should leave it alone.
    """
    raw = _read(path)
    if raw is None:
        return None
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(document, dict):
        return None

    cursor: Any = document
    parents: List[Tuple[Dict[str, Any], str]] = []
    for key in keys[:-1]:
        if not isinstance(cursor, dict) or not isinstance(cursor.get(key), dict):
            return None
        parents.append((cursor, key))
        cursor = cursor[key]
    if not isinstance(cursor, dict) or keys[-1] not in cursor:
        return None
    del cursor[keys[-1]]

    # Drop containers this plugin emptied, so uninstalling a fresh config leaves a fresh one.
    for parent, key in reversed(parents):
        if not parent[key]:
            del parent[key]
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def merge_block(path: Path, fmt: str, body: str) -> str:
    """Replace (or append) the trainpilot block in a TOML or Markdown file."""
    begin, end = MARKERS[fmt]
    block = f"{begin}\n{body.rstrip()}\n{end}"
    existing = _read(path)
    if existing is None or not existing.strip():
        return block + "\n"

    start = existing.find(begin)
    if start != -1:
        stop = existing.find(end, start)
        if stop != -1:
            stop += len(end)
            # Swallow the newline that ends the block, but not a blank line separating it
            # from unrelated text below.
            if existing[stop:stop + 1] == "\n":
                stop += 1
            return existing[:start] + block + "\n" + existing[stop:].lstrip("\n")

    separator = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
    return existing + separator + block + "\n"


def unmerge_block(path: Path, fmt: str) -> Optional[str]:
    """Remove the trainpilot block; return the new text, or ``None`` if there was none."""
    begin, end = MARKERS[fmt]
    existing = _read(path)
    if existing is None:
        return None
    start = existing.find(begin)
    if start == -1:
        return None
    stop = existing.find(end, start)
    if stop == -1:
        return None
    stop += len(end)
    if existing[stop:stop + 1] == "\n":
        stop += 1
    remainder = (existing[:start] + existing[stop:]).strip("\n")
    return remainder + "\n" if remainder else ""


# --------------------------------------------------------------------------- planning


def plan(bundle: Bundle, staging_dir: Optional[Path]) -> List[Change]:
    """Work out every file's final content without writing anything.

    ``staging_dir`` is required for :data:`catalog.BUNDLE` targets, whose paths are relative
    to the directory handed to the vendor CLI, and ignored for :data:`catalog.IN_PLACE`
    targets, whose paths are already real destinations.
    """
    changes: List[Change] = []
    for item in bundle.files:
        path = _destination(item, bundle.kind, staging_dir)
        changes.append(Change(str(path), _classify(path, item.content), item.content))
    for item in bundle.merges:
        path = expand(item.path)
        content = _merged_content(path, item)
        changes.append(Change(str(path), _classify(path, content), content))
    return changes


def plan_revert(bundle: Bundle) -> List[Change]:
    """Work out how to take the plugin back out.

    Merged files are edited surgically -- the trainpilot JSON key or marker block is removed
    and everything around it is left alone -- because the user may have written into them
    since. A file the plugin created and left empty afterwards is deleted rather than left
    behind as a blank ``AGENTS.md``.
    """
    changes: List[Change] = []
    for item in bundle.files:
        path = expand(item.path)
        if path.exists():
            changes.append(Change(str(path), DELETE))
    for item in bundle.merges:
        path = expand(item.path)
        if item.fmt == JSON:
            content = unmerge_json(path, item.remove_keys)
        else:
            content = unmerge_block(path, item.fmt)
        if content is None:
            continue
        if content.strip():
            changes.append(Change(str(path), UPDATE, content))
        else:
            changes.append(Change(str(path), DELETE))
    return changes


def _destination(item: File, kind: str, staging_dir: Optional[Path]) -> Path:
    if kind == catalog.BUNDLE:
        if staging_dir is None:
            raise ValueError("a bundle-kind target needs a staging directory")
        return staging_dir / item.path
    return expand(item.path)


def _merged_content(path: Path, item: Merge) -> str:
    if item.fmt == JSON:
        return merge_json(path, item.payload)
    return merge_block(path, item.fmt, str(item.payload))


# --------------------------------------------------------------------------- applying


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

        if change.action == DELETE:
            path.unlink(missing_ok=True)
            _prune_empty_parents(path)
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        if change.action == UPDATE:
            _backup(path)
        path.write_text(change.content, encoding="utf-8")
    return log


def _backup(path: Path) -> None:
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + BACKUP_SUFFIX))


def _prune_empty_parents(path: Path) -> None:
    """Remove directories left empty by a delete, stopping at the user's home or the cwd."""
    protected = {Path(os.path.expanduser("~")).resolve(), Path.cwd().resolve()}
    parent = path.parent
    while parent not in protected and parent.is_dir():
        try:
            parent.rmdir()  # only succeeds while empty
        except OSError:
            return
        parent = parent.parent


def display_path(path: Path) -> str:
    home = os.path.expanduser("~")
    text = str(path)
    return "~" + text[len(home):] if text.startswith(home) else text


def expand_argv(template: Sequence[str], *, bundle: Path, scope: str, mcp_url: str) -> List[str]:
    """Fill in an argv template's ``{bundle}``, ``{scope}`` and ``{mcp_url}`` placeholders."""
    return [
        part.replace("{bundle}", str(bundle)).replace("{scope}", scope).replace("{mcp_url}", mcp_url)
        for part in template
    ]
