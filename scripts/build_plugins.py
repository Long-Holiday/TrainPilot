#!/usr/bin/env python3
"""Regenerate the checked-in Antigravity plugin; standard library only.

The plugin `agy plugin install <repository-url>` picks up is generated from the same
renderer as the one-shot installer, so the two cannot drift. Nothing here is hand-edited:
edit the renderer or the templates and run this again.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trainpilot.agent.plugin.release import published_files


if __name__ == "__main__":
    for relative, content in published_files().items():
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(relative)
