#!/usr/bin/env python3
"""Regenerate all checked-in native plugins and marketplaces; standard library only.

Published assets use the canonical renderer. Edit the renderer or shared templates and
run this script; do not hand-edit the generated files.
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
