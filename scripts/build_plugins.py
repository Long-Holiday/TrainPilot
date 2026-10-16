#!/usr/bin/env python3
"""Regenerate checked-in native plugin assets without third-party dependencies."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trainpilot.agent.plugin.release import marketplace_files


if __name__ == "__main__":
    for relative, content in marketplace_files().items():
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(relative)
