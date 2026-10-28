"""Credential-free published plugins, derived from the canonical renderer.

Antigravity keeps its existing assets-only directory. Claude Code and Codex have separate
native plugin directories and repository marketplace indexes for their own CLIs.
"""

from typing import Dict

from trainpilot.agent.plugin import catalog
from trainpilot.agent.plugin.render import PluginContext, marketplace_file, render_bundle
from trainpilot.common.gateway import DEFAULT_GATEWAY_URL


def published_files() -> Dict[str, str]:
    """Repository-relative path to content, for every generated asset."""
    ctx = PluginContext(gateway_url=DEFAULT_GATEWAY_URL, api_token=None)
    files = {}
    for target in catalog.TARGETS.values():
        directory = "plugins/trainpilot" if target == catalog.ANTIGRAVITY else f"plugins/{target.key}"
        bundle = render_bundle(ctx, target)
        for item in bundle.files:
            files[f"{directory}/{item.path}"] = item.content
        if target != catalog.ANTIGRAVITY:
            index = marketplace_file(target, "trainpilot", f"./{directory}")
            files[index.path] = index.content
    return files
