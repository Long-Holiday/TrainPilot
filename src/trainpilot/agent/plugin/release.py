"""Publishable Claude marketplace assets, derived from the canonical renderer."""

import json
from typing import Dict

from trainpilot.agent.plugin.render import PluginContext, render_bundle


def marketplace_files() -> Dict[str, str]:
    """No credentials or server address are baked into the GitHub-distributed plugin."""
    bundle = render_bundle(
        "claude-code",
        PluginContext(gateway_url="${TRAINPILOT_GATEWAY_URL}", api_token=None, task_id=""),
    )
    files = {}
    for item in bundle.files:
        if item.path == ".claude-plugin/marketplace.json":
            marketplace = json.loads(item.content)
            marketplace["plugins"][0]["source"] = "./plugins/claude-code"
            files[item.path] = json.dumps(marketplace, indent=2, ensure_ascii=False) + "\n"
        else:
            content = item.content
            if item.path == ".mcp.json":
                mcp = json.loads(content)
                # An unset token is an empty X-API-Token, not a malformed 'Bearer '.
                mcp["mcpServers"]["trainpilot"]["headers"] = {
                    "X-API-Token": "${TRAINPILOT_API_TOKEN:-}",
                }
                content = json.dumps(mcp, indent=2, ensure_ascii=False) + "\n"
            files[f"plugins/claude-code/{item.path}"] = content
    return files
