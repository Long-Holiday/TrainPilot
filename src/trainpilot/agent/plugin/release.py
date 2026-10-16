"""The checked-in Antigravity plugin, derived from the canonical renderer.

``agy plugin install <repository-url>`` installs the plugin from this published copy, so it
has to be installable as-is: a literal default gateway address and no credentials. Anyone
pointing at a gateway somewhere else edits ``serverUrl`` in the installed
``mcp_config.json``, or runs the one-shot installer, which renders the real address and
token into the bundle it stages.
"""

from typing import Dict

from trainpilot.agent.plugin.render import PluginContext, render_bundle
from trainpilot.common.gateway import DEFAULT_GATEWAY_URL


def published_files() -> Dict[str, str]:
    """Repository-relative path to content, for every published plugin asset."""
    bundle = render_bundle(
        PluginContext(gateway_url=DEFAULT_GATEWAY_URL, api_token=None, task_id="")
    )
    return {f"plugins/trainpilot/{item.path}": item.content for item in bundle.files}
