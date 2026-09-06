"""Run the TrainPilot Control Plane Gateway."""

import os
import sys

# Ensure src is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from trainpilot.server.config import settings
from trainpilot.server.main import start

if __name__ == "__main__":
    print(f"Starting TrainPilot MCP Gateway on http://{settings.effective_bind_host}:{settings.port}")
    print(f"MCP Streamable HTTP endpoint: http://{settings.public_host}:{settings.port}/mcp")
    print(f"Feishu configured: {settings.is_feishu_configured}")
    print(f"Docs available at: http://{settings.public_host}:{settings.port}/docs")
    start()
