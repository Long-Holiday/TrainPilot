"""Run the TrainPilot Control Plane Gateway."""

import os
import sys

# Ensure src is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from trainpilot.server.config import settings
from trainpilot.server.main import start

if __name__ == "__main__":
    print(f"Starting TrainPilot Gateway on http://{settings.effective_bind_host}:{settings.port}")
    print(f"Public gateway URL: {settings.advertised_gateway_url}")
    print(f"Feishu configured: {settings.is_feishu_configured}")
    print(f"Docs available at: http://{settings.public_host}:{settings.port}/docs")
    start()
