"""TrainPilot server API routes."""

from trainpilot.server.routes.tasks import router as tasks_router
from trainpilot.server.routes.webhook import router as webhook_router

__all__ = ["tasks_router", "webhook_router"]
