"""FastAPI application entrypoint for TrainPilot Control Plane."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from trainpilot import __version__
from trainpilot.server.config import settings
from trainpilot.server.mailbox import default_mailbox
from trainpilot.server.routes.tasks import router as tasks_router
from trainpilot.server.routes.webhook import router as webhook_router

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trainpilot.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    logger.info("Starting TrainPilot Control Plane Gateway...")
    logger.info("Feishu configured: %s (Receiver: %s)", settings.is_feishu_configured, settings.feishu_receiver_id)
    yield
    logger.info("TrainPilot Control Plane Gateway stopped.")


app = FastAPI(
    title="TrainPilot Control Plane",
    description="Centralized HITL Control Gateway for GPU Distributed Training with Feishu Card Integration",
    version=__version__,
    lifespan=lifespan,
)

# Cross-Origin Resource Sharing
# NOTE: single-worker in-memory mailbox; do not run with --workers>1 (see README).
# '*' cannot be combined with credentials per Fetch spec, enforced in config.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parsed_cors_origins,
    allow_credentials=settings.effective_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(tasks_router)
app.include_router(webhook_router)


@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe (never fails on downstream Feishu errors)."""
    try:
        stale = default_mailbox.get_stale_tasks(timeout_seconds=settings.task_heartbeat_timeout_seconds)
        stale_count = len(stale)
        tasks_count = len(default_mailbox.list_tasks(limit=100000, offset=0))
    except Exception:
        stale_count = 0
        tasks_count = -1
    return {
        "status": "healthy",
        "service": "trainpilot-control-plane",
        "version": __version__,
        "feishu_ready": settings.is_feishu_configured,
        "auth_enforced": settings.is_api_token_configured,
        "tasks_count": tasks_count,
        "stale_tasks_count": stale_count,
    }


@app.get("/ready", tags=["System"])
def readiness_check():
    """Readiness probe: mailbox accessible."""
    try:
        default_mailbox.list_tasks(limit=1, offset=0)
        return {"ready": True, "service": "trainpilot-control-plane", "version": __version__}
    except Exception as exc:
        logger.error("Readiness check failed: %s", exc)
        return {"ready": False, "error": str(exc)}


@app.get("/", tags=["System"])
def root():
    """Root info endpoint."""
    return {
        "message": "Welcome to TrainPilot Control Plane Gateway",
        "docs_url": "/docs",
        "health_url": "/health",
    }


def start():
    """Helper entrypoint to start uvicorn programmatically."""
    import uvicorn

    uvicorn.run(
        "trainpilot.server.main:app",
        host=settings.effective_bind_host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    start()
