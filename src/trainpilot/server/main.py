"""FastAPI application entrypoint for TrainPilot Control Plane."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from trainpilot.server.config import settings
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
    version="0.1.0",
    lifespan=lifespan,
)

# Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(tasks_router)
app.include_router(webhook_router)


@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe."""
    return {
        "status": "healthy",
        "service": "trainpilot-control-plane",
        "feishu_ready": settings.is_feishu_configured,
    }


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
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    start()
