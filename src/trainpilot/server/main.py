"""FastAPI application entrypoint for TrainPilot Control Plane and MCP Server."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from trainpilot import __version__
from trainpilot.server.config import settings
from trainpilot.server.mailbox import default_mailbox
from trainpilot.server.mcp_server import mcp_server
from trainpilot.server.routes.tasks import router as tasks_router
from trainpilot.server.routes.webhook import router as webhook_router
from trainpilot.server.watchdog import default_watchdog

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trainpilot.server")


# Initialize MCP Server Streamable HTTP transport (latest MCP standard)
streamable_mcp_app = mcp_server.streamable_http_app(
    streamable_http_path="/mcp",
    transport_security=settings.mcp_transport_security,
)

# Extract ASGI endpoint to support safe session_manager reload across test/process lifespans
_streamable_asgi_handler = None
for _r in streamable_mcp_app.routes:
    if hasattr(_r, "endpoint") and hasattr(_r.endpoint, "session_manager"):
        _streamable_asgi_handler = _r.endpoint
        break


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    logger.info("Starting TrainPilot Control Plane MCP Gateway (Streamable HTTP)...")
    logger.info("Feishu configured: %s (Receiver: %s)", settings.is_feishu_configured, settings.feishu_receiver_id)
    default_watchdog.start()

    # Ensure clean StreamableHTTPSessionManager instance for each lifespan cycle
    sm = getattr(mcp_server._lowlevel_server, "_session_manager", None)
    if sm is not None and getattr(sm, "_has_started", False):
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        sm = StreamableHTTPSessionManager(
            app=mcp_server._lowlevel_server,
            security_settings=settings.mcp_transport_security,
        )
        mcp_server._lowlevel_server._session_manager = sm
        if _streamable_asgi_handler is not None:
            _streamable_asgi_handler.session_manager = sm

    async with sm.run():
        yield

    default_watchdog.stop()
    logger.info("TrainPilot Control Plane MCP Gateway stopped.")


app = FastAPI(
    title="TrainPilot Control Plane (MCP Server)",
    description="Centralized Model Context Protocol (MCP) Server and HITL Control Gateway for GPU Distributed Training with Feishu Card Integration",
    version=__version__,
    lifespan=lifespan,
)

# Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parsed_cors_origins,
    allow_credentials=settings.effective_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount HTTP routes
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
        "mcp_enabled": True,
        "mcp_transport": "streamable_http",
        "mcp_endpoint": "/mcp",
        "version": __version__,
        "feishu_ready": settings.is_feishu_configured,
        "auth_enforced": settings.is_api_token_configured,
        "sqlite_enabled": settings.enable_sqlite,
        "tasks_count": tasks_count,
        "stale_tasks_count": stale_count,
        "watchdog": default_watchdog.get_stats(),
    }


@app.get("/ready", tags=["System"])
def readiness_check():
    """Readiness probe: mailbox accessible."""
    try:
        default_mailbox.list_tasks(limit=1, offset=0)
        return {"ready": True, "service": "trainpilot-control-plane", "mcp_enabled": True, "version": __version__}
    except Exception as exc:
        logger.error("Readiness check failed: %s", exc)
        return {"ready": False, "error": str(exc)}


@app.get("/", tags=["System"])
def root():
    """Root info endpoint."""
    return {
        "message": "Welcome to TrainPilot Control Plane MCP Server",
        "docs_url": "/docs",
        "health_url": "/health",
        "mcp_url": "/mcp",
        "transport": "streamable_http",
    }


# Mount MCP Server Streamable HTTP transport at root so /mcp endpoint is exposed
app.mount("/", streamable_mcp_app)


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
