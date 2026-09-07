import hmac
import logging
from typing import Optional

from fastapi import Header, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from trainpilot.server.config import settings

logger = logging.getLogger("trainpilot.auth")


def is_api_token_valid(provided_token: Optional[str]) -> bool:
    """Validate token using constant-time comparison against settings.api_token."""
    if not settings.is_api_token_configured:
        return True
    if not provided_token:
        return False
    expected = settings.api_token or ""
    return hmac.compare_digest(provided_token, expected)


async def verify_api_token(
    authorization: Optional[str] = Header(default=None),
    x_api_token: Optional[str] = Header(default=None),
) -> None:
    """FastAPI dependency: enforce Bearer / X-API-Token when configured.

    - If `TRAINPILOT_API_TOKEN` is unset: open dev mode, always allow.
    - If set: require `Authorization: Bearer <token>` or `X-API-Token: <token>`.
    """
    if not settings.is_api_token_configured:
        return None
    provided: Optional[str] = None
    if authorization and authorization.startswith("Bearer "):
        provided = authorization[len("Bearer "):].strip()
    elif x_api_token:
        provided = x_api_token.strip()
    if not is_api_token_valid(provided):
        logger.warning("Rejected request with invalid/missing API token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token",
        )
    return None


class McpAuthMiddleware(BaseHTTPMiddleware):
    """ASGI middleware to enforce API token authentication on MCP endpoints (/mcp)."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path == "/mcp" or path.startswith("/mcp/"):
            # Allow CORS preflight OPTIONS requests
            if request.method == "OPTIONS":
                return await call_next(request)

            if settings.is_api_token_configured:
                auth_header = request.headers.get("Authorization")
                token: Optional[str] = None
                if auth_header and auth_header.startswith("Bearer "):
                    token = auth_header[len("Bearer "):].strip()
                elif "X-API-Token" in request.headers:
                    token = request.headers["X-API-Token"].strip()
                elif "token" in request.query_params:
                    token = request.query_params["token"].strip()

                if not is_api_token_valid(token):
                    logger.warning("Rejected unauthorized request to MCP endpoint '%s'", path)
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "Unauthorized: Invalid or missing API token for MCP endpoint"},
                    )

        return await call_next(request)

