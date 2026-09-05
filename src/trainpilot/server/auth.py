"""Optional API token authentication for Agent-facing endpoints."""

import logging
from typing import Optional

from fastapi import Header, HTTPException, status

from trainpilot.server.config import settings

logger = logging.getLogger("trainpilot.auth")


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
    expected = settings.api_token or ""
    provided: Optional[str] = None
    if authorization and authorization.startswith("Bearer "):
        provided = authorization[len("Bearer "):].strip()
    elif x_api_token:
        provided = x_api_token.strip()
    if not provided or provided != expected:
        logger.warning("Rejected request with invalid/missing API token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token",
        )
    return None
