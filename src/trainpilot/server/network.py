"""Network request analysis utilities for TrainPilot Control Plane.

Extracts and manages the GPU Agent's client IP from incoming HTTP and MCP requests,
enabling the watchdog to ping the GPU host without requiring the agent to self-report its IP.
"""

import contextvars
import ipaddress
import logging
from typing import Optional
from starlette.requests import Request

logger = logging.getLogger("trainpilot.network")

# ContextVar storing the client IP extracted from the current active HTTP/MCP request.
_client_ip_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "trainpilot_client_ip", default=None
)


def normalize_client_ip(raw_ip: Optional[str]) -> Optional[str]:
    """Clean and validate an IP address extracted from a socket or header."""
    if not raw_ip:
        return None
    ip = raw_ip.strip().strip("'\"")
    if not ip or ip.lower() in ("unknown", "none", "null"):
        return None

    # Strip IPv4-mapped IPv6 prefix (e.g., ::ffff:192.168.1.1 -> 192.168.1.1)
    if ip.lower().startswith("::ffff:"):
        ip = ip[7:].strip()

    # Strip enclosing brackets if any (e.g., [127.0.0.1] -> 127.0.0.1)
    if ip.startswith("[") and ip.endswith("]"):
        ip = ip[1:-1].strip()

    # Strip port if present in IPv4 (e.g., 192.168.1.10:45678)
    if ":" in ip and ip.count(":") == 1:
        maybe_ip, maybe_port = ip.rsplit(":", 1)
        if maybe_port.isdigit():
            ip = maybe_ip.strip()

    # Verify IP validity
    try:
        ipaddress.ip_address(ip)
        return ip
    except ValueError:
        # If not a strict IP (e.g. hostname), allow if non-empty and valid host chars
        if ip and all(c.isalnum() or c in ".-_" for c in ip):
            return ip
        return None


def extract_client_ip(request: Request) -> Optional[str]:
    """Analyze an incoming HTTP/MCP network request to determine the client's source IP.

    Checks:
    1. X-Forwarded-For header (first hop IP if behind reverse proxy/load balancer)
    2. X-Real-IP header
    3. Underlying ASGI client host (request.client.host)
    """
    # 1. Check X-Forwarded-For
    xff = request.headers.get("x-forwarded-for")
    if xff:
        # Format can be: "client, proxy1, proxy2"
        first_ip = xff.split(",")[0].strip()
        cleaned = normalize_client_ip(first_ip)
        if cleaned:
            return cleaned

    # 2. Check X-Real-IP
    x_real_ip = request.headers.get("x-real-ip")
    if x_real_ip:
        cleaned = normalize_client_ip(x_real_ip)
        if cleaned:
            return cleaned

    # 3. Check socket connection client host
    if request.client and request.client.host:
        cleaned = normalize_client_ip(request.client.host)
        if cleaned:
            return cleaned

    return None


def get_current_client_ip() -> Optional[str]:
    """Retrieve the client IP analyzed from the current request context."""
    return _client_ip_ctx.get()


def set_current_client_ip(ip: Optional[str]) -> contextvars.Token:
    """Set the analyzed client IP for the current request context."""
    cleaned = normalize_client_ip(ip) if ip else None
    return _client_ip_ctx.set(cleaned)


def reset_current_client_ip(token: contextvars.Token) -> None:
    """Reset the client IP context variable."""
    _client_ip_ctx.reset(token)


from starlette.middleware.base import BaseHTTPMiddleware


class ClientIpMiddleware(BaseHTTPMiddleware):
    """ASGI middleware to analyze incoming network requests, set context IP, and update tasks."""

    async def dispatch(self, request: Request, call_next):
        client_ip = extract_client_ip(request)
        token = set_current_client_ip(client_ip)
        try:
            # Check if task_id can be extracted from headers or path
            task_id = request.headers.get("X-Task-ID")
            if not task_id:
                path_parts = request.url.path.strip("/").split("/")
                if len(path_parts) >= 3 and path_parts[0] == "api" and path_parts[1] == "tasks":
                    candidate = path_parts[2]
                    if candidate not in ("notify", "events"):
                        task_id = candidate
            if task_id and client_ip:
                from trainpilot.server.mailbox import default_mailbox
                default_mailbox.update_task_client_ip(task_id, client_ip)

            return await call_next(request)
        finally:
            reset_current_client_ip(token)
