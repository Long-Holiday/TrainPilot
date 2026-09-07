"""Unit and integration tests for MCP Server authentication middleware."""

import pytest
from fastapi.testclient import TestClient

from trainpilot.server.auth import is_api_token_valid
from trainpilot.server.config import settings
from trainpilot.server.main import app


@pytest.fixture(autouse=True)
def reset_auth_settings():
    old_token = settings.api_token
    yield
    settings.api_token = old_token


def test_is_api_token_valid_logic():
    settings.api_token = "my-secret-token-123"
    assert is_api_token_valid("my-secret-token-123") is True
    assert is_api_token_valid("wrong-token") is False
    assert is_api_token_valid(None) is False
    assert is_api_token_valid("") is False

    # Dev mode
    settings.api_token = None
    assert is_api_token_valid(None) is True
    assert is_api_token_valid("any") is True


def test_mcp_endpoint_auth_enforced_when_token_configured():
    settings.api_token = "test-mcp-key-xyz"

    with TestClient(app) as client:
        # 1. No token -> 401 Unauthorized
        resp = client.get("/mcp")
        assert resp.status_code == 401
        assert "Unauthorized" in resp.json()["detail"]

        # 2. Invalid token -> 401 Unauthorized
        resp_invalid = client.get("/mcp", headers={"Authorization": "Bearer wrong-token"})
        assert resp_invalid.status_code == 401

        # 3. Valid Bearer token -> Passed through (returns non-401, e.g. 405 Method Not Allowed or 200/400 streamable response)
        resp_valid = client.get("/mcp", headers={"Authorization": "Bearer test-mcp-key-xyz"})
        assert resp_valid.status_code != 401

        # 4. Valid X-API-Token -> Passed through
        resp_x = client.get("/mcp", headers={"X-API-Token": "test-mcp-key-xyz"})
        assert resp_x.status_code != 401

        # 5. OPTIONS preflight -> Allowed without credentials (for browser/CORS)
        resp_opt = client.options("/mcp")
        assert resp_opt.status_code != 401


def test_mcp_endpoint_open_in_dev_mode():
    settings.api_token = None

    with TestClient(app) as client:
        resp = client.get("/mcp")
        # In dev mode without api_token, request is never blocked by auth (status != 401)
        assert resp.status_code != 401
