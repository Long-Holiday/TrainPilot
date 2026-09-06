"""Server configuration management using pydantic-settings."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from trainpilot.common.gateway import resolve_bind_host, resolve_gateway_url, resolve_public_host


class ServerSettings(BaseSettings):
    """Configuration settings for the TrainPilot control plane server.

    Web vs GPU 环境变量约定 (v0.2+):
    - Web 服务器只用 ``host``/``bind_host`` + ``port`` 决定监听地址。
      优先 ``TRAINPILOT_BIND_HOST``, 回退 ``TRAINPILOT_HOST`` (仅本地值有效),
      否则默认 ``0.0.0.0``。即使误填公网 IP 也不会导致 bind 失败。
    - GPU 服务器只用 ``TRAINPILOT_GATEWAY_URL`` 或 ``TRAINPILOT_HOST`` +
      ``TRAINPILOT_PORT`` 拼接网关地址, 与本机的 bind 无关。
      例: ``TRAINPILOT_HOST=35.202.16.245`` + ``TRAINPILOT_PORT=28780``。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="TRAINPILOT_",
    )

    # Server binding
    # TRAINPILOT_HOST: 历史字段, Web 侧表示绑定地址 (建议 0.0.0.0), GPU 侧表示网关 Host。
    host: str = Field(default="0.0.0.0", description="Host to bind the server (Web 侧绑定地址; GPU 侧请改用网关地址, 见文档)")
    # TRAINPILOT_BIND_HOST: Web 侧专用绑定地址, 优先级高于 host, GPU 侧忽略。
    bind_host: Optional[str] = Field(default=None, description="Web-only bind address override (TRAINPILOT_BIND_HOST)")
    port: int = Field(default=28780, description="Port to listen on (custom non-conflicting port)")
    debug: bool = Field(default=False, description="Enable debug logging")

    # Feishu App Credentials
    feishu_app_id: Optional[str] = Field(default=None, description="Feishu App ID (cli_xxx)")
    feishu_app_secret: Optional[str] = Field(default=None, description="Feishu App Secret")
    feishu_verification_token: Optional[str] = Field(default=None, description="Feishu Webhook Verification Token")
    feishu_encrypt_key: Optional[str] = Field(default=None, description="Feishu Webhook Encrypt Key if configured")

    # Feishu Target Receiver (where alerts/milestones are pushed)
    # receive_id_type can be 'chat_id', 'open_id', 'user_id', or 'email'
    feishu_receive_id_type: str = Field(default="chat_id", description="Receiver ID type")
    feishu_receiver_id: Optional[str] = Field(
        default=None,
        description="Target chat_id (e.g. oc_xxx) or open_id (ou_xxx) to receive cards",
    )

    # Mock & Fallback Options
    enable_mock_feishu: bool = Field(
        default=False,
        description="Force Feishu mock mode regardless of credentials for testing",
    )

    # Task Timeout
    task_heartbeat_timeout_seconds: int = Field(
        default=300,
        description="Seconds without heartbeat before marking task as potentially stalled",
    )

    # HITL alert decision timeout: seconds without human click before auto self-resolve.
    alert_decision_timeout_seconds: int = Field(
        default=30,
        description="Seconds an alert card waits for human decision before auto self-resolve",
    )

    # API Authentication (optional, but strongly recommended for public gateways)
    # When set, GPU Agents must send `Authorization: Bearer <token>` or `X-API-Token: <token>`.
    # Webhook uses Feishu verification token separately and is not affected.
    api_token: Optional[str] = Field(
        default=None,
        description="Shared secret for Agent-facing APIs. If unset, APIs are open (dev mode).",
    )

    # CORS settings (fix allow-all + credentials invalid combo)
    cors_allow_origins: str = Field(
        default="*",
        description="Comma-separated allowed origins, e.g. 'https://console.example.com'. '*' means all (no credentials).",
    )
    cors_allow_credentials: bool = Field(
        default=False,
        description="Whether to allow cookies/auth headers. Must be false when origins='*'.",
    )

    # Reliability caps
    max_events_per_task: int = Field(
        default=500,
        description="Max retained events per task (ring buffer, oldest dropped).",
    )

    # SQLite Persistence
    sqlite_path: str = Field(
        default="trainpilot.db",
        description="Path to SQLite database file for local persistence",
    )
    enable_sqlite: bool = Field(
        default=True,
        description="Enable SQLite persistence for tasks and events",
    )

    # Server-side Watchdog
    enable_watchdog: bool = Field(
        default=True,
        description="Enable server-side heartbeat watchdog and periodic housekeeping",
    )
    watchdog_interval_seconds: int = Field(
        default=15,
        description="Watchdog background check interval in seconds",
    )

    # Long Polling
    long_poll_timeout_seconds: float = Field(
        default=20.0,
        description="Default max timeout in seconds for server-side long polling",
    )

    # MCP Transport Security (DNS rebinding protection)
    mcp_enable_dns_rebinding_protection: bool = Field(
        default=False,
        description="Whether to enable DNS rebinding protection for MCP transport (default False to allow remote cluster access)",
    )
    mcp_allowed_hosts: str = Field(
        default="*",
        description="Comma-separated allowed Host headers when DNS rebinding protection is enabled",
    )

    @property
    def effective_bind_host(self) -> str:
        """Web 侧实际用于 uvicorn --host 的绑定地址。"""
        return resolve_bind_host(bind_host_env=self.bind_host, host_env=self.host)

    @property
    def public_host(self) -> str:
        """对外展示/拼接用的网关 Host (0.0.0.0 自动折叠为 127.0.0.1)。"""
        return resolve_public_host(self.host)

    @property
    def advertised_gateway_url(self) -> str:
        """Web 侧对外公布的网关地址 (优先 TRAINPILOT_GATEWAY_URL, 否则由 HOST+PORT 拼接)。"""
        gateway_env = os.environ.get("TRAINPILOT_GATEWAY_URL")
        if gateway_env and gateway_env.strip():
            return gateway_env.strip().rstrip("/")
        return resolve_gateway_url(host=self.host, port=self.port)

    @property
    def is_feishu_configured(self) -> bool:
        """Return True if Feishu credentials and target receiver are set."""
        return (
            not self.enable_mock_feishu
            and bool(self.feishu_app_id)
            and bool(self.feishu_app_secret)
            and bool(self.feishu_receiver_id)
        )

    @property
    def is_api_token_configured(self) -> bool:
        """Return True if API token auth is enforced."""
        return bool(self.api_token)

    @property
    def parsed_cors_origins(self) -> list[str]:
        """Parse comma-separated origins into a list."""
        raw = (self.cors_allow_origins or "*").strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    @property
    def effective_cors_allow_credentials(self) -> bool:
        """Browsers reject '*' + credentials; force False in that case."""
        if self.parsed_cors_origins == ["*"]:
            return False
        return self.cors_allow_credentials

    @property
    def mcp_transport_security(self):
        """MCP transport security settings for DNS rebinding protection."""
        from mcp.server.transport_security import TransportSecuritySettings

        if not self.mcp_enable_dns_rebinding_protection:
            return TransportSecuritySettings(enable_dns_rebinding_protection=False)

        raw = (self.mcp_allowed_hosts or "*").strip()
        if raw == "*":
            allowed = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
            if self.host and self.host != "0.0.0.0":
                allowed.append(f"{self.host}:*")
            return TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=allowed,
                allowed_origins=["*"],
            )

        allowed = [h.strip() for h in raw.split(",") if h.strip()]
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed,
            allowed_origins=["*"],
        )


settings = ServerSettings()
