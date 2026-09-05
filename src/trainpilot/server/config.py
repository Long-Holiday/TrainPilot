"""Server configuration management using pydantic-settings."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    """Configuration settings for the TrainPilot control plane server."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="TRAINPILOT_",
    )

    # Server binding
    host: str = Field(default="0.0.0.0", description="Host to bind the server")
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


settings = ServerSettings()
