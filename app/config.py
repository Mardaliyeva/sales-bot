from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    debug_panel_enabled: bool = False

    database_url: str
    test_database_url: str | None = None

    customer_azure_openai_endpoint: str
    customer_azure_openai_api_key: SecretStr
    azure_text_model: str = "gpt-5.4-mini"
    azure_embedding_model: str = "text-embedding-3-large"

    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_name: str = Field(
        default="sales_bot_products_semantic_v2",
        pattern=r"^[a-zA-Z0-9_-]{1,255}$",
    )
    alternative_min_score: float = Field(default=0.39, ge=0, le=1)

    product_catalog_path: Path = PROJECT_ROOT / "data" / "catalog" / "products.jsonl"
    mode_name: str = "ecommerce_assistant_v1"
    reasoning_effort: str = "low"
    max_tool_count: int = Field(default=3, ge=0, le=10)
    max_model_rounds: int = Field(default=4, ge=1, le=12)
    max_output_tokens: int = Field(default=800, ge=64, le=4096)
    history_message_limit: int = Field(default=12, ge=2, le=50)
    session_ttl_hours: int = Field(default=168, ge=1, le=24 * 90)
    llm_timeout_seconds: float = Field(default=30, gt=0, le=120)
    tool_timeout_seconds: float = Field(default=10, gt=0, le=60)

    @field_validator("database_url", "test_database_url")
    @classmethod
    def validate_database_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("postgresql+psycopg://"):
            raise ValueError("PostgreSQL URL postgresql+psycopg:// ilə başlamalıdır")
        return value

    @field_validator("customer_azure_openai_api_key")
    @classmethod
    def validate_azure_key(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value().strip()
        if not secret or secret in {"YOUR_AZURE_API_KEY", "CHANGE_ME"}:
            raise ValueError("CUSTOMER_AZURE_OPENAI_API_KEY konfiqurasiya edilməyib")
        return SecretStr(secret)

    @field_validator("customer_azure_openai_endpoint")
    @classmethod
    def normalize_azure_endpoint(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if normalized == "YOUR_AZURE_ENDPOINT":
            raise ValueError("CUSTOMER_AZURE_OPENAI_ENDPOINT konfiqurasiya edilməyib")
        if not normalized.startswith("https://"):
            raise ValueError("Azure endpoint HTTPS olmalıdır")
        return normalized

    @field_validator("qdrant_url")
    @classmethod
    def normalize_qdrant_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().rstrip("/")
        if not normalized.startswith("https://"):
            raise ValueError("Qdrant URL HTTPS olmalıdır")
        return normalized

    @field_validator("qdrant_api_key")
    @classmethod
    def normalize_optional_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        return SecretStr(value.get_secret_value().strip())

    @field_validator("azure_text_model", "azure_embedding_model")
    @classmethod
    def validate_azure_deployment(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Azure deployment adı boş ola bilməz")
        return normalized

    @field_validator("reasoning_effort")
    @classmethod
    def validate_reasoning_effort(cls, value: str) -> str:
        allowed = {"none", "low", "medium", "high", "xhigh"}
        if value not in allowed:
            raise ValueError(f"reasoning_effort bunlardan biri olmalıdır: {sorted(allowed)}")
        return value

    @field_validator("product_catalog_path")
    @classmethod
    def resolve_catalog_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
