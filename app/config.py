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

    database_url: str
    test_database_url: str | None = None

    openrouter_api_key: SecretStr
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-5.4-mini"

    customer_azure_openai_endpoint: str | None = None
    customer_azure_openai_api_key: SecretStr | None = None
    azure_embedding_model: str = "text-embedding-3-large"

    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_name: str = Field(
        default="sales_bot_products_v1",
        pattern=r"^[a-zA-Z0-9_-]{1,255}$",
    )

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

    @field_validator("openrouter_api_key")
    @classmethod
    def validate_openrouter_key(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value().strip()
        if not secret or secret == "CHANGE_ME":
            raise ValueError("OPENROUTER_API_KEY konfiqurasiya edilməyib")
        return SecretStr(secret)

    @field_validator("openrouter_base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        value = value.rstrip("/")
        if not value.startswith("https://"):
            raise ValueError("OPENROUTER_BASE_URL HTTPS olmalıdır")
        return value

    @field_validator("customer_azure_openai_endpoint", "qdrant_url")
    @classmethod
    def normalize_optional_https_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().rstrip("/")
        if not normalized.startswith("https://"):
            raise ValueError("Azure və Qdrant URL-ləri HTTPS olmalıdır")
        return normalized

    @field_validator("customer_azure_openai_api_key", "qdrant_api_key")
    @classmethod
    def normalize_optional_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        return SecretStr(value.get_secret_value().strip())

    @field_validator("azure_embedding_model")
    @classmethod
    def validate_embedding_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("AZURE_EMBEDDING_MODEL boş ola bilməz")
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
