from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_placeholder_api_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            database_url="postgresql+psycopg://user:pass@127.0.0.1/db",
            openrouter_api_key="CHANGE_ME",
        )


def test_non_postgres_database_url_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(database_url="sqlite:///test.db", openrouter_api_key="test")


def test_optional_vector_services_are_normalized() -> None:
    settings = Settings(
        database_url="postgresql+psycopg://user:pass@127.0.0.1/db",
        openrouter_api_key="test",
        customer_azure_openai_endpoint="https://example.openai.azure.com/",
        customer_azure_openai_api_key=" azure-secret ",
        qdrant_url="https://example.cloud.qdrant.io/",
        qdrant_api_key=" qdrant-secret ",
    )

    assert settings.customer_azure_openai_endpoint == "https://example.openai.azure.com"
    assert settings.customer_azure_openai_api_key is not None
    assert settings.customer_azure_openai_api_key.get_secret_value() == "azure-secret"
    assert settings.qdrant_url == "https://example.cloud.qdrant.io"
    assert settings.qdrant_api_key is not None
    assert settings.qdrant_api_key.get_secret_value() == "qdrant-secret"


def test_vector_service_urls_must_use_https() -> None:
    with pytest.raises(ValidationError):
        Settings(
            database_url="postgresql+psycopg://user:pass@127.0.0.1/db",
            openrouter_api_key="test",
            qdrant_url="http://example.local",
        )
