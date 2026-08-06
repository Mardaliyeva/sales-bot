from __future__ import annotations

from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings


@pytest.fixture
def catalog_path() -> Path:
    return PROJECT_ROOT / "data" / "catalog" / "products.jsonl"


@pytest.fixture
def settings(catalog_path: Path) -> Settings:
    return Settings(
        database_url="postgresql+psycopg://assistant_app:test@127.0.0.1:5432/ecommerce_assistant",
        customer_azure_openai_endpoint="https://test-resource.openai.azure.com",
        customer_azure_openai_api_key="test-azure-key",
        azure_text_model="gpt-5.4-mini",
        product_catalog_path=catalog_path,
        llm_timeout_seconds=1,
        qdrant_url=None,
        qdrant_api_key=None,
    )
