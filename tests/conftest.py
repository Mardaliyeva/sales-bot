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
        openrouter_api_key="test-openrouter-key",
        product_catalog_path=catalog_path,
        llm_timeout_seconds=1,
    )
