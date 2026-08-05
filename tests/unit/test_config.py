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
