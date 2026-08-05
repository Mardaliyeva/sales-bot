from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest

from app.config import Settings
from app.db.repositories import ConversationRepository
from app.db.session import Database


@pytest.mark.integration
@pytest.mark.asyncio
async def test_migration_and_session_persistence() -> None:
    test_url = os.getenv("TEST_DATABASE_URL")
    if not test_url:
        pytest.skip("TEST_DATABASE_URL təyin edilməyib")
    if not test_url.rsplit("/", 1)[-1].endswith("_test"):
        pytest.fail("Integration test yalnız _test sonluqlu database-də işlədilə bilər")

    env = os.environ.copy()
    env["DATABASE_URL"] = test_url
    await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        check=True,
        env=env,
    )
    settings = Settings(database_url=test_url, openrouter_api_key="test-key")
    database = Database(test_url)
    try:
        assert await database.is_ready() is True
        repository = ConversationRepository(database, settings)
        session = await repository.create_session()
        loaded = await repository.get_session(session.id)
        assert loaded.id == session.id
        assert loaded.status == "active"
    finally:
        await database.dispose()
