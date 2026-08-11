from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

from app.config import Settings
from app.db.models import ChatSession
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
    settings = Settings(
        database_url=test_url,
        customer_azure_openai_endpoint="https://test-resource.openai.azure.com",
        customer_azure_openai_api_key="test-key",
    )
    database = Database(test_url)
    try:
        assert await database.is_ready() is True
        repository = ConversationRepository(database, settings)
        session = await repository.create_session()
        loaded = await repository.get_session(session.id)
        assert loaded.id == session.id
        assert loaded.status == "active"

        expired_at = datetime.now(UTC) - timedelta(minutes=1)
        async with database.session() as db, db.begin():
            await db.execute(
                update(ChatSession)
                .where(ChatSession.id == session.id)
                .values(
                    expires_at=expired_at,
                    context={
                        "last_product_ids": ["prd_test_001"],
                        "memory": {
                            "version": 2,
                            "continuation_summary": "Silinməli sessiya konteksti",
                        },
                    },
                )
            )

        assert await repository.scrub_expired_session_contexts() == 1
        async with database.session() as db:
            scrubbed = await db.get(ChatSession, session.id)
            assert scrubbed is not None
            assert scrubbed.status == "expired"
            assert scrubbed.context == {}
        assert await repository.scrub_expired_session_contexts() == 0
    finally:
        await database.dispose()
