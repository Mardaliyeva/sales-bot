from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


class Database:
    def __init__(self, url: str) -> None:
        self.engine: AsyncEngine = create_async_engine(url, pool_pre_ping=True)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    def session(self) -> AsyncSession:
        return self.session_factory()

    async def is_ready(self) -> bool:
        try:
            async with self.engine.connect() as connection:
                result = await connection.execute(
                    text(
                        "SELECT to_regclass('public.chat_sessions') IS NOT NULL "
                        "AND to_regclass('public.agent_runs') IS NOT NULL "
                        "AND to_regclass('public.chat_messages') IS NOT NULL"
                    )
                )
                return bool(result.scalar_one())
        except Exception:
            return False

    async def dispose(self) -> None:
        await self.engine.dispose()
