from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class SessionBusyError(RuntimeError):
    pass


class SessionLockManager:
    def __init__(self) -> None:
        self._locks: dict[uuid.UUID, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self, session_id: uuid.UUID) -> AsyncIterator[None]:
        async with self._guard:
            lock = self._locks.setdefault(session_id, asyncio.Lock())
            if lock.locked():
                raise SessionBusyError
            await lock.acquire()
        try:
            yield
        finally:
            lock.release()
