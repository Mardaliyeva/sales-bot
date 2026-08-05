from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends

from app.agent.locks import SessionBusyError, SessionLockManager
from app.agent.runtime import AgentRuntime
from app.agent.types import AgentRuntimeError
from app.api.deps import get_agent_runtime, get_database, get_lock_manager, get_repository
from app.api.errors import ApiError
from app.api.schemas import ChatRequest, ChatResponse
from app.db.repositories import (
    ConcurrentRunError,
    ConversationRepository,
    SessionClosedError,
    SessionExpiredError,
    SessionNotFoundError,
)
from app.db.session import Database
from app.safety.input_gate import InputRejected, validate_and_clean_message

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])
DatabaseDep = Annotated[Database, Depends(get_database)]
RepositoryDep = Annotated[ConversationRepository, Depends(get_repository)]
RuntimeDep = Annotated[AgentRuntime, Depends(get_agent_runtime)]
LockManagerDep = Annotated[SessionLockManager, Depends(get_lock_manager)]


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    database: DatabaseDep,
    repository: RepositoryDep,
    runtime: RuntimeDep,
    lock_manager: LockManagerDep,
) -> ChatResponse:
    request_id = uuid.uuid4()
    try:
        message = validate_and_clean_message(payload.message)
    except InputRejected as exc:
        raise ApiError(422, exc.code, exc.message, request_id=request_id) from exc

    if not await database.is_ready():
        raise ApiError(
            503,
            "database_unavailable",
            "Məlumat bazası hazır deyil.",
            request_id=request_id,
        )

    logger.info(
        "chat.request_received",
        extra={"request_id": str(request_id), "session_id": str(payload.session_id)},
    )
    try:
        async with lock_manager.acquire(payload.session_id):
            run, _, session = await repository.start_run(payload.session_id, request_id, message)
            result = await runtime.run(run=run, session=session, user_message=message)
    except SessionBusyError as exc:
        raise ApiError(
            409,
            "session_busy",
            "Bu sessiyada başqa sorğu icra olunur.",
            request_id=request_id,
        ) from exc
    except ConcurrentRunError as exc:
        raise ApiError(
            409,
            "session_busy",
            "Bu sessiyada başqa sorğu icra olunur.",
            request_id=request_id,
        ) from exc
    except SessionNotFoundError as exc:
        raise ApiError(404, "session_not_found", "Sessiya tapılmadı.", request_id=request_id) from exc
    except SessionExpiredError as exc:
        raise ApiError(
            410,
            "session_expired",
            "Sessiyanın vaxtı bitib. Yeni sessiya yaradın.",
            request_id=request_id,
        ) from exc
    except SessionClosedError as exc:
        raise ApiError(410, "session_closed", "Sessiya bağlıdır.", request_id=request_id) from exc
    except AgentRuntimeError as exc:
        raise ApiError(exc.http_status, exc.code, exc.user_message, request_id=request_id) from exc
    except ApiError:
        raise
    except Exception as exc:
        logger.exception(
            "chat.failed",
            extra={
                "request_id": str(request_id),
                "session_id": str(payload.session_id),
                "error_type": "unhandled_chat_error",
            },
        )
        raise ApiError(
            503,
            "assistant_temporarily_unavailable",
            "Hazırda cavab hazırlamaq mümkün olmadı. Bir qədər sonra yenidən yoxlayın.",
            request_id=request_id,
        ) from exc

    return ChatResponse(
        request_id=request_id,
        session_id=payload.session_id,
        message_id=result.message_id,
        answer=result.answer,
        used_tools=result.used_tools,
        finish_reason="completed",
    )
