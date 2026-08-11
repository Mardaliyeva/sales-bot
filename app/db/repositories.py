from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, or_, select, update

from app.agent.memory import SessionMemory
from app.config import Settings
from app.db.models import AgentRun, ChatMessage, ChatSession
from app.db.session import Database


class SessionNotFoundError(LookupError):
    pass


class SessionExpiredError(LookupError):
    pass


class SessionClosedError(LookupError):
    pass


class ConcurrentRunError(RuntimeError):
    pass


@dataclass(slots=True)
class HistoryMessage:
    role: str
    content: str


@dataclass(slots=True)
class DebugRunSnapshot:
    run: AgentRun
    messages: list[ChatMessage]
    final_message_id: uuid.UUID | None


class ConversationRepository:
    def __init__(self, database: Database, settings: Settings) -> None:
        self.database = database
        self.settings = settings

    async def create_session(self) -> ChatSession:
        now = datetime.now(UTC)
        item = ChatSession(
            status="active",
            mode_name=self.settings.mode_name,
            model=self.settings.azure_text_model,
            reasoning_effort=self.settings.reasoning_effort,
            max_tool_count=self.settings.max_tool_count,
            context={
                "last_product_ids": [],
                "focused_product_id": None,
                "memory": SessionMemory().model_dump(mode="json", exclude_none=True),
            },
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(hours=self.settings.session_ttl_hours),
        )
        async with self.database.session() as db:
            db.add(item)
            await db.commit()
            await db.refresh(item)
        return item

    async def get_session(self, session_id: uuid.UUID) -> ChatSession:
        async with self.database.session() as db:
            item = await db.get(ChatSession, session_id)
            if item is None:
                raise SessionNotFoundError
            now = datetime.now(UTC)
            if item.status == "active" and item.expires_at <= now:
                item.status = "expired"
                item.context = {}
                item.updated_at = now
                await db.commit()
            if item.status == "expired":
                raise SessionExpiredError
            if item.status != "active":
                raise SessionClosedError
            return item

    async def scrub_expired_session_contexts(
        self,
        *,
        now: datetime | None = None,
    ) -> int:
        """Expire sessions and remove their cache-like context without deleting audit runs."""
        cutoff = now or datetime.now(UTC)
        async with self.database.session() as db, db.begin():
            result = await db.execute(
                update(ChatSession)
                .where(
                    ChatSession.expires_at <= cutoff,
                    or_(
                        ChatSession.status != "expired",
                        ChatSession.context != {},
                    ),
                )
                .values(status="expired", context={}, updated_at=cutoff)
            )
        return int(result.rowcount or 0)

    async def start_run(
        self,
        session_id: uuid.UUID,
        request_id: uuid.UUID,
        user_message: str,
    ) -> tuple[AgentRun, ChatMessage, ChatSession]:
        # This preflight persists a lazily expired session before the write transaction starts.
        await self.get_session(session_id)
        now = datetime.now(UTC)
        async with self.database.session() as db, db.begin():
            session = (
                await db.execute(select(ChatSession).where(ChatSession.id == session_id).with_for_update())
            ).scalar_one_or_none()
            if session is None:
                raise SessionNotFoundError
            if session.expires_at <= now or session.status == "expired":
                session.status = "expired"
                session.updated_at = now
                raise SessionExpiredError
            if session.status != "active":
                raise SessionClosedError

            running = await db.scalar(
                select(func.count())
                .select_from(AgentRun)
                .where(AgentRun.session_id == session_id, AgentRun.status == "running")
            )
            if running:
                raise ConcurrentRunError

            run = AgentRun(
                request_id=request_id,
                session_id=session_id,
                status="running",
                model=session.model,
                tool_count=0,
                model_rounds=0,
                input_tokens=0,
                output_tokens=0,
                reasoning_tokens=0,
                started_at=now,
            )
            db.add(run)
            await db.flush()
            sequence = await self._next_sequence(db, session_id)
            message = ChatMessage(
                session_id=session_id,
                run_id=run.id,
                sequence_no=sequence,
                role="user",
                content=user_message,
                created_at=now,
            )
            db.add(message)
            session.updated_at = now
        return run, message, session

    async def get_final_history(
        self,
        session_id: uuid.UUID,
        *,
        exclude_run_id: uuid.UUID,
        limit: int,
    ) -> list[HistoryMessage]:
        async with self.database.session() as db:
            rows = (
                (
                    await db.execute(
                        select(ChatMessage)
                        .where(
                            ChatMessage.session_id == session_id,
                            ChatMessage.run_id != exclude_run_id,
                            ChatMessage.role.in_(["user", "assistant"]),
                            ChatMessage.tool_calls.is_(None),
                            ChatMessage.content.is_not(None),
                        )
                        .order_by(ChatMessage.sequence_no.desc())
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
        return [HistoryMessage(role=row.role, content=row.content or "") for row in reversed(rows)]

    async def store_tool_exchange(
        self,
        *,
        run_id: uuid.UUID,
        session_id: uuid.UUID,
        provider_response_id: str | None,
        assistant_content: str | None,
        tool_call: dict[str, Any],
        tool_name: str,
        tool_arguments: dict[str, Any],
        tool_result: dict[str, Any],
        tool_count: int,
        model_rounds: int,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
    ) -> None:
        now = datetime.now(UTC)
        async with self.database.session() as db, db.begin():
            await db.execute(select(ChatSession).where(ChatSession.id == session_id).with_for_update())
            sequence = await self._next_sequence(db, session_id)
            db.add(
                ChatMessage(
                    session_id=session_id,
                    run_id=run_id,
                    sequence_no=sequence,
                    role="assistant",
                    content=assistant_content,
                    tool_calls=[tool_call],
                    tool_name=tool_name,
                    tool_arguments=tool_arguments,
                    provider_response_id=provider_response_id,
                    created_at=now,
                )
            )
            db.add(
                ChatMessage(
                    session_id=session_id,
                    run_id=run_id,
                    sequence_no=sequence + 1,
                    role="tool",
                    content=json.dumps(tool_result, ensure_ascii=False),
                    tool_call_id=tool_call.get("id"),
                    tool_name=tool_name,
                    tool_arguments=tool_arguments,
                    tool_result=tool_result,
                    created_at=now,
                )
            )
            run = await db.get(AgentRun, run_id)
            if run is not None:
                run.tool_count = tool_count
                run.model_rounds = model_rounds
                run.input_tokens = input_tokens
                run.output_tokens = output_tokens
                run.reasoning_tokens = reasoning_tokens

    async def complete_run(
        self,
        *,
        run_id: uuid.UUID,
        session_id: uuid.UUID,
        answer: str,
        provider_response_id: str | None,
        tool_count: int,
        model_rounds: int,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        latency_ms: int,
        last_product_ids: list[str] | None,
        session_memory: dict[str, Any] | None,
        debug_trace: dict[str, Any] | None,
    ) -> ChatMessage:
        now = datetime.now(UTC)
        async with self.database.session() as db, db.begin():
            session = (
                await db.execute(select(ChatSession).where(ChatSession.id == session_id).with_for_update())
            ).scalar_one()
            sequence = await self._next_sequence(db, session_id)
            message = ChatMessage(
                session_id=session_id,
                run_id=run_id,
                sequence_no=sequence,
                role="assistant",
                content=answer,
                provider_response_id=provider_response_id,
                created_at=now,
            )
            db.add(message)
            run = await db.get(AgentRun, run_id)
            if run is None:
                raise RuntimeError("Agent run tapılmadı")
            run.status = "completed"
            run.tool_count = tool_count
            run.model_rounds = model_rounds
            run.input_tokens = input_tokens
            run.output_tokens = output_tokens
            run.reasoning_tokens = reasoning_tokens
            run.latency_ms = latency_ms
            run.debug_trace = debug_trace
            run.completed_at = now
            session.updated_at = now
            if last_product_ids is not None or session_memory is not None:
                context = dict(session.context or {})
                if last_product_ids is not None:
                    context["last_product_ids"] = last_product_ids
                    context["focused_product_id"] = (
                        last_product_ids[0] if len(last_product_ids) == 1 else None
                    )
                if session_memory is not None:
                    context["memory"] = session_memory
                session.context = context
            await db.flush()
        return message

    async def fail_run(
        self,
        *,
        run_id: uuid.UUID,
        error_type: str,
        error_message: str,
        model_rounds: int,
        tool_count: int,
        latency_ms: int,
        debug_trace: dict[str, Any] | None,
    ) -> None:
        now = datetime.now(UTC)
        async with self.database.session() as db, db.begin():
            run = await db.get(AgentRun, run_id)
            if run is None:
                return
            run.status = "failed"
            run.error_type = error_type[:80]
            run.error_message = error_message[:2000]
            run.debug_trace = debug_trace
            run.model_rounds = model_rounds
            run.tool_count = tool_count
            run.latency_ms = latency_ms
            run.completed_at = now

    async def get_debug_run(
        self,
        *,
        session_id: uuid.UUID,
        request_id: uuid.UUID | None = None,
        message_id: uuid.UUID | None = None,
    ) -> DebugRunSnapshot | None:
        if (request_id is None) == (message_id is None):
            raise ValueError("Dəqiq bir request_id və ya message_id verilməlidir")
        async with self.database.session() as db:
            query = select(AgentRun).where(AgentRun.session_id == session_id)
            if request_id is not None:
                query = query.where(AgentRun.request_id == request_id)
            else:
                query = query.join(ChatMessage, ChatMessage.run_id == AgentRun.id).where(
                    ChatMessage.id == message_id
                )
            run = (await db.execute(query)).scalar_one_or_none()
            if run is None:
                return None
            messages = list(
                (
                    await db.execute(
                        select(ChatMessage)
                        .where(ChatMessage.run_id == run.id)
                        .order_by(ChatMessage.sequence_no)
                    )
                )
                .scalars()
                .all()
            )
        final_message_id = next(
            (
                message.id
                for message in reversed(messages)
                if message.role == "assistant"
                and message.tool_calls is None
                and message.content is not None
            ),
            None,
        )
        return DebugRunSnapshot(
            run=run,
            messages=messages,
            final_message_id=final_message_id,
        )

    @staticmethod
    async def _next_sequence(db: Any, session_id: uuid.UUID) -> int:
        current = await db.scalar(
            select(func.coalesce(func.max(ChatMessage.sequence_no), 0)).where(
                ChatMessage.session_id == session_id
            )
        )
        return int(current or 0) + 1
