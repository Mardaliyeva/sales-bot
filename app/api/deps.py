from __future__ import annotations

from fastapi import Request

from app.agent.locks import SessionLockManager
from app.agent.runtime import AgentRuntime
from app.db.repositories import ConversationRepository
from app.db.session import Database
from app.tools.catalog import ProductCatalog


def get_database(request: Request) -> Database:
    return request.app.state.database


def get_repository(request: Request) -> ConversationRepository:
    return request.app.state.repository


def get_agent_runtime(request: Request) -> AgentRuntime:
    return request.app.state.agent_runtime


def get_lock_manager(request: Request) -> SessionLockManager:
    return request.app.state.lock_manager


def get_catalog(request: Request) -> ProductCatalog:
    return request.app.state.catalog
