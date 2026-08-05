from __future__ import annotations

import json
from typing import Any

from app.agent.prompt import SYSTEM_PROMPT
from app.db.repositories import HistoryMessage


def build_context(
    *,
    session_context: dict[str, Any],
    history: list[HistoryMessage],
    user_message: str,
) -> list[dict[str, Any]]:
    safe_context = {
        "last_product_ids": session_context.get("last_product_ids", []),
        "focused_product_id": session_context.get("focused_product_id"),
    }
    system = f"{SYSTEM_PROMPT}\nDaxili sessiya konteksti: {json.dumps(safe_context, ensure_ascii=False)}"
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    messages.extend({"role": item.role, "content": item.content} for item in history)
    messages.append({"role": "user", "content": user_message})
    return messages
