from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.agent.context_builder import build_context
from app.agent.prompt import FINAL_WITHOUT_TOOLS
from app.agent.types import AgentResult, AgentRuntimeError
from app.config import Settings
from app.db.models import AgentRun, ChatSession
from app.db.repositories import ConversationRepository
from app.llm.azure_client import AzureChatClient, ProviderError, ProviderTimeoutError
from app.tools.registry import ToolArgumentsError, ToolRegistry, UnknownToolError

logger = logging.getLogger(__name__)


class AgentRuntime:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: ConversationRepository,
        llm: AzureChatClient,
        tools: ToolRegistry,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.llm = llm
        self.tools = tools

    async def run(
        self,
        *,
        run: AgentRun,
        session: ChatSession,
        user_message: str,
    ) -> AgentResult:
        started = time.perf_counter()
        model_rounds = 0
        tool_count = 0
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        used_tools: list[str] = []
        last_product_ids: list[str] | None = None
        provider_response_id: str | None = None

        history = await self.repository.get_final_history(
            session.id,
            exclude_run_id=run.id,
            limit=self.settings.history_message_limit,
        )
        messages = build_context(
            session_context=session.context or {},
            history=history,
            user_message=user_message,
        )

        try:
            for round_number in range(1, self.settings.max_model_rounds + 1):
                model_rounds = round_number
                allow_tools = round_number < self.settings.max_model_rounds and tool_count < min(
                    session.max_tool_count, self.settings.max_tool_count
                )
                if not allow_tools:
                    messages.append({"role": "system", "content": FINAL_WITHOUT_TOOLS})

                response = await self.llm.chat(
                    messages=messages,
                    tools=self.tools.specs() if allow_tools else None,
                    tool_choice="auto" if allow_tools else "none",
                    request_id=str(run.request_id),
                    model_round=round_number,
                )
                provider_response_id = response.id
                input_tokens += response.usage.prompt_tokens
                output_tokens += response.usage.completion_tokens
                reasoning_tokens += response.usage.reasoning_tokens
                assistant = response.first_choice.message
                tool_calls = assistant.tool_calls or []

                if not tool_calls:
                    answer = (assistant.content or "").strip()
                    if not answer:
                        raise AgentRuntimeError(
                            "provider_protocol_error",
                            "Hazırda cavab hazırlamaq mümkün olmadı. Bir qədər sonra yenidən yoxlayın.",
                        )
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    message = await self.repository.complete_run(
                        run_id=run.id,
                        session_id=session.id,
                        answer=answer,
                        provider_response_id=provider_response_id,
                        tool_count=tool_count,
                        model_rounds=model_rounds,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        reasoning_tokens=reasoning_tokens,
                        latency_ms=latency_ms,
                        last_product_ids=last_product_ids,
                    )
                    logger.info(
                        "agent.completed",
                        extra={
                            "request_id": str(run.request_id),
                            "session_id": str(session.id),
                            "run_id": str(run.id),
                            "model": session.model,
                            "tool_count": tool_count,
                            "model_round": model_rounds,
                            "duration_ms": latency_ms,
                            "status": "completed",
                        },
                    )
                    return AgentResult(message_id=message.id, answer=answer, used_tools=used_tools)

                if not allow_tools or len(tool_calls) != 1:
                    raise AgentRuntimeError(
                        "provider_protocol_error",
                        "Hazırda cavab hazırlamaq mümkün olmadı. Bir qədər sonra yenidən yoxlayın.",
                    )

                call = tool_calls[0]
                tool_count += 1
                tool_name = call.function.name
                raw_arguments: dict[str, Any] = {}
                try:
                    decoded = json.loads(call.function.arguments)
                    if not isinstance(decoded, dict):
                        raise ValueError
                    raw_arguments = decoded
                    tool_result = await self.tools.execute(tool_name, raw_arguments)
                except json.JSONDecodeError, ValueError, ToolArgumentsError:
                    tool_result = {
                        "status": "error",
                        "code": "invalid_tool_arguments",
                        "message": "Tool arqumentləri etibarsızdır.",
                    }
                except UnknownToolError:
                    tool_result = {
                        "status": "error",
                        "code": "unknown_tool",
                        "message": "Bu alət sistemdə mövcud deyil.",
                    }
                except TimeoutError:
                    tool_result = {
                        "status": "error",
                        "code": "tool_timeout",
                        "message": "Məhsul axtarışı vaxtında tamamlanmadı.",
                    }
                except Exception:
                    logger.exception(
                        "tool.failed",
                        extra={
                            "request_id": str(run.request_id),
                            "session_id": str(session.id),
                            "run_id": str(run.id),
                            "tool_name": tool_name,
                            "tool_count": tool_count,
                            "model_round": model_rounds,
                            "status": "failed",
                            "error_type": "tool_internal_error",
                        },
                    )
                    tool_result = {
                        "status": "error",
                        "code": "tool_failed",
                        "message": "Məhsul axtarışı tamamlanmadı.",
                    }

                if tool_name == "product_search" and tool_name not in used_tools:
                    used_tools.append(tool_name)
                if tool_result.get("status") == "success":
                    last_product_ids = [item["product_id"] for item in tool_result.get("items", [])]

                tool_call_payload = call.model_dump(mode="json", exclude_none=True)
                await self.repository.store_tool_exchange(
                    run_id=run.id,
                    session_id=session.id,
                    provider_response_id=provider_response_id,
                    assistant_content=assistant.content,
                    tool_call=tool_call_payload,
                    tool_name=tool_name,
                    tool_arguments=raw_arguments,
                    tool_result=tool_result,
                    tool_count=tool_count,
                    model_rounds=model_rounds,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                )

                assistant_payload: dict[str, Any] = {
                    "role": "assistant",
                    "content": assistant.content,
                    "tool_calls": [tool_call_payload],
                }
                if assistant.reasoning_details is not None:
                    assistant_payload["reasoning_details"] = assistant.reasoning_details
                messages.append(assistant_payload)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

            raise AgentRuntimeError(
                "model_round_limit",
                "Hazırda cavab hazırlamaq mümkün olmadı. Bir qədər sonra yenidən yoxlayın.",
            )
        except AgentRuntimeError as exc:
            await self._mark_failed(run, exc.code, model_rounds, tool_count, started)
            raise
        except ProviderTimeoutError as exc:
            await self._mark_failed(run, exc.error_type, model_rounds, tool_count, started)
            raise AgentRuntimeError(
                "assistant_timeout",
                "Cavabın hazırlanması vaxtı keçdi. Bir qədər sonra yenidən yoxlayın.",
                http_status=504,
            ) from exc
        except ProviderError as exc:
            await self._mark_failed(run, exc.error_type, model_rounds, tool_count, started)
            raise AgentRuntimeError(
                "assistant_temporarily_unavailable",
                "Hazırda cavab hazırlamaq mümkün olmadı. Bir qədər sonra yenidən yoxlayın.",
            ) from exc
        except Exception as exc:
            logger.exception(
                "agent.failed",
                extra={
                    "request_id": str(run.request_id),
                    "session_id": str(session.id),
                    "run_id": str(run.id),
                    "model": session.model,
                    "tool_count": tool_count,
                    "model_round": model_rounds,
                    "status": "failed",
                    "error_type": "agent_internal_error",
                },
            )
            await self._mark_failed(run, "agent_internal_error", model_rounds, tool_count, started)
            raise AgentRuntimeError(
                "assistant_temporarily_unavailable",
                "Hazırda cavab hazırlamaq mümkün olmadı. Bir qədər sonra yenidən yoxlayın.",
            ) from exc

    async def _mark_failed(
        self,
        run: AgentRun,
        error_type: str,
        model_rounds: int,
        tool_count: int,
        started: float,
    ) -> None:
        try:
            await self.repository.fail_run(
                run_id=run.id,
                error_type=error_type,
                error_message=error_type,
                model_rounds=model_rounds,
                tool_count=tool_count,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception:
            logger.exception("agent.failure_persistence_failed")
