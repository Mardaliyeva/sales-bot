from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.agent.context_builder import build_context
from app.agent.presentation import build_product_cards
from app.agent.prompt import FINAL_WITHOUT_TOOLS
from app.agent.types import AgentResult, AgentRuntimeError
from app.config import Settings
from app.db.models import AgentRun, ChatSession
from app.db.repositories import ConversationRepository
from app.llm.azure_client import AzureChatClient, ProviderError, ProviderTimeoutError
from app.tools.registry import ToolArgumentsError, ToolRegistry, UnknownToolError

logger = logging.getLogger(__name__)
TRACE_VERSION = 2


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
        last_tool_result: dict[str, Any] | None = None
        last_retrieval_trace: dict[str, Any] | None = None
        trace = self._new_trace(user_message)

        try:
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
            trace["timeline"].append(
                {
                    "stage": "context_build",
                    "status": "completed",
                    "history_message_count": len(history),
                    "session_product_context_count": len(
                        (session.context or {}).get("last_product_ids", [])
                    ),
                }
            )

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
                decision = "tool_call" if tool_calls else (
                    "direct_answer" if allow_tools else "forced_final"
                )
                trace["timeline"].append(
                    {
                        "stage": "model_round",
                        "status": "completed",
                        "round": round_number,
                        "tools_allowed": allow_tools,
                        "decision": decision,
                        "tool_name": tool_calls[0].function.name if len(tool_calls) == 1 else None,
                    }
                )

                if not tool_calls:
                    answer = (assistant.content or "").strip()
                    if not answer:
                        raise AgentRuntimeError(
                            "provider_protocol_error",
                            "Hazırda cavab hazırlamaq mümkün olmadı. Bir qədər sonra yenidən yoxlayın.",
                        )
                    answer = self._guard_product_answer(answer, last_tool_result)
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    self._finalize_trace(
                        trace,
                        status="completed",
                        model_rounds=model_rounds,
                        tool_count=tool_count,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        reasoning_tokens=reasoning_tokens,
                        latency_ms=latency_ms,
                        last_tool_result=last_tool_result,
                        last_retrieval_trace=last_retrieval_trace,
                    )
                    trace["timeline"].append(
                        {
                            "stage": "final_answer",
                            "status": "completed",
                            "used_tools": used_tools,
                        }
                    )
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
                        debug_trace=trace,
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
                    return AgentResult(
                        message_id=message.id,
                        answer=answer,
                        used_tools=used_tools,
                        presentation=build_product_cards(last_tool_result),
                    )

                if not allow_tools or len(tool_calls) != 1:
                    raise AgentRuntimeError(
                        "provider_protocol_error",
                        "Hazırda cavab hazırlamaq mümkün olmadı. Bir qədər sonra yenidən yoxlayın.",
                    )

                call = tool_calls[0]
                tool_count += 1
                tool_name = call.function.name
                raw_arguments: dict[str, Any] = {}
                retrieval_trace: dict[str, Any] | None = None
                try:
                    decoded = json.loads(call.function.arguments)
                    if not isinstance(decoded, dict):
                        raise ValueError
                    raw_arguments = decoded
                    tool_result, retrieval_trace = await self._execute_tool(
                        tool_name,
                        raw_arguments,
                    )
                except (json.JSONDecodeError, ValueError, ToolArgumentsError):
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

                last_tool_result = tool_result
                last_retrieval_trace = retrieval_trace
                trace["timeline"].append(
                    self._tool_trace_event(
                        tool_name=tool_name,
                        arguments=raw_arguments,
                        result=tool_result,
                        retrieval_trace=retrieval_trace,
                    )
                )

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
            await self._mark_failed(
                run,
                exc.code,
                model_rounds,
                tool_count,
                started,
                trace,
                input_tokens,
                output_tokens,
                reasoning_tokens,
            )
            raise
        except ProviderTimeoutError as exc:
            await self._mark_failed(
                run,
                exc.error_type,
                model_rounds,
                tool_count,
                started,
                trace,
                input_tokens,
                output_tokens,
                reasoning_tokens,
            )
            raise AgentRuntimeError(
                "assistant_timeout",
                "Cavabın hazırlanması vaxtı keçdi. Bir qədər sonra yenidən yoxlayın.",
                http_status=504,
            ) from exc
        except ProviderError as exc:
            await self._mark_failed(
                run,
                exc.error_type,
                model_rounds,
                tool_count,
                started,
                trace,
                input_tokens,
                output_tokens,
                reasoning_tokens,
            )
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
            await self._mark_failed(
                run,
                "agent_internal_error",
                model_rounds,
                tool_count,
                started,
                trace,
                input_tokens,
                output_tokens,
                reasoning_tokens,
            )
            raise AgentRuntimeError(
                "assistant_temporarily_unavailable",
                "Hazırda cavab hazırlamaq mümkün olmadı. Bir qədər sonra yenidən yoxlayın.",
            ) from exc

    async def _execute_tool(
        self,
        tool_name: str,
        raw_arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        execute_with_trace = getattr(self.tools, "execute_with_trace", None)
        if callable(execute_with_trace):
            execution = await execute_with_trace(tool_name, raw_arguments)
            return execution.result, execution.debug_trace
        return await self.tools.execute(tool_name, raw_arguments), None

    def _new_trace(self, user_message: str) -> dict[str, Any]:
        source_state: dict[str, Any] = {}
        debug_source_state = getattr(self.tools, "debug_source_state", None)
        if callable(debug_source_state):
            try:
                source_state = debug_source_state()
            except Exception:
                logger.warning("agent.debug_source_state_failed")
        semantic = source_state.get("semantic_qdrant")
        if isinstance(semantic, dict) and semantic.get("configured"):
            semantic["collection"] = self.settings.qdrant_collection_name
        return {
            "trace_version": TRACE_VERSION,
            "detail_level": "full",
            "status": "running",
            "model": {
                "provider": "azure_openai",
                "deployment": self.settings.azure_text_model,
                "reasoning_effort": self.settings.reasoning_effort,
            },
            "data_sources": source_state,
            "timeline": [
                {
                    "stage": "input_validation",
                    "status": "completed",
                    "accepted": True,
                    "message_length": len(user_message),
                }
            ],
            "diagnosis": None,
            "warnings": [],
            "metrics": {},
        }

    @staticmethod
    def _tool_trace_event(
        *,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        retrieval_trace: dict[str, Any] | None,
    ) -> dict[str, Any]:
        items = result.get("items", []) if result.get("status") == "success" else []
        returned_products = [
            {"product_id": item.get("product_id"), "name": item.get("name")}
            for item in items
        ]
        return {
            "stage": tool_name,
            "status": "completed" if result.get("status") == "success" else "failed",
            "arguments": arguments,
            "result": {
                "status": result.get("status"),
                "code": result.get("code"),
                "total": result.get("total"),
                "match_status": result.get("match_status"),
                "strict_total": result.get("strict_total"),
                "relaxed_fields": result.get("relaxed_fields", []),
                "returned_products": returned_products,
            },
            "retrieval": retrieval_trace,
        }

    @staticmethod
    def _guard_product_answer(answer: str, result: dict[str, Any] | None) -> str:
        if not result or result.get("status") != "success":
            return answer
        match_status = result.get("match_status")
        requested_label = str(result.get("requested_label") or "").strip()
        if match_status == "alternatives":
            subject = requested_label or "İstədiyiniz dəqiq məhsul"
            items = result.get("items") if isinstance(result.get("items"), list) else []
            first_name = ""
            if items and isinstance(items[0], dict):
                first_name = str(items[0].get("name") or "").strip()
            recommendation = (
                f" Ən yaxın seçim kimi {first_name} göstərilir."
                if first_name
                else " Aşağıda ən yaxın alternativləri təqdim edirəm."
            )
            return f"{subject} kataloqda tapılmadı.{recommendation}"
        if match_status == "not_found":
            subject = requested_label or "Sorğunuza uyğun məhsul"
            return f"{subject} kataloqda tapılmadı və etibarlı alternativ müəyyən edilmədi."
        return answer

    @staticmethod
    def _finalize_trace(
        trace: dict[str, Any],
        *,
        status: str,
        model_rounds: int,
        tool_count: int,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        latency_ms: int,
        last_tool_result: dict[str, Any] | None,
        last_retrieval_trace: dict[str, Any] | None,
        error_type: str | None = None,
    ) -> None:
        trace["status"] = status
        trace["metrics"] = {
            "model_rounds": model_rounds,
            "tool_count": tool_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "latency_ms": latency_ms,
        }
        if status == "failed":
            trace["diagnosis"] = {
                "code": "run_error",
                "title": "Agent run tamamlanmadı",
                "detail": "Azure və ya agent icrası xəta ilə dayandı.",
                "catalog_checked": tool_count > 0,
                "data_status": "Müəyyən edilə bilmədi",
                "result_count": None,
                "error_type": error_type,
            }
            return
        if last_tool_result is None:
            trace["diagnosis"] = {
                "code": "catalog_not_checked",
                "title": "Model birbaşa cavab verdi",
                "detail": "product_search çağırılmadı və məhsul kataloqu yoxlanılmadı.",
                "catalog_checked": False,
                "data_status": "Yoxlanılmayıb",
                "result_count": None,
            }
            return
        if last_tool_result.get("status") != "success":
            semantic_state = (last_retrieval_trace or {}).get("semantic_state")
            detail = "Tool xətası səbəbindən data nəticəsi müəyyən edilə bilmədi."
            if last_tool_result.get("code") == "product_search_unavailable":
                detail = "Azure embedding və ya Qdrant əlçatan olmadığı üçün axtarış tamamlanmadı."
                trace["warnings"].append(
                    {
                        "code": "semantic_unavailable",
                        "detail": (
                            f"Semantic qat {semantic_state or 'failed'} vəziyyətində idi; "
                            "fallback edilmədi."
                        ),
                    }
                )
            trace["diagnosis"] = {
                "code": "tool_error",
                "title": "Məhsul axtarışı tamamlanmadı",
                "detail": detail,
                "catalog_checked": True,
                "data_status": "Müəyyən edilə bilmədi",
                "result_count": None,
                "error_type": last_tool_result.get("code"),
            }
            return

        items = last_tool_result.get("items", [])
        result_count = len(items)
        match_status = last_tool_result.get("match_status")
        if match_status == "alternatives":
            code = "alternatives_returned"
            title = "Dəqiq məhsul tapılmadı, alternativlər seçildi"
            detail = (
                "Sərt uyğunluq boş qaldı; təhlükəsiz filterlər saxlanılaraq "
                "yaxın alternativlər qaytarıldı."
            )
            data_status = f"Dəqiq: 0; alternativ: {result_count}"
        elif match_status == "not_found":
            code = "not_found"
            title = "Məhsul və etibarlı alternativ tapılmadı"
            detail = "Kataloq yoxlanıldı, lakin sərt uyğunluq və relevance həddini keçən alternativ yoxdur."
            data_status = "Uyğun nəticə yoxdur"
        elif result_count:
            code = "products_found"
            title = "Uyğun məhsullar tapıldı"
            detail = "Axtarış tamamlandı və uyğun məhsullar cavaba ötürüldü."
            data_status = f"Tapılıb: {result_count}"
        elif last_retrieval_trace and last_retrieval_trace.get("exact_filter_conflict"):
            code = "exact_filter_conflict"
            title = "Identifier və filter ziddiyyəti"
            detail = "Məhsul identifier-i tapıldı, lakin verilmiş filter həmin məhsulu istisna etdi."
            data_status = "Uyğun nəticə yoxdur"
        elif last_retrieval_trace and last_retrieval_trace.get("filters") and not last_retrieval_trace.get(
            "filtered_count"
        ):
            code = "no_match_after_filters"
            title = "Filterlərdən sonra məhsul qalmadı"
            detail = "Kataloq yoxlanıldı, lakin bütün filterlərə uyğun məhsul tapılmadı."
            data_status = "Uyğun nəticə yoxdur"
        else:
            code = "no_retrieval_match"
            title = "Axtarış uyğunluğu tapılmadı"
            detail = "Qdrant yoxlanıldı, lakin semantic uyğunluq qaytarılmadı."
            data_status = "Uyğun nəticə yoxdur"
        trace["diagnosis"] = {
            "code": code,
            "title": title,
            "detail": detail,
            "catalog_checked": True,
            "data_status": data_status,
            "result_count": result_count,
            "match_status": match_status,
            "strict_total": last_tool_result.get("strict_total"),
            "relaxed_fields": last_tool_result.get("relaxed_fields", []),
        }
        semantic_state = (last_retrieval_trace or {}).get("semantic_state")
        if semantic_state in {"failed", "not_configured"}:
            trace["warnings"].append(
                {
                    "code": "semantic_unavailable",
                    "detail": f"Semantic qat {semantic_state} vəziyyətində idi; fallback edilmədi.",
                }
            )

    async def _mark_failed(
        self,
        run: AgentRun,
        error_type: str,
        model_rounds: int,
        tool_count: int,
        started: float,
        trace: dict[str, Any],
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
    ) -> None:
        latency_ms = int((time.perf_counter() - started) * 1000)
        self._finalize_trace(
            trace,
            status="failed",
            model_rounds=model_rounds,
            tool_count=tool_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            latency_ms=latency_ms,
            last_tool_result=None,
            last_retrieval_trace=None,
            error_type=error_type,
        )
        trace["timeline"].append(
            {"stage": "run_error", "status": "failed", "error_type": error_type}
        )
        try:
            await self.repository.fail_run(
                run_id=run.id,
                error_type=error_type,
                error_message=error_type,
                model_rounds=model_rounds,
                tool_count=tool_count,
                latency_ms=latency_ms,
                debug_trace=trace,
            )
        except Exception:
            logger.exception("agent.failure_persistence_failed")
