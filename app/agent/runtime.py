from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any

from app.agent.context_builder import build_context, build_system_message
from app.agent.explanation import build_decision_explanation, build_failed_explanation
from app.agent.memory import (
    load_session_memory,
    memory_context_payload,
    preserved_transition,
    semantic_memory_hash,
    update_session_memory,
)
from app.agent.presentation import build_product_cards
from app.agent.prompt import prompt_debug_metadata, prompt_hash
from app.agent.types import AgentResult, AgentRuntimeError
from app.config import Settings
from app.db.models import AgentRun, ChatSession
from app.db.repositories import ConversationRepository
from app.llm.azure_client import AzureChatClient, ProviderError, ProviderTimeoutError
from app.tools.registry import ToolArgumentsError, ToolRegistry, UnknownToolError

logger = logging.getLogger(__name__)
TRACE_VERSION = 6
SEMANTIC_PLAN_CACHE_SIZE = 256


def _plan_memory_references(plan: dict[str, Any] | None) -> set[str]:
    if not plan:
        return set()
    references = {str(item) for item in plan.get("referenced_memory_ids", []) if item}

    def collect_expression(expression: Any) -> None:
        if not isinstance(expression, dict):
            return
        predicate = expression.get("predicate")
        if isinstance(predicate, dict):
            references.update(str(item) for item in predicate.get("memory_refs", []) if item)
        for child in expression.get("expressions") or []:
            collect_expression(child)
        for key in ("expression", "primary", "secondary"):
            collect_expression(expression.get(key))

    for entity in plan.get("entities", []):
        if isinstance(entity, dict):
            references.update(str(item) for item in entity.get("memory_refs", []) if item)
    for expression_name in (
        "selection_expression",
        "filter_expression",
        "preference_expression",
    ):
        collect_expression(plan.get(expression_name))
    for question in plan.get("fact_questions", []):
        if isinstance(question, dict):
            references.update(str(item) for item in question.get("memory_refs", []) if item)
    return references


class AgentRuntime:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: ConversationRepository,
        llm: AzureChatClient,
        tools: ToolRegistry,
        runtime_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.llm = llm
        self.tools = tools
        self.runtime_metadata = runtime_metadata or {}
        self._semantic_plan_cache: OrderedDict[
            str,
            tuple[str, float, dict[str, Any]],
        ] = OrderedDict()

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
        last_product_result: dict[str, Any] | None = None
        last_product_retrieval_trace: dict[str, Any] | None = None
        last_product_plan: dict[str, Any] | None = None
        last_document_result: dict[str, Any] | None = None
        last_document_retrieval_trace: dict[str, Any] | None = None
        last_document_arguments: dict[str, Any] | None = None
        last_tool_name: str | None = None
        blocked_tool_names: set[str] = set()
        trace = self._new_trace(user_message)
        session_memory = load_session_memory(session.context or {})
        trace["memory_transition"] = preserved_transition(session_memory)
        trace["memory_transition"]["cache_expires_at"] = (
            session.expires_at.isoformat() if session.expires_at else None
        )
        trace["continuation_context_before"] = session_memory.continuation_summary
        trace["continuation_context_after"] = session_memory.continuation_summary
        history_message_count = 0

        try:
            history = await self.repository.get_final_history(
                session.id,
                exclude_run_id=run.id,
                limit=self.settings.history_message_limit,
            )
            history_message_count = len(history)
            messages = build_context(
                session_context=session.context or {},
                history=history,
                user_message=user_message,
                include_session_memory=self.settings.session_memory_context_enabled,
                prompt_phase="tool",
                modular_prompt_enabled=bool(self.settings.modular_prompt_enabled),
            )
            semantic_cache_key = self._semantic_cache_key(
                user_message=user_message,
                history=history,
                session_context=session.context or {},
                model=session.model,
            )
            semantic_cache_owner = str(session.id)
            semantic_cache_expiry = (
                session.expires_at.timestamp()
                if session.expires_at
                else time.time() + (self.settings.session_ttl_hours * 3600)
            )
            self._prune_semantic_plan_cache(now=time.time())
            trace["timeline"].append(
                {
                    "stage": "context_build",
                    "status": "completed",
                    "history_message_count": len(history),
                    "session_product_context_count": len(
                        (session.context or {}).get("last_product_ids", [])
                    ),
                    "memory_revision": session_memory.revision,
                    "memory_context_enabled": self.settings.session_memory_context_enabled,
                }
            )

            for round_number in range(1, self.settings.max_model_rounds + 1):
                model_rounds = round_number
                allow_tools = round_number < self.settings.max_model_rounds and tool_count < min(
                    session.max_tool_count, self.settings.max_tool_count
                )
                tool_specs = self.tools.specs() if allow_tools else []
                if blocked_tool_names:
                    tool_specs = [
                        spec
                        for spec in tool_specs
                        if spec.get("function", {}).get("name") not in blocked_tool_names
                    ]
                allow_tools = allow_tools and bool(tool_specs)
                prompt_phase = "tool" if allow_tools else "response"
                messages[0] = {
                    "role": "system",
                    "content": build_system_message(
                        session_context=session.context or {},
                        include_session_memory=self.settings.session_memory_context_enabled,
                        prompt_phase=prompt_phase,
                        modular_prompt_enabled=bool(self.settings.modular_prompt_enabled),
                    ),
                }
                trace["prompt"]["active_phase"] = prompt_phase

                response = await self.llm.chat(
                    messages=[dict(message) for message in messages],
                    tools=tool_specs if allow_tools else None,
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
                provider_finish_reason = response.first_choice.finish_reason
                protocol_issue = None
                if provider_finish_reason == "content_filter":
                    protocol_issue = "content_filter"
                elif not tool_calls and not (assistant.content or "").strip():
                    protocol_issue = "empty_content"
                elif tool_calls and (not allow_tools or len(tool_calls) != 1):
                    protocol_issue = "invalid_tool_protocol"
                if protocol_issue:
                    retry_status = "failed"
                    try:
                        safe_messages = [dict(item) for item in messages]
                        safe_messages[0] = {
                            "role": "system",
                            "content": build_system_message(
                                session_context=session.context or {},
                                include_session_memory=self.settings.session_memory_context_enabled,
                                prompt_phase="safe_final",
                                modular_prompt_enabled=bool(self.settings.modular_prompt_enabled),
                            ),
                        }
                        trace["prompt"]["active_phase"] = "safe_final"
                        safe_retry = await self.llm.chat(
                            messages=safe_messages,
                            tools=None,
                            tool_choice="none",
                            request_id=str(run.request_id),
                            model_round=round_number,
                        )
                        provider_response_id = safe_retry.id
                        input_tokens += safe_retry.usage.prompt_tokens
                        output_tokens += safe_retry.usage.completion_tokens
                        reasoning_tokens += safe_retry.usage.reasoning_tokens
                        retry_choice = safe_retry.first_choice
                        retry_answer = (retry_choice.message.content or "").strip()
                        if (
                            retry_answer
                            and not retry_choice.message.tool_calls
                            and retry_choice.finish_reason != "content_filter"
                        ):
                            assistant = retry_choice.message
                            retry_status = "completed"
                        else:
                            raise ValueError("safe retry provider protocol invalid")
                    except Exception:
                        assistant = assistant.model_copy(
                            update={
                                "content": self._safe_degraded_answer(
                                    last_product_result,
                                    last_document_result,
                                ),
                                "tool_calls": None,
                            }
                        )
                    tool_calls = []
                    trace["timeline"].append(
                        {
                            "stage": "safe_response_retry",
                            "status": retry_status,
                            "tools_allowed": False,
                            "provider_finish_reason": provider_finish_reason,
                        }
                    )
                    trace["warnings"].append(
                        {
                            "code": "degraded_safe_response",
                            "detail": "Provider cavabı təhlükəsiz deterministik cavabla əvəz edildi.",
                            "provider_finish_reason": provider_finish_reason,
                            "protocol_issue": protocol_issue,
                        }
                    )
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
                        "prompt_phase": prompt_phase,
                    }
                )

                if not tool_calls:
                    answer = (assistant.content or "").strip()
                    answer = self._guard_product_answer(answer, last_product_result)
                    answer = self._guard_document_answer(answer, last_document_result)
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
                        last_tool_name=last_tool_name,
                        last_product_result=last_product_result,
                        last_product_retrieval_trace=last_product_retrieval_trace,
                        last_document_result=last_document_result,
                        last_document_retrieval_trace=last_document_retrieval_trace,
                    )
                    try:
                        memory_update = update_session_memory(
                            session_memory,
                            request_id=str(run.request_id),
                            product_plan=last_product_plan,
                            product_result=last_product_result,
                            document_arguments=last_document_arguments,
                            document_result=last_document_result,
                            max_bytes=self.settings.session_memory_max_bytes,
                        )
                        memory_update.transition["cache_expires_at"] = (
                            session.expires_at.isoformat() if session.expires_at else None
                        )
                        decision_explanation = build_decision_explanation(
                            history_message_count=history_message_count,
                            memory_before_revision=session_memory.revision,
                            product_plan=last_product_plan,
                            product_result=last_product_result,
                            product_retrieval=last_product_retrieval_trace,
                            document_arguments=last_document_arguments,
                            document_result=last_document_result,
                            document_retrieval=last_document_retrieval_trace,
                            used_tools=used_tools,
                            warnings=trace["warnings"],
                            memory_transition=memory_update.transition,
                            memory_context_enabled=self.settings.session_memory_context_enabled,
                        )
                    except Exception:
                        logger.exception("agent.decision_explanation_failed")
                        memory_update = update_session_memory(
                            session_memory,
                            request_id=str(run.request_id),
                            product_plan=None,
                            product_result=None,
                            document_arguments=None,
                            document_result=None,
                            max_bytes=self.settings.session_memory_max_bytes,
                        )
                        memory_update.transition["cache_expires_at"] = (
                            session.expires_at.isoformat() if session.expires_at else None
                        )
                        trace["warnings"].append(
                            {
                                "code": "decision_explanation_failed",
                                "detail": "Əsaslandırma qurulmadı; sessiya yaddaşı dəyişdirilmədi.",
                            }
                        )
                        decision_explanation = build_decision_explanation(
                            history_message_count=history_message_count,
                            memory_before_revision=session_memory.revision,
                            product_plan=None,
                            product_result=None,
                            product_retrieval=None,
                            document_arguments=None,
                            document_result=None,
                            document_retrieval=None,
                            used_tools=used_tools,
                            warnings=trace["warnings"],
                            memory_transition=memory_update.transition,
                            memory_context_enabled=self.settings.session_memory_context_enabled,
                        )
                    trace["decision_explanation"] = decision_explanation
                    trace["memory_transition"] = memory_update.transition
                    trace["continuation_context_after"] = (
                        memory_update.memory.continuation_summary
                    )
                    trace["metrics"].update(
                        {
                            "memory_revision": memory_update.memory.revision,
                            "memory_size_bytes": memory_update.transition["size_bytes"],
                            "memory_refs_used": len(
                                _plan_memory_references(last_product_plan)
                            ),
                            "memory_action": memory_update.transition["action"],
                            "explanation_build_status": (
                                "failed_safe"
                                if any(
                                    item.get("code") == "decision_explanation_failed"
                                    for item in trace["warnings"]
                                )
                                else "completed"
                            ),
                        }
                    )
                    trace["timeline"].append(
                        {
                            "stage": "session_memory_update",
                            "status": "completed",
                            **memory_update.transition,
                        }
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
                        session_memory=memory_update.memory.model_dump(
                            mode="json", exclude_none=True
                        ),
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
                        presentation=build_product_cards(last_product_result),
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
                model_supplied_query: str | None = None
                try:
                    decoded = json.loads(call.function.arguments)
                    if not isinstance(decoded, dict):
                        raise ValueError
                    raw_arguments = decoded
                    if tool_name == "product_search":
                        cached_entry = self._semantic_plan_cache.get(semantic_cache_key)
                        cached_plan = (
                            cached_entry[2]
                            if cached_entry is not None
                            and cached_entry[0] == semantic_cache_owner
                            and cached_entry[1] > time.time()
                            else None
                        )
                        if cached_entry is not None and cached_plan is None:
                            self._semantic_plan_cache.pop(semantic_cache_key, None)
                        if cached_plan is not None and (
                            "operation" in raw_arguments or "entities" in raw_arguments
                        ):
                            raw_arguments = dict(cached_plan)
                            self._semantic_plan_cache.move_to_end(semantic_cache_key)
                            trace["timeline"].append(
                                {
                                    "stage": "semantic_plan_cache",
                                    "status": "hit",
                                    "cache_key": semantic_cache_key,
                                }
                            )
                        elif "operation" in raw_arguments or "entities" in raw_arguments:
                            trace["timeline"].append(
                                {
                                    "stage": "semantic_plan_cache",
                                    "status": "miss",
                                    "cache_key": semantic_cache_key,
                                }
                            )
                        supplied_query = raw_arguments.get("query")
                        model_supplied_query = (
                            supplied_query if isinstance(supplied_query, str) else None
                        )
                        raw_arguments["query"] = user_message[:500]
                        if "operation" in raw_arguments or "entities" in raw_arguments:
                            context_ids = [
                                str(product_id)
                                for product_id in (session.context or {}).get(
                                    "last_product_ids", []
                                )
                            ]
                            context_ids.extend(session_memory.product.display_product_ids)
                            context_ids.extend(
                                entity.product_id
                                for entity in session_memory.product.entities
                                if entity.product_id
                            )
                            if session_memory.pending_intent is not None:
                                context_ids.extend(
                                    session_memory.pending_intent.state.display_product_ids
                                )
                                context_ids.extend(
                                    entity.product_id
                                    for entity in session_memory.pending_intent.state.entities
                                    if entity.product_id
                                )
                            raw_arguments["context_product_ids"] = list(
                                dict.fromkeys(context_ids)
                            )[:3]
                            raw_arguments["context_memory"] = (
                                memory_context_payload(session_memory)
                                if self.settings.session_memory_context_enabled
                                else {}
                            )
                    tool_result, retrieval_trace = await self._execute_tool(
                        tool_name,
                        raw_arguments,
                    )
                    if tool_name == "product_search" and (
                        "operation" in raw_arguments or "entities" in raw_arguments
                    ):
                        plan_content_hash = tool_result.get("canonical_query_hash")
                        tool_result["canonical_query_hash"] = semantic_cache_key
                        if retrieval_trace is not None:
                            retrieval_trace = {
                                **retrieval_trace,
                                "semantic_plan_content_hash": plan_content_hash,
                                "canonical_query_hash": semantic_cache_key,
                            }
                    if retrieval_trace is not None and model_supplied_query != raw_arguments.get(
                        "query"
                    ):
                        retrieval_trace = {
                            **retrieval_trace,
                            "runtime_query_correction": {
                                "action": "restored_original_user_query",
                                "model_supplied_query": model_supplied_query,
                                "canonical_query": raw_arguments["query"],
                            },
                        }
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
                        "message": "Tool əməliyyatı vaxtında tamamlanmadı.",
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
                        "message": "Tool əməliyyatı tamamlanmadı.",
                    }

                last_tool_name = tool_name
                if tool_name == "product_search":
                    last_product_result = tool_result
                    last_product_retrieval_trace = retrieval_trace
                    canonical_plan = (retrieval_trace or {}).get(
                        "canonical_semantic_plan"
                    )
                    last_product_plan = (
                        dict(canonical_plan)
                        if isinstance(canonical_plan, dict)
                        else {
                            key: value
                            for key, value in raw_arguments.items()
                            if key not in {"context_memory", "context_product_ids"}
                        }
                    )
                elif tool_name == "document_search":
                    last_document_result = tool_result
                    last_document_retrieval_trace = retrieval_trace
                    last_document_arguments = dict(raw_arguments)
                trace["timeline"].append(
                    self._tool_trace_event(
                        tool_name=tool_name,
                        arguments=raw_arguments,
                        result=tool_result,
                        retrieval_trace=retrieval_trace,
                    )
                )

                if tool_name in {"product_search", "document_search"} and tool_name not in used_tools:
                    used_tools.append(tool_name)
                if tool_name == "product_search" and tool_result.get("status") == "success":
                    if isinstance(tool_result.get("clarification"), dict):
                        blocked_tool_names.add("product_search")
                    if (
                        ("operation" in raw_arguments or "entities" in raw_arguments)
                        and not (retrieval_trace or {}).get("semantic_plan_invalid")
                    ):
                        self._semantic_plan_cache[semantic_cache_key] = (
                            semantic_cache_owner,
                            semantic_cache_expiry,
                            dict(raw_arguments),
                        )
                        self._semantic_plan_cache.move_to_end(semantic_cache_key)
                        while len(self._semantic_plan_cache) > SEMANTIC_PLAN_CACHE_SIZE:
                            self._semantic_plan_cache.popitem(last=False)
                    display_ids = tool_result.get("display_product_ids")
                    last_product_ids = (
                        [str(product_id) for product_id in display_ids]
                        if isinstance(display_ids, list)
                        else [item["product_id"] for item in tool_result.get("items", [])[:3]]
                    )

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
                        "content": json.dumps(
                            self._tool_result_for_model(tool_result),
                            ensure_ascii=False,
                        ),
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

    def _semantic_cache_key(
        self,
        *,
        user_message: str,
        history: list[Any],
        session_context: dict[str, Any],
        model: str,
    ) -> str:
        debug_source_state = getattr(self.tools, "debug_source_state", None)
        source_state = debug_source_state() if callable(debug_source_state) else {}
        catalog = source_state.get("product_catalog_json", {})
        payload = {
            "query": user_message,
            "relevant_history": [
                {"role": item.role, "content": item.content} for item in history
            ],
            "context_product_ids": session_context.get("last_product_ids", []),
            "focused_product_id": session_context.get("focused_product_id"),
            "session_memory_hash": (
                semantic_memory_hash(load_session_memory(session_context))
                if self.settings.session_memory_context_enabled
                else None
            ),
            "model": model,
            "planner_prompt_hash": prompt_hash(
                "tool",
                modular=bool(self.settings.modular_prompt_enabled),
            ),
            "product_tool_schema_hash": self._product_tool_schema_hash(),
            "catalog_schema_version": catalog.get("dataset_version"),
            "catalog_checksum": catalog.get("catalog_checksum"),
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _product_tool_schema_hash(self) -> str | None:
        try:
            product_spec = next(
                spec
                for spec in self.tools.specs()
                if spec.get("function", {}).get("name") == "product_search"
            )
        except (StopIteration, TypeError):
            return None
        return hashlib.sha256(
            json.dumps(
                product_spec,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _prune_semantic_plan_cache(self, *, now: float) -> None:
        expired = [
            key
            for key, (_, expires_at, _) in self._semantic_plan_cache.items()
            if expires_at <= now
        ]
        for key in expired:
            self._semantic_plan_cache.pop(key, None)

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
            "runtime": dict(self.runtime_metadata),
            "prompt": {
                **prompt_debug_metadata(
                    modular=bool(self.settings.modular_prompt_enabled)
                ),
                "active_phase": "tool",
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
        chunks = result.get("chunks", []) if result.get("status") == "success" else []
        returned_chunks = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "title": chunk.get("title"),
                "heading": chunk.get("heading"),
                "score": chunk.get("score"),
            }
            for chunk in chunks
            if isinstance(chunk, dict)
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
                "constraint_conflicts": result.get("constraint_conflicts", []),
                "argument_corrections": result.get("argument_corrections", []),
                "recommended_product_id": result.get("recommended_product_id"),
                "display_product_ids": result.get("display_product_ids", []),
                "operation": result.get("operation"),
                "resolved_entities": result.get("resolved_entities", []),
                "entity_results": result.get("entity_results", []),
                "canonical_query_hash": result.get("canonical_query_hash"),
                "clarification": result.get("clarification"),
                "unavailable_requested_values": result.get(
                    "unavailable_requested_values", []
                ),
                "requested_product": (
                    {
                        "product_id": result["requested_item"].get("product_id"),
                        "name": result["requested_item"].get("name"),
                    }
                    if isinstance(result.get("requested_item"), dict)
                    else None
                ),
                "returned_products": returned_products,
                "returned_chunks": returned_chunks,
            },
            "retrieval": retrieval_trace,
        }

    @staticmethod
    def _guard_product_answer(answer: str, result: dict[str, Any] | None) -> str:
        if AgentRuntime._is_json_answer(answer) or AgentRuntime._contains_internal_payload_terms(
            answer
        ):
            return AgentRuntime._safe_degraded_answer(result, None)
        if not result or result.get("status") != "success":
            return answer
        clarification = result.get("clarification")
        if isinstance(clarification, dict):
            question = str(clarification.get("question") or "").strip()
            return question or "Sorğunu bir qədər dəqiqləşdirə bilərsiniz?"
        match_status = result.get("match_status")
        requested_label = str(result.get("requested_label") or "").strip()
        if match_status == "exact_conflict":
            requested_item = result.get("requested_item")
            product_name = (
                str(requested_item.get("name") or "").strip()
                if isinstance(requested_item, dict)
                else requested_label
            )
            conflicts = [
                str(conflict)
                for conflict in result.get("constraint_conflicts", [])
                if isinstance(conflict, str) and conflict.strip()
            ]
            conflict_text = "; ".join(conflicts) or "verilən şərtlə uyğun gəlmir"
            alternatives = result.get("items") if isinstance(result.get("items"), list) else []
            suffix = (
                " Aşağıda şərtlərə daha yaxın alternativlər ayrıca göstərilir."
                if alternatives
                else " Etibarlı alternativ müəyyən edilmədi."
            )
            return f"{product_name or 'Dəqiq məhsul'} kataloqda mövcuddur, lakin {conflict_text}.{suffix}"
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
    def _guard_document_answer(answer: str, result: dict[str, Any] | None) -> str:
        if not result:
            return answer
        if result.get("status") != "success":
            if result.get("code") == "document_search_unavailable":
                return "Sənəd axtarışı müvəqqəti əlçatan deyil. Bir qədər sonra yenidən yoxlayın."
            return answer
        if result.get("match_status") == "not_found":
            return "Yüklənmiş sənədlərdə bu məlumatı tapa bilmədim."
        return answer

    @staticmethod
    def _is_json_answer(answer: str) -> bool:
        stripped = answer.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            stripped = "\n".join(lines[1:-1]).strip() if len(lines) >= 3 else ""
        if not stripped or stripped[0] not in "[{":
            return False
        try:
            json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return False
        return True

    @staticmethod
    def _contains_internal_payload_terms(answer: str) -> bool:
        normalized = answer.casefold()
        return any(
            marker in normalized
            for marker in (
                "filter_payload",
                "embedding_text",
                "system_prompt",
                "qdrant_api_key",
                "customer_azure_openai_api_key",
                "database_url",
            )
        )

    @staticmethod
    def _deterministic_product_answer(result: dict[str, Any]) -> str:
        clarification = result.get("clarification")
        if isinstance(clarification, dict):
            return str(clarification.get("question") or "").strip() or (
                "Sorğunu bir qədər dəqiqləşdirə bilərsiniz?"
            )
        status = result.get("match_status")
        if status in {"alternatives", "not_found", "exact_conflict"}:
            return AgentRuntime._guard_product_answer("", result)
        display_ids = {
            str(product_id)
            for product_id in result.get("display_product_ids", [])
            if product_id
        }
        items = [item for item in result.get("items", []) if isinstance(item, dict)]
        recommended_id = str(result.get("recommended_product_id") or "")
        selected = next(
            (item for item in items if str(item.get("product_id")) == recommended_id),
            items[0] if items else None,
        )
        if selected is None:
            return "Sorğunuza uyğun məhsul məlumatı müəyyən edilmədi."
        if display_ids and str(selected.get("product_id")) not in display_ids:
            return "Sorğunuza uyğun məhsullar kartlarda göstərilir."
        stock = "stokdadır" if selected.get("stock_status") == "in_stock" else "stokda deyil"
        return (
            f"{selected.get('name')} {selected.get('sale_price')} "
            f"{selected.get('currency', 'AZN')} qiymətindədir və {stock}."
        )

    @staticmethod
    def _safe_degraded_answer(
        product_result: dict[str, Any] | None,
        document_result: dict[str, Any] | None,
    ) -> str:
        if product_result and product_result.get("status") == "success":
            return AgentRuntime._deterministic_product_answer(product_result)
        if document_result and document_result.get("status") == "success":
            if document_result.get("match_status") == "not_found":
                return "Yüklənmiş sənədlərdə bu məlumatı tapa bilmədim."
            return "Sənəd məlumatı tapıldı, lakin hazırda təhlükəsiz yekun cavab hazırlamaq mümkün olmadı."
        return (
            "Daxili məlumatı və tool payload-unu JSON kimi göstərə bilmərəm. "
            "Sualınızı normal mətnlə yazın, qısa cavab verim."
        )

    @staticmethod
    def _tool_result_for_model(result: dict[str, Any]) -> dict[str, Any]:
        if result.get("status") != "success":
            return result
        if "chunks" in result:
            return {**result, "chunks": result.get("chunks", [])[:5]}
        display_ids = {
            str(product_id)
            for product_id in result.get("display_product_ids", [])
            if product_id
        }
        visible_items = [
            item
            for item in result.get("items", [])
            if isinstance(item, dict) and str(item.get("product_id")) in display_ids
        ]
        return {**result, "items": visible_items[:3]}

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
        last_tool_name: str | None,
        last_product_result: dict[str, Any] | None,
        last_product_retrieval_trace: dict[str, Any] | None,
        last_document_result: dict[str, Any] | None,
        last_document_retrieval_trace: dict[str, Any] | None,
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
                "catalog_checked": last_product_result is not None,
                "documents_checked": last_document_result is not None,
                "data_status": "Müəyyən edilə bilmədi",
                "result_count": None,
                "error_type": error_type,
            }
            return
        if last_product_result is None and last_document_result is None:
            trace["diagnosis"] = {
                "code": "catalog_not_checked",
                "title": "Model birbaşa cavab verdi",
                "detail": "product_search və document_search çağırılmadı; data mənbələri yoxlanılmadı.",
                "catalog_checked": False,
                "documents_checked": False,
                "data_status": "Yoxlanılmayıb",
                "result_count": None,
            }
            return

        if last_tool_name == "document_search" and last_document_result is not None:
            document_result = last_document_result
            document_trace = last_document_retrieval_trace or {}
            if document_result.get("status") != "success":
                trace["diagnosis"] = {
                    "code": "document_search_failed",
                    "title": "Sənəd axtarışı tamamlanmadı",
                    "detail": "Azure embedding və ya Qdrant document collection əlçatan olmadı.",
                    "catalog_checked": last_product_result is not None,
                    "documents_checked": True,
                    "data_status": "Müəyyən edilə bilmədi",
                    "result_count": None,
                    "error_type": document_result.get("code"),
                }
                trace["warnings"].append(
                    {
                        "code": "document_semantic_unavailable",
                        "detail": (
                            f"Document semantic qat {document_trace.get('semantic_state') or 'failed'} "
                            "vəziyyətində idi; fallback edilmədi."
                        ),
                    }
                )
                return
            chunks = document_result.get("chunks", [])
            result_count = len(chunks) if isinstance(chunks, list) else 0
            found = document_result.get("match_status") == "found" and result_count > 0
            trace["diagnosis"] = {
                "code": "document_chunks_found" if found else "no_document_match",
                "title": "Uyğun sənəd hissələri tapıldı" if found else "Sənədlərdə uyğun məlumat tapılmadı",
                "detail": (
                    "Document Qdrant axtarışı tamamlandı və seçilmiş chunk-lar modelə ötürüldü."
                    if found
                    else "Qdrant yoxlanıldı, lakin relevance həddini keçən sənəd hissəsi olmadı."
                ),
                "catalog_checked": last_product_result is not None,
                "documents_checked": True,
                "data_status": f"Tapılıb: {result_count}" if found else "Uyğun nəticə yoxdur",
                "result_count": result_count,
                "match_status": document_result.get("match_status"),
            }
            return

        last_tool_result = last_product_result
        last_retrieval_trace = last_product_retrieval_trace
        if last_tool_result is None:
            trace["diagnosis"] = {
                "code": "document_not_checked",
                "title": "Sənəd mənbəyi yoxlanılmadı",
                "detail": "document_search çağırılmadı.",
                "catalog_checked": False,
                "documents_checked": False,
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
                "documents_checked": last_document_result is not None,
                "data_status": "Müəyyən edilə bilmədi",
                "result_count": None,
                "error_type": last_tool_result.get("code"),
            }
            return

        items = last_tool_result.get("items", [])
        result_count = len(items)
        match_status = last_tool_result.get("match_status")
        if isinstance(last_tool_result.get("clarification"), dict):
            code = "clarification_required"
            title = "Sorğu dəqiqləşdirilməlidir"
            detail = "Semantic plan təhlükəsiz və unikal kataloq sorğusuna çevrilmədi; kart göstərilmədi."
            data_status = "Aydınlaşdırma gözlənilir"
        elif match_status == "exact_conflict":
            code = "exact_filter_conflict"
            title = "Dəqiq məhsul mövcuddur, lakin şərtlə ziddiyyət var"
            detail = "Identifier filtersiz tapıldı; konfliktlər və alternativlər ayrıca qaytarıldı."
            data_status = f"Dəqiq məhsul mövcuddur; alternativ: {result_count}"
        elif match_status == "alternatives":
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
            "documents_checked": last_document_result is not None,
            "data_status": data_status,
            "result_count": result_count,
            "match_status": match_status,
            "strict_total": last_tool_result.get("strict_total"),
            "relaxed_fields": last_tool_result.get("relaxed_fields", []),
            "operation": last_tool_result.get("operation"),
            "canonical_query_hash": last_tool_result.get("canonical_query_hash"),
            "clarification": last_tool_result.get("clarification"),
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
            last_tool_name=None,
            last_product_result=None,
            last_product_retrieval_trace=None,
            last_document_result=None,
            last_document_retrieval_trace=None,
            error_type=error_type,
        )
        trace["timeline"].append(
            {"stage": "run_error", "status": "failed", "error_type": error_type}
        )
        transition = trace.get("memory_transition") or {
            "revision_before": 0,
            "revision_after": 0,
            "action": "preserve",
            "changed_ids": [],
            "removed_ids": [],
            "size_bytes": 0,
        }
        history_event = next(
            (
                item
                for item in trace.get("timeline", [])
                if item.get("stage") == "context_build"
            ),
            {},
        )
        trace["decision_explanation"] = build_failed_explanation(
            history_message_count=int(history_event.get("history_message_count") or 0),
            memory_revision=int(transition.get("revision_before") or 0),
            error_type=error_type,
            memory_transition=transition,
            memory_context_enabled=bool(history_event.get("memory_context_enabled")),
        )
        trace["metrics"].update(
            {
                "memory_revision": transition.get("revision_after", 0),
                "memory_size_bytes": transition.get("size_bytes", 0),
                "memory_refs_used": 0,
                "memory_action": "preserve",
                "explanation_build_status": "completed",
            }
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
