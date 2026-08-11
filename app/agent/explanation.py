from __future__ import annotations

from typing import Any


def build_decision_explanation(
    *,
    history_message_count: int,
    memory_before_revision: int,
    product_plan: dict[str, Any] | None,
    product_result: dict[str, Any] | None,
    product_retrieval: dict[str, Any] | None,
    document_arguments: dict[str, Any] | None,
    document_result: dict[str, Any] | None,
    document_retrieval: dict[str, Any] | None,
    used_tools: list[str],
    warnings: list[dict[str, Any]],
    memory_transition: dict[str, Any],
    status: str = "completed",
    error_type: str | None = None,
    memory_context_enabled: bool = True,
) -> dict[str, Any]:
    basis = _basis(used_tools, warnings, status)
    understood = _understood_request(product_plan, document_arguments)
    memory_refs = sorted(_memory_refs(product_plan))
    context_used = [
        {
            "source": "current_message",
            "detail": "Cari istifadəçi mesajı semantic planın əsas mənbəyidir.",
        }
    ]
    if history_message_count:
        context_used.append(
            {
                "source": "conversation_history",
                "detail": f"{history_message_count} relevant tarixçə mesajı contextə daxil edildi.",
            }
        )
    if memory_refs:
        context_used.append(
            {
                "source": "session_memory",
                "detail": "Yalnız server tərəfindən təsdiqlənmiş memory istinadları istifadə edildi.",
                "memory_ids": memory_refs,
                "revision": memory_before_revision,
            }
        )
    elif memory_before_revision > 0 and memory_context_enabled:
        context_used.append(
            {
                "source": "session_memory",
                "detail": (
                    "Server tərəfindən təsdiqlənmiş sessiya yaddaşı təhlükəsiz context kimi "
                    "ötürüldü; semantic memory istinadı istifadə edilmədi."
                ),
                "memory_ids": [],
                "revision": memory_before_revision,
            }
        )

    steps: list[dict[str, Any]] = []
    if product_plan:
        steps.append(
            {
                "code": "semantic_plan",
                "status": "completed",
                "detail": "Sorğu entity, şərt, üstünlük və fakt suallarına ayrıldı.",
            }
        )
    if product_retrieval:
        steps.append(
            {
                "code": "catalog_resolution",
                "status": str(product_retrieval.get("semantic_state") or "completed"),
                "detail": "Entity və facet dəyərləri kataloq metadata-sı ilə yoxlanıldı.",
            }
        )
    if product_result:
        steps.append(
            {
                "code": "product_decision",
                "status": str(product_result.get("match_status") or product_result.get("status")),
                "detail": _product_outcome_detail(product_result),
            }
        )
    if document_result:
        steps.append(
            {
                "code": "document_decision",
                "status": str(document_result.get("match_status") or document_result.get("status")),
                "detail": "Cavab üçün yalnız seçilmiş document chunk-ları əsas götürüldü.",
            }
        )
    if not used_tools and status == "completed":
        steps.append(
            {
                "code": "direct_answer",
                "status": "completed",
                "detail": "Model tool çağırmadan cari mesaj və təhlükəsiz söhbət konteksti ilə cavab verdi.",
            }
        )
    if status == "failed":
        steps.append(
            {
                "code": "run_error",
                "status": "failed",
                "detail": f"Run tamamlanmadı: {error_type or 'unknown_error'}.",
            }
        )
    if warnings:
        steps.append(
            {
                "code": "runtime_guard",
                "status": "applied",
                "detail": "Runtime təhlükəsizlik və protokol guard-larını tətbiq etdi.",
            }
        )

    evidence = {
        "product_ids": _product_ids(product_result),
        "document_source_ids": _document_ids(document_result),
        "canonical_fields": _canonical_fields(product_plan),
        "retrieval_executed": bool(
            (product_retrieval or {}).get("retrieval_executed")
        ),
        "unavailable_requested_values": list(
            (product_result or {}).get("unavailable_requested_values")
            or (product_retrieval or {}).get("unavailable_requested_values")
            or []
        ),
    }
    limitations: list[str] = []
    if not used_tools:
        limitations.append("Kataloq və document mənbələri yoxlanılmadı.")
    if any(item.get("code") == "degraded_safe_response" for item in warnings):
        limitations.append("Provider cavabı deterministik təhlükəsiz cavabla əvəz edildi.")
    if product_result and isinstance(product_result.get("clarification"), dict):
        limitations.append("Unikal və təhlükəsiz nəticə üçün istifadəçi dəqiqləşdirməsi tələb olunur.")

    narrative = _narrative(
        basis=basis,
        history_message_count=history_message_count,
        memory_before_revision=memory_before_revision,
        memory_context_enabled=memory_context_enabled,
        product_plan=product_plan,
        product_result=product_result,
        product_retrieval=product_retrieval,
        document_arguments=document_arguments,
        document_result=document_result,
        warnings=warnings,
        memory_transition=memory_transition,
        status=status,
        error_type=error_type,
    )
    return {
        "version": 2,
        "basis": basis,
        "summary": _summary(basis, product_result, document_result, status),
        "narrative": narrative,
        "understood_request": understood,
        "context_used": context_used,
        "decision_path": steps,
        "evidence": evidence,
        "outcome": _outcome(product_result, document_result, status, error_type),
        "memory_effect": memory_transition,
        "limitations": limitations,
    }


def build_failed_explanation(
    *,
    history_message_count: int,
    memory_revision: int,
    error_type: str,
    memory_transition: dict[str, Any],
    memory_context_enabled: bool = True,
) -> dict[str, Any]:
    return build_decision_explanation(
        history_message_count=history_message_count,
        memory_before_revision=memory_revision,
        product_plan=None,
        product_result=None,
        product_retrieval=None,
        document_arguments=None,
        document_result=None,
        document_retrieval=None,
        used_tools=[],
        warnings=[],
        memory_transition=memory_transition,
        status="failed",
        error_type=error_type,
        memory_context_enabled=memory_context_enabled,
    )


def _basis(used_tools: list[str], warnings: list[dict[str, Any]], status: str) -> str:
    if status == "failed":
        return "runtime_guard"
    if any(item.get("code") == "degraded_safe_response" for item in warnings):
        return "runtime_guard"
    has_product = "product_search" in used_tools
    has_document = "document_search" in used_tools
    if has_product and has_document:
        return "combined_tools"
    if has_product:
        return "product_search"
    if has_document:
        return "document_search"
    return "direct_answer"


def _understood_request(
    plan: dict[str, Any] | None,
    document_arguments: dict[str, Any] | None,
) -> dict[str, Any]:
    if not plan:
        return {
            "operation": "document_lookup" if document_arguments else "direct_answer",
            "entities": [],
            "hard_constraints": [],
            "preferences": [],
            "fact_questions": [],
        }
    return {
        "operation": plan.get("operation"),
        "memory_action": plan.get("memory_action", "replace"),
        "entities": [
            {
                key: item.get(key)
                for key in ("entity_id", "raw_text", "state", "identifier_type", "memory_refs")
                if item.get(key) is not None
            }
            for item in plan.get("entities", [])
            if isinstance(item, dict)
        ],
        "hard_constraints": _predicate_summaries(plan.get("filter_expression")),
        "preferences": _predicate_summaries(plan.get("preference_expression")),
        "fact_questions": [
            {
                key: item.get(key)
                for key in ("field", "operator", "value", "unit", "memory_refs")
                if item.get(key) is not None
            }
            for item in plan.get("fact_questions", [])
            if isinstance(item, dict)
        ],
        "memory_removals": [
            {
                key: item.get(key)
                for key in ("memory_id", "evidence_text")
                if item.get(key) is not None
            }
            for item in plan.get("memory_removals", [])
            if isinstance(item, dict)
        ],
    }


def _predicate_summaries(expression: Any) -> list[dict[str, Any]]:
    return [
        {
            key: item.get(key)
            for key in ("field", "operator", "value", "strength", "unit", "memory_refs")
            if item.get(key) is not None
        }
        for item in _predicate_dicts(expression)
    ]


def _predicate_dicts(expression: Any) -> list[dict[str, Any]]:
    if not isinstance(expression, dict):
        return []
    result: list[dict[str, Any]] = []
    if expression.get("kind") == "predicate" and isinstance(expression.get("predicate"), dict):
        result.append(expression["predicate"])
    for child in expression.get("expressions") or []:
        result.extend(_predicate_dicts(child))
    for key in ("expression", "primary", "secondary"):
        result.extend(_predicate_dicts(expression.get(key)))
    return result


def _memory_refs(plan: dict[str, Any] | None) -> set[str]:
    if not plan:
        return set()
    result = {str(item) for item in plan.get("referenced_memory_ids", [])}
    for entity in plan.get("entities", []):
        if isinstance(entity, dict):
            result.update(str(item) for item in entity.get("memory_refs", []))
    for predicate in [
        *_predicate_dicts(plan.get("filter_expression")),
        *_predicate_dicts(plan.get("preference_expression")),
    ]:
        result.update(str(item) for item in predicate.get("memory_refs", []))
    for question in plan.get("fact_questions", []):
        if isinstance(question, dict):
            result.update(str(item) for item in question.get("memory_refs", []))
    return result


def _canonical_fields(plan: dict[str, Any] | None) -> list[str]:
    if not plan:
        return []
    values = [item.get("field") for item in _predicate_dicts(plan.get("filter_expression"))]
    values.extend(item.get("field") for item in _predicate_dicts(plan.get("preference_expression")))
    values.extend(
        item.get("field")
        for item in plan.get("fact_questions", [])
        if isinstance(item, dict)
    )
    return list(dict.fromkeys(str(item) for item in values if item))


def _product_ids(result: dict[str, Any] | None) -> list[str]:
    if not result:
        return []
    values = [str(item) for item in result.get("display_product_ids", []) if item]
    requested = result.get("requested_item")
    if isinstance(requested, dict) and requested.get("product_id"):
        values.append(str(requested["product_id"]))
    return list(dict.fromkeys(values))[:4]


def _document_ids(result: dict[str, Any] | None) -> list[str]:
    if not result:
        return []
    return list(
        dict.fromkeys(
            str(item.get("document_id") or item.get("chunk_id"))
            for item in result.get("chunks", [])
            if isinstance(item, dict) and (item.get("document_id") or item.get("chunk_id"))
        )
    )[:5]


def _product_outcome_detail(result: dict[str, Any]) -> str:
    if isinstance(result.get("clarification"), dict):
        return "Sorğu təhlükəsiz şəkildə icra edilə bilmədiyi üçün aydınlaşdırma tələb olundu."
    status = result.get("match_status")
    if status == "exact_match":
        return "Dəqiq kataloq məhsulu filtersiz yoxlandı və uyğun tapıldı."
    if status == "exact_conflict":
        return "Dəqiq məhsul mövcuddur, lakin bir və ya daha çox hard şərtlə ziddiyyət var."
    if status == "alternatives":
        return "Dəqiq nəticə tapılmadı; qorunan hard şərtlərlə yaxın alternativlər seçildi."
    if status == "clarification_required":
        return "Sorğu təhlükəsiz şəkildə icra edilə bilmədiyi üçün aydınlaşdırma tələb olundu."
    if status == "not_found":
        return "Etibarlı nəticə və alternativ tapılmadı."
    return "Kataloq filterləri və retrieval sıralaması ilə uyğun məhsullar seçildi."


def _summary(
    basis: str,
    product_result: dict[str, Any] | None,
    document_result: dict[str, Any] | None,
    status: str,
    ) -> str:
    if status == "failed":
        return "Run xəta ilə dayandı; sessiya yaddaşı dəyişdirilmədi."
    if product_result:
        if isinstance(product_result.get("clarification"), dict):
            return "Sorğu aydınlaşdırma tələb etdi; məhsul axtarışı tamamlanmadı."
        return _product_outcome_detail(product_result)
    if document_result:
        return "Cavab seçilmiş document nəticələrinə əsasən hazırlandı."
    if basis == "runtime_guard":
        return "Cavab runtime təhlükəsizlik guard-ı ilə qorundu."
    return "Model cari mesaj və təhlükəsiz söhbət konteksti ilə birbaşa cavab verdi."


def _outcome(
    product_result: dict[str, Any] | None,
    document_result: dict[str, Any] | None,
    status: str,
    error_type: str | None,
) -> dict[str, Any]:
    if status == "failed":
        return {"status": "failed", "error_type": error_type}
    if product_result:
        return {
            "status": product_result.get("status"),
            "match_status": product_result.get("match_status"),
            "display_product_ids": product_result.get("display_product_ids", []),
            "recommended_product_id": product_result.get("recommended_product_id"),
            "constraint_conflicts": product_result.get("constraint_conflicts", []),
            "relaxed_fields": product_result.get("relaxed_fields", []),
            "clarification": product_result.get("clarification"),
            "unavailable_requested_values": product_result.get(
                "unavailable_requested_values", []
            ),
        }
    if document_result:
        return {
            "status": document_result.get("status"),
            "match_status": document_result.get("match_status"),
            "source_ids": _document_ids(document_result),
        }
    return {"status": "completed", "match_status": None}


_FIELD_LABELS = {
    "brand": "brend",
    "category_id": "kateqoriya",
    "color_code": "rəng",
    "in_stock": "stok",
    "model": "model",
    "model_family": "məhsul ailəsi",
    "price": "qiymət",
    "ram_gb": "RAM",
    "sale_price": "qiymət",
    "stock_status": "stok",
    "storage_gb": "yaddaş",
}


def _narrative(
    *,
    basis: str,
    history_message_count: int,
    memory_before_revision: int,
    memory_context_enabled: bool,
    product_plan: dict[str, Any] | None,
    product_result: dict[str, Any] | None,
    product_retrieval: dict[str, Any] | None,
    document_arguments: dict[str, Any] | None,
    document_result: dict[str, Any] | None,
    warnings: list[dict[str, Any]],
    memory_transition: dict[str, Any],
    status: str,
    error_type: str | None,
) -> str:
    if status == "failed":
        return (
            "İstifadəçinin sorğusu emal edilərkən run tamamlanmadı. "
            f"Xəta növü {error_type or 'müəyyən edilməmiş xəta'} kimi qeydə alındı və "
            "sessiya yaddaşı dəyişdirilmədi."
        )
    if product_result is not None:
        sentences = [_request_sentence(product_plan, product_result)]
        context_sentence = _context_sentence(
            product_plan,
            history_message_count=history_message_count,
            memory_before_revision=memory_before_revision,
            memory_context_enabled=memory_context_enabled,
        )
        if context_sentence:
            sentences.append(context_sentence)

        clarification = product_result.get("clarification")
        retrieval_executed = bool((product_retrieval or {}).get("retrieval_executed"))
        if isinstance(clarification, dict):
            sentences.append(_clarification_reason(clarification))
            sentences.append(
                "Buna görə real məhsul axtarışı başlamadı və sistem istifadəçidən "
                "sorğunu dəqiqləşdirməsini istədi."
                if not retrieval_executed
                else "Namizədlərin yoxlanması unikal nəticə vermədiyi üçün sistem "
                "istifadəçidən sorğunu dəqiqləşdirməsini istədi."
            )
        else:
            sentences.append(_result_sentence(product_result, retrieval_executed))
            unavailable = _unavailable_sentence(product_result, product_retrieval)
            if unavailable:
                sentences.append(unavailable)

        memory_sentence = _memory_effect_sentence(memory_transition)
        if memory_sentence:
            sentences.append(memory_sentence)
        return " ".join(item for item in sentences if item)

    if document_result is not None:
        topic = str((document_arguments or {}).get("query") or "sənəd məlumatı").strip()
        if document_result.get("match_status") == "found":
            return (
                f"İstifadəçi {topic} barədə məlumat istəyir. Sistem sənəd mənbələrini yoxladı "
                "və cavabı yalnız tapılmış mənbə hissələrinə əsasən hazırladı."
            )
        return (
            f"İstifadəçi {topic} barədə məlumat istəyir. Sistem sənəd mənbələrini yoxladı, "
            "lakin uyğun məlumat tapılmadı."
        )

    if basis == "runtime_guard" or warnings:
        return (
            "İstifadəçinin sorğusu cavablandırıldı, lakin provider və ya cavab formatı "
            "təhlükəsizlik yoxlamasından keçmədiyi üçün runtime təhlükəsiz cavab tətbiq etdi. "
            "Kataloq faktı təsdiqlənməyibsə əlavə məhsul iddiası verilmədi."
        )
    return (
        "İstifadəçi məhsul kataloqunun yoxlanmasını tələb etməyən birbaşa mesaj göndərdi. "
        "Model cari mesaj və təhlükəsiz söhbət konteksti ilə cavab verdi; kataloq və sənəd "
        "mənbələri yoxlanılmadı."
    )


def _request_sentence(
    plan: dict[str, Any] | None,
    result: dict[str, Any] | None,
) -> str:
    if not plan:
        return "İstifadəçi məhsulla bağlı sorğu göndərdi."
    product_names: dict[str, str] = {}
    if result:
        result_products = [*result.get("items", []), result.get("requested_item")]
        for product in result_products:
            if not isinstance(product, dict) or not product.get("product_id"):
                continue
            name = str(product.get("name") or "").strip()
            if name:
                product_names[str(product["product_id"])] = name
    resolution_ids = {
        str(item.get("entity_id")): str(item.get("product_id"))
        for item in (result or {}).get("resolved_entities", [])
        if isinstance(item, dict) and item.get("entity_id") and item.get("product_id")
    }
    selected = []
    for item in plan.get("entities", []):
        if not isinstance(item, dict) or item.get("state", "selected") != "selected":
            continue
        resolved_name = product_names.get(resolution_ids.get(str(item.get("entity_id")), ""))
        raw_text = str(item.get("raw_text") or "").strip()
        if resolved_name or raw_text:
            selected.append(resolved_name or raw_text)
    selected = list(dict.fromkeys(selected))[:3]
    operation = plan.get("operation")
    if operation == "compare" and selected:
        return f"İstifadəçi {_join_words(selected)} məhsullarını müqayisə etmək istəyir."
    if selected and plan.get("fact_questions"):
        return f"İstifadəçi {_join_words(selected)} haqqında məlumat istəyir."
    if selected:
        return f"İstifadəçi {_join_words(selected)} seçiminə uyğun məhsul istəyir."
    constraints = _constraint_phrases(plan)
    if constraints:
        return f"İstifadəçi {_join_words(constraints)} şərtlərinə uyğun məhsul istəyir."
    return "İstifadəçi kataloqdan uyğun məhsul tapılmasını istəyir."


def _context_sentence(
    plan: dict[str, Any] | None,
    *,
    history_message_count: int,
    memory_before_revision: int,
    memory_context_enabled: bool,
) -> str:
    memory_refs = _memory_refs(plan)
    if memory_refs:
        return (
            "Əvvəlki təsdiqlənmiş məhsul və şərtlər sessiya yaddaşındakı istinadlarla "
            "davam etdirildi."
        )
    if memory_before_revision > 0 and memory_context_enabled:
        return (
            "Sessiya yaddaşı kontekstə verildi, lakin bu sorğunun planında əvvəlki "
            "yaddaşa istinad istifadə edilmədi."
        )
    if history_message_count:
        return (
            "Söhbət tarixçəsi kontekstə daxil edildi, lakin əvvəlki məhsul şərtləri "
            "təsdiqlənmiş sessiya yaddaşı ilə əsaslandırılmadı."
        )
    return ""


def _clarification_reason(clarification: dict[str, Any]) -> str:
    reasons = set(str(clarification.get("reason") or "").split(","))
    if "evidence_not_grounded" in reasons:
        return (
            "Plan daxilindəki bəzi məhsul və ya şərtlərin mənbəyi cari mesajdan və "
            "ya sessiya yaddaşından təsdiqlənə bilmədi."
        )
    if "invalid_memory_reference" in reasons:
        return "Plan cari sessiyaya aid olmayan və ya köhnəlmiş yaddaş istinadından istifadə etdi."
    if "ambiguous_entity" in reasons or "ambiguous_facet_value" in reasons:
        return "Kataloqda bir-birinə çox yaxın bir neçə mümkün məhsul və ya facet dəyəri tapıldı."
    if "unmapped_required_value" in reasons:
        return "Məcburi şərtin dəyəri kataloqdakı unikal facet dəyərinə çevrilə bilmədi."
    if "model_requested_clarification" in reasons:
        return "İstifadəçinin nəzərdə tutduğu seçim cümlədən təhlükəsiz şəkildə müəyyən edilmədi."
    return "Semantic plan təhlükəsiz və unikal kataloq sorğusuna çevrilə bilmədi."


def _result_sentence(result: dict[str, Any], retrieval_executed: bool) -> str:
    status = result.get("match_status")
    items = [item for item in result.get("items", []) if isinstance(item, dict)]
    names = [str(item.get("name") or "").strip() for item in items if item.get("name")][:3]
    requested = result.get("requested_item")
    requested_name = (
        str(requested.get("name") or "").strip()
        if isinstance(requested, dict)
        else str(result.get("requested_label") or "").strip()
    )
    if status == "exact_match":
        if len(names) > 1:
            return (
                "Müqayisə edilən məhsullar kataloqdan ID üzrə təsdiqləndi: "
                f"{_join_words(names)}."
            )
        return f"Kataloqda dəqiq uyğunluq tapıldı: {requested_name or _join_words(names)}."
    if status == "exact_conflict":
        conflicts = [str(item) for item in result.get("constraint_conflicts", []) if item]
        detail = _join_words(conflicts) if conflicts else "verilən məcburi şərtlərlə ziddiyyət"
        return (
            f"{requested_name or 'Dəqiq məhsul'} kataloqda mövcuddur, lakin {detail} aşkarlandı."
        )
    if status == "alternatives":
        return (
            "Dəqiq seçim tapılmadı; qorunan məcburi şərtlərlə "
            f"{len(items)} yaxın alternativ seçildi."
        )
    if status == "not_found":
        return (
            "Kataloq axtarışı icra edildi, lakin uyğun məhsul və etibarlı alternativ tapılmadı."
            if retrieval_executed
            else "Məhsul axtarışı icra edilmədiyi üçün kataloq nəticəsi müəyyən edilmədi."
        )
    if status == "clarification_required":
        return "Sorğu aydınlaşdırılmadan məhsul axtarışı icra edilmədi."
    return (
        f"Kataloq axtarışı tamamlandı və {len(items)} uyğun məhsul seçildi."
        if retrieval_executed
        else "Kataloq nəticəsi müəyyən edilmədi."
    )


def _unavailable_sentence(
    result: dict[str, Any],
    retrieval: dict[str, Any] | None,
) -> str:
    unavailable = (
        result.get("unavailable_requested_values")
        or (retrieval or {}).get("unavailable_requested_values")
        or []
    )
    values = []
    for item in unavailable:
        if not isinstance(item, dict):
            continue
        value = str(item.get("evidence_text") or item.get("value") or "").strip()
        if value:
            values.append(value)
    values = list(dict.fromkeys(values))[:3]
    if not values:
        return ""
    return (
        f"{_join_words(values)} seçimi kataloq facet-lərində mövcud olmadığı üçün "
        "nəticəyə daxil edilmədi."
    )


def _memory_effect_sentence(transition: dict[str, Any]) -> str:
    if transition.get("revision_after") == transition.get("revision_before"):
        return "Sessiya yaddaşındakı son təsdiqlənmiş vəziyyət dəyişdirilmədi."
    action = transition.get("action")
    if action == "merge":
        return "Nəticə və davam edən seçimlər sessiya yaddaşındakı vəziyyətlə birləşdirildi."
    if action == "replace":
        return "Yeni müstəqil məhsul məqsədi sessiya yaddaşındakı aktiv vəziyyəti əvəz etdi."
    return "Sessiya yaddaşındakı təsdiqlənmiş vəziyyət qorundu."


def _constraint_phrases(plan: dict[str, Any]) -> list[str]:
    result = []
    for predicate in _predicate_dicts(plan.get("filter_expression"))[:4]:
        field = _FIELD_LABELS.get(str(predicate.get("field")), str(predicate.get("field")))
        value = predicate.get("value")
        unit = str(predicate.get("unit") or "").strip()
        rendered = f"{value} {unit}".strip()
        result.append(f"{field}: {rendered}")
    return result


def _join_words(values: list[str]) -> str:
    cleaned = [value for value in values if value]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return f"{', '.join(cleaned[:-1])} və {cleaned[-1]}"
