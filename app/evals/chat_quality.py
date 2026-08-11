from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.config import PROJECT_ROOT

CASES_PATH = PROJECT_ROOT / "data" / "evals" / "chat_quality_50.json"
LEAK_MARKERS = (
    "filter_payload",
    "embedding_text",
    "system_prompt",
    "customer_azure_openai_api_key",
    "qdrant_api_key",
    "database_url",
)


class QualityCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    query: str
    expected_product_id: str | None = None
    expected_result_kinds: list[str] = Field(default_factory=list)
    expected_stock: str | None = None
    expected_highlight: str | None = None
    expected_no_presentation: bool = False
    security: bool = False
    repeat: int = Field(default=1, ge=1, le=3)


def _is_json_answer(answer: str) -> bool:
    text = answer.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip() if len(lines) >= 3 else ""
    if not text or text[0] not in "[{":
        return False
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


def _presentation_products(presentation: dict[str, Any]) -> list[dict[str, Any]]:
    products = [item for item in presentation.get("items", []) if isinstance(item, dict)]
    requested = presentation.get("requested_item")
    if isinstance(requested, dict):
        products.append(requested)
    return products


def _evaluate(case: QualityCase, status: int, payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if status != 200:
        return [f"HTTP {status}"]
    answer = str(payload.get("answer") or "")
    if not answer:
        failures.append("empty_answer")
    if _is_json_answer(answer):
        failures.append("public_json")
    lower_answer = answer.casefold()
    if any(marker in lower_answer for marker in LEAK_MARKERS):
        failures.append("internal_marker_leak")

    presentation = payload.get("presentation")
    if case.expected_no_presentation and presentation is not None:
        failures.append("unexpected_presentation")
    if case.expected_result_kinds or case.expected_product_id:
        if not isinstance(presentation, dict):
            failures.append("missing_presentation")
            return failures
        result_kind = str(presentation.get("result_kind") or "matches")
        if case.expected_result_kinds and result_kind not in case.expected_result_kinds:
            failures.append(f"result_kind={result_kind}")
        products = _presentation_products(presentation)
        if case.expected_product_id and not any(
            item.get("product_id") == case.expected_product_id for item in products
        ):
            failures.append("expected_product_missing")
        if case.expected_stock and not any(
            item.get("product_id") == case.expected_product_id
            and item.get("stock_status") == case.expected_stock
            for item in products
        ):
            failures.append("stock_mismatch")
        if case.expected_highlight and not any(
            case.expected_highlight in str(highlight)
            for item in products
            for highlight in item.get("highlights", [])
        ):
            failures.append("canonical_highlight_missing")
        items = presentation.get("items", [])
        if isinstance(items, list) and len(items) > 3:
            failures.append("more_than_three_cards")
        recommended = presentation.get("recommended_product_id")
        if recommended and not any(item.get("product_id") == recommended for item in products):
            failures.append("recommendation_not_displayed")
        if result_kind == "alternatives" and any(
            not item.get("differences") for item in items if isinstance(item, dict)
        ):
            failures.append("alternative_difference_missing")
    return failures


async def run_quality(base_url: str, case_ids: set[str] | None = None) -> int:
    cases = [QualityCase.model_validate(item) for item in json.loads(CASES_PATH.read_text(encoding="utf-8"))]
    if case_ids:
        cases = [case for case in cases if case.id in case_ids]
    results: list[tuple[str, int, list[str]]] = []
    five_xx = 0
    security_leaks = 0
    concurrency = int(os.getenv("SALES_BOT_QA_CONCURRENCY", "4"))
    semaphore = asyncio.Semaphore(max(1, min(concurrency, 8)))
    result_lock = asyncio.Lock()

    async with httpx.AsyncClient(base_url=base_url, timeout=90) as client:
        async def run_case(case: QualityCase) -> None:
            nonlocal five_xx, security_leaks
            for attempt in range(1, case.repeat + 1):
                async with semaphore:
                    try:
                        session_response = await client.post("/v1/sessions", json={})
                        if session_response.status_code != 201:
                            failures = [f"session_http={session_response.status_code}"]
                            status_code = session_response.status_code
                        else:
                            session_id = session_response.json()["session_id"]
                            response = await client.post(
                                "/v1/chat",
                                json={"session_id": session_id, "message": case.query},
                            )
                            status_code = response.status_code
                            try:
                                payload = response.json()
                            except json.JSONDecodeError:
                                payload = {}
                            failures = _evaluate(case, status_code, payload)
                    except (httpx.HTTPError, KeyError, TypeError) as exc:
                        status_code = 0
                        failures = [f"runner_error={type(exc).__name__}"]
                async with result_lock:
                    five_xx += int(status_code >= 500)
                    security_leaks += int(
                        case.security
                        and any(
                            failure in {"public_json", "internal_marker_leak"}
                            for failure in failures
                        )
                    )
                    results.append((case.id, attempt, failures))
                    print(
                        f"{case.id}#{attempt}: "
                        f"{'PASS' if not failures else 'FAIL ' + ','.join(failures)}",
                        flush=True,
                    )

        await asyncio.gather(*(run_case(case) for case in cases))

    case_pass = {
        case.id: all(not failures for case_id, _, failures in results if case_id == case.id)
        for case in cases
    }
    passed = sum(case_pass.values())
    required_passes = 48 if len(cases) == 50 else len(cases)
    gate = passed >= required_passes and five_xx == 0 and security_leaks == 0
    print(
        f"Gate: {'KEÇDİ' if gate else 'UĞURSUZ'}; cases={passed}/{len(cases)}; "
        f"security_leaks={security_leaks}; http_5xx={five_xx}"
    )
    return 0 if gate else 1


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="50 suallıq live chat quality acceptance suite")
    parser.add_argument("--base-url")
    parser.add_argument("--case", action="append", dest="case_ids")
    args = parser.parse_args()
    port = os.getenv("SALES_BOT_API_PORT", "8001")
    base_url = (args.base_url or os.getenv("SALES_BOT_API_URL") or f"http://127.0.0.1:{port}").rstrip("/")
    return asyncio.run(run_quality(base_url, set(args.case_ids or [])))


if __name__ == "__main__":
    raise SystemExit(main())
