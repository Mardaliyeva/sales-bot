from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from app.embeddings.azure import AzureEmbeddingClient, EmbeddingError
from app.embeddings.cache import EmbeddingCache


def _response(vectors: list[list[float]], *, reverse: bool = False) -> httpx.Response:
    data = [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)]
    if reverse:
        data.reverse()
    return httpx.Response(200, json={"data": data})


def test_embedding_request_is_batched_and_response_order_is_restored() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        vectors = [[float(len(text)), float(index)] for index, text in enumerate(payload["input"])]
        return _response(vectors, reverse=True)

    client = AzureEmbeddingClient(
        endpoint="https://example.openai.azure.com/",
        api_key="secret",
        deployment="text-embedding-3-large",
        dimensions=2,
        batch_size=2,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    try:
        result = client.embed(["a", "bb", "ccc"])
    finally:
        client.close()

    assert result == [[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]
    assert len(requests) == 2
    assert requests[0] == {
        "input": ["a", "bb"],
        "model": "text-embedding-3-large",
        "dimensions": 2,
        "encoding_format": "float",
    }


def test_embedding_endpoint_is_normalized() -> None:
    captured_url = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_url
        captured_url = str(request.url)
        return _response([[0.1, 0.2]])

    with AzureEmbeddingClient(
        endpoint="https://example.services.ai.azure.com",
        api_key="secret",
        deployment="deployment-name",
        dimensions=2,
        transport=httpx.MockTransport(handler),
    ) as client:
        client.embed(["query"])

    assert captured_url == "https://example.services.ai.azure.com/openai/v1/embeddings"


def test_retryable_error_is_retried_without_exposing_provider_body() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": "provider-secret-detail"})
        return _response([[0.1, 0.2]])

    with AzureEmbeddingClient(
        endpoint="https://example.openai.azure.com",
        api_key="secret",
        deployment="deployment-name",
        dimensions=2,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    ) as client:
        assert client.embed(["query"]) == [[0.1, 0.2]]

    assert calls == 2


def test_auth_error_is_not_retried() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": "provider-secret-detail"})

    with AzureEmbeddingClient(
        endpoint="https://example.openai.azure.com",
        api_key="secret",
        deployment="deployment-name",
        dimensions=2,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    ) as client:
        with pytest.raises(EmbeddingError) as captured:
            client.embed(["query"])

    assert calls == 1
    assert captured.value.error_type == "embedding_auth_error"
    assert "provider-secret-detail" not in str(captured.value)


def test_invalid_vector_dimensions_are_rejected() -> None:
    with AzureEmbeddingClient(
        endpoint="https://example.openai.azure.com",
        api_key="secret",
        deployment="deployment-name",
        dimensions=2,
        transport=httpx.MockTransport(lambda _: _response([[0.1]])),
    ) as client:
        with pytest.raises(EmbeddingError) as captured:
            client.embed(["query"])

    assert captured.value.error_type == "embedding_dimension_error"


def test_incomplete_batch_response_is_rejected() -> None:
    with AzureEmbeddingClient(
        endpoint="https://example.openai.azure.com",
        api_key="secret",
        deployment="deployment-name",
        dimensions=2,
        transport=httpx.MockTransport(lambda _: _response([[0.1, 0.2]])),
    ) as client:
        with pytest.raises(EmbeddingError) as captured:
            client.embed(["first", "second"])

    assert captured.value.error_type == "embedding_protocol_error"


def test_timeout_is_retried_up_to_max_attempts() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=request)

    with AzureEmbeddingClient(
        endpoint="https://example.openai.azure.com",
        api_key="secret",
        deployment="deployment-name",
        dimensions=2,
        max_attempts=3,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    ) as client:
        with pytest.raises(EmbeddingError) as captured:
            client.embed(["query"])

    assert calls == 3
    assert captured.value.error_type == "embedding_timeout"


def _cache_path() -> Path:
    return Path.cwd() / f".test-embedding-cache-{uuid4().hex}.sqlite3"


def test_cached_embedding_skips_remote_call() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response([[0.1, 0.2]])

    cache_path = _cache_path()
    try:
        cache = EmbeddingCache(cache_path)
        with AzureEmbeddingClient(
            endpoint="https://example.openai.azure.com",
            api_key="secret",
            deployment="deployment-name",
            dimensions=2,
            cache=cache,
            transport=httpx.MockTransport(handler),
        ) as client:
            first = client.embed(["same text"], text_version="v1")
            second = client.embed(["same text"], text_version="v1")
    finally:
        cache_path.unlink(missing_ok=True)

    assert first == second == [[0.1, 0.2]]
    assert calls == 1


def test_refresh_ignores_cached_embedding() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response([[float(calls), 0.0]])

    cache_path = _cache_path()
    try:
        with AzureEmbeddingClient(
            endpoint="https://example.openai.azure.com",
            api_key="secret",
            deployment="deployment-name",
            dimensions=2,
            cache=EmbeddingCache(cache_path),
            transport=httpx.MockTransport(handler),
        ) as client:
            first = client.embed(["same text"])
            second = client.embed(["same text"], refresh=True)
    finally:
        cache_path.unlink(missing_ok=True)

    assert first == [[1.0, 0.0]]
    assert second == [[2.0, 0.0]]
    assert calls == 2
