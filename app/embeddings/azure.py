from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from app.config import PROJECT_ROOT, Settings
from app.embeddings.cache import EmbeddingCache

logger = logging.getLogger(__name__)

DEFAULT_DIMENSIONS = 3072
DEFAULT_BATCH_SIZE = 32
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TEXT_VERSION = "name_description_v2"
DEFAULT_CACHE_PATH = PROJECT_ROOT / ".cache" / "azure_embeddings.sqlite3"


class EmbeddingError(RuntimeError):
    def __init__(self, error_type: str, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code


class EmbeddingItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    data: list[EmbeddingItem]


class AzureEmbeddingClient:
    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        deployment: str,
        dimensions: int = DEFAULT_DIMENSIONS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        cache: EmbeddingCache | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("Embedding dimensions müsbət olmalıdır")
        if batch_size <= 0:
            raise ValueError("Embedding batch size müsbət olmalıdır")
        if max_attempts <= 0:
            raise ValueError("Embedding cəhd sayı müsbət olmalıdır")
        normalized_endpoint = self._normalize_endpoint(endpoint)
        if not api_key.strip() or api_key in {"YOUR_AZURE_API_KEY", "CHANGE_ME"}:
            raise EmbeddingError("embedding_configuration_error", "Azure embedding API açarı yoxdur")
        if not deployment.strip():
            raise EmbeddingError("embedding_configuration_error", "Azure embedding deployment adı yoxdur")

        self.deployment = deployment.strip()
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.cache = cache
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=normalized_endpoint,
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
            headers={"api-key": api_key.strip(), "Content-Type": "application/json"},
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        cache_path: Path = DEFAULT_CACHE_PATH,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> AzureEmbeddingClient:
        endpoint = settings.customer_azure_openai_endpoint
        api_key = settings.customer_azure_openai_api_key
        if endpoint is None or endpoint == "YOUR_AZURE_ENDPOINT":
            raise EmbeddingError("embedding_configuration_error", "Azure embedding endpoint yoxdur")
        if api_key is None:
            raise EmbeddingError("embedding_configuration_error", "Azure embedding API açarı yoxdur")
        return cls(
            endpoint=endpoint,
            api_key=api_key.get_secret_value(),
            deployment=settings.azure_embedding_model,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            cache=EmbeddingCache(cache_path),
            transport=transport,
            sleep=sleep,
        )

    def embed(
        self,
        texts: Sequence[str],
        *,
        text_version: str = DEFAULT_TEXT_VERSION,
        refresh: bool = False,
    ) -> list[list[float]]:
        normalized = list(texts)
        if not normalized:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in normalized):
            raise EmbeddingError("embedding_input_error", "Embedding mətni boş ola bilməz")
        if not text_version.strip():
            raise EmbeddingError("embedding_input_error", "Embedding mətn versiyası boş ola bilməz")

        results: list[list[float] | None] = [None] * len(normalized)
        missing: list[tuple[int, str]] = []
        for index, text in enumerate(normalized):
            cached = None
            if self.cache is not None and not refresh:
                cached = self.cache.get(
                    text=text,
                    deployment=self.deployment,
                    dimensions=self.dimensions,
                    text_version=text_version,
                )
            if cached is None:
                missing.append((index, text))
            else:
                results[index] = cached

        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = self._request_batch([text for _, text in batch])
            for (result_index, text), vector in zip(batch, vectors, strict=True):
                results[result_index] = vector
                if self.cache is not None:
                    self.cache.put(
                        text=text,
                        deployment=self.deployment,
                        dimensions=self.dimensions,
                        text_version=text_version,
                        vector=vector,
                    )

        if any(vector is None for vector in results):
            raise EmbeddingError("embedding_protocol_error", "Bəzi embedding nəticələri yaranmadı")
        return [vector for vector in results if vector is not None]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> AzureEmbeddingClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _request_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "input": texts,
            "model": self.deployment,
            "dimensions": self.dimensions,
            "encoding_format": "float",
        }
        for attempt in range(1, self.max_attempts + 1):
            logger.info(
                "embedding.request_started",
                extra={"model": self.deployment, "batch_size": len(texts), "attempt": attempt},
            )
            try:
                response = self._client.post("/embeddings", json=payload)
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                if attempt < self.max_attempts:
                    self._sleep(self._retry_delay(attempt, None))
                    continue
                error_type = (
                    "embedding_timeout"
                    if isinstance(exc, httpx.TimeoutException)
                    else "embedding_network_error"
                )
                raise EmbeddingError(error_type, "Azure embedding xidməti əlçatan deyil") from exc

            if response.status_code == 429 or response.status_code >= 500:
                if attempt < self.max_attempts:
                    self._sleep(self._retry_delay(attempt, response.headers.get("Retry-After")))
                    continue
            if response.is_error:
                raise self._http_error(response.status_code)

            vectors = self._parse_response(response, expected_count=len(texts))
            logger.info(
                "embedding.request_completed",
                extra={"model": self.deployment, "batch_size": len(texts), "status": "success"},
            )
            return vectors

        raise EmbeddingError("embedding_unavailable", "Azure embedding xidməti əlçatan deyil")

    def _parse_response(self, response: httpx.Response, *, expected_count: int) -> list[list[float]]:
        try:
            parsed = EmbeddingResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise EmbeddingError("embedding_protocol_error", "Azure embedding cavabı etibarsızdır") from exc
        if len(parsed.data) != expected_count:
            raise EmbeddingError("embedding_protocol_error", "Azure embedding nəticələrinin sayı uyğun deyil")
        ordered = sorted(parsed.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(expected_count)):
            raise EmbeddingError("embedding_protocol_error", "Azure embedding indeksləri uyğun deyil")
        vectors = [item.embedding for item in ordered]
        if any(len(vector) != self.dimensions for vector in vectors):
            raise EmbeddingError("embedding_dimension_error", "Azure embedding ölçüsü uyğun deyil")
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise EmbeddingError("embedding_protocol_error", "Azure embedding qeyri-sonlu dəyər qaytardı")
        return vectors

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        normalized = endpoint.strip().rstrip("/")
        if not normalized.startswith("https://"):
            raise EmbeddingError("embedding_configuration_error", "Azure embedding endpoint HTTPS olmalıdır")
        if normalized.endswith("/openai/v1"):
            return normalized
        if normalized.endswith("/openai"):
            return f"{normalized}/v1"
        return f"{normalized}/openai/v1"

    @staticmethod
    def _retry_delay(attempt: int, retry_after: str | None) -> float:
        if retry_after is not None:
            try:
                return min(max(float(retry_after), 0.0), DEFAULT_TIMEOUT_SECONDS)
            except ValueError:
                pass
        return min(0.25 * (2 ** (attempt - 1)), 2.0)

    @staticmethod
    def _http_error(status_code: int) -> EmbeddingError:
        if status_code in {401, 403}:
            error_type = "embedding_auth_error"
        elif status_code == 400:
            error_type = "embedding_bad_request"
        elif status_code == 429:
            error_type = "embedding_rate_limit"
        elif status_code >= 500:
            error_type = "embedding_unavailable"
        else:
            error_type = "embedding_http_error"
        return EmbeddingError(
            error_type,
            f"Azure embedding request statusu: {status_code}",
            status_code=status_code,
        )
