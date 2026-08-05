from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path


class EmbeddingCache:
    """Small local cache that keeps Azure calls resumable and repeatable."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @staticmethod
    def key_for(*, text: str, deployment: str, dimensions: int, text_version: str) -> str:
        payload = {
            "deployment": deployment,
            "dimensions": dimensions,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text_version": text_version,
        }
        encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get(
        self,
        *,
        text: str,
        deployment: str,
        dimensions: int,
        text_version: str,
    ) -> list[float] | None:
        key = self.key_for(
            text=text,
            deployment=deployment,
            dimensions=dimensions,
            text_version=text_version,
        )
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT vector_json FROM embedding_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        try:
            vector = json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(vector, list) or len(vector) != dimensions:
            return None
        try:
            return [float(value) for value in vector]
        except (TypeError, ValueError):
            return None

    def put(
        self,
        *,
        text: str,
        deployment: str,
        dimensions: int,
        text_version: str,
        vector: list[float],
    ) -> None:
        if len(vector) != dimensions:
            raise ValueError("Cache-ə yazılan embedding ölçüsü uyğun deyil")
        key = self.key_for(
            text=text,
            deployment=deployment,
            dimensions=dimensions,
            text_version=text_version,
        )
        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        vector_json = json.dumps(vector, ensure_ascii=True, separators=(",", ":"))
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO embedding_cache (
                    cache_key, text_sha256, deployment, dimensions, text_version, vector_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET vector_json = excluded.vector_json
                """,
                (key, text_sha256, deployment, dimensions, text_version, vector_json),
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                cache_key TEXT PRIMARY KEY,
                text_sha256 TEXT NOT NULL,
                deployment TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                text_version TEXT NOT NULL,
                vector_json TEXT NOT NULL
            )
            """
        )
        connection.commit()
        return connection
