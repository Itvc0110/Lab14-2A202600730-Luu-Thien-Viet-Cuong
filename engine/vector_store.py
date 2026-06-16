from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from engine.real_config import get_runtime_config

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDING_CACHE_PATH = ROOT / "data" / "embedding_cache_openrouter.jsonl"
DEFAULT_VECTOR_INDEX_PATH = ROOT / "data" / "vector_index_openrouter.jsonl"


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class MockEmbedder:
    """Deterministic Day07-style fallback embedder for tests and local dry runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[index] += sign
        return normalize_vector(vector)


class OpenRouterEmbedder:
    """OpenRouter embedding API helper, matching the Day07 vector retrieval path."""

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter embeddings.")
        config = get_runtime_config()
        self.api_url = f"{config.openrouter_api_base.rstrip('/')}/embeddings"
        self._backend_name = f"openrouter:{model_name}"

    def __call__(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model_name, "input": text}, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(3):
            request = urllib.request.Request(
                self.api_url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    data = json.loads(response.read().decode("utf-8"))
                embedding = data["data"][0]["embedding"]
                return normalize_vector([float(value) for value in embedding])
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code not in {429, 500, 502, 503, 504}:
                    break
                time.sleep(2**attempt)
            except urllib.error.URLError as error:
                last_error = error
                time.sleep(2**attempt)
        raise RuntimeError(f"OpenRouter embedding request failed: {last_error}") from last_error


def make_embedder() -> Any:
    config = get_runtime_config()
    if config.embedding_provider == "openrouter":
        return OpenRouterEmbedder(model_name=config.embedding_model, api_key=config.openrouter_api_key)
    return MockEmbedder()


def row_content_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(row.get("content", "").encode("utf-8")).hexdigest()


def build_vector_index(
    rows: list[dict[str, Any]],
    index_path: Path = DEFAULT_VECTOR_INDEX_PATH,
    embedding_fn: Any | None = None,
) -> list[dict[str, Any]]:
    embedder = embedding_fn or make_embedder()
    backend_name = getattr(embedder, "_backend_name", embedder.__class__.__name__)
    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "id": row["id"],
                "content": row.get("content", ""),
                "metadata": row.get("metadata", {}),
                "content_hash": row_content_hash(row),
                "embedding_backend": backend_name,
                "embedding": embedder(row.get("content", "")),
            }
        )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def load_vector_index(index_path: Path = DEFAULT_VECTOR_INDEX_PATH) -> list[dict[str, Any]]:
    if not index_path.exists():
        raise FileNotFoundError(f"Vector index not found: {index_path}")
    return [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class VectorStore:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        embedding_fn: Any | None = None,
        cache_path: Path = DEFAULT_EMBEDDING_CACHE_PATH,
    ) -> None:
        self.rows = rows
        self.embedding_fn = embedding_fn or make_embedder()
        self.cache_path = cache_path
        self._backend_name = getattr(self.embedding_fn, "_backend_name", self.embedding_fn.__class__.__name__)
        self.records = self._build_records()

    @classmethod
    def from_index(cls, index_records: list[dict[str, Any]], embedding_fn: Any | None = None) -> "VectorStore":
        store = cls.__new__(cls)
        store.rows = [
            {"id": record["id"], "content": record.get("content", ""), "metadata": record.get("metadata", {})}
            for record in index_records
        ]
        store.embedding_fn = embedding_fn or make_embedder()
        store.cache_path = DEFAULT_EMBEDDING_CACHE_PATH
        store._backend_name = getattr(store.embedding_fn, "_backend_name", store.embedding_fn.__class__.__name__)
        store.records = [
            {
                "row": {"id": record["id"], "content": record.get("content", ""), "metadata": record.get("metadata", {})},
                "embedding": [float(value) for value in record.get("embedding", [])],
            }
            for record in index_records
        ]
        return store

    def _cache_key(self, row: dict[str, Any]) -> str:
        digest = hashlib.sha256(row.get("content", "").encode("utf-8")).hexdigest()
        return f"{self._backend_name}:{row['id']}:{digest}"

    def _read_cache(self) -> dict[str, list[float]]:
        if not self.cache_path.exists():
            return {}
        cache: dict[str, list[float]] = {}
        for line in self.cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            cache[item["key"]] = [float(value) for value in item["embedding"]]
        return cache

    def _write_cache(self, cache: dict[str, list[float]]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as handle:
            for key, embedding in sorted(cache.items()):
                handle.write(json.dumps({"key": key, "embedding": embedding}, ensure_ascii=False) + "\n")

    def _build_records(self) -> list[dict[str, Any]]:
        cache = self._read_cache()
        changed = False
        records = []
        for row in self.rows:
            key = self._cache_key(row)
            embedding = cache.get(key)
            if embedding is None:
                embedding = self.embedding_fn(row.get("content", ""))
                cache[key] = embedding
                changed = True
            records.append({"row": row, "embedding": embedding})
        if changed:
            self._write_cache(cache)
        return records

    def search(self, query: str, top_k: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        query_embedding = self.embedding_fn(query)
        candidates = self.records
        if metadata_filter:
            candidates = [
                record
                for record in candidates
                if all(record["row"].get("metadata", {}).get(key) == value for key, value in metadata_filter.items())
            ]
        scored = []
        for record in candidates:
            row = record["row"]
            scored.append(
                {
                    "id": row["id"],
                    "text": row["content"],
                    "content": row["content"],
                    "metadata": row["metadata"],
                    "score": dot(query_embedding, record["embedding"]),
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]
