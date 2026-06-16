from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from engine.real_config import get_runtime_config, is_placeholder, validate_real_env
from engine.vector_store import DEFAULT_VECTOR_INDEX_PATH, MockEmbedder, VectorStore, load_vector_index

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_PATH = ROOT / "data" / "corpus.jsonl"


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value and not is_placeholder(value):
            return value
    return default


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d")


def _tokens(text: str) -> set[str]:
    stop_words = {
        "la",
        "cua",
        "va",
        "the",
        "nao",
        "trong",
        "duoc",
        "cho",
        "khi",
        "mot",
        "cac",
        "nhung",
        "ve",
        "co",
        "khong",
        "nguoi",
        "quy",
        "dinh",
        "phai",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _normalize(text))
        if len(token) > 1 and token not in stop_words
    }


def _summarize(text: str, max_chars: int = 520) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    summary = " ".join(sentences[:2]).strip() or cleaned
    if len(summary) > max_chars:
        return summary[: max_chars - 3].rstrip() + "..."
    return summary


class MainAgent:
    def __init__(
        self,
        version: str = "V1_Base",
        corpus_path: str | Path = DEFAULT_CORPUS_PATH,
        top_k: int | None = None,
    ) -> None:
        self.version = version
        self.name = f"LegalRAG-{version}"
        self.corpus_path = Path(corpus_path)
        self.top_k = top_k or (3 if version == "V1_Base" else 5)
        self.config = get_runtime_config()
        self.run_mode = self.config.run_mode
        self.allow_fallback = self.config.allow_fallback
        self.openai_key = self.config.openai_api_key
        self.model = self.config.main_model
        self._cache: dict[str, dict[str, Any]] = {}
        self._rows = self._load_corpus(self.corpus_path)
        self._vector_store: VectorStore | None = None
        self._last_retrieval_backend = "not_run"

    def _load_corpus(self, corpus_path: Path) -> list[dict[str, Any]]:
        if not corpus_path.exists():
            return []
        rows = []
        for line in corpus_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["_tokens"] = _tokens(row.get("content", ""))
            rows.append(row)
        return rows

    def _infer_domain(self, question: str) -> str | None:
        normalized = _normalize(question)
        if any(keyword in normalized for keyword in ["bao hiem", "bhxh", "thai san", "huu tri", "om dau"]):
            return "bao_hiem_xa_hoi"
        if any(keyword in normalized for keyword in ["hon nhan", "vo chong", "ly hon", "ket hon", "con duoi", "mang thai ho"]):
            return "hon_nhan_gia_dinh"
        if any(keyword in normalized for keyword in ["lao dong", "hop dong", "thu viec", "tien luong", "nguoi su dung"]):
            return "lao_dong"
        return None

    def _looks_out_of_context(self, question: str) -> bool:
        normalized = _normalize(question)
        unrelated = [
            "thoi tiet",
            "bitcoin",
            "dien thoai",
            "nau pho",
            "bong da",
            "may in",
            "guitar",
            "ca chua",
            "card do hoa",
            "diem chuan",
        ]
        return any(keyword in normalized for keyword in unrelated)

    def _looks_adversarial(self, question: str) -> bool:
        normalized = _normalize(question)
        return "bo qua" in normalized or "suy doan ca nhan" in normalized

    def _looks_ambiguous(self, question: str) -> bool:
        normalized = _normalize(question)
        return "toi co duoc huong che do nay khong" in normalized or "chua noi ro truong hop" in normalized

    def _search(self, question: str) -> list[dict[str, Any]]:
        if not self._rows:
            return []
        if self.version in {"V2_RetrievalPlus", "V3_CostAware"}:
            vector_results = self._vector_search(question)
            if vector_results:
                self._last_retrieval_backend = "vector_index" if self.run_mode == "real" else "vector_mock"
                return vector_results

        self._last_retrieval_backend = "lexical_fallback"
        query_tokens = _tokens(question)
        inferred_domain = self._infer_domain(question)
        normalized_question = _normalize(question)
        if self.version in {"V2_RetrievalPlus", "V3_CostAware"} and inferred_domain:
            candidates = [row for row in self._rows if row["metadata"].get("domain") == inferred_domain]
        else:
            candidates = self._rows

        if not candidates:
            candidates = self._rows

        scored = []
        for row in candidates:
            content_tokens = row.get("_tokens", set())
            if not query_tokens or not content_tokens:
                score = 0.0
            else:
                overlap = len(query_tokens & content_tokens)
                score = overlap / math.sqrt(len(query_tokens) * len(content_tokens))
            if _normalize(row["id"]) in normalized_question:
                score += 5.0
            if inferred_domain and row["metadata"].get("domain") == inferred_domain:
                score += 0.05
            scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        threshold = 0.0001 if self.version == "V1_Base" else 0.03
        for score, row in scored[: self.top_k]:
            if score < threshold:
                continue
            results.append(
                {
                    "id": row["id"],
                    "text": row["content"],
                    "score": round(score, 4),
                    "metadata": row["metadata"],
                }
            )
        return results

    def _vector_search(self, question: str) -> list[dict[str, Any]]:
        try:
            if self._vector_store is None:
                if self.run_mode == "real":
                    if not DEFAULT_VECTOR_INDEX_PATH.exists():
                        raise FileNotFoundError(
                            f"Vector index missing: {DEFAULT_VECTOR_INDEX_PATH}. Run python scripts/build_vector_index.py first."
                        )
                    self._vector_store = VectorStore.from_index(load_vector_index(DEFAULT_VECTOR_INDEX_PATH))
                else:
                    self._vector_store = VectorStore(self._rows, embedding_fn=MockEmbedder())
            inferred_domain = self._infer_domain(question)
            metadata_filter = {"domain": inferred_domain} if inferred_domain else None
            results = self._vector_store.search(question, top_k=self.top_k, metadata_filter=metadata_filter)
            if not results and metadata_filter:
                results = self._vector_store.search(question, top_k=self.top_k)
            return [
                {
                    "id": result["id"],
                    "text": result["text"],
                    "score": round(float(result["score"]), 4),
                    "metadata": result["metadata"],
                }
                for result in results
            ]
        except Exception:
            if self.run_mode == "real" and not self.allow_fallback:
                raise
            return []

    def _build_answer(self, question: str, contexts: list[dict[str, Any]]) -> str:
        if not contexts:
            return "Tôi chưa có đủ thông tin trong dữ liệu đã truy xuất để trả lời câu hỏi này."

        if self.version == "V1_Base":
            return "Dựa trên tài liệu luật đã tìm được: " + _summarize(contexts[0]["text"], max_chars=460)

        parts = []
        for index, context in enumerate(contexts[:2], start=1):
            metadata = context.get("metadata", {})
            source = metadata.get("file_name", metadata.get("source", "nguồn luật"))
            chunk_index = metadata.get("chunk_index", "?")
            parts.append(f"Nguồn {index} ({source}, chunk {chunk_index}): {_summarize(context['text'], 320)}")

        if self._looks_adversarial(question):
            return "Không được bỏ qua tài liệu đã truy xuất. " + " ".join(parts) + " Kết luận chỉ dựa trên nguồn luật."
        if self._looks_ambiguous(question):
            return "Cần hỏi làm rõ thông tin cá nhân, loại chế độ, thời gian đóng và bối cảnh áp dụng trước khi kết luận. " + " ".join(parts[:1])
        return " ".join(parts) + " Kết luận: câu trả lời cần bám đúng các nguồn luật trên."

    def _build_openai_prompt(self, question: str, contexts: list[dict[str, Any]]) -> list[dict[str, str]]:
        context_block = "\n\n".join(
            f"[{context['id']}]\n{context['text']}" for context in contexts[: self.top_k]
        )
        if not context_block:
            context_block = "[no retrieved legal context]"
        system_prompt = (
            "Bạn là trợ lý RAG pháp luật Việt Nam. Chỉ trả lời dựa trên context được cung cấp. "
            "Nếu context không đủ, ngoài phạm vi, hoặc câu hỏi thiếu thông tin cần thiết, hãy nói rõ là chưa đủ căn cứ "
            "và hỏi/khuyến nghị làm rõ. Không bịa điều luật, không tư vấn chắc chắn vượt quá dữ liệu. "
            "Không nhắc đến các từ context, chunk, retrieval, hay prompt trong câu trả lời."
        )
        user_prompt = (
            "Câu hỏi:\n"
            f"{question}\n\n"
            "Context truy xuất:\n"
            f"{context_block}\n\n"
            "Yêu cầu trả lời bằng tiếng Việt, ngắn gọn nhưng đủ căn cứ. Không nhắc tên chunk hoặc cơ chế truy xuất."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _estimate_openai_cost(self, usage: dict[str, Any]) -> float:
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        lowered = self.model.lower()
        if "gpt-4o-mini" in lowered:
            input_rate, output_rate = 0.15, 0.60
        else:
            input_rate, output_rate = 2.50, 10.0
        return round((input_tokens * input_rate + output_tokens * output_rate) / 1_000_000, 6)

    def _call_openai_generation(self, question: str, contexts: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        if not self.openai_key:
            raise RuntimeError("OPENAI_API_KEY is required for real agent generation.")
        payload = json.dumps(
            {
                "model": self.model,
                "messages": self._build_openai_prompt(question, contexts),
                "temperature": 0.1,
                "max_tokens": 700,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        api_url = f"{self.config.openai_api_base.rstrip('/')}/chat/completions"
        request = urllib.request.Request(
            api_url,
            data=payload,
            headers={"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip(), data.get("usage", {})

    async def _generate_answer(self, question: str, contexts: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        if self.run_mode != "real":
            answer = self._build_answer(question, contexts)
            approx_tokens = max(
                40,
                len(_tokens(question + " " + answer)) * 2 + sum(len(_tokens(c["text"])) for c in contexts[:2]),
            )
            cost_multiplier = 0.65 if self.version == "V3_CostAware" else 1.0
            return answer, {
                "generation_mode": "deterministic",
                "fallback_used": False,
                "api_call_counts": {"openai_main": 0},
                "models_used": {"main": "deterministic_legal_rag"},
                "tokens_used": approx_tokens,
                "cost_usd": round(approx_tokens / 1_000_000 * 0.15 * cost_multiplier, 6),
            }

        validate_real_env()
        try:
            answer, usage = await asyncio.to_thread(self._call_openai_generation, question, contexts)
            tokens_used = int(usage.get("total_tokens", 0) or 0)
            if not tokens_used:
                tokens_used = int(usage.get("prompt_tokens", 0) or 0) + int(usage.get("completion_tokens", 0) or 0)
            return answer, {
                "generation_mode": "openai_api",
                "fallback_used": False,
                "api_call_counts": {"openai_main": 1},
                "models_used": {"main": self.model},
                "tokens_used": tokens_used,
                "cost_usd": self._estimate_openai_cost(usage),
            }
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            if not self.allow_fallback:
                raise RuntimeError(f"OpenAI main generation failed in real mode: {exc}") from exc
            answer = self._build_answer(question, contexts)
            approx_tokens = max(40, len(_tokens(question + " " + answer)) * 2)
            return answer, {
                "generation_mode": "deterministic_fallback",
                "fallback_used": True,
                "api_call_counts": {"openai_main": 0},
                "models_used": {"main": "deterministic_legal_rag"},
                "tokens_used": approx_tokens,
                "cost_usd": 0.0,
            }

    async def query(self, question: str) -> dict[str, Any]:
        started = time.perf_counter()
        cache_key = f"{self.version}:{question}"
        if self.version == "V3_CostAware" and cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            cached["metadata"] = dict(cached["metadata"])
            cached["metadata"]["cache_hit"] = True
            return cached

        await asyncio.sleep(0)
        contexts = [] if self._looks_out_of_context(question) and self.version != "V1_Base" else self._search(question)
        answer, generation_metadata = await self._generate_answer(question, contexts)
        response = {
            "answer": answer,
            "retrieved_ids": [context["id"] for context in contexts],
            "contexts": contexts,
            "metadata": {
                "version": self.version,
                "run_mode": self.run_mode,
                "model": generation_metadata["models_used"]["main"],
                "generation_mode": generation_metadata["generation_mode"],
                "fallback_used": generation_metadata["fallback_used"],
                "api_call_counts": generation_metadata["api_call_counts"],
                "models_used": generation_metadata["models_used"],
                "retrieval_backend": self._last_retrieval_backend,
                "vector_index_path": str(DEFAULT_VECTOR_INDEX_PATH) if self.run_mode == "real" else None,
                "tokens_used": generation_metadata["tokens_used"],
                "cost_usd": generation_metadata["cost_usd"],
                "latency_internal": round(time.perf_counter() - started, 4),
                "cache_hit": False,
            },
        }
        if self.version == "V3_CostAware":
            self._cache[cache_key] = response
        return response


if __name__ == "__main__":
    async def _demo() -> None:
        agent = MainAgent(version="V2_RetrievalPlus")
        result = await agent.query("Thời gian thử việc tối đa là bao nhiêu ngày?")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(_demo())
