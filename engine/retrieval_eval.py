from __future__ import annotations

import re
import unicodedata
from typing import Any


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


class RetrievalEvaluator:
    def calculate_hit_rate(
        self,
        expected_ids: list[str],
        retrieved_ids: list[str],
        top_k: int = 3,
    ) -> float:
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0
        top_retrieved = set(retrieved_ids[:top_k])
        return 1.0 if any(doc_id in top_retrieved for doc_id in expected_ids) else 0.0

    def calculate_mrr(self, expected_ids: list[str], retrieved_ids: list[str]) -> float:
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0
        expected = set(expected_ids)
        for index, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in expected:
                return 1.0 / index
        return 0.0

    def calculate_context_precision(self, expected_ids: list[str], retrieved_ids: list[str]) -> float:
        if not retrieved_ids:
            return 1.0 if not expected_ids else 0.0
        if not expected_ids:
            return 0.0
        expected = set(expected_ids)
        relevant = sum(1 for doc_id in retrieved_ids if doc_id in expected)
        return relevant / len(retrieved_ids)

    def calculate_context_recall(self, expected_ids: list[str], retrieved_ids: list[str]) -> float:
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0
        retrieved = set(retrieved_ids)
        relevant = sum(1 for doc_id in expected_ids if doc_id in retrieved)
        return relevant / len(expected_ids)

    def calculate_faithfulness(self, answer: str, contexts: list[dict[str, Any]]) -> float:
        answer_tokens = _tokens(answer)
        if "chua co du thong tin" in _normalize(answer):
            return 1.0
        if not answer_tokens:
            return 0.0
        context_tokens = _tokens(" ".join(context.get("text", "") for context in contexts))
        if not context_tokens:
            return 0.0
        overlap = len(answer_tokens & context_tokens)
        return min(1.0, overlap / max(1, len(answer_tokens) * 0.55))

    def calculate_relevancy(self, question: str, expected_answer: str, answer: str) -> float:
        if "chua co du thong tin" in _normalize(expected_answer):
            return 1.0 if "chua co du thong tin" in _normalize(answer) else 0.2
        expected_tokens = _tokens(f"{question} {expected_answer}")
        answer_tokens = _tokens(answer)
        if not expected_tokens or not answer_tokens:
            return 0.0
        overlap = len(expected_tokens & answer_tokens)
        return min(1.0, overlap / max(1, len(expected_tokens) * 0.35))

    def classify_failure(self, case: dict[str, Any], retrieved_ids: list[str], answer: str) -> str:
        expected_ids = case.get("expected_retrieval_ids", [])
        if not expected_ids and "chua co du thong tin" not in _normalize(answer):
            return "out_of_context_not_refused"
        if expected_ids and not any(doc_id in set(retrieved_ids[:5]) for doc_id in expected_ids):
            return "retrieval_miss"
        expected_domain = case.get("domain")
        if expected_domain and expected_domain != "out_of_context":
            wrong_domain = [
                doc_id
                for doc_id in retrieved_ids[:3]
                if expected_domain not in doc_id and expected_domain != "bao_hiem_xa_hoi"
            ]
            if len(wrong_domain) == len(retrieved_ids[:3]) and retrieved_ids:
                return "wrong_domain"
        if "chua co du thong tin" in _normalize(answer) and expected_ids:
            return "incomplete_answer"
        return "none"

    async def score(self, case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
        expected_ids = list(case.get("expected_retrieval_ids", []))
        retrieved_ids = list(response.get("retrieved_ids", []))
        contexts = list(response.get("contexts", []))
        answer = str(response.get("answer", ""))

        hit_rate_3 = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=3)
        hit_rate_5 = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=5)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)
        precision = self.calculate_context_precision(expected_ids, retrieved_ids[:5])
        recall = self.calculate_context_recall(expected_ids, retrieved_ids[:5])
        faithfulness = self.calculate_faithfulness(answer, contexts)
        relevancy = self.calculate_relevancy(
            str(case.get("question", "")),
            str(case.get("expected_answer", "")),
            answer,
        )

        return {
            "retrieval": {
                "hit_rate": hit_rate_3,
                "hit_rate_at_3": hit_rate_3,
                "hit_rate_at_5": hit_rate_5,
                "mrr": mrr,
                "context_precision": precision,
                "context_recall": recall,
            },
            "generation": {
                "faithfulness": faithfulness,
                "relevancy": relevancy,
            },
            "failure_type": self.classify_failure(case, retrieved_ids, answer),
        }

    async def evaluate_batch(self, dataset: list[dict[str, Any]], responses: list[dict[str, Any]]) -> dict[str, float]:
        scores = [await self.score(case, response) for case, response in zip(dataset, responses)]
        if not scores:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0}
        return {
            "avg_hit_rate": sum(item["retrieval"]["hit_rate"] for item in scores) / len(scores),
            "avg_mrr": sum(item["retrieval"]["mrr"] for item in scores) / len(scores),
        }
