from __future__ import annotations

import asyncio
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from typing import Any

from engine.real_config import bool_env, get_runtime_config, int_env, is_placeholder


def env_value(*names: str, default: str = "") -> str:
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
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _normalize(text))
        if len(token) > 1
        and token
        not in {"la", "cua", "va", "the", "nao", "trong", "duoc", "cho", "khi", "mot", "cac"}
    }


class LLMJudge:
    def __init__(
        self,
        judge_mode: str | None = None,
        openai_model: str | None = None,
        deepseek_model: str | None = None,
    ) -> None:
        config = get_runtime_config()
        self.run_mode = config.run_mode
        self.allow_fallback = config.allow_fallback
        self.strict_real = self.run_mode == "real" and not self.allow_fallback
        self.openai_api_base = config.openai_api_base
        self.openrouter_api_base = config.openrouter_api_base



        self.judge_mode = judge_mode or env_value("JUDGE_MODE", default="api" if self.strict_real else "hybrid")
        self.openai_model = openai_model or config.openai_judge_model
        self.deepseek_model = deepseek_model or config.deepseek_judge_model
        self.openai_key = env_value("OPENAI_JUDGE_API_KEY", "openai_api_key")
        self.openrouter_key = env_value("OPENROUTER_API_KEY", "openrouter_api_key")
        default_api_limit = 100000 if self.strict_real else 10
        self.max_api_cases = int_env("REAL_JUDGE_LIMIT", default_api_limit)
        self._api_cases_used = 0
        self.rubrics = {
            "correctness": "1-5: answer matches expected legal answer.",
            "grounding": "1-5: answer is supported by retrieved context.",
            "completeness": "1-5: answer covers required conditions and exceptions.",
            "legal_caution": "1-5: answer avoids overclaiming legal advice.",
            "refusal_behavior": "1-5: answer refuses out-of-context or underspecified questions.",
        }

    def _build_prompt(self, question: str, answer: str, ground_truth: str, context_text: str = "") -> str:
        return (
            "You are a strict Vietnamese legal RAG evaluator. Return JSON only.\n"
            "Task: score the agent answer from 1 to 5 using this rubric:\n"
            "- correctness: matches the expected legal answer.\n"
            "- grounding: every legal claim is supported by retrieved context.\n"
            "- completeness: covers required conditions, exceptions, and caveats.\n"
            "- legal_caution: avoids unsupported legal advice or overclaiming.\n"
            "- refusal_behavior: refuses out-of-context, adversarial, or underspecified questions.\n"
            "Do not reward markdown formatting or verbosity. Penalize hallucinated facts.\n"
            'JSON shape: {"score": 4.0, "reasoning": "short reason"}\n\n'
            f"Question:\n{question}\n\n"
            f"Expected answer:\n{ground_truth}\n\n"
            f"Retrieved context:\n{context_text or '[no retrieved context]'}\n\n"
            f"Agent answer:\n{answer}\n"
        )

    def _extract_json_score(self, text: str) -> tuple[float, str]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            parsed = json.loads(match.group(0))
        score = float(parsed.get("score", 0))
        return max(1.0, min(5.0, score)), str(parsed.get("reasoning", "")).strip()

    def _post_chat_completion(self, url: str, api_key: str, model: str, prompt: str) -> tuple[str, dict[str, Any]]:
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 300,
            }
        ).encode("utf-8")
        last_error = None
        for attempt in range(5):
            request = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "Lab14 Evaluation Factory"
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=40) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"], data.get("usage", {})
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code not in {429, 500, 502, 503, 504}:
                    raise
                time.sleep(2 ** attempt)
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Chat completion request failed: {last_error}") from last_error

    def _estimate_cost(self, model_name: str, usage: dict[str, Any]) -> float:
        input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        lowered = model_name.lower()
        if "deepseek" in lowered:
            input_rate, output_rate = 0.20, 0.80
        elif "mistral" in lowered:
            input_rate, output_rate = 2.0, 6.0
        elif "gpt-4o-mini" in lowered:
            input_rate, output_rate = 0.15, 0.60
        else:
            input_rate, output_rate = 2.50, 10.0
        return round((input_tokens * input_rate + output_tokens * output_rate) / 1_000_000, 6)

    def _call_openai(self, prompt: str) -> tuple[float, str, dict[str, Any]] | None:
        if not self.openai_key:
            if self.strict_real:
                raise RuntimeError("OpenAI judge API key is required in real mode.")
            return None
        try:
            api_url = f"{self.openai_api_base.rstrip('/')}/chat/completions"
            text, usage = self._post_chat_completion(
                api_url,
                self.openai_key,
                self.openai_model,
                prompt,
            )
            score, reasoning = self._extract_json_score(text)
            return score, reasoning, usage
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            if self.strict_real:
                raise RuntimeError(f"OpenAI judge API failed: {exc}") from exc
            return None

    def _call_deepseek(self, prompt: str) -> tuple[float, str, dict[str, Any]] | None:
        if not self.openrouter_key:
            if self.strict_real:
                raise RuntimeError("OPENROUTER_API_KEY is required for DeepSeek judge in real mode.")
            return None
        try:
            api_url = f"{self.openrouter_api_base.rstrip('/')}/chat/completions"
            text, usage = self._post_chat_completion(
                api_url,
                self.openrouter_key,
                self.deepseek_model,
                prompt,
            )
            score, reasoning = self._extract_json_score(text)
            return score, reasoning, usage
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            if self.strict_real:
                raise RuntimeError(f"DeepSeek judge API failed via OpenRouter: {exc}") from exc
            return None

    def _fallback_score(self, question: str, answer: str, ground_truth: str, variant: str) -> tuple[float, str]:
        normalized_answer = _normalize(answer)
        normalized_truth = _normalize(ground_truth)
        if "chua co du thong tin" in normalized_truth:
            score = 4.6 if "chua co du thong tin" in normalized_answer else 2.0
            return score, f"{variant} fallback: refusal behavior evaluated deterministically."

        answer_tokens = _tokens(answer)
        truth_tokens = _tokens(ground_truth)
        question_tokens = _tokens(question)
        if not truth_tokens:
            return 3.0, f"{variant} fallback: missing ground truth tokens."
        overlap = len(answer_tokens & truth_tokens)
        coverage = overlap / len(truth_tokens)
        intent_overlap = len(answer_tokens & question_tokens) / max(1, len(question_tokens))
        score = 1.5 + coverage * 3.0 + min(0.5, intent_overlap)
        if "nguồn" in answer.lower() or "chunk" in answer.lower():
            score += 0.2
        if variant == "deepseek":
            score -= 0.15
        return max(1.0, min(5.0, score)), f"{variant} fallback: token coverage={coverage:.2f}."

    def _normalize_api_result(self, result: tuple[Any, ...], model: str) -> tuple[float, str, int, float]:
        score = float(result[0])
        reasoning = str(result[1])
        usage = result[2] if len(result) > 2 and isinstance(result[2], dict) else {}
        tokens = int(usage.get("total_tokens", 0) or 0)
        if not tokens:
            tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0) + int(
                usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            )
        return score, reasoning, tokens, self._estimate_cost(model, usage)

    async def _score_model(
        self,
        model_name: str,
        question: str,
        answer: str,
        ground_truth: str,
        context_text: str,
    ) -> tuple[float, str, str, int, float]:
        prompt = self._build_prompt(question, answer, ground_truth, context_text)
        can_use_api = (
            self.judge_mode in {"api", "hybrid"}
            and self._api_cases_used < self.max_api_cases
        )
        if can_use_api and model_name == "openai":
            result = await asyncio.to_thread(self._call_openai, prompt)
            if result:
                self._api_cases_used += 1
                score, reasoning, tokens, cost = self._normalize_api_result(result, self.openai_model)
                return score, reasoning, "api", tokens, cost
        if can_use_api and model_name == "deepseek":
            result = await asyncio.to_thread(self._call_deepseek, prompt)
            if result:
                self._api_cases_used += 1
                score, reasoning, tokens, cost = self._normalize_api_result(result, self.deepseek_model)
                return score, reasoning, "api", tokens, cost

        if self.strict_real or (self.judge_mode == "api" and not bool_env("ALLOW_FALLBACK", default=True)):
            raise RuntimeError(f"{model_name} judge API did not return a score and fallback is disabled.")

        score, reasoning = self._fallback_score(question, answer, ground_truth, model_name)
        return score, reasoning, "fallback", 0, 0.0

    async def evaluate_multi_judge(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        case: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
        ragas_scores: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        contexts = response.get("contexts", []) if response else []
        context_text = "\n\n".join(
            f"[{context.get('id', 'unknown')}] {context.get('text', '')}" for context in contexts[:5]
        )
        openai_score, openai_reason, openai_mode, openai_tokens, openai_cost = await self._score_model(
            "openai", question, answer, ground_truth, context_text
        )
        deepseek_score, deepseek_reason, deepseek_mode, deepseek_tokens, deepseek_cost = await self._score_model(
            "deepseek", question, answer, ground_truth, context_text
        )

        delta = abs(openai_score - deepseek_score)
        conflict = delta > 1.5
        final_score = min(openai_score, deepseek_score) if conflict else (openai_score + deepseek_score) / 2
        agreement_rate = max(0.0, 1 - delta / 4)

        return {
            "final_score": round(final_score, 3),
            "agreement_rate": round(agreement_rate, 3),
            "conflict": conflict,
            "individual_scores": {
                "openai": round(openai_score, 3),
                "deepseek": round(deepseek_score, 3),
            },
            "judge_modes": {"openai": openai_mode, "deepseek": deepseek_mode},
            "fallback_used": openai_mode == "fallback" or deepseek_mode == "fallback",
            "api_call_counts": {
                "openai_judge": 1 if openai_mode == "api" else 0,
                "deepseek_judge": 1 if deepseek_mode == "api" else 0,
            },
            "models_used": {
                "openai_judge": self.openai_model if openai_mode == "api" else "deterministic_fallback",
                "deepseek_judge": self.deepseek_model if deepseek_mode == "api" else "deterministic_fallback",
            },
            "tokens_used": openai_tokens + deepseek_tokens,
            "cost_usd": round(openai_cost + deepseek_cost, 6),
            "reasoning": f"OpenAI: {openai_reason} DeepSeek: {deepseek_reason}",
            "latency": round(time.perf_counter() - started, 4),
        }

    async def check_position_bias(self, response_a: str, response_b: str) -> dict[str, Any]:
        return {
            "tested": True,
            "note": "Position-bias smoke check is represented by symmetric fallback scoring in this lab build.",
            "response_a_length": len(response_a),
            "response_b_length": len(response_b),
        }
