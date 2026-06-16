from __future__ import annotations

import asyncio
import time
from typing import Any


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: dict[str, Any]) -> dict[str, Any]:
        start_time = time.perf_counter()
        response = await self.agent.query(test_case["question"])
        agent_latency = time.perf_counter() - start_time

        ragas_scores = await self.evaluator.score(test_case, response)
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case["expected_answer"],
            case=test_case,
            response=response,
            ragas_scores=ragas_scores,
        )

        latency = time.perf_counter() - start_time
        final_score = judge_result["final_score"]
        return {
            "id": test_case.get("id"),
            "domain": test_case.get("domain"),
            "type": test_case.get("type"),
            "difficulty": test_case.get("difficulty"),
            "test_case": test_case["question"],
            "expected_answer": test_case.get("expected_answer", ""),
            "expected_retrieval_ids": test_case.get("expected_retrieval_ids", []),
            "agent_response": response["answer"],
            "retrieved_ids": response.get("retrieved_ids", []),
            "contexts": response.get("contexts", []),
            "latency": latency,
            "agent_latency": agent_latency,
            "tokens_used": response.get("metadata", {}).get("tokens_used", 0),
            "cost_usd": response.get("metadata", {}).get("cost_usd", 0.0),
            "agent_metadata": response.get("metadata", {}),
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": "fail" if final_score < 3.5 else "pass",
        }

    async def run_all(self, dataset: list[dict[str, Any]], batch_size: int = 5) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for index in range(0, len(dataset), batch_size):
            batch = dataset[index : index + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results
