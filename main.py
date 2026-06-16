from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent.main_agent import MainAgent
from data.synthetic_gen import CORPUS_PATH, GOLDEN_PATH, generate_lab_data
from engine.llm_judge import LLMJudge
from engine.real_config import RuntimeConfig, get_runtime_config, validate_real_env
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
ANALYSIS_PATH = ROOT / "analysis" / "failure_analysis.md"
VERSIONS = ["V1_Base", "V2_RetrievalPlus", "V3_CostAware"]
REAL_VERSION = "V3_CostAware"
REAL_CASE_TYPE_TARGETS = {
    "fact": 8,
    "reasoning": 8,
    "multi_hop": 7,
    "adversarial": 7,
    "out_of_context": 0,
    "ambiguous": 0,
}


def load_dataset() -> list[dict[str, Any]]:
    if not GOLDEN_PATH.exists() or not CORPUS_PATH.exists():
        print("Golden dataset or corpus missing; generating from Day 7 law data...")
        generate_lab_data()
    dataset = [
        json.loads(line)
        for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not dataset:
        raise ValueError("data/golden_set.jsonl is empty")
    return dataset


def select_real_benchmark_dataset(dataset: list[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("REAL_EVAL_CASE_LIMIT must be positive")

    available_types = sorted({str(case.get("type", "fact")) for case in dataset})
    preferred_types = [case_type for case_type in REAL_CASE_TYPE_TARGETS if case_type in available_types]
    extra_types = [case_type for case_type in available_types if case_type not in preferred_types]
    ordered_types = preferred_types + extra_types
    if not ordered_types:
        raise ValueError("Dataset has no case types")

    base = limit // len(ordered_types)
    remainder = limit % len(ordered_types)
    targets = {
        case_type: base + (1 if index < remainder else 0)
        for index, case_type in enumerate(ordered_types)
    }

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for case_type, target in targets.items():
        bucket = [case for case in dataset if case.get("type") == case_type]
        if case_type == "fact":
            bucket.sort(key=lambda case: (0 if str(case.get("id", "")).startswith("law_seed") else 1, case.get("id", "")))
        else:
            bucket.sort(key=lambda case: case.get("id", ""))
        for case in bucket[:target]:
            selected.append(case)
            selected_ids.add(case["id"])

    if len(selected) < limit:
        for case in dataset:
            if case["id"] in selected_ids:
                continue
            selected.append(case)
            selected_ids.add(case["id"])
            if len(selected) == limit:
                break

    if len(selected) != limit:
        raise ValueError(f"Could not build focused real benchmark of {limit} cases; got {len(selected)}")
    return selected


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sum_nested_counts(rows: list[dict[str, Any]], key_path: tuple[str, ...]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value: Any = row
        for key in key_path:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        if isinstance(value, dict):
            for name, count in value.items():
                counts[name] += int(count or 0)
    return dict(counts)


def _merge_models_used(results: list[dict[str, Any]]) -> dict[str, str]:
    models: dict[str, str] = {}
    for row in results:
        agent_models = row.get("agent_metadata", {}).get("models_used", {})
        judge_models = row.get("judge", {}).get("models_used", {})
        if isinstance(agent_models, dict):
            models.update({str(key): str(value) for key, value in agent_models.items()})
        if isinstance(judge_models, dict):
            models.update({str(key): str(value) for key, value in judge_models.items()})
    return models


def summarize_results(version: str, results: list[dict[str, Any]], config: RuntimeConfig) -> dict[str, Any]:
    total = len(results)
    pass_count = sum(1 for row in results if row["status"] == "pass")
    failure_clusters = Counter(
        row["ragas"].get("failure_type", "none")
        for row in results
        if row["status"] == "fail" or row["ragas"].get("failure_type") != "none"
    )
    judge_modes = Counter(
        mode
        for row in results
        for mode in row.get("judge", {}).get("judge_modes", {}).values()
    )
    generation_modes = Counter(row.get("agent_metadata", {}).get("generation_mode", "unknown") for row in results)
    api_call_counts = _sum_nested_counts(results, ("agent_metadata", "api_call_counts"))
    judge_api_counts = _sum_nested_counts(results, ("judge", "api_call_counts"))
    for name, count in judge_api_counts.items():
        api_call_counts[name] = api_call_counts.get(name, 0) + count
    fallback_used = any(
        row.get("agent_metadata", {}).get("fallback_used", False)
        or row.get("judge", {}).get("fallback_used", False)
        for row in results
    )
    agent_tokens = sum(row.get("tokens_used", 0) for row in results)
    judge_tokens = sum(row.get("judge", {}).get("tokens_used", 0) for row in results)
    agent_cost = sum(row.get("cost_usd", 0.0) for row in results)
    judge_cost = sum(row.get("judge", {}).get("cost_usd", 0.0) for row in results)

    metrics = {
        "avg_score": average([row["judge"]["final_score"] for row in results]),
        "hit_rate": average([row["ragas"]["retrieval"]["hit_rate"] for row in results]),
        "hit_rate_at_5": average([row["ragas"]["retrieval"]["hit_rate_at_5"] for row in results]),
        "mrr": average([row["ragas"]["retrieval"]["mrr"] for row in results]),
        "context_precision": average([row["ragas"]["retrieval"]["context_precision"] for row in results]),
        "context_recall": average([row["ragas"]["retrieval"]["context_recall"] for row in results]),
        "faithfulness": average([row["ragas"]["generation"]["faithfulness"] for row in results]),
        "relevancy": average([row["ragas"]["generation"]["relevancy"] for row in results]),
        "agreement_rate": average([row["judge"]["agreement_rate"] for row in results]),
        "avg_latency": average([row["latency"] for row in results]),
        "total_tokens": agent_tokens + judge_tokens,
        "total_cost_usd": round(agent_cost + judge_cost, 6),
        "pass_rate": pass_count / total if total else 0.0,
    }
    return {
        "metadata": {
            "version": version,
            "run_mode": config.run_mode,
            "agent_generation_mode": next(iter(generation_modes), "unknown") if len(generation_modes) == 1 else dict(generation_modes),
            "total": total,
            "pass": pass_count,
            "fail": total - pass_count,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "judge_modes": dict(judge_modes),
            "fallback_used": fallback_used,
            "api_call_counts": api_call_counts,
            "models_used": _merge_models_used(results),
            "real_case_limit": config.real_case_limit if config.run_mode == "real" else None,
        },
        "metrics": metrics,
        "failure_clusters": dict(failure_clusters),
    }


def decide_release(v1: dict[str, Any], v2: dict[str, Any], v3: dict[str, Any]) -> dict[str, Any]:
    v2_metrics = v2["metrics"]
    v3_metrics = v3["metrics"]
    score_delta_v3_v2 = v3_metrics["avg_score"] - v2_metrics["avg_score"]
    score_delta_v2_v1 = v2_metrics["avg_score"] - v1["metrics"]["avg_score"]
    cost_growth = 0.0
    if v2_metrics["total_cost_usd"] > 0:
        cost_growth = (v3_metrics["total_cost_usd"] - v2_metrics["total_cost_usd"]) / v2_metrics["total_cost_usd"]

    blockers = []
    if v3_metrics["avg_score"] < 4.0:
        blockers.append("avg_score below 4.0")
    if v3_metrics["hit_rate"] < 0.8:
        blockers.append("hit_rate below 0.8")
    if score_delta_v3_v2 < -0.05:
        blockers.append("V3 quality regressed from V2")
    if cost_growth > 0.30 and score_delta_v3_v2 <= 0:
        blockers.append("cost increased over 30% without quality gain")

    return {
        "delta_v2_vs_v1": round(score_delta_v2_v1, 4),
        "delta_v3_vs_v2": round(score_delta_v3_v2, 4),
        "cost_growth_v3_vs_v2": round(cost_growth, 4),
        "release_decision": "APPROVE" if not blockers else "BLOCK_RELEASE",
        "blockers": blockers,
    }


def decide_real_release(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    metadata = summary["metadata"]
    blockers = []
    if metadata.get("fallback_used"):
        blockers.append("fallback used in real evaluation")
    if metrics["avg_score"] < 4.0:
        blockers.append("avg_score below 4.0")
    if metrics["hit_rate"] < 0.8:
        blockers.append("hit_rate below 0.8")
    if metrics["agreement_rate"] < 0.65:
        blockers.append("agreement_rate below 0.65")
    api_counts = metadata.get("api_call_counts", {})
    if api_counts.get("openai_main", 0) < metadata.get("total", 0):
        blockers.append("not every case used OpenAI main generation")
    if api_counts.get("openai_judge", 0) < metadata.get("total", 0):
        blockers.append("not every case used OpenAI judge")
    if api_counts.get("deepseek_judge", 0) < metadata.get("total", 0):
        blockers.append("not every case used DeepSeek judge")

    return {
        "release_decision": "APPROVE" if not blockers else "BLOCK_RELEASE",
        "blockers": blockers,
        "quality_thresholds": {
            "avg_score_min": 4.0,
            "hit_rate_min": 0.8,
            "agreement_rate_min": 0.65,
            "fallback_allowed": False,
        },
        "regression_note": "Focused real run gates V3 directly; local V1/V2/V3 summaries remain available for regression comparison.",
    }


def validate_real_report(summary: dict[str, Any]) -> None:
    metadata = summary.get("metadata", {})
    if metadata.get("run_mode") != "real":
        return

    errors = []
    if metadata.get("fallback_used"):
        errors.append("fallback_used must be false")
    if "fallback" in metadata.get("judge_modes", {}):
        errors.append("judge_modes must not contain fallback")
    if metadata.get("agent_generation_mode") != "openai_api":
        errors.append("agent_generation_mode must be openai_api")
    api_counts = metadata.get("api_call_counts", {})
    total = int(metadata.get("total", 0) or 0)
    for key in ["openai_main", "openai_judge", "deepseek_judge"]:
        if api_counts.get(key, 0) < total:
            errors.append(f"{key} API call count must be at least total cases")
    if errors:
        raise ValueError("Invalid real evaluation report: " + "; ".join(errors))


def worst_cases(results: list[dict[str, Any]], count: int = 3) -> list[dict[str, Any]]:
    return sorted(
        results,
        key=lambda row: (
            row["judge"]["final_score"],
            row["ragas"]["retrieval"]["hit_rate"],
            row["ragas"]["generation"]["faithfulness"],
        ),
    )[:count]


def write_failure_analysis(final_summary: dict[str, Any], all_results: dict[str, list[dict[str, Any]]]) -> None:
    v3_results = all_results["V3_CostAware"]
    worst = worst_cases(v3_results)
    metrics = final_summary["metrics"]
    clusters = final_summary.get("failure_clusters", {})

    lines = [
        "# Báo cáo Phân tích Thất bại (Failure Analysis Report)",
        "",
        "## 1. Tổng quan Benchmark",
        f"- **Tổng số cases:** {final_summary['metadata']['total']}",
        f"- **Tỉ lệ Pass:** {metrics['pass_rate'] * 100:.1f}%",
        f"- **Điểm LLM-Judge trung bình:** {metrics['avg_score']:.2f} / 5.0",
        f"- **Hit Rate@3:** {metrics['hit_rate'] * 100:.1f}%",
        f"- **MRR:** {metrics['mrr']:.3f}",
        f"- **Faithfulness:** {metrics['faithfulness']:.3f}",
        f"- **Agreement Rate:** {metrics['agreement_rate']:.3f}",
        f"- **Avg latency:** {metrics['avg_latency']:.3f}s/case",
        f"- **Estimated total cost:** ${metrics['total_cost_usd']:.6f}",
        f"- **Run mode:** {final_summary['metadata'].get('run_mode', 'local')}",
        f"- **Agent generation:** {final_summary['metadata'].get('agent_generation_mode', 'unknown')}",
        f"- **Judge modes:** {json.dumps(final_summary['metadata'].get('judge_modes', {}), ensure_ascii=False)}",
        f"- **Fallback used:** {final_summary['metadata'].get('fallback_used', False)}",
        f"- **API call counts:** {json.dumps(final_summary['metadata'].get('api_call_counts', {}), ensure_ascii=False)}",
        "",
        "## 2. Phân nhóm lỗi",
        "| Nhóm lỗi | Số lượng | Hướng xử lý |",
        "|---|---:|---|",
    ]
    if clusters:
        for name, value in sorted(clusters.items()):
            lines.append(f"| {name} | {value} | Ưu tiên kiểm tra retrieval trace và prompt theo nhóm lỗi này. |")
    else:
        lines.append("| none | 0 | Không có cụm lỗi nghiêm trọng trong V3. |")

    lines.extend(
        [
            "",
            "## 3. Phân tích 5 Whys cho 3 case yếu nhất",
        ]
    )
    for index, row in enumerate(worst, start=1):
        failure_type = row["ragas"].get("failure_type", "none")
        lines.extend(
            [
                f"### Case #{index}: {row['id']} ({failure_type})",
                f"1. **Symptom:** Judge score = {row['judge']['final_score']:.2f}; câu hỏi: {row['test_case']}",
                f"2. **Why 1:** Retrieval trả về ids: {', '.join(row.get('retrieved_ids', [])[:3]) or 'không có'}.",
                f"3. **Why 2:** Expected ids: {', '.join(row.get('expected_retrieval_ids', [])[:3]) or 'out-of-context/refusal case'}.",
                "4. **Why 3:** Nếu chunk đúng không nằm top-k, nguyên nhân nằm ở lexical matching/chunk boundary.",
                "5. **Why 4:** Nếu answer thiếu ý, prompt cần ép trả lời theo điều kiện/ngoại lệ từ context.",
                f"6. **Root Cause:** {failure_type}.",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. Kế hoạch cải tiến",
            "- Giữ recursive chunking làm baseline chính, nhưng chuẩn hóa metadata theo điều/khoản nếu có thêm thời gian.",
            "- Chỉ dùng judge API thật cho các case fail/borderline để giảm ít nhất 30% chi phí.",
            "- Cache corpus, retrieval results và judge results theo `case_id + version`.",
            "- Bổ sung red-team cases sau mỗi vòng failure analysis.",
            "",
            "## 5. Đóng góp nhóm",
            "- Cường: integration, versioning, release gate.",
            "- Thành: Day 7 law data, corpus, golden dataset.",
            "- Quân: retrieval/RAGAS-style metrics.",
            "- Chi: multi-judge consensus.",
            "- Minh: failure analysis, cost/latency report.",
            "- Toàn: data QA, report review, reflection.",
        ]
    )
    ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PATH.write_text("\n".join(lines), encoding="utf-8")


async def run_benchmark_for_version(
    version: str,
    dataset: list[dict[str, Any]],
    config: RuntimeConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    print(f"Running benchmark for {version}...")
    batch_size = int(os.getenv("BENCHMARK_BATCH_SIZE", "5"))
    runner = BenchmarkRunner(
        MainAgent(version=version, corpus_path=CORPUS_PATH),
        RetrievalEvaluator(),
        LLMJudge(),
    )
    results = await runner.run_all(dataset, batch_size=batch_size)
    return results, summarize_results(version, results, config)


async def main() -> None:
    load_dotenv(ROOT / ".env", override=True)
    config = get_runtime_config()
    if config.run_mode == "real":
        config = validate_real_env()
        print("Real evaluation mode enabled: using OpenAI main generation and OpenAI+DeepSeek judges.")

    dataset = load_dataset()
    if config.run_mode == "real":
        dataset = select_real_benchmark_dataset(dataset, limit=config.real_case_limit)
        versions = [REAL_VERSION]
        print(f"Focused real benchmark selected: {len(dataset)} cases.")
    else:
        versions = VERSIONS

    all_results: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}

    for version in versions:
        results, summary = await run_benchmark_for_version(version, dataset, config)
        all_results[version] = results
        summaries[version] = summary
        print(
            f"{version}: avg_score={summary['metrics']['avg_score']:.2f}, "
            f"hit_rate={summary['metrics']['hit_rate']:.2f}, mrr={summary['metrics']['mrr']:.2f}"
        )

    final_summary = dict(summaries[REAL_VERSION])
    if config.run_mode == "real":
        regression = decide_real_release(final_summary)
    else:
        regression = decide_release(summaries["V1_Base"], summaries["V2_RetrievalPlus"], summaries["V3_CostAware"])
    final_summary["regression"] = regression
    final_summary["version_summaries"] = summaries
    final_summary["team_contributions"] = {
        "Cuong": "Integration, versioning, release gate",
        "Thanh": "Data ingestion and golden dataset",
        "Quan": "Retrieval and RAGAS-style metrics",
        "Chi": "Multi-judge consensus",
        "Minh": "Failure analysis and QA",
        "Toan": "Light QA and reflection",
    }
    final_summary["metrics"]["release_decision"] = regression["release_decision"]
    validate_real_report(final_summary)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (REPORTS_DIR / "benchmark_results.json").write_text(
        json.dumps(
            {
                "run_mode": config.run_mode,
                "case_count": len(dataset),
                "versions": all_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_failure_analysis(final_summary, all_results)

    print("\n--- REGRESSION RELEASE GATE ---")
    print(json.dumps(regression, ensure_ascii=False, indent=2))
    print(f"Saved {REPORTS_DIR / 'summary.json'}")
    print(f"Saved {REPORTS_DIR / 'benchmark_results.json'}")
    print(f"Updated {ANALYSIS_PATH}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
