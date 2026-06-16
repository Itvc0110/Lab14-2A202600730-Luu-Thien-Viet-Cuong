import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestLab14Pipeline(unittest.TestCase):
    def test_recursive_chunk_text_uses_stable_overlap(self):
        from data.synthetic_gen import recursive_chunk_text

        text = " ".join(f"word{i}" for i in range(220))
        chunks = recursive_chunk_text(text, chunk_size=120, overlap=20)

        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(chunk) <= 140 for chunk in chunks))
        self.assertTrue(any(chunks[0][-10:].strip() in chunk for chunk in chunks[1:2]))

    def test_build_corpus_from_law_dir_creates_stable_chunk_ids(self):
        from data.synthetic_gen import build_corpus_from_law_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            law_dir = root / "law"
            law_dir.mkdir()
            (law_dir / "luat_bhxh_X.txt").write_text(
                "Điều 1. Người lao động đóng bảo hiểm xã hội.\n\n"
                "Điều 2. Thời gian đóng bảo hiểm xã hội được tính theo tháng.",
                encoding="utf-8",
            )
            corpus_path = root / "corpus.jsonl"

            rows = build_corpus_from_law_dir(law_dir, corpus_path, chunk_size=80, overlap=10)

            self.assertGreaterEqual(len(rows), 1)
            self.assertTrue(rows[0]["id"].startswith("luat_bhxh_X_chunk_"))
            self.assertEqual(rows[0]["metadata"]["domain"], "bao_hiem_xa_hoi")
            self.assertTrue(corpus_path.exists())

    def test_natural_golden_set_2_is_normalized_to_lab14_schema(self):
        from data.synthetic_gen import normalize_natural_golden_cases

        corpus_rows = [
            {
                "id": "luat_lao_dong_I_chunk_0001",
                "content": "Độ tuổi lao động tối thiểu của người lao động là đủ 15 tuổi.",
                "metadata": {"domain": "lao_dong", "source": "data/law_corpus/luat_lao_dong_I.txt"},
            }
        ]
        raw_cases = [
            {
                "question": "Tuổi lao động tối thiểu là bao nhiêu?",
                "expected_answer": "Độ tuổi lao động tối thiểu là đủ 15 tuổi.",
                "context": "Độ tuổi lao động tối thiểu của người lao động là đủ 15 tuổi.",
                "expected_retrieval_ids": ["doc_002"],
                "metadata": {"difficulty": "easy", "type": "fact-check", "domain": "legal", "source_file": "synthetic_legal"},
            }
        ]

        normalized = normalize_natural_golden_cases(raw_cases, corpus_rows)

        self.assertEqual(normalized[0]["id"], "law_natural_001")
        self.assertEqual(normalized[0]["type"], "fact")
        self.assertEqual(normalized[0]["domain"], "lao_dong")
        self.assertEqual(normalized[0]["expected_retrieval_ids"], ["luat_lao_dong_I_chunk_0001"])
        self.assertNotIn("chunk_", normalized[0]["question"])

    def test_real_mode_requires_openrouter_embedding_key(self):
        from engine.real_config import validate_real_env

        with self.assertRaises(RuntimeError):
            validate_real_env(
                {
                    "EVAL_MODE": "real",
                    "ALLOW_FALLBACK": "false",
                    "OPENAI_API_KEY": "sk-real-main",
                    "OPENAI_JUDGE_API_KEY": "sk-real-judge",
                    "EMBEDDING_PROVIDER": "openrouter",
                    "EMBEDDING_MODEL": "openai/text-embedding-3-small",
                    "OPENROUTER_JUDGE_MODEL": "deepseek/deepseek-v4-flash",
                }
            )

    def test_vector_index_can_be_built_and_loaded_for_search(self):
        from engine.vector_store import MockEmbedder, VectorStore, build_vector_index, load_vector_index

        rows = [
            {
                "id": "doc_a",
                "content": "quy định về tuổi lao động tối thiểu là đủ 15 tuổi",
                "metadata": {"domain": "lao_dong"},
            },
            {
                "id": "doc_b",
                "content": "quy định về tài sản chung của vợ chồng",
                "metadata": {"domain": "hon_nhan_gia_dinh"},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "vector_index.jsonl"
            build_vector_index(rows, index_path=index_path, embedding_fn=MockEmbedder())
            loaded = load_vector_index(index_path)
            store = VectorStore.from_index(loaded, embedding_fn=MockEmbedder())
            results = store.search("tuổi lao động tối thiểu", top_k=1)

            self.assertTrue(index_path.exists())
            self.assertEqual(len(loaded), 2)
            self.assertEqual(results[0]["id"], "doc_a")


    def test_retrieval_metrics_score_expected_ids(self):
        from engine.retrieval_eval import RetrievalEvaluator

        evaluator = RetrievalEvaluator()
        case = {"expected_retrieval_ids": ["doc_b"], "expected_answer": "quy định thai sản"}
        response = {
            "answer": "quy định thai sản được áp dụng theo tài liệu",
            "retrieved_ids": ["doc_a", "doc_b", "doc_c"],
            "contexts": [
                {"id": "doc_a", "text": "nội dung khác"},
                {"id": "doc_b", "text": "quy định thai sản được áp dụng theo tài liệu"},
                {"id": "doc_c", "text": "nội dung bổ sung"},
            ],
        }

        scores = asyncio.run(evaluator.score(case, response))

        self.assertEqual(scores["retrieval"]["hit_rate"], 1.0)
        self.assertAlmostEqual(scores["retrieval"]["mrr"], 0.5)
        self.assertGreater(scores["generation"]["faithfulness"], 0.5)

    def test_main_agent_returns_required_response_contract(self):
        from agent.main_agent import MainAgent

        with tempfile.TemporaryDirectory() as tmp:
            corpus_path = Path(tmp) / "corpus.jsonl"
            rows = [
                {
                    "id": "luat_lao_dong_X_chunk_0001",
                    "content": "Thời gian thử việc tối đa là 60 ngày đối với công việc cần trình độ cao đẳng.",
                    "metadata": {
                        "domain": "lao_dong",
                        "source": "data/law_corpus/luat_lao_dong_X.txt",
                        "chunk_index": 1,
                    },
                }
            ]
            corpus_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

            agent = MainAgent(version="V2_RetrievalPlus", corpus_path=corpus_path)
            response = asyncio.run(agent.query("Thời gian thử việc tối đa là bao nhiêu ngày?"))

            self.assertIn("answer", response)
            self.assertIn("retrieved_ids", response)
            self.assertIn("contexts", response)
            self.assertIn("metadata", response)
            self.assertEqual(response["retrieved_ids"], ["luat_lao_dong_X_chunk_0001"])

    def test_llm_judge_fallback_returns_consensus_shape(self):
        from engine.llm_judge import LLMJudge

        judge = LLMJudge(judge_mode="fallback")
        result = asyncio.run(
            judge.evaluate_multi_judge(
                "Điều kiện hưởng thai sản là gì?",
                "Người lao động nữ phải đóng bảo hiểm xã hội đủ 06 tháng trong 12 tháng trước sinh.",
                "Người lao động nữ phải đóng bảo hiểm xã hội đủ 06 tháng trong 12 tháng trước sinh.",
            )
        )

        self.assertGreaterEqual(result["final_score"], 4.0)
        self.assertIn("openai", result["individual_scores"])
        self.assertIn("deepseek", result["individual_scores"])
        self.assertIn("agreement_rate", result)

    def test_placeholder_env_is_rejected_for_real_mode(self):
        from engine.real_config import validate_real_env

        with self.assertRaises(RuntimeError):
            validate_real_env(
                {
                    "EVAL_MODE": "real",
                    "OPENAI_API_KEY": "fill_me",
                    "OPENAI_JUDGE_API_KEY": "sk-real-judge",
                    "OPENROUTER_API_KEY": "openrouter-real",
                    "OPENROUTER_JUDGE_MODEL": "deepseek/deepseek-v4-flash",
                }
            )

    def test_real_mode_refuses_missing_keys(self):
        from engine.real_config import validate_real_env

        with self.assertRaises(RuntimeError):
            validate_real_env({"EVAL_MODE": "real", "ALLOW_FALLBACK": "false"})

    def test_llm_judge_api_mode_records_api_when_mocked_calls_succeed(self):
        from engine.llm_judge import LLMJudge

        judge = LLMJudge(judge_mode="api")
        judge.openai_key = "sk-real-openai"
        judge.openrouter_key = "openrouter-real"
        judge.max_api_cases = 10
        with patch.object(judge, "_call_openai", return_value=(4.5, "OpenAI API rubric pass")), patch.object(
            judge, "_call_deepseek", return_value=(4.0, "DeepSeek API rubric pass")
        ):
            result = asyncio.run(
                judge.evaluate_multi_judge(
                    "What is the rule?",
                    "The answer is grounded in the retrieved context.",
                    "The answer is grounded in the retrieved context.",
                )
            )

        self.assertEqual(result["judge_modes"], {"openai": "api", "deepseek": "api"})
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["api_call_counts"], {"openai_judge": 1, "deepseek_judge": 1})

    def test_select_real_benchmark_dataset_is_stratified_to_30_cases(self):
        from main import select_real_benchmark_dataset

        dataset = []
        for case_type in ["fact", "reasoning", "multi_hop", "adversarial"]:
            for index in range(8):
                dataset.append(
                    {
                        "id": f"{case_type}_{index}",
                        "question": "q",
                        "expected_answer": "a",
                        "expected_retrieval_ids": ["doc_1"],
                        "domain": "legal",
                        "difficulty": "medium",
                        "type": case_type,
                        "source": "test",
                    }
                )

        selected = select_real_benchmark_dataset(dataset, limit=30)
        counts = {}
        for case in selected:
            counts[case["type"]] = counts.get(case["type"], 0) + 1

        self.assertEqual(len(selected), 30)
        self.assertGreaterEqual(counts["fact"], 7)
        self.assertGreaterEqual(counts["reasoning"], 7)
        self.assertGreaterEqual(counts["multi_hop"], 7)
        self.assertGreaterEqual(counts["adversarial"], 7)

    def test_real_report_validation_rejects_fallback(self):
        from main import validate_real_report

        summary = {
            "metadata": {
                "run_mode": "real",
                "fallback_used": True,
                "judge_modes": {"fallback": 2},
            },
            "metrics": {"avg_score": 4.2, "hit_rate": 0.9, "agreement_rate": 0.9},
        }

        with self.assertRaises(ValueError):
            validate_real_report(summary)

    def test_custom_api_base_url_resolution(self):
        from engine.real_config import get_runtime_config
        config = get_runtime_config({
            "OPENAI_API_BASE": "https://api.shopaikey.com/v1",
            "OPENROUTER_API_BASE": "https://api.shopaikey.com/v1"
        })
        self.assertEqual(config.openai_api_base, "https://api.shopaikey.com/v1")
        self.assertEqual(config.openrouter_api_base, "https://api.shopaikey.com/v1")


if __name__ == "__main__":
    unittest.main()

