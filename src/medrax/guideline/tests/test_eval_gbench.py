"""Tests for the ECG-GBench eval harness.

    PYTHONPATH=src python -m unittest medrax.guideline.tests.test_eval_gbench -v
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

from medrax.guideline import GuidelineRetriever
from medrax.guideline.eval_gbench import evaluate, load_benchmark, _first_gold_rank

GDIR = Path(__file__).resolve().parent.parent
CORPUS = GDIR / "corpus" / "ecg_guidelines_cn.jsonl"
BENCH = GDIR / "benchmark" / "ecg_gbench_v3.jsonl"


class MetricHelperTests(unittest.TestCase):
    def test_first_gold_rank(self):
        self.assertEqual(_first_gold_rank(["a", "b", "c"], {"b"}), 2)
        self.assertEqual(_first_gold_rank(["a", "b"], {"z"}), None)
        self.assertEqual(_first_gold_rank(["x", "a"], {"x"}), 1)

    def test_ndcg_formula_matches_rank(self):
        # a synthetic 2-item run: gold at rank1 and rank3
        class _Hit:
            def __init__(self, rid): self.record = type("R", (), {"id": rid, "status": "current", "superseded_by": []})()
            change_note = None
        class _Retr:
            corpus_version = "x"
            def search(self, q, k=5, only_current=True, **kw):
                return [_Hit(i) for i in (["g", "b", "c"] if q == "first" else ["b", "c", "g"])][:k]
        items = [{"id": "1", "question": "first", "gold_ids": ["g"], "type": "current"},
                 {"id": "2", "question": "third", "gold_ids": ["g"], "type": "current"}]
        agg, per = evaluate(_Retr(), items, k=5)
        self.assertEqual(per[0]["rank"], 1)
        self.assertEqual(per[1]["rank"], 3)
        self.assertAlmostEqual(per[1]["ndcg"], 1.0 / math.log2(4), places=4)
        self.assertAlmostEqual(agg["MRR"], (1.0 + 1.0 / 3) / 2, places=4)


class BenchmarkRunTests(unittest.TestCase):
    def setUp(self):
        self.retr = GuidelineRetriever.from_corpus(CORPUS, enable_dense=False)
        self.items = load_benchmark(BENCH)

    def test_bench_well_formed(self):
        ids = {r.id for r in self.retr.records}
        self.assertGreaterEqual(len(self.items), 10)
        for it in self.items:
            for g in it["gold_ids"]:
                self.assertIn(g, ids, f"{it['id']}: gold {g} not in corpus")
            for s in it.get("stale_ids", []):
                self.assertIn(s, ids, f"{it['id']}: stale {s} not in corpus")

    def test_retrieval_quality_reasonable(self):
        agg, _ = evaluate(self.retr, self.items, k=5)
        # BM25 on this seed should find most golds in top-5
        self.assertGreaterEqual(agg["recall@k"], 0.7)
        self.assertGreaterEqual(agg["hit_rate@k"], 0.7)

    def test_temporal_correctness(self):
        agg, _ = evaluate(self.retr, self.items, k=5)
        self.assertGreaterEqual(agg["temporal_n"], 3)
        # superseded versions must never be served under only_current
        self.assertEqual(agg["temporal_stale_excluded"], 1.0)
        # forward-resolution note must fire for the stale versions
        self.assertGreaterEqual(agg["forward_resolution_fires"], 0.85)  # ≥most; misses = retrieval recall, not note bug

    def test_findings_mode_runs(self):
        # items with a 'findings' field should map to a CN query and retrieve
        agg, _ = evaluate(self.retr, self.items, k=5, query_mode="findings")
        self.assertIn("n", agg)   # v3 has no 'findings' fields; just ensure it runs


class AnswerEvalTests(unittest.TestCase):
    """answer-quality harness with a mock LLM client (no network)."""

    def setUp(self):
        self.retr = GuidelineRetriever.from_corpus(CORPUS, enable_dense=False)
        self.items = load_benchmark(BENCH)

    def _fake_client(self, prompt, max_tokens, temperature=0.0):
        # judge prompt asks for JSON; generation prompt does not
        if "只返回 JSON" in prompt:
            # no-RAG answer wrong but faithful; with-RAG answer correct + faithful
            return '{"A":{"correct":0,"faithful":1},"B":{"correct":1,"faithful":1}}'
        return "（模拟回答）"

    def test_answer_eval_aggregates_and_delta(self):
        from medrax.guideline.eval_answer import run
        res = run(self.retr, self.items, k=3, limit=3, client=self._fake_client)
        agg = res["aggregate"]
        self.assertEqual(agg["n"], 3)
        self.assertEqual(agg["no_rag"]["correct"], 0.0)
        self.assertEqual(agg["with_rag"]["correct"], 1.0)
        self.assertEqual(agg["delta"]["correct"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
