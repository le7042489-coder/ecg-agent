"""Dependency-free tests for the guideline retriever.

Runs with the standard library only (no pytest, no numpy required for the core
paths), so it works on the bare-python on-device profile:

    PYTHONPATH=src python -m unittest medrax.guideline.tests.test_retriever -v

Tests run against the bundled *development* corpus (paraphrased placeholder
content); they assert retrieval *behaviour* (filtering, version resolution,
graceful degradation), not clinical correctness of the text.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from medrax.guideline.retriever import GuidelineRetriever
from medrax.guideline.schema import load_corpus, record_from_dict
from medrax.guideline.tokenize import tokenize

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "ecg_guidelines_cn.jsonl"

# Strong AF-anticoagulation query: should rank the current AF anticoag record top.
_AF_QUERY = "非瓣膜性心房颤动 口服抗凝 预防卒中"


def _retr() -> GuidelineRetriever:
    # BM25-only, no embeddings — the zero-dependency on-device profile.
    return GuidelineRetriever.from_corpus(CORPUS, enable_dense=False)


class CorpusLoadingTests(unittest.TestCase):
    def test_loads_all_records_and_version(self):
        retr = _retr()
        self.assertGreaterEqual(len(retr.records), 50)
        self.assertTrue(retr.corpus_version.startswith("egb-"))

    def test_superseded_records_link_to_current(self):
        recs = load_corpus(CORPUS)
        by_id = {r.id: r for r in recs}
        superseded = [r for r in recs if r.status == "superseded"]
        self.assertTrue(superseded, "corpus should have superseded (version-pair) records")
        for r in superseded:
            self.assertTrue(r.superseded_by, f"{r.id} superseded but no superseded_by")
            self.assertIn(r.superseded_by[0], by_id, f"{r.id} links to missing record")
            self.assertEqual(by_id[r.superseded_by[0]].status, "current")


class VersioningTests(unittest.TestCase):
    def test_only_current_excludes_superseded(self):
        hits = _retr().search(_AF_QUERY, k=10, only_current=True)
        self.assertTrue(hits)
        self.assertTrue(all(h.record.status == "current" for h in hits))

    def test_all_versions_includes_superseded_with_forward_note(self):
        recs = load_corpus(CORPUS)
        by_id = {r.id: r for r in recs}
        sup = next(r for r in recs if r.status == "superseded")  # any superseded record
        cur = by_id[sup.superseded_by[0]]
        hits = _retr().search(sup.text, k=10, only_current=False)
        hit = next((h for h in hits if h.record.id == sup.id), None)
        self.assertIsNotNone(hit, "superseded record should be retrievable with --all-versions")
        self.assertIsNotNone(hit.change_note, "superseded hit must carry a forward-resolution note")
        self.assertIn("取代", hit.change_note)
        self.assertIn(cur.citation, hit.change_note)

    def test_current_records_have_no_change_note(self):
        hits = _retr().search(_AF_QUERY, k=10, only_current=True)
        self.assertTrue(all(h.change_note is None for h in hits))


class FilterTests(unittest.TestCase):
    def test_topic_filter(self):
        hits = _retr().search("起搏 适应证", k=10, topic="bradycardia_conduction")
        self.assertTrue(hits)
        self.assertTrue(all(h.record.topic == "bradycardia_conduction" for h in hits))

    def test_society_filter_is_case_insensitive(self):
        hits = _retr().search("房颤", k=10, society="csc")
        self.assertTrue(hits)
        self.assertTrue(all(h.record.society == "CSC" for h in hits))

    def test_unknown_society_returns_empty(self):
        self.assertEqual(_retr().search("房颤", society="ESC"), [])

    def test_unknown_topic_returns_empty(self):
        self.assertEqual(_retr().search("房颤", topic="does_not_exist"), [])

    def test_k_limits_result_count(self):
        self.assertLessEqual(len(_retr().search("心电图", k=2)), 2)


class RankingTests(unittest.TestCase):
    def test_bm25_ranks_af_anticoag_first(self):
        hits = _retr().search(_AF_QUERY, k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0].record.topic, "atrial_fibrillation")
        self.assertEqual(hits[0].retrieval, "bm25")
        self.assertGreater(hits[0].score, 0.0)

    def test_no_token_overlap_returns_empty(self):
        """Min-score floor: a BM25-only query with zero token overlap returns []
        rather than arbitrary file-order records."""
        self.assertEqual(_retr().search("zzzqqq xkcdnomatch foobar", k=3), [])

    def test_english_labels_dont_match_chinese_corpus(self):
        """The real classifier emits EN labels; against the CN corpus BM25 finds
        nothing → empty (not the first-3 current records by file order)."""
        self.assertEqual(_retr().search("PACE AFIB ILMI QWAVE", k=3), [])

    def test_partial_token_match_still_returns(self):
        """Floor only drops zero-match: a query sharing ≥1 token still returns,
        and every returned hit has a positive score."""
        hits = _retr().search("房颤", k=5)
        self.assertTrue(hits)
        self.assertTrue(all(h.score > 0.0 for h in hits))

    def test_dense_mode_without_index_falls_back_to_bm25(self):
        """Regression: mode='dense' on a BM25-only index must degrade to BM25,
        not return arbitrary zero-scored hits mislabeled 'hybrid'."""
        hits = _retr().search(_AF_QUERY, k=3, mode="dense")
        self.assertTrue(hits, "dense mode without vectors should still return BM25 hits")
        self.assertEqual(hits[0].retrieval, "bm25")
        self.assertGreater(hits[0].score, 0.0, "scores must be real BM25, not 0.0")
        self.assertEqual(hits[0].record.topic, "atrial_fibrillation")

    def test_dense_mode_with_unloadable_encoder_falls_back(self):
        """vectors present but the query encoder can't be loaded (the realistic
        on-device failure) must also degrade to BM25 rather than raise."""
        try:
            import numpy as np
            import sentence_transformers  # noqa: F401
            self.skipTest("sentence-transformers importable; can't force encoder failure")
        except ImportError:
            pass
        import numpy as np

        records = load_corpus(CORPUS)
        fake = np.zeros((len(records), 4), dtype="float32")
        retr = GuidelineRetriever(records, vectors=fake,
                                  embed_model_name="__nonexistent_model__",
                                  enable_dense=True)
        hits = retr.search(_AF_QUERY, k=3, mode="dense")
        self.assertTrue(hits)
        self.assertEqual(hits[0].retrieval, "bm25")


class SchemaTests(unittest.TestCase):
    def test_roundtrip_preserves_class_and_loe(self):
        rec = load_corpus(CORPUS)[0]
        rt = record_from_dict(rec.to_dict())
        self.assertEqual(rt.recommendation_class, rec.recommendation_class)
        self.assertEqual(rt.level_of_evidence, rec.level_of_evidence)
        self.assertIn(rt.recommendation_class, ("I", "IIa", "IIb", "III"))

    def test_to_dict_uses_clinical_json_keys(self):
        d = next(r for r in load_corpus(CORPUS)).to_dict()
        self.assertIn("class", d)   # not "recommendation_class"
        self.assertIn("LOE", d)     # not "level_of_evidence"


class TokenizeTests(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(tokenize(""), [])

    def test_ascii_lowercased_and_cjk_present(self):
        toks = tokenize("NOAC 抗凝治疗")
        self.assertIn("noac", toks)                       # ASCII run, lowercased
        self.assertTrue(any("一" <= c <= "鿿" for t in toks for c in t),
                        "expected at least one CJK token")


if __name__ == "__main__":
    unittest.main(verbosity=2)
