"""Versioned, ECG-only guideline retrieval for ECG-Agent.

Design goals (see src/medrax/guideline/README.md):
  - small + versioned + ECG-only corpus
  - on-device / offline / light (DK-2500, 8GB, CPU): query-time path has NO hard
    third-party dependency (pure-Python BM25 fallback); dense + jieba are optional
    accelerators used automatically when present.
  - index building is an offline batch job (run on the dev laptop, ship the index).

Public API:
    from medrax.guideline import GuidelineRetriever
    retr = GuidelineRetriever.from_corpus(".../ecg_guidelines_cn.jsonl")
    hits = retr.search("房颤患者口服抗凝", k=3)
"""

from .schema import GuidelineRecord, load_corpus
from .retriever import GuidelineRetriever, SearchResult
from .findings import findings_to_query, parse_findings, load_findings_map, ground_by_findings

__all__ = [
    "GuidelineRecord", "load_corpus", "GuidelineRetriever", "SearchResult",
    "findings_to_query", "parse_findings", "load_findings_map", "ground_by_findings",
]
