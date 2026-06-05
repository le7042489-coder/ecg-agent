"""ECG-GBench: retrieval + temporal-correctness evaluation for the guideline tool.

Runs NOW with zero heavy deps (corpus + a QA set with gold record ids):
  * Retrieval quality: Recall@k, MRR, nDCG@k, hit-rate@k.
  * Temporal correctness (the version-aware contribution): for `type=="temporal"`
    items, does retrieval serve the CURRENT record and exclude the superseded
    (`stale_ids`) one under only_current, and does forward-resolution fire when
    superseded versions are allowed?

NOT covered yet (see benchmark/README.md): dense/hybrid baseline (needs the
offline dense index), MedRAG / RAG² baselines (external, M2), and answer
correctness / citation faithfulness (needs the generation model + an LLM judge —
reuse src/eval/faithfulness.py's DeepSeek client).

    PYTHONPATH=src python -m medrax.guideline.eval_gbench \
        --corpus  src/medrax/guideline/corpus/ecg_guidelines_cn.jsonl \
        --bench   src/medrax/guideline/benchmark/ecg_gbench_v0.jsonl  --k 5
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional

from .retriever import GuidelineRetriever
from .findings import findings_to_query


def load_benchmark(path) -> List[dict]:
    items = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("//"):
            items.append(json.loads(ln))
    return items


def _first_gold_rank(ranked_ids: List[str], gold: set) -> Optional[int]:
    """1-based rank of the first gold id in ranked_ids, or None."""
    for i, rid in enumerate(ranked_ids):
        if rid in gold:
            return i + 1
    return None


def evaluate(retr: GuidelineRetriever, items: List[dict], k: int = 5,
             query_mode: str = "question") -> tuple[dict, List[dict]]:
    """query_mode: 'question' (use item['question']) or 'findings' (map SCP codes
    in item['findings'] → CN query; items without findings are skipped)."""
    per: List[dict] = []
    for it in items:
        if query_mode == "findings":
            q, _ = findings_to_query(it.get("findings", []))
            if not q:
                continue
        else:
            q = it["question"]
        gold = set(it["gold_ids"])
        ranked = [h.record.id for h in retr.search(q, k=k, only_current=True)]
        rank = _first_gold_rank(ranked, gold)
        stale = set(it.get("stale_ids", []))
        row = {
            "id": it.get("id"), "type": it.get("type", "current"), "rank": rank,
            "hit": rank is not None,
            "recall": (len(gold & set(ranked)) / len(gold)) if gold else 0.0,
            "rr": (1.0 / rank) if rank else 0.0,
            "ndcg": (1.0 / math.log2(rank + 1)) if rank else 0.0,  # single-grade, IDCG=1
            "stale_served": bool(stale & set(ranked)),
            "fwd_ok": None,
        }
        if it.get("type") == "temporal" and stale:
            allv = retr.search(q, k=max(k, 10), only_current=False)
            notes = {h.record.id: (h.change_note is not None) for h in allv}
            seen = [s for s in stale if s in notes]
            row["fwd_ok"] = bool(seen) and all(notes[s] for s in seen)
        per.append(row)

    n = len(per) or 1
    agg = {
        "n": len(per), "k": k, "query_mode": query_mode,
        "recall@k": round(sum(p["recall"] for p in per) / n, 4),
        "MRR": round(sum(p["rr"] for p in per) / n, 4),
        "nDCG@k": round(sum(p["ndcg"] for p in per) / n, 4),
        "hit_rate@k": round(sum(1 for p in per if p["hit"]) / n, 4),
    }
    temps = [p for p in per if p["type"] == "temporal"]
    if temps:
        tn = len(temps)
        agg["temporal_n"] = tn
        agg["temporal_current_served@k"] = round(sum(1 for p in temps if p["hit"]) / tn, 4)
        agg["temporal_stale_excluded"] = round(sum(1 for p in temps if not p["stale_served"]) / tn, 4)
        fo = [p for p in temps if p["fwd_ok"] is not None]
        if fo:
            agg["forward_resolution_fires"] = round(sum(1 for p in fo if p["fwd_ok"]) / len(fo), 4)
    return agg, per


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Run ECG-GBench retrieval/temporal evaluation.")
    ap.add_argument("--corpus", default=str(here / "corpus" / "ecg_guidelines_cn.jsonl"))
    ap.add_argument("--bench", default=str(here / "benchmark" / "ecg_gbench_v0.jsonl"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", choices=["question", "findings"], default="question")
    ap.add_argument("--index", default=None, help="prebuilt index dir → hybrid (dense+BM25+RRF); else BM25-only")
    args = ap.parse_args()

    if args.index:
        retr = GuidelineRetriever.from_index(args.index, enable_dense=True)
        method = "hybrid"
    else:
        retr = GuidelineRetriever.from_corpus(args.corpus, enable_dense=False)
        method = "bm25"
    items = load_benchmark(args.bench)
    agg, per = evaluate(retr, items, k=args.k, query_mode=args.mode)
    agg["method"] = method

    print(f"\nECG-GBench  corpus={retr.corpus_version}  bench={Path(args.bench).name}  "
          f"mode={args.mode}  method={method}\n" + "=" * 72)
    for p in per:
        flag = "" if p["hit"] else "  ✗MISS"
        t = " [temporal]" if p["type"] == "temporal" else ""
        print(f"  {p['id']:18s} rank={str(p['rank']):4s} ndcg={p['ndcg']:.3f}{t}{flag}")
    print("-" * 72)
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print("\n(注: dense/hybrid/MedRAG/RAG² 基线与 回答质量/引用忠实 见 benchmark/README.md，需后续接入)")


if __name__ == "__main__":
    main()
