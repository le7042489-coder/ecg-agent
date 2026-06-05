"""CLI smoke test / manual query tool for the guideline retriever.

    # zero-dependency BM25 over the bundled dev corpus:
    python -m medrax.guideline.search "房颤患者口服抗凝"

    # against a prebuilt index dir (uses dense + RRF if embeddings present):
    python -m medrax.guideline.search "房颤抗凝" --index src/medrax/guideline/index

    # show superseded versions too (demonstrates forward-resolution):
    python -m medrax.guideline.search "房颤抗凝" --all-versions
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .retriever import GuidelineRetriever

_DEFAULT_CORPUS = Path(__file__).parent / "corpus" / "ecg_guidelines_cn.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description="Query the ECG guideline corpus.")
    ap.add_argument("query", help="search query")
    ap.add_argument("--index", default=None, help="prebuilt index dir (else use bundled corpus, BM25-only)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--society", default=None)
    ap.add_argument("--topic", default=None)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--all-versions", action="store_true", help="include superseded recommendations")
    ap.add_argument("--no-dense", action="store_true", help="force BM25-only even if index has vectors")
    args = ap.parse_args()

    if args.index:
        retr = GuidelineRetriever.from_index(args.index, enable_dense=not args.no_dense)
    else:
        retr = GuidelineRetriever.from_corpus(_DEFAULT_CORPUS, enable_dense=False)

    hits = retr.search(
        args.query, k=args.k, society=args.society, topic=args.topic,
        only_current=not args.all_versions, lang=args.lang,
    )

    print(f"\nquery: {args.query!r}   corpus_version={retr.corpus_version}   "
          f"results={len(hits)}\n" + "=" * 72)
    if not hits:
        print("（无匹配结果）")
        return
    for j, h in enumerate(hits, 1):
        r = h.record
        flag = "" if r.status == "current" else f"  [{r.status}]"
        print(f"\n{j}. ({h.retrieval} score={h.score:.4f}){flag} {r.text}")
        print(f"   来源: {r.society} {r.version} · {r.section} · "
              f"推荐类别 {r.recommendation_class} / 证据级别 {r.level_of_evidence} · {r.citation}")
        if h.change_note:
            print(f"   ⚠ {h.change_note}")
    print()


if __name__ == "__main__":
    main()
