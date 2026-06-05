"""ECG-GBench answer-quality track: does grounding on retrieved guideline improve
answer **correctness** and **citation faithfulness**?

For each QA item we generate two answers — NO-RAG (model alone) and WITH-RAG
(model + retrieved guideline context) — then an LLM judge scores each against the
**gold** recommendation. The headline is the no-RAG → with-RAG *delta*.

Generator + judge default to DeepSeek (deepseek-v4-pro). Caveat: same model
generates and judges, so absolute numbers are soft; the *delta* (constant bias
across both conditions) is the signal. Swap in the device 3B as generator for the
deployed-system number. The LLM client is injectable (`client=`) for testing.

    DEEPSEEK_API_KEY=... PYTHONPATH=src python -m medrax.guideline.eval_answer --limit 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional

from .retriever import GuidelineRetriever
from .eval_gbench import load_benchmark

MODEL, ENDPOINT = "deepseek-v4-pro", "https://api.deepseek.com/chat/completions"


def _deepseek(prompt: str, max_tokens: int = 2000, temperature: float = 0.0) -> str:
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    import time
    last = None
    for attempt in range(4):                       # retry transient network errors
        try:
            req = urllib.request.Request(ENDPOINT, data=body, headers={
                "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last = e; time.sleep(2 * (attempt + 1))
    raise last


def _context(retr: GuidelineRetriever, question: str, k: int = 3) -> str:
    lines = []
    for h in retr.search(question, k=k, only_current=True):
        r = h.record
        lines.append(f"- {r.text}（{r.society} {r.version}, 推荐类别 {r.recommendation_class}"
                     f"/证据级别 {r.level_of_evidence}, {r.citation}）")
    return "\n".join(lines)


def _gen(client, question: str, context: Optional[str]) -> str:
    if context:
        p = ("你是心电图可穿戴设备助手，回答简洁。**根据下面的【参考指南】**回答用户问题，"
             "并在合适处注明推荐类别/证据级别/出处。\n【参考指南】\n" + context + f"\n【问题】{question}")
    else:
        p = f"你是心电图可穿戴设备助手，回答简洁。回答用户问题。\n【问题】{question}"
    return client(p, 1500)


def _judge(client, question: str, gold, ans_norag: str, ans_rag: str) -> dict:
    p = ("你是严格的医学评审。下面是【标准答案】(摘自指南)以及两条助手回答。"
         "分别判断每条回答：correct=是否与标准答案的核心推荐一致(1/0)；"
         "faithful=是否未编造、且引用的推荐类别/证据级别/出处与标准答案不矛盾(1/0)。\n"
         '严格只返回 JSON：{"A":{"correct":0/1,"faithful":0/1},"B":{"correct":0/1,"faithful":0/1}}\n'
         f"【标准答案】{gold.text}（推荐类别{gold.recommendation_class}/证据级别{gold.level_of_evidence}，{gold.citation}）\n"
         f"【问题】{question}\n【回答A（无指南）】{ans_norag}\n【回答B（有指南）】{ans_rag}")
    out = client(p, 2500).strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(out)


def run(retr: GuidelineRetriever, items: List[dict], k: int = 3,
        limit: Optional[int] = None, client: Callable = _deepseek) -> dict:
    by_id = {r.id: r for r in retr.records}
    items = items[:limit] if limit else items
    rows = []
    for it in items:
        gold = by_id[it["gold_ids"][0]]
        ctx = _context(retr, it["question"], k=k)
        a = _gen(client, it["question"], None)
        b = _gen(client, it["question"], ctx)
        j = _judge(client, it["question"], gold, a, b)
        rows.append({"id": it.get("id"), "norag": j.get("A", {}), "rag": j.get("B", {})})
    return {"aggregate": _aggregate(rows), "per_item": rows}


def _aggregate(rows: List[dict]) -> dict:
    n = len(rows) or 1
    def _mean(cond, key):
        return round(sum(int(r[cond].get(key, 0)) for r in rows) / n, 4)
    agg = {"n": len(rows),
           "no_rag": {"correct": _mean("norag", "correct"), "faithful": _mean("norag", "faithful")},
           "with_rag": {"correct": _mean("rag", "correct"), "faithful": _mean("rag", "faithful")}}
    agg["delta"] = {"correct": round(agg["with_rag"]["correct"] - agg["no_rag"]["correct"], 4),
                    "faithful": round(agg["with_rag"]["faithful"] - agg["no_rag"]["faithful"], 4)}
    return agg


def judge_file(answers_path, retr: GuidelineRetriever, client: Callable = _deepseek) -> dict:
    """Judge pre-generated answers (from eval_gen_3b.py: {id, question, gold_id,
    ans_norag, ans_rag} per line) with the LLM judge against gold."""
    by_id = {r.id: r for r in retr.records}
    rows = []
    for ln in Path(answers_path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        a = json.loads(ln)
        try:
            j = _judge(client, a["question"], by_id[a["gold_id"]], a["ans_norag"], a["ans_rag"])
            rows.append({"id": a.get("id"), "norag": j.get("A", {}), "rag": j.get("B", {})})
        except Exception as e:
            import sys as _s; print(f"[judge skip {a.get('id')}] {e}", file=_s.stderr)
    return {"aggregate": _aggregate(rows), "per_item": rows}


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="ECG-GBench answer-quality (no-RAG vs RAG, LLM-judged).")
    ap.add_argument("--corpus", default=str(here / "corpus" / "ecg_guidelines_cn.jsonl"))
    ap.add_argument("--bench", default=str(here / "benchmark" / "ecg_gbench_v1.jsonl"))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--judge-file", default=None,
                    help="judge a pre-generated answers jsonl (from eval_gen_3b.py) instead of generating")
    args = ap.parse_args()
    retr = GuidelineRetriever.from_corpus(args.corpus, enable_dense=False)
    if args.judge_file:
        res = judge_file(args.judge_file, retr)
    else:
        res = run(retr, load_benchmark(args.bench), k=args.k, limit=args.limit)
    print(json.dumps(res["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
