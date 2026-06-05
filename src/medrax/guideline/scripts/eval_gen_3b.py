"""DEVICE-side: generate NO-RAG and WITH-RAG answers for the ECG-GBench items
using the on-device 3B (llama-cpp GGUF). Writes one JSON line per item; the
laptop then judges them with DeepSeek (eval_answer.py --judge-file).

Split design: the slow part (3B generation, ~CPU) runs here and needs no API key;
judging (DeepSeek) runs on the laptop. Output is appended + flushed per item so
progress is visible and a crash keeps partial results.

    PYTHONPATH=src python src/medrax/guideline/scripts/eval_gen_3b.py \
        --gguf ~/workspace/ECG-Agent/ecg_agent_llama3b_q4km.gguf \
        --bench src/medrax/guideline/benchmark/ecg_gbench_v2.jsonl \
        --out /tmp/ans3b.jsonl --max-tokens 128 [--limit N]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

from medrax.guideline.retriever import GuidelineRetriever
from medrax.guideline.eval_gbench import load_benchmark

SYS = "你是心电图可穿戴设备助手，用简洁的中文直接回答用户的问题（2-4 句）。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="/tmp/ans3b.jsonl")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from llama_cpp import Llama
    llm = Llama(model_path=args.gguf, n_ctx=4096, n_threads=os.cpu_count(), verbose=False)
    retr = GuidelineRetriever.from_corpus(args.corpus, enable_dense=False)
    items = load_benchmark(args.bench)
    if args.limit:
        import random
        random.Random(0).shuffle(items)   # seeded → representative span across topics
        items = items[:args.limit]

    def ctx_for(q):
        out = []
        for h in retr.search(q, k=args.k, only_current=True):
            r = h.record
            out.append(f"- {r.text}（{r.society} {r.version}, 推荐类别 {r.recommendation_class}"
                       f"/证据级别 {r.level_of_evidence}, {r.citation}）")
        return "\n".join(out)

    def gen(question, context):
        if context:
            user = "根据下面的【参考指南】回答问题，可注明推荐类别/证据级别/出处。\n【参考指南】\n" + context + f"\n【问题】{question}"
        else:
            user = question
        r = llm.create_chat_completion(
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            max_tokens=args.max_tokens, temperature=0.0)
        return r["choices"][0]["message"]["content"].strip()

    open(args.out, "w").close()  # fresh
    t0 = time.perf_counter()
    with open(args.out, "a", encoding="utf-8") as f:
        for i, it in enumerate(items, 1):
            ctx = ctx_for(it["question"])
            rec = {"id": it.get("id"), "question": it["question"], "gold_id": it["gold_ids"][0],
                   "ans_norag": gen(it["question"], None), "ans_rag": gen(it["question"], ctx)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if i % 5 == 0 or i == len(items):
                print(f"[{i}/{len(items)}] {time.perf_counter()-t0:.0f}s", flush=True)
    print(f"=== GEN DONE {len(items)} items in {time.perf_counter()-t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
