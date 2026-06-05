"""Offline: DeepSeek expands the ECG-GBench seed by generating several natural
Chinese question phrasings per CURRENT corpus record. We own the structure
(gold_id = the source record, so gold is never wrong); DeepSeek only varies the
question wording. Output: benchmark/ecg_gbench_v1.jsonl candidate (review then keep).

    DEEPSEEK_API_KEY=... python src/medrax/guideline/scripts/expand_benchmark.py
"""
from __future__ import annotations
import json, os, sys, urllib.request
from pathlib import Path

GDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GDIR.parents[1]))  # .../src
from medrax.guideline.schema import load_corpus

CORPUS = GDIR / "corpus" / "ecg_guidelines_cn.jsonl"
OUT = GDIR / "benchmark" / "ecg_gbench_v3.jsonl"
N_PER = 2
MODEL, ENDPOINT = "deepseek-v4-pro", "https://api.deepseek.com/chat/completions"


def call(prompt, max_tokens=12000):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.3}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={
        "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read())


def main():
    recs = load_corpus(CORPUS)
    current = [r for r in recs if r.status == "current"]
    # current -> superseded counterpart (for temporal items)
    stale_of = {}
    for r in recs:
        if r.status == "superseded" and r.superseded_by:
            stale_of.setdefault(r.superseded_by[0], []).append(r.id)

    from collections import defaultdict
    bytopic = defaultdict(list)
    for r in current:
        bytopic[r.topic].append(r)

    items, n = [], 0
    for topic, rs in bytopic.items():       # chunk per topic (avoid truncation)
        ask = [{"id": r.id, "推荐": r.text} for r in rs]
        prompt = (
            f"你在为一个心电图指南问答基准出题。下面每条是一条中文指南推荐。"
            f"为每条写 {N_PER} 个**不同问法**的中文问题，要求：问题的正确答案就是该条推荐；"
            f"口吻多样（患者口语 / 医生提问）；不要直接照抄推荐原文的句子；每题一句话。\n"
            f'严格只返回 JSON：{{"id": ["问题1",...]}}，不要 markdown、不要多余文字。\n\n'
            + json.dumps(ask, ensure_ascii=False))
        try:
            resp = call(prompt)
            ch = resp["choices"][0]
            content = ch["message"]["content"].strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            qmap = json.loads(content)
        except Exception as e:
            print(f"[{topic}] FAIL: {e}", file=sys.stderr); continue
        for r in rs:
            for i, q in enumerate(qmap.get(r.id, [])):
                if not isinstance(q, str) or not q.strip():
                    continue
                item = {"id": f"{r.id}__{i+1}", "question": q.strip(), "gold_ids": [r.id], "topic": r.topic}
                if r.id in stale_of:
                    item["type"] = "temporal"; item["stale_ids"] = stale_of[r.id]
                else:
                    item["type"] = "current"
                items.append(item); n += 1
        print(f"[{topic}] ok", file=sys.stderr)
    OUT.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")
    print(f"wrote {n} items -> {OUT.name} | temporal={sum(1 for it in items if it['type']=='temporal')}", file=sys.stderr)
    # small sample to stderr for review
    for it in items[:3] + items[-2:]:
        print(f"  {it['id']:22s} [{it['type']}] {it['question']}", file=sys.stderr)


if __name__ == "__main__":
    main()
