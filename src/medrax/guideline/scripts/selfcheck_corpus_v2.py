"""Stage 2: DeepSeek critically self-checks each Stage-1 record (Class/LOE vs the
cited guideline, citation/version realness, clinical/internal consistency).
Per-record verdict ok|conflict + reason. Batched per topic. Conflicts (+ Stage-1
low-confidence + all version pairs) go to Claude for Stage-3 review.

    DEEPSEEK_API_KEY=... python src/medrax/guideline/scripts/selfcheck_corpus_v2.py
Output: /tmp/selfcheck_v2.jsonl  [{_idx, verdict, reason}]
"""
from __future__ import annotations
import json, os, sys, urllib.request
from collections import defaultdict

MODEL, ENDPOINT = "deepseek-v4-pro", "https://api.deepseek.com/chat/completions"
CAND, OUT = "/tmp/corpus_v2_candidate.jsonl", "/tmp/selfcheck_v2.jsonl"


def call(prompt, max_tokens=8000):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={
        "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    recs = [json.loads(l) for l in open(CAND, encoding="utf-8") if l.strip()]
    by_topic = defaultdict(list)
    for r in recs:
        by_topic[r["topic"]].append(r)

    open(OUT, "w").close()
    total_conf = 0
    for topic, rs in by_topic.items():
        batch = [{"_idx": r["_idx"], "text": r.get("text"), "class": r.get("class"),
                  "LOE": r.get("LOE"), "citation": r.get("citation"), "version": r.get("version"),
                  "status": r.get("status")} for r in rs]
        prompt = ("你是严格的中国心血管指南审稿人。下面是一批中文指南推荐(含推荐类别 class、证据级别 LOE、出处 citation、版本 version、状态 status)。"
                  "请逐条**批判性复核**：① class/LOE 是否符合所引指南的实际推荐；② 出处/版本是否真实存在(不存在或张冠李戴算 conflict)；"
                  "③ 有无临床错误或自相矛盾；④ status=superseded 的是否确为旧版本(应是同一推荐的更早**版本年份**，而非另一条不同推荐——否则 conflict)。"
                  "对每条给 verdict(ok|conflict) 和简短 reason。严格只返回 JSON 数组 "
                  '[{"_idx":"...","verdict":"ok|conflict","reason":"..."}]，不要 markdown、不要多余文字。\n\n'
                  + json.dumps(batch, ensure_ascii=False))
        try:
            resp = call(prompt)
            ch = resp["choices"][0]
            content = ch["message"]["content"].strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            verdicts = json.loads(content)
        except Exception as e:
            print(f"[{topic}] FAILED: {e}", file=sys.stderr); continue
        with open(OUT, "a", encoding="utf-8") as f:
            for v in verdicts:
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
        nc = sum(1 for v in verdicts if v.get("verdict") == "conflict")
        total_conf += nc
        print(f"[{topic}] checked {len(verdicts)}, conflicts {nc}", file=sys.stderr)
    print(f"=== SELFCHECK DONE: {total_conf} conflicts -> {OUT} ===", file=sys.stderr)


if __name__ == "__main__":
    main()
