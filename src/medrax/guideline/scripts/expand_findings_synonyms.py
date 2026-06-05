"""Offline helper: ask DeepSeek to SUGGEST extra Chinese synonyms for the
actionable SCP codes in ``findings_map.json`` (query-expansion terms for BM25
against a Chinese cardiology-guideline corpus).

It only PRINTS suggestions (JSON of additions, already de-duped against existing
terms) — a human curates and merges into findings_map.json. Never auto-merges:
clinical terms need review.

    DEEPSEEK_API_KEY=... python src/medrax/guideline/scripts/expand_findings_synonyms.py

deepseek-v4-pro is a reasoning model, so max_tokens is set high (reasoning tokens
are spent before the answer).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

MAP_PATH = Path(__file__).resolve().parent.parent / "findings_map.json"
ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-pro"


def _call(prompt: str, max_tokens: int = 8000) -> dict:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("DEEPSEEK_API_KEY not set")
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main() -> None:
    fmap = json.loads(MAP_PATH.read_text(encoding="utf-8"))["map"]
    actionable = {c: e for c, e in fmap.items() if e.get("topic")}

    lines = ["你是心电图/心血管临床术语专家。下面是 ECG 分类器的 SCP 标签（英文）及其当前中文检索词。",
             "请为每个标签补充**额外的**中文检索同义词/别名（医生或患者常用、适合在中文心血管指南语料里做 BM25 检索扩展）。",
             "要求：只给中文词、简短(一般2-8字)；不要重复已有词；不确定就给空数组；",
             '严格只返回 JSON：{"CODE":["词",...],...}，不要 markdown、不要多余文字。', "", "标签："]
    for c, e in actionable.items():
        lines.append(f"{c} ({e['en']}) 当前: {json.dumps(e.get('cn', []), ensure_ascii=False)}")
    prompt = "\n".join(lines)

    resp = _call(prompt)
    ch = resp["choices"][0]
    content = ch["message"]["content"]
    print(f"[finish={ch.get('finish_reason')} usage={resp.get('usage', {}).get('completion_tokens')} tok]",
          file=sys.stderr)
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        sugg = json.loads(content)
    except Exception:
        print(content)  # raw, for manual salvage
        sys.exit("could not parse JSON; raw above")

    # keep only genuine additions
    additions = {}
    for c, terms in sugg.items():
        cur = set(fmap.get(c, {}).get("cn", []))
        new = [t for t in terms if isinstance(t, str) and t and t not in cur]
        if new:
            additions[c] = new
    print(json.dumps(additions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
