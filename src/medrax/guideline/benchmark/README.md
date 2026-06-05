# ECG-GBench (v0 seed)

A small versioned ECG **grounding/QA** benchmark for the guideline tool. v0 is a
**14-item hand-authored seed** over the v0.1 Chinese corpus (covers the 6 topics +
the 3 version pairs as `temporal` items). Expand with DeepSeek bulk-gen later.

## Item format (`ecg_gbench_v0.jsonl`)
```json
{"id": "...", "question": "中文问题",
 "gold_ids": ["af_anticoag_2023"],        // correct corpus record(s)
 "topic": "atrial_fibrillation",
 "type": "current" | "temporal",
 "stale_ids": ["af_anticoag_2018"],       // (temporal) the superseded record that must NOT be served
 "findings": ["AFIB"]}                     // (optional) SCP codes, for the findings→query path
```

## Run (now, no model / no dense index)
```bash
PYTHONPATH=src python -m medrax.guideline.eval_gbench           # question→retrieval, k=5
PYTHONPATH=src python -m medrax.guideline.eval_gbench --mode findings   # SCP-codes→CN query path
```
Reports: **Recall@k / MRR / nDCG@k / hit-rate@k**, and for `temporal` items
**current-served@k**, **stale-excluded**, **forward-resolution-fires** — i.e. the
version-aware contribution, measured directly.

## What's covered now vs later
| track | status |
|---|---|
| retrieval (BM25) + temporal correctness | ✅ runs now |
| dense / hybrid baseline | ⬜ needs the offline dense index (deferred) |
| MedRAG / RAG² baselines | ⬜ external systems (M2) |
| answer correctness / citation faithfulness / hallucination | ⬜ needs the generation model + LLM judge (reuse `src/eval/faithfulness.py` DeepSeek client) |
| efficiency frontier, multi-turn | ⬜ M2 |

## Caveats
- Corpus is **AI-drafted, `verified=false`** — these scores measure *retrieval
  behaviour*, not clinical correctness of the answers.
- nDCG uses single-grade relevance (gold = 1); fine for mostly single-gold items.
