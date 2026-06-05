# ECG Guideline Retrieval (versioned, on-device)

A small, **versioned, ECG-only** guideline corpus + a lightweight hybrid
retriever, exposed to ECG-Agent as a tool. Built for the DK-2500 bedside target
(x86_64, 8GB, CPU, offline).

**Status: M0 vertical slice + M0.5 engineering done.** Relocated into the
ECG-Agent repo, BM25 path test-covered (`tests/`), and wired into the bedside
loop (auto-grounding, **off by default**). The dense index is **deferred to the
M1 corpus** (see `HANDOFF.md`). The corpus under `corpus/` is a *development
placeholder* — paraphrased, illustrative, with placeholder citations; real
curation + licensing sign-off is M1.

## Why it's shaped this way
- **Version-aware**: every record carries `version` / `status` / `supersedes` /
  `superseded_by`. Retrieval defaults to `only_current=True` so superseded
  recommendations are never served; a superseded hit (with `--all-versions`) is
  annotated with a forward-resolution note to the current one.
- **Citation-faithful**: every hit returns society + version + recommendation
  `class` + `LOE` + citation, so the agent can ground claims.
- **Graceful degradation**: the query path needs **no third-party deps** (pure-
  Python BM25). Dense retrieval (bge-small-zh) + `jieba` are used automatically
  when present, fused with RRF.

## Develop on the laptop, ship the index to DK-2500
Index building is an **offline batch job** — never run it on the 8GB device.

> All paths below are relative to the **ECG-Agent repo root**. On the dev laptop
> the repo is nested at `~/workspace/ECG-Agent/ECG-Agent/` — `cd` into it first.
> See `HANDOFF.md` § 代码位置 for the two-tree layout (and why not to use the
> workspace-top-level `src/`).

```bash
# 1) (laptop) build a dense index from the corpus
pip install -r src/medrax/guideline/requirements.txt
python -m medrax.guideline.build_index \
    --corpus src/medrax/guideline/corpus/ecg_guidelines_cn.jsonl \
    --out    src/medrax/guideline/index \
    --model  BAAI/bge-small-zh-v1.5
#   → src/medrax/guideline/index/{records.jsonl, embeddings.npy, index_manifest.json}

# 2) ship the index dir alongside the GGUF (same tar/scp flow as the deploy guide)
scp -r src/medrax/guideline/index user@10.157.225.11:~/workspace/ECG-Agent/src/medrax/guideline/

# 3) (DK-2500) point the tool at the prebuilt index
export ECG_GUIDELINE_INDEX=~/workspace/ECG-Agent/src/medrax/guideline/index
```

The index for a few-thousand-record corpus is a few MB. On the 8GB device, use
`bge-small-zh-v1.5` (or an int8 ONNX export) and/or run BM25-only
(`enable_dense=False`) if RAM is tight.

## Query it (CLI smoke test)
```bash
# zero-dependency BM25 over the bundled dev corpus:
PYTHONPATH=src python -m medrax.guideline.search "房颤患者口服抗凝预防卒中" --k 3

# against a prebuilt index (dense + RRF):
PYTHONPATH=src python -m medrax.guideline.search "房颤抗凝" --index src/medrax/guideline/index

# show superseded versions + forward-resolution:
PYTHONPATH=src python -m medrax.guideline.search "房颤抗凝" --all-versions
```

## Tests
Dependency-free `unittest` suite (no pytest / numpy needed) covering the BM25
path, version filtering, forward-resolution, metadata filters, schema
round-trip, and the dense→BM25 fallback:
```bash
PYTHONPATH=src python -m unittest medrax.guideline.tests.test_retriever -v
```

## Integration

### A) Research / LangGraph `Agent` path
Add the tool to the list passed to `Agent(...)` (`src/medrax/agent/agent.py`):
```python
from medrax.tools import GuidelineRetrievalTool
tools = [..., GuidelineRetrievalTool()]
```

### B) Bedside loop (`dk2500_deploy/bedside_agent.py`) — implemented, off by default
The fine-tuned 3B wasn't trained to emit a guideline tool action, so this uses
**auto-grounding**: the bedside loop retrieves current-version guideline context
for the cached classification findings and appends it to the **classification
`Tool_Output`** — the same channel the model was trained to read tool results
from — so the response turn grounds on it without a new message type.

```bash
# off by default; enable explicitly. Uses ECG_GUIDELINE_INDEX / --guideline-index
# if given, else the bundled placeholder corpus (BM25-only, no model download):
python bedside_agent.py --mat ecg.mat --guideline
python bedside_agent.py --mat ecg.mat --guideline --guideline-index ~/…/guideline/index
```

**Off by default on purpose**: the bundled corpus has `占位-待核` citations and
must not reach a clinical response. Loading/retrieval failures degrade silently
(feature off, session unaffected); empty findings / no hits inject nothing.
`corpus_version` is printed at load and embedded in the injected block header for
provenance. The helpers (`load_guideline_retriever`, `guideline_context`) live in
`bedside_agent.py` if you want to call them elsewhere.

## Files
| file | role |
|------|------|
| `schema.py` | `GuidelineRecord` + JSONL loader |
| `tokenize.py` | CJK-aware tokenizer (jieba optional) |
| `retriever.py` | hybrid retriever, RRF, metadata filter, version resolution |
| `build_index.py` | offline index builder (laptop) |
| `search.py` | CLI |
| `corpus/` | versioned JSONL + `manifest.json` (dev placeholder) |
| `tests/` | stdlib `unittest` suite (no deps) |
| `findings_map.json` | 72 SCP classifier codes → CN query terms + topic (clinical map, human-owned) |
| `findings.py` | `findings_to_query` / `ground_by_findings`: classifier EN labels → CN, topic-scoped retrieval |
| `../tools/guideline_retrieval.py` | LangChain `BaseTool` wrapper (exported from `tools/__init__.py`) |
