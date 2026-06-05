"""Offline index builder. Run on the DEV LAPTOP, ship the output dir to DK-2500.

It precomputes corpus embeddings so the on-device path only encodes the query.

    python -m medrax.guideline.build_index \
        --corpus src/medrax/guideline/corpus/ecg_guidelines_cn.jsonl \
        --out    src/medrax/guideline/index \
        --model  BAAI/bge-small-zh-v1.5

Use --no-dense to build a BM25-only index (no embedding model needed). The
output dir contains:
    records.jsonl          normalized copy of the corpus
    embeddings.npy         float32 [N, d], L2-normalized (omitted with --no-dense)
    index_manifest.json    corpus_version, embed_model, dim, counts, checksum
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .schema import load_corpus, load_corpus_version


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build(corpus: str, out: str, model: str = "BAAI/bge-small-zh-v1.5", no_dense: bool = False) -> None:
    corpus_path = Path(corpus)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_corpus(corpus_path)
    corpus_version = load_corpus_version(corpus_path)
    print(f"[build] {len(records)} records | corpus_version={corpus_version}")

    # normalized records copy
    records_path = out_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    embed_model, dim = None, None
    if not no_dense:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        print(f"[build] encoding with {model} (this runs offline on the laptop)...")
        st = SentenceTransformer(model)
        texts = [f"{r.text} {r.section}".strip() for r in records]
        vecs = st.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        vecs = np.asarray(vecs, dtype="float32")
        np.save(out_dir / "embeddings.npy", vecs)
        embed_model, dim = model, int(vecs.shape[1])
        print(f"[build] wrote embeddings.npy {vecs.shape}")
    else:
        print("[build] --no-dense: BM25-only index (no embeddings)")

    manifest = {
        "corpus_version": corpus_version,
        "embed_model": embed_model,
        "embed_dim": dim,
        "record_count": len(records),
        "records_sha256": _sha256(records_path),
        "dense": not no_dense,
    }
    (out_dir / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[build] index ready at {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a guideline retrieval index (offline).")
    ap.add_argument("--corpus", required=True, help="corpus JSONL path")
    ap.add_argument("--out", required=True, help="output index dir")
    ap.add_argument("--model", default="BAAI/bge-small-zh-v1.5", help="sentence-transformers model")
    ap.add_argument("--no-dense", action="store_true", help="BM25-only index, skip embeddings")
    args = ap.parse_args()
    build(args.corpus, args.out, args.model, args.no_dense)


if __name__ == "__main__":
    main()
