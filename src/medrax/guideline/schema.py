"""Guideline corpus record schema + loaders.

One record == one recommendation (or one coherent section). The schema is
metadata-first so retrieval can filter by version/society/topic and so every
returned snippet carries a citation with recommendation class + level of evidence.

JSON keys use ``class`` / ``LOE`` (clinical convention); they map to the Python
field names ``recommendation_class`` / ``level_of_evidence`` (``class`` is a
reserved word).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class GuidelineRecord:
    """A single versioned guideline recommendation."""

    id: str
    society: str                 # e.g. CSC / ESC / ACC-AHA-HRS
    doc_title: str
    version: str                 # e.g. "2023"
    status: str                  # "current" | "superseded"
    topic: str                   # coarse ECG topic, see manifest.topics
    section: str
    recommendation_class: str    # JSON: "class"  (I / IIa / IIb / III)
    level_of_evidence: str       # JSON: "LOE"    (A / B / C)
    citation: str
    text: str
    lang: str = "zh"
    effective_date: str = ""
    supersedes: List[str] = field(default_factory=list)
    superseded_by: List[str] = field(default_factory=list)
    verified: bool = False        # False = AI-drafted, not yet clinically verified (比赛阶段)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "society": self.society,
            "doc_title": self.doc_title,
            "version": self.version,
            "effective_date": self.effective_date,
            "status": self.status,
            "topic": self.topic,
            "section": self.section,
            "class": self.recommendation_class,
            "LOE": self.level_of_evidence,
            "lang": self.lang,
            "citation": self.citation,
            "text": self.text,
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
            "verified": self.verified,
        }


def record_from_dict(obj: dict) -> GuidelineRecord:
    """Build a record from a corpus JSON object (tolerant of missing optionals)."""
    try:
        return GuidelineRecord(
            id=obj["id"],
            society=obj.get("society", ""),
            doc_title=obj.get("doc_title", ""),
            version=str(obj.get("version", "")),
            status=obj.get("status", "current"),
            topic=obj.get("topic", ""),
            section=obj.get("section", ""),
            recommendation_class=str(obj.get("class", obj.get("recommendation_class", ""))),
            level_of_evidence=str(obj.get("LOE", obj.get("level_of_evidence", ""))),
            citation=obj.get("citation", ""),
            text=obj.get("text", ""),
            lang=obj.get("lang", "zh"),
            effective_date=obj.get("effective_date", ""),
            supersedes=list(obj.get("supersedes", []) or []),
            superseded_by=list(obj.get("superseded_by", []) or []),
            verified=bool(obj.get("verified", False)),
        )
    except KeyError as e:
        raise ValueError(f"guideline record missing required key {e}: {obj!r}") from e


def load_corpus(path: str | Path) -> List[GuidelineRecord]:
    """Load a JSONL corpus file (one record per line)."""
    path = Path(path)
    records: List[GuidelineRecord] = []
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln}: invalid JSON: {e}") from e
            records.append(record_from_dict(obj))
    return records


def load_corpus_version(corpus_path: str | Path) -> str:
    """Read corpus_version from a sibling manifest.json, if any."""
    manifest = Path(corpus_path).parent / "manifest.json"
    if manifest.is_file():
        try:
            return json.loads(manifest.read_text(encoding="utf-8")).get("corpus_version", "unknown")
        except Exception:
            return "unknown"
    return "unknown"
