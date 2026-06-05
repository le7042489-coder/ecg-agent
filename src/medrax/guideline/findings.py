"""Turn ECG classifier findings (SCP-ECG codes) into a Chinese guideline query.

The on-device classifier emits English SCP codes with probabilities, e.g. the
cached string ``"['PACE(0.49)', 'ILMI(0.46)', 'AFIB(0.29)']"``. The guideline
corpus is Chinese, so feeding raw codes to BM25 yields zero token overlap
(the cross-lingual gap observed on the device). This module maps codes to
curated Chinese terms (+ a coarse topic) via ``findings_map.json`` so retrieval
grounds on the right recommendations.

``findings_to_query`` returns ``("", set())`` when nothing maps (e.g. only
NORM/SR/PACE) — the caller then injects no guideline context, instead of
grounding on arbitrary text.

Clinical code->topic decisions live in ``findings_map.json`` (human-owned).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

_MAP_PATH = Path(__file__).parent / "findings_map.json"

# A single "CODE" or "CODE(prob)" token (codes may contain _ / ( ) per SCP set).
_TOKEN_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_/()]*?)\s*(?:\(\s*([0-9.]+)\s*\))?\s*$")


@lru_cache(maxsize=1)
def load_findings_map() -> dict:
    """Load the SCP-code → {en, cn, topic} map (cached)."""
    return json.loads(_MAP_PATH.read_text(encoding="utf-8")).get("map", {})


def parse_findings(findings: Union[str, List, Tuple]) -> List[Tuple[str, Optional[float]]]:
    """Parse classifier findings into ``[(code, prob_or_None), ...]``.

    Accepts a list/tuple of ``"CODE"`` / ``"CODE(prob)"`` items, or the
    ``repr``-style cached string ``"['AFIB(0.95)', '3AVB(0.40)']"``.
    """
    items: List[str]
    if isinstance(findings, (list, tuple)):
        items = [str(x) for x in findings]
    else:
        s = str(findings)
        items = re.findall(r"'([^']*)'", s) or re.findall(r'"([^"]*)"', s)
        if not items:  # bare/space/comma separated, no quotes
            items = re.split(r"[,\s]+", s.strip().strip("[]"))

    out: List[Tuple[str, Optional[float]]] = []
    for it in items:
        m = _TOKEN_RE.match(it)
        if not m:
            continue
        code, prob = m.group(1), m.group(2)
        out.append((code, float(prob) if prob is not None else None))
    return out


def findings_to_query(
    findings: Union[str, List, Tuple],
    prob_threshold: float = 0.0,
) -> Tuple[str, Set[str]]:
    """Build ``(cn_query, topics)`` for guideline retrieval from classifier findings.

    Codes below ``prob_threshold`` (when a probability is given), unmapped codes,
    and non-actionable codes (``topic`` is null, e.g. NORM/SR/PACE) are skipped.
    Returns ``("", set())`` if nothing maps — caller should then inject nothing.
    CN terms are de-duplicated preserving first-seen order.
    """
    fmap = load_findings_map()
    terms: List[str] = []
    topics: Set[str] = set()
    seen: Set[str] = set()
    for code, prob in parse_findings(findings):
        if prob is not None and prob < prob_threshold:
            continue
        entry = fmap.get(code)
        if not entry or not entry.get("topic"):
            continue
        topics.add(entry["topic"])
        for t in entry.get("cn", []):
            if t not in seen:
                seen.add(t)
                terms.append(t)
    return " ".join(terms), topics


def ground_by_findings(retr, findings, k: int = 3, prob_threshold: float = 0.0,
                       only_current: bool = True):
    """Map classifier ``findings`` to a CN query + topics, then retrieve **scoped
    to those topics**. Returns ``[]`` when nothing maps (caller injects nothing).

    Topic scoping matters: a query mixing several findings' CN terms would
    otherwise let the bigram tokenizer bleed across topics (e.g. 心肌 in both
    心肌梗死 and 心肌细胞膜); restricting to the findings' own topics removes that.
    ``retr`` is any object with a ``.search(query, k, topic, only_current)`` API.
    """
    query, topics = findings_to_query(findings, prob_threshold)
    if not query:
        return []
    return retr.search(query, k=k, topic=(topics or None), only_current=only_current)
