"""LangChain tool wrapping the versioned ECG guideline retriever.

Used by the research / LangGraph `Agent` path (src/medrax/agent/agent.py): add an
instance to the tools list passed to `Agent(...)`. The bedside loop
(dk2500_deploy/bedside_agent.py) can instead call `medrax.guideline.GuidelineRetriever`
directly (see src/medrax/guideline/README.md).

The retriever is loaded lazily and prefers a prebuilt index dir given by the
``ECG_GUIDELINE_INDEX`` env var (dense + RRF); otherwise it falls back to the
bundled dev corpus with BM25-only (no model download needed).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Type

from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from medrax.guideline import GuidelineRetriever

__all__ = ["GuidelineRetrievalTool", "GuidelineSearchInput"]

_RETRIEVER: Optional[GuidelineRetriever] = None


def _get_retriever() -> GuidelineRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        idx = os.environ.get("ECG_GUIDELINE_INDEX")
        if idx and Path(idx).is_dir():
            _RETRIEVER = GuidelineRetriever.from_index(idx, enable_dense=True)
        else:
            corpus = (Path(__file__).resolve().parent.parent
                      / "guideline" / "corpus" / "ecg_guidelines_cn.jsonl")
            _RETRIEVER = GuidelineRetriever.from_corpus(corpus, enable_dense=False)
    return _RETRIEVER


class GuidelineSearchInput(BaseModel):
    """Input schema for the ECG guideline search tool."""

    query: str = Field(..., description="临床问题或 ECG 发现，例如『房颤患者是否需要抗凝』")
    society: Optional[str] = Field(None, description="按学会过滤，如 CSC / ESC / ACC-AHA-HRS")
    topic: Optional[str] = Field(
        None,
        description="按主题过滤: atrial_fibrillation / bradycardia_conduction / "
                    "qt_channelopathy / ischemia_stemi / svt / electrolyte",
    )
    only_current: bool = Field(True, description="仅检索现行版本指南（默认 True，避免过期推荐）")
    k: int = Field(5, description="返回条数")


class GuidelineRetrievalTool(BaseTool):
    """Retrieve version-aware, citation-bearing ECG guideline recommendations."""

    name: str = "ecg_guideline_search"
    description: str = (
        "检索 ECG 相关临床指南推荐。当需要依据指南回答诊断、处置、适应证、"
        "抗凝、起搏、再灌注等临床问题时使用。返回的每条推荐都带有学会、版本、"
        "推荐类别(Class)和证据级别(LOE)及出处。"
        "重要：回答中引用指南时必须注明 Class/LOE/版本；没有检索结果时不得编造指南内容。"
    )
    args_schema: Type[BaseModel] = GuidelineSearchInput
    return_direct: bool = False

    def _run(
        self,
        query: str,
        society: Optional[str] = None,
        topic: Optional[str] = None,
        only_current: bool = True,
        k: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        retr = _get_retriever()
        hits = retr.search(query, k=k, society=society, topic=topic, only_current=only_current)
        if not hits:
            return f"[指南检索 · corpus={retr.corpus_version}] 未检索到相关推荐。"

        lines = [f"[指南检索结果 · corpus_version={retr.corpus_version}]"]
        for j, h in enumerate(hits, 1):
            r = h.record
            lines.append(
                f"{j}. {r.text}\n"
                f"   来源: {r.society} {r.version} · 推荐类别 {r.recommendation_class}"
                f" / 证据级别 {r.level_of_evidence} · {r.citation}"
            )
            if h.change_note:
                lines.append(f"   注意: {h.change_note}")
        return "\n".join(lines)

    async def _arun(self, *args, **kwargs) -> str:  # pragma: no cover - simple delegation
        return self._run(*args, **kwargs)
