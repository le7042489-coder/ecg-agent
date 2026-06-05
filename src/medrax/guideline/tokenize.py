"""Lightweight CJK-aware tokenization for BM25.

Uses ``jieba`` when available (better Chinese segmentation); otherwise falls back
to a dependency-free tokenizer: ASCII word runs + CJK unigrams and bigrams. The
bigrams let pure-Python BM25 match multi-character Chinese terms reasonably well
without any external dependency.
"""

from __future__ import annotations

import re
from typing import List

_ASCII = re.compile(r"[a-zA-Z0-9]+")
_CJK_RUN = re.compile(r"[一-鿿]+")

_jieba = None
_jieba_tried = False


def _get_jieba():
    global _jieba, _jieba_tried
    if not _jieba_tried:
        _jieba_tried = True
        try:
            import jieba  # type: ignore

            jieba.setLogLevel(60)  # silence
            _jieba = jieba
        except Exception:
            _jieba = None
    return _jieba


def _fallback_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    tokens.extend(_ASCII.findall(text.lower()))
    for run in _CJK_RUN.findall(text):
        chars = list(run)
        tokens.extend(chars)  # unigrams
        tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))  # bigrams
    return tokens


def tokenize(text: str, lang: str = "zh") -> List[str]:
    """Tokenize text for BM25. ``lang`` is advisory; mixed scripts are handled."""
    if not text:
        return []
    jb = _get_jieba()
    if jb is not None:
        toks = [t.strip().lower() for t in jb.lcut(text)]
        toks = [t for t in toks if t and not t.isspace()]
        # keep ASCII numbers/words and CJK words; drop pure punctuation
        return [t for t in toks if _ASCII.match(t) or _CJK_RUN.search(t)]
    return _fallback_tokens(text)
