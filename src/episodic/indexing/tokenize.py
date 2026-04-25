"""Shared tokenization for BM25 / token-level features."""

from __future__ import annotations

import re
from functools import lru_cache

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

_DEFAULT_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "by", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "i", "you", "he", "she",
    "we", "they", "them", "his", "her", "their", "my", "our", "your",
    "at", "from", "into", "about", "than", "then", "so", "if", "not",
    "do", "does", "did", "have", "has", "had", "will", "would", "can",
    "could", "should", "may", "might", "must", "what", "which", "who",
    "whom", "whose", "where", "when", "why", "how", "use", "using",
}


@lru_cache(maxsize=1)
def stopwords() -> frozenset[str]:
    """Try NLTK stopwords; fall back to a built-in list if unavailable."""
    try:
        from nltk.corpus import stopwords as nltk_sw  # type: ignore

        try:
            return frozenset(nltk_sw.words("english"))
        except LookupError:
            import nltk  # type: ignore

            nltk.download("stopwords", quiet=True)
            return frozenset(nltk_sw.words("english"))
    except Exception:
        return frozenset(_DEFAULT_STOPWORDS)


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    toks = _WORD_RE.findall((text or "").lower())
    if drop_stopwords:
        sw = stopwords()
        toks = [t for t in toks if t not in sw]
    return toks
