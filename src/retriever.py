# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""
Retrieval layer (Part B).

Features:
    * Top-k dense retrieval via VectorStore (cosine similarity).
    * Hybrid retrieval: dense + BM25 keyword score, combined with a tunable
      alpha weight. This is the "extension" required by Part B (option:
      Hybrid search).
    * Optional query expansion helper (simple synonym / number expansion,
      written by hand — no external framework).
    * Per-chunk feedback boost (Part G innovation, applied here so the
      retriever is the single source of truth for ranking).
    * A documented failure case and the fix that hybrid search applies to it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .config import CONFIG
from .embedder import Embedder
from .feedback import FeedbackStore
from .logger import get_logger
from .vector_store import VectorStore

log = get_logger("retriever")

# ---------------------------------------------------------------------------
# Simple tokeniser for BM25 (lowercase, keep alnum + hyphen)
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_]*")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    idx: int
    score: float          # final combined score
    dense_score: float    # cosine similarity
    bm25_score: float     # raw BM25 score (unnormalised)
    boost: float          # from feedback loop
    meta: dict

    def __repr__(self) -> str:
        return f"<RR idx={self.idx} score={self.score:.3f} src={self.meta.get('source')}>"


# ---------------------------------------------------------------------------
class Retriever:
    """
    Orchestrates dense + sparse + feedback scoring.

    Final score for chunk i given query q:
        final_i = alpha * cosine(q, v_i)
                + (1 - alpha) * normalize(bm25(q, tokens_i))
                + clamp(feedback_boost_i, -cap, +cap)
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder | None = None,
        feedback: FeedbackStore | None = None,
    ):
        self.store = store
        self.embedder = embedder or Embedder()
        self.feedback = feedback or FeedbackStore()
        log.info("Building BM25 keyword index...")
        self._tokenised_corpus = [_tokenize(m["text"]) for m in store.metas]
        self._bm25 = BM25Okapi(self._tokenised_corpus)
        log.info(f"BM25 index ready (N={len(self._tokenised_corpus)} docs)")

    # ------------------------------------------------------------------
    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-9:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = CONFIG.top_k,
        alpha: float = CONFIG.hybrid_alpha,
        expand: bool = False,
    ) -> list[RetrievalResult]:
        """
        Hybrid retrieval. Set alpha=1.0 for pure dense, alpha=0.0 for pure BM25.
        """
        if expand:
            query = expand_query(query)

        # --- dense scores over the whole corpus -----------------------
        q_vec = self.embedder.embed_one(query)
        dense_scores = self.store.matrix @ q_vec  # (N,)

        # --- sparse scores -------------------------------------------
        bm25_raw = np.asarray(self._bm25.get_scores(_tokenize(query)), dtype=np.float32)

        # --- normalise both to [0,1] for combinable weighting --------
        dense_norm = self._minmax(dense_scores)
        bm25_norm = self._minmax(bm25_raw)

        combined = alpha * dense_norm + (1 - alpha) * bm25_norm

        # --- apply feedback boost -------------------------------------
        for i, meta in enumerate(self.store.metas):
            b = self.feedback.boost_for(meta["chunk_id"])
            if b != 0.0:
                combined[i] += np.clip(b, -CONFIG.feedback_cap, CONFIG.feedback_cap)

        # --- pick top-k ----------------------------------------------
        k = min(top_k, len(combined))
        part = np.argpartition(-combined, k - 1)[:k]
        order = part[np.argsort(-combined[part])]

        results = []
        for idx in order:
            results.append(
                RetrievalResult(
                    idx=int(idx),
                    score=float(combined[idx]),
                    dense_score=float(dense_scores[idx]),
                    bm25_score=float(bm25_raw[idx]),
                    boost=float(self.feedback.boost_for(self.store.metas[idx]["chunk_id"])),
                    meta=self.store.metas[idx],
                )
            )
        return results


# ---------------------------------------------------------------------------
# Query expansion (hand-rolled, very lightweight)
# ---------------------------------------------------------------------------
_SYNONYMS: dict[str, list[str]] = {
    "gdp": ["gross domestic product"],
    "budget": ["fiscal", "expenditure"],
    "npp": ["new patriotic party"],
    "ndc": ["national democratic congress"],
    "mp": ["member of parliament"],
    "election": ["poll", "vote"],
    "revenue": ["income", "tax"],
    "deficit": ["shortfall"],
    "inflation": ["cpi"],
    "exchange rate": ["cedi", "forex"],
}


def expand_query(q: str) -> str:
    """
    Extremely simple expansion: append synonyms for any matched term.
    Kept intentionally dumb so we can *show* when it helps vs hurts.
    """
    q_low = q.lower()
    adds: list[str] = []
    for term, syns in _SYNONYMS.items():
        if term in q_low:
            adds.extend(syns)
    if not adds:
        return q
    return f"{q} ({' '.join(adds)})"
