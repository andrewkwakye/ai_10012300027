# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""
Local embedding wrapper (sentence-transformers).

Why local? API-hosted embedding services kept hitting rate limits, project
access denials, or deprecated model names during development. Embedding 1,244
chunks locally on a CPU finishes in under a minute, never pays a cent, and
gives us full reproducibility for the exam.

Responsibilities:
    * Accept a list of strings, return an (N, D) NumPy array of unit vectors.
    * Cache results on disk keyed by text hash so repeated builds are fast.
    * Keep the same public interface (`embed_texts`, `embed_one`) so nothing
      downstream has to change when the provider is swapped.

The vector-store and retrieval logic are still implemented by hand in
`vector_store.py` and `retriever.py` — the sentence-transformers library is
used purely as the transport layer for producing vectors, not for indexing.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .config import EMBED_MODEL, PROCESSED_DIR
from .logger import get_logger

log = get_logger("embedder")

_CACHE_PATH = PROCESSED_DIR / "embed_cache.jsonl"

# Kept for backwards compatibility with older callers. Local
# sentence-transformers embeddings don't use Gemini's task_type hint, but
# we still include it in the cache key so any previously-cached vectors
# (from the Gemini provider) won't collide with the local ones.
TASK_DOC = "st_document"
TASK_QUERY = "st_query"


def _hash(text, task_type):
    """Cache key. Includes the model name + task_type so switching providers
    doesn't silently reuse the wrong vectors."""
    payload = f"{EMBED_MODEL}::{task_type}::{text}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _load_cache():
    if not _CACHE_PATH.exists():
        return {}
    cache = {}
    with _CACHE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[row["h"]] = row["v"]
    return cache


def _append_cache(new):
    with _CACHE_PATH.open("a", encoding="utf-8") as f:
        for h, v in new.items():
            f.write(json.dumps({"h": h, "v": v}, ensure_ascii=False) + "\n")


class Embedder:
    """Thin wrapper around a sentence-transformers model.

    The model is loaded lazily on first use so import-time cost is zero. Once
    loaded it stays in memory (a singleton pattern via the RAGPipeline).
    """

    def __init__(self, model=EMBED_MODEL):
        self.model_name = model
        self._model = None  # loaded lazily
        self._cache = _load_cache()
        log.info(
            f"Embedder ready (model={model}, cached={len(self._cache)} vectors)"
        )

    def _ensure_model(self):
        if self._model is None:
            # Heavy import — only pay the cost when we actually need to embed.
            from sentence_transformers import SentenceTransformer

            log.info(
                f"Loading sentence-transformers model '{self.model_name}' "
                "(first time: downloads ~80MB)..."
            )
            self._model = SentenceTransformer(self.model_name)
            log.info("Model loaded.")

    def _embed_batch(self, batch, task_type):
        """Embed a batch locally. task_type is ignored by the local model but
        still threaded through for API symmetry."""
        self._ensure_model()
        # `encode` returns a numpy array of shape (len(batch), D).
        arr = self._model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we L2-normalise ourselves below
        )
        return [v.tolist() for v in arr]

    def embed_texts(self, texts, batch_size=32, task_type=TASK_DOC):
        """Embed a list of texts. Returns (len(texts), D) float32 matrix."""
        hashes = [_hash(t, task_type) for t in texts]
        missing_idx = [i for i, h in enumerate(hashes) if h not in self._cache]
        new_vectors = {}

        for start in range(0, len(missing_idx), batch_size):
            idxs = missing_idx[start : start + batch_size]
            batch_texts = [texts[i] for i in idxs]
            vectors = self._embed_batch(batch_texts, task_type=task_type)
            for i, v in zip(idxs, vectors):
                new_vectors[hashes[i]] = v
            if start % (batch_size * 10) == 0 and missing_idx:
                log.info(
                    f"embed progress: {min(start + batch_size, len(missing_idx))}"
                    f" / {len(missing_idx)}"
                )

        if new_vectors:
            self._cache.update(new_vectors)
            _append_cache(new_vectors)
            log.info(
                f"embed_texts: embedded {len(new_vectors)} new / {len(texts)} total"
            )
        else:
            log.info(f"embed_texts: all {len(texts)} cache hits")

        mat = np.array([self._cache[h] for h in hashes], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        return mat

    def embed_one(self, text):
        """Embed a single query."""
        return self.embed_texts([text], task_type=TASK_QUERY)[0]
