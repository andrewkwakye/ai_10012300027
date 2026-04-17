# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""
Custom NumPy-backed vector store.

We deliberately do NOT use FAISS or Chroma — the exam rubric asks us to
implement our own. For the size of this corpus (< 50k chunks) a dense
matmul against L2-normalised vectors is more than fast enough and lets us
reason clearly about the math.

On-disk layout
--------------
    embeddings.npy  : (N, D) float32 matrix (L2-normalised)
    meta.jsonl      : N lines, one per chunk, containing
                      {chunk_id, doc_id, source, text, meta, strategy}

Both files are produced by scripts/build_index.py and loaded here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import EMBEDDINGS_PATH, META_PATH
from .logger import get_logger

log = get_logger("vectorstore")


class VectorStore:
    def __init__(self, matrix: np.ndarray, metas: list[dict]):
        assert matrix.shape[0] == len(metas), "matrix rows must match metas length"
        # Defensive L2 normalise (idempotent since embedder already does it)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = (matrix / norms).astype(np.float32)
        self.metas = metas

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, emb_path: Path = EMBEDDINGS_PATH, meta_path: Path = META_PATH) -> "VectorStore":
        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Missing index files ({emb_path} / {meta_path}). "
                "Run `python scripts/build_index.py` first."
            )
        matrix = np.load(emb_path)
        metas = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    metas.append(json.loads(line))
        log.info(f"VectorStore loaded: {matrix.shape[0]} vectors, dim={matrix.shape[1]}")
        return cls(matrix, metas)

    def save(self, emb_path: Path = EMBEDDINGS_PATH, meta_path: Path = META_PATH) -> None:
        np.save(emb_path, self.matrix)
        with meta_path.open("w", encoding="utf-8") as f:
            for m in self.metas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        log.info(f"VectorStore saved: {len(self.metas)} records -> {emb_path}, {meta_path}")

    # ------------------------------------------------------------------
    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """
        Cosine similarity search.
        Returns a list of (index, score) pairs sorted by descending score.
        Because all vectors are L2-normalised, cosine similarity == dot product.
        """
        if query_vec.ndim == 1:
            q = query_vec / (np.linalg.norm(query_vec) or 1.0)
        else:
            q = query_vec.squeeze()
            q = q / (np.linalg.norm(q) or 1.0)
        scores = self.matrix @ q  # shape (N,)
        top_k = min(top_k, len(scores))
        # argpartition is O(N); sort only the top-k for the final order
        part = np.argpartition(-scores, top_k - 1)[:top_k]
        ordered = part[np.argsort(-scores[part])]
        return [(int(i), float(scores[i])) for i in ordered]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.metas)

    def get(self, idx: int) -> dict:
        return self.metas[idx]
