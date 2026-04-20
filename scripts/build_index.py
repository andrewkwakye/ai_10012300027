# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""
One-shot index builder.

    1. Load CSV + PDF (src.data_loader).
    2. Chunk with the routed strategy (src.chunker.chunk_documents).
    3. Embed every chunk (src.embedder).
    4. Persist to data/processed/ as embeddings.npy + meta.jsonl.

Usage:
    python scripts/build_index.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunker import chunk_documents  # noqa: E402
from src.config import CHUNKS_PATH, EMBEDDINGS_PATH, META_PATH  # noqa: E402
from src.data_loader import load_all  # noqa: E402
from src.embedder import Embedder  # noqa: E402
from src.logger import get_logger  # noqa: E402

log = get_logger("build_index")


def main() -> None:
    log.info("Loading documents...")
    docs = load_all()

    log.info("Chunking...")
    chunks = chunk_documents(docs)
    # persist raw chunk list for reproducibility / debugging
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(chunks)} chunks -> {CHUNKS_PATH}")

    log.info("Embedding chunks (runs sentence-transformers locally — no API calls)...")
    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    matrix = embedder.embed_texts(texts)
    log.info(f"Embeddings matrix shape: {matrix.shape}")

    np.save(EMBEDDINGS_PATH, matrix)
    with META_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    log.info(f"Saved index: {EMBEDDINGS_PATH}, {META_PATH}")

    log.info("Done.")


if __name__ == "__main__":
    main()
