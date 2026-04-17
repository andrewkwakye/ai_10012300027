# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""
Three hand-rolled chunking strategies (Part A).

We deliberately avoid LangChain's splitters. Each function takes a list of
documents (as produced by data_loader) and returns a list of chunk dicts:

    { "chunk_id": str, "doc_id": str, "source": str, "text": str,
      "meta": {...}, "strategy": str }

Strategies implemented
----------------------
1) fixed_token_chunks   — hard split every N tokens with M overlap.
2) recursive_char_chunks — recursively split on decreasing-strength
                           separators (paragraph -> sentence -> word) while
                           respecting a max character budget and overlap.
                           This is what LangChain calls "RecursiveCharacterTextSplitter",
                           re-implemented from scratch.
3) row_chunks           — one row == one chunk. Used for CSV data where
                           a row is already the natural semantic unit.

Why three? Per the exam rubric we must justify chunk size / overlap AND
produce a comparative analysis of chunking impact on retrieval quality.
See docs/chunking_comparison.md.
"""

from __future__ import annotations

import hashlib
import re
from typing import Callable, Iterable

import tiktoken

from .config import CONFIG
from .logger import get_logger

log = get_logger("chunker")

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
# cl100k_base is the tokenizer for text-embedding-3-* and GPT-4o-mini.
_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Strategy 1: fixed-token sliding window
# ---------------------------------------------------------------------------
def fixed_token_chunks(
    docs: list[dict],
    chunk_size: int = CONFIG.chunk_size_tokens,
    overlap: int = CONFIG.chunk_overlap_tokens,
) -> list[dict]:
    """
    Hard-tokenize each document then slice with a sliding window.
    Pros: predictable size, easy embedding budgeting.
    Cons: cuts through sentences / sections.
    """
    assert overlap < chunk_size, "overlap must be < chunk_size"
    chunks: list[dict] = []
    for d in docs:
        tokens = _ENC.encode(d["text"])
        if not tokens:
            continue
        start = 0
        step = chunk_size - overlap
        while start < len(tokens):
            window = tokens[start : start + chunk_size]
            text = _ENC.decode(window).strip()
            if len(text) < 30:
                start += step
                continue
            chunks.append(
                {
                    "chunk_id": f"{d['doc_id']}::fix::{_hash(text)}",
                    "doc_id": d["doc_id"],
                    "source": d["source"],
                    "text": text,
                    "meta": {**d.get("meta", {}), "token_start": start},
                    "strategy": "fixed_token",
                }
            )
            if start + chunk_size >= len(tokens):
                break
            start += step
    log.info(f"fixed_token_chunks: {len(docs)} docs -> {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: recursive character splitter (re-implemented from scratch)
# ---------------------------------------------------------------------------
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def _recursive_split(text: str, max_chars: int, separators: list[str]) -> list[str]:
    """
    Split text by the first separator that yields pieces below max_chars.
    If nothing works, fall back to hard character slicing.
    """
    if len(text) <= max_chars:
        return [text]
    for sep in separators:
        if sep not in text:
            continue
        parts = text.split(sep)
        pieces: list[str] = []
        buf = ""
        for p in parts:
            candidate = (buf + sep + p) if buf else p
            if len(candidate) <= max_chars:
                buf = candidate
            else:
                if buf:
                    pieces.append(buf)
                # recurse on oversized part
                if len(p) > max_chars:
                    pieces.extend(
                        _recursive_split(p, max_chars, separators[separators.index(sep) + 1 :])
                    )
                    buf = ""
                else:
                    buf = p
        if buf:
            pieces.append(buf)
        # if splitting worked (no piece still oversized), return
        if all(len(x) <= max_chars for x in pieces):
            return pieces
    # final fallback: brute-force char slice
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def recursive_char_chunks(
    docs: list[dict],
    chunk_size_tokens: int = CONFIG.chunk_size_tokens,
    overlap_tokens: int = CONFIG.chunk_overlap_tokens,
) -> list[dict]:
    """
    Prefer to split at paragraph boundaries, then sentences, then words.
    Overlap is applied by re-appending the last `overlap_tokens` of chunk N
    to the front of chunk N+1. Produces more coherent chunks than fixed_token
    but slightly uneven sizes.
    """
    # Approximate: tokens ≈ chars/4 for English. We recurse on characters
    # because that's a simpler invariant, then enforce token budget at end.
    max_chars = chunk_size_tokens * 4
    overlap_chars = overlap_tokens * 4

    chunks: list[dict] = []
    for d in docs:
        pieces = _recursive_split(d["text"], max_chars, _SEPARATORS)
        # add overlap
        merged: list[str] = []
        for i, p in enumerate(pieces):
            if i == 0 or overlap_chars == 0:
                merged.append(p.strip())
            else:
                tail = pieces[i - 1][-overlap_chars:]
                merged.append((tail + " " + p).strip())
        for p in merged:
            if len(p) < 30:
                continue
            # enforce token ceiling (safety — embeddings cap at ~8k)
            if count_tokens(p) > chunk_size_tokens * 2:
                tokens = _ENC.encode(p)
                p = _ENC.decode(tokens[: chunk_size_tokens * 2])
            chunks.append(
                {
                    "chunk_id": f"{d['doc_id']}::rec::{_hash(p)}",
                    "doc_id": d["doc_id"],
                    "source": d["source"],
                    "text": p,
                    "meta": d.get("meta", {}).copy(),
                    "strategy": "recursive_char",
                }
            )
    log.info(f"recursive_char_chunks: {len(docs)} docs -> {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# Strategy 3: one-row-per-chunk (for CSV)
# ---------------------------------------------------------------------------
def row_chunks(docs: list[dict]) -> list[dict]:
    """
    A no-op splitter for already-atomic units (CSV rows). Keeps row semantics
    intact — splitting a single election result across chunks would be silly.
    """
    chunks = []
    for d in docs:
        chunks.append(
            {
                "chunk_id": f"{d['doc_id']}::row",
                "doc_id": d["doc_id"],
                "source": d["source"],
                "text": d["text"],
                "meta": d.get("meta", {}).copy(),
                "strategy": "row",
            }
        )
    log.info(f"row_chunks: {len(docs)} docs -> {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# Router: pick strategy per source
# ---------------------------------------------------------------------------
def chunk_documents(docs: list[dict]) -> list[dict]:
    """
    Default strategy used by the build script:
        - CSV rows  -> row_chunks (semantic unit = 1 row)
        - PDF pages -> recursive_char_chunks (preserves paragraph structure)
    Returns a single flat list of chunks.
    """
    csv_docs = [d for d in docs if d["source"].endswith(".csv")]
    pdf_docs = [d for d in docs if d["source"].endswith(".pdf")]

    chunks = []
    if csv_docs:
        chunks.extend(row_chunks(csv_docs))
    if pdf_docs:
        chunks.extend(recursive_char_chunks(pdf_docs))
    log.info(f"chunk_documents (routed): total={len(chunks)}")
    return chunks


STRATEGIES: dict[str, Callable[[list[dict]], list[dict]]] = {
    "fixed_token": fixed_token_chunks,
    "recursive_char": recursive_char_chunks,
    "row": row_chunks,
    "routed": chunk_documents,
}
