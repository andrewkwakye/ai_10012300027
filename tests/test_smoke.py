# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""
Smoke test — verifies every module imports cleanly and pure-Python
helpers work without an API key. Does NOT hit the Groq API.

Usage:
    python -m tests.test_smoke
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_imports():
    modules = [
        "src.config",
        "src.logger",
        "src.data_loader",
        "src.chunker",
        "src.vector_store",
        "src.retriever",
        "src.prompt_builder",
        "src.feedback",
        "src.evaluator",
        # Note: src.embedder lazy-loads sentence-transformers and src.llm imports
        # the Groq SDK — both import fine without an API key, only instantiation
        # (or, for the embedder, first .embed() call) would fail.
        "src.embedder",
        "src.llm",
        "src.pipeline",
    ]
    for m in modules:
        importlib.import_module(m)
        print(f"  [OK] import {m}")


def test_chunker_without_api():
    from src.chunker import fixed_token_chunks, recursive_char_chunks, row_chunks

    docs = [
        {
            "doc_id": "t1",
            "source": "test.txt",
            "text": "This is a small document. " * 80,
            "meta": {},
        }
    ]

    fixed = fixed_token_chunks(docs, chunk_size=100, overlap=20)
    assert len(fixed) >= 2, f"fixed chunker produced {len(fixed)} chunks"
    print(f"  [OK] fixed_token_chunks -> {len(fixed)} chunks")

    rec = recursive_char_chunks(docs, chunk_size_tokens=100, overlap_tokens=20)
    assert len(rec) >= 1, "recursive chunker produced nothing"
    print(f"  [OK] recursive_char_chunks -> {len(rec)} chunks")

    rows = row_chunks(docs)
    assert len(rows) == 1, "row chunker should be 1:1"
    print(f"  [OK] row_chunks -> {len(rows)} chunks")


def test_prompt_templates():
    from src.prompt_builder import TEMPLATES

    ctx = "[#1] example chunk"
    for name, fn in TEMPLATES.items():
        p = fn("What is Ghana's capital?", ctx)
        assert p.system and p.user
        assert "CONTEXT" in p.user or "Context" in p.user
        print(f"  [OK] prompt {name}")


def test_feedback_roundtrip(tmp_path_factory=None):
    import tempfile
    from pathlib import Path

    from src import feedback as fmod
    from src.config import CONFIG

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "feedback.jsonl"
        store = fmod.FeedbackStore(path=p)
        store.record("c1", "q", +1)
        store.record("c1", "q", +1)
        store.record("c2", "q", -1)
        assert abs(store.boost_for("c1") - 2 * CONFIG.feedback_boost) < 1e-6
        assert abs(store.boost_for("c2") - (-CONFIG.feedback_boost)) < 1e-6
        store2 = fmod.FeedbackStore(path=p)
        assert abs(store2.boost_for("c1") - 2 * CONFIG.feedback_boost) < 1e-6
        print("  [OK] feedback roundtrip")


if __name__ == "__main__":
    print("== imports ==")
    test_imports()
    print("\n== chunker ==")
    test_chunker_without_api()
    print("\n== prompts ==")
    test_prompt_templates()
    print("\n== feedback ==")
    test_feedback_roundtrip()
    print("\nAll smoke tests passed.")
