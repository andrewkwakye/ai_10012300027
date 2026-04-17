# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""
Prompt engineering layer (Part C).

We ship three prompt templates so we can run the rubric-required
"same query with different prompts" experiment. See docs/prompt_iterations.md.

All templates are plain Python f-strings — no Jinja, no framework —
and they share a single context-assembly routine that:
    * Truncates the retrieved chunk list to fit CONFIG.max_context_tokens
    * Numbers each chunk so the model can cite it by id
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import tiktoken

from .config import CONFIG
from .logger import get_logger

if TYPE_CHECKING:
    from .retriever import RetrievalResult

log = get_logger("prompt")

_ENC = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Context packing
# ---------------------------------------------------------------------------
def pack_context(
    results: list[RetrievalResult],
    budget_tokens: int = CONFIG.max_context_tokens,
) -> tuple[str, list[RetrievalResult]]:
    """
    Concatenate retrieved chunks in rank order, stopping when the token
    budget is exhausted. Returns (context_string, chunks_actually_used).

    Each chunk is framed as:
        [#1 | source=... | score=0.82]
        <text>
    so the generator can cite it.
    """
    used: list[RetrievalResult] = []
    pieces: list[str] = []
    total = 0
    for i, r in enumerate(results, start=1):
        header = f"[#{i} | source={r.meta.get('source')} | score={r.score:.2f}]"
        body = r.meta.get("text", "")
        block = f"{header}\n{body}"
        tks = len(_ENC.encode(block))
        if total + tks > budget_tokens and used:
            # budget exhausted; stop adding
            break
        pieces.append(block)
        used.append(r)
        total += tks
    ctx = "\n\n---\n\n".join(pieces)
    log.info(f"pack_context: used {len(used)}/{len(results)} chunks, {total} tokens")
    return ctx, used


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
@dataclass
class Prompt:
    system: str
    user: str


# --- v1 -- minimal (baseline) ----------------------------------------------
_V1_SYSTEM = "You are a helpful assistant for Academic City University. Answer the user's question using the provided context."

def build_v1(question: str, context: str) -> Prompt:
    return Prompt(
        system=_V1_SYSTEM,
        user=f"Context:\n{context}\n\nQuestion: {question}\nAnswer:",
    )


# --- v2 -- anti-hallucination + citation ------------------------------------
_V2_SYSTEM = (
    "You are an evidence-grounded research assistant for Academic City University. "
    "You answer questions strictly using the CONTEXT provided below. "
    "If the context does not contain enough information to answer, say exactly: "
    "\"I don't have enough information in my source documents to answer that.\" "
    "Never invent numbers, names, dates, or policies that are not in the context. "
    "When you give a fact, cite the chunk number in square brackets, e.g. [#2]."
)

def build_v2(question: str, context: str) -> Prompt:
    return Prompt(
        system=_V2_SYSTEM,
        user=(
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer concisely, using only the context. Cite chunk numbers for each fact."
        ),
    )


# --- v3 -- structured thinking with grounding check ------------------------
_V3_SYSTEM = (
    "You are Academic City University's evidence-grounded research assistant.\n"
    "Follow this procedure on every question:\n"
    "1. Read the CONTEXT carefully.\n"
    "2. Identify which chunks, if any, contain information that directly answers the question.\n"
    "3. If no chunk is directly relevant, reply exactly: "
    "\"I don't have enough information in my source documents to answer that.\"\n"
    "4. Otherwise, produce a short factual answer grounded entirely in the chunks you identified, "
    "and cite them in square brackets [#n]. Do NOT use outside knowledge.\n"
    "5. Do not speculate about causes, motivations, or future events unless the context states them.\n"
    "Numbers, dates, and proper nouns must come verbatim from the context."
)

def build_v3(question: str, context: str) -> Prompt:
    return Prompt(
        system=_V3_SYSTEM,
        user=(
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Remember: evidence-grounded only, cite chunk numbers, refuse if unsupported."
        ),
    )


TEMPLATES = {
    "v1_minimal": build_v1,
    "v2_guarded": build_v2,
    "v3_structured": build_v3,
}

DEFAULT_TEMPLATE = "v3_structured"
