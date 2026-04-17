# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""
Evaluation harness (Part E).

Supplies:
    * A small test set of queries covering:
        - answerable factual queries (ground-truth in the corpus)
        - adversarial queries (ambiguous / misleading / unanswerable)
    * Metrics:
        - accuracy       : does the answer mention the expected substring?
        - hallucination  : did the model answer confidently on an unanswerable query?
        - consistency    : run the same query 3x, count distinct top-line answers.
    * A comparator that runs the same query with use_retrieval=True and then False.

No framework — pure string matching + a consistency hash. The exam asks
for evidence-based comparison, so we persist everything to CSV.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from .config import LOGS_DIR
from .logger import get_logger
from .pipeline import RAGPipeline

log = get_logger("evaluator")


# ---------------------------------------------------------------------------
@dataclass
class EvalCase:
    qid: str
    query: str
    kind: str             # "answerable" | "ambiguous" | "misleading" | "unanswerable"
    expect_substrings: list[str]   # any-of substrings we expect in a correct answer
    unanswerable: bool = False     # True -> a good answer is a refusal


# A small but representative set. Expand before final submission.
DEFAULT_CASES: list[EvalCase] = [
    EvalCase(
        qid="Q1",
        query="Which constituency did the NDC win in 2020 in Ashanti Region?",
        kind="answerable",
        expect_substrings=[],  # fill with actual constituency once you inspect CSV
    ),
    EvalCase(
        qid="Q2",
        query="What is the projected fiscal deficit for 2025 in Ghana?",
        kind="answerable",
        expect_substrings=["deficit", "2025"],
    ),
    EvalCase(
        qid="Q3_ambiguous",
        query="What did the government do about taxes?",
        kind="ambiguous",
        expect_substrings=["tax"],
    ),
    EvalCase(
        qid="Q4_misleading",
        query="How many votes did the NPP win in the 2030 Ghanaian election?",
        kind="misleading",  # 2030 hasn't happened
        expect_substrings=[],
        unanswerable=True,
    ),
    EvalCase(
        qid="Q5_unanswerable",
        query="What is the exchange rate of the Ghana cedi to Japanese yen today?",
        kind="unanswerable",
        expect_substrings=[],
        unanswerable=True,
    ),
]


REFUSAL_TOKENS = [
    "don't have enough information",
    "do not have enough information",
    "I don't know",
    "cannot find",
    "not in the provided context",
]


# ---------------------------------------------------------------------------
def _looks_like_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(tok.lower() in low for tok in REFUSAL_TOKENS)


def _first_line(text: str) -> str:
    return (text.strip().split("\n", 1)[0]).strip()


def _hash_line(line: str) -> str:
    return hashlib.md5(line.lower().encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
def run_case(
    pipe: RAGPipeline,
    case: EvalCase,
    *,
    use_retrieval: bool,
    runs: int = 1,
) -> dict:
    """Run a single case `runs` times and return aggregated metrics."""
    answers: list[str] = []
    sources_all: list[list[str]] = []
    for _ in range(runs):
        r = pipe.ask(case.query, use_retrieval=use_retrieval)
        answers.append(r.answer)
        sources_all.append([ret.meta.get("source", "") for ret in r.retrieved])

    # accuracy: first-run answer contains at least one expected substring
    acc = False
    if case.expect_substrings and not case.unanswerable:
        low = answers[0].lower()
        acc = any(s.lower() in low for s in case.expect_substrings)

    # hallucination: unanswerable queries answered confidently (no refusal)
    refused_first = _looks_like_refusal(answers[0])
    halluc = case.unanswerable and not refused_first

    # consistency: distinct first-line hashes across runs
    unique_first_lines = len({_hash_line(_first_line(a)) for a in answers})

    return {
        "qid": case.qid,
        "kind": case.kind,
        "query": case.query,
        "use_retrieval": use_retrieval,
        "runs": runs,
        "accuracy": int(acc),
        "hallucination": int(halluc),
        "consistency_distinct_lines": unique_first_lines,
        "first_answer": answers[0][:400],
        "refused": int(refused_first),
        "top_source": sources_all[0][0] if sources_all and sources_all[0] else "",
    }


def run_suite(
    pipe: RAGPipeline,
    cases: list[EvalCase] | None = None,
    runs: int = 2,
    out_path: Path | None = None,
) -> list[dict]:
    cases = cases or DEFAULT_CASES
    rows: list[dict] = []
    for case in cases:
        log.info(f"--- {case.qid} [{case.kind}] ---")
        rag_row = run_case(pipe, case, use_retrieval=True, runs=runs)
        baseline_row = run_case(pipe, case, use_retrieval=False, runs=runs)
        rows.append({**rag_row, "system": "RAG"})
        rows.append({**baseline_row, "system": "LLM-only"})

    if out_path is None:
        out_path = LOGS_DIR / "evaluation_results.csv"
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log.info(f"Wrote {len(rows)} eval rows -> {out_path}")
    return rows
