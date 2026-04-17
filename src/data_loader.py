# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""
Loads and cleans the two raw sources.

Outputs a uniform list of "documents" — dicts with at minimum:
    { "doc_id": str, "source": str, "text": str, "meta": {...} }

Each document is a *semantic unit*:
    - For the CSV   : one row per document (a natural record boundary)
    - For the PDF   : one page per document (then chunked later)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from pypdf import PdfReader

from .config import CSV_PATH, PDF_PATH
from .logger import get_logger

log = get_logger("loader")


# ---------------------------------------------------------------------------
# CSV: Ghana election results
# ---------------------------------------------------------------------------
def load_election_csv(path: Path = CSV_PATH) -> list[dict]:
    """
    Cleans the Ghana election CSV and renders each row as a short natural-
    language sentence. Embedding a narrative sentence works better than
    embedding raw column-value pairs because the embedding model was
    trained on prose.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/download_data.py first")

    df = pd.read_csv(path)
    original_rows = len(df)

    # --- Cleaning ------------------------------------------------------
    # 1. strip whitespace in string columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # 2. standardise column names to lowercase_snake
    normalized = [re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in df.columns]

    # 2b. de-duplicate column names (two raw headers can normalise to the same
    # token, e.g. "Candidate Name" and "Candidate_Name" both -> candidate_name).
    # If we leave duplicates in place, `row[col]` returns a Series instead of a
    # scalar and everything downstream breaks.
    seen: dict[str, int] = {}
    dedup_cols: list[str] = []
    for c in normalized:
        if c in seen:
            seen[c] += 1
            dedup_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 1
            dedup_cols.append(c)
    df.columns = dedup_cols

    # 3. drop fully-empty rows
    df = df.dropna(how="all")

    # 4. drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    log.info(
        f"CSV cleaned: {original_rows} -> {len(df)} rows "
        f"({before - len(df)} dupes dropped)"
    )

    docs: list[dict] = []
    for idx, row in df.iterrows():
        # iterate positionally via .items() so we always see scalars
        parts = []
        for k, v in row.items():
            if pd.notna(v):
                parts.append(f"{k.replace('_', ' ')}: {v}")
        text = ". ".join(parts) + "."
        docs.append(
            {
                "doc_id": f"csv_row_{idx}",
                "source": "Ghana_Election_Result.csv",
                "text": text,
                "meta": {"row_index": int(idx), **row.to_dict()},
            }
        )
    return docs


# ---------------------------------------------------------------------------
# PDF: Budget statement
# ---------------------------------------------------------------------------
_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"-\n([a-z])")  # word-wrapped hyphen


def _clean_page_text(raw: str) -> str:
    # fix hyphenated line breaks: "informa-\ntion" -> "information"
    text = _HYPHEN_BREAK.sub(r"\1", raw)
    # collapse runs of spaces/tabs
    text = _WS_RE.sub(" ", text)
    # collapse 3+ newlines to 2
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def load_budget_pdf(path: Path = PDF_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/download_data.py first")

    reader = PdfReader(str(path))
    docs: list[dict] = []
    empty_pages = 0
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        clean = _clean_page_text(raw)
        if len(clean) < 50:  # junk pages (covers, blank)
            empty_pages += 1
            continue
        docs.append(
            {
                "doc_id": f"pdf_page_{i+1}",
                "source": "2025_Budget_Statement.pdf",
                "text": clean,
                "meta": {"page": i + 1},
            }
        )
    log.info(
        f"PDF cleaned: {len(reader.pages)} pages -> {len(docs)} kept "
        f"({empty_pages} skipped as too-short)"
    )
    return docs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_all() -> list[dict]:
    docs = load_election_csv() + load_budget_pdf()
    log.info(f"Total documents loaded: {len(docs)}")
    return docs
