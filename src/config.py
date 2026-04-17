# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""
Central configuration for the RAG system.

All magic numbers live here so we can tune them from one place.
Reads secrets from environment variables (.env locally, st.secrets on Streamlit Cloud).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Load .env from the project root BEFORE any secret lookup.
# Safe no-op if python-dotenv isn't installed or no .env exists.
try:
    from dotenv import load_dotenv  # type: ignore

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_PROJECT_ROOT / ".env")
except Exception:
    pass

# --- Paths ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = ROOT_DIR / "logs"

for _d in (RAW_DIR, PROCESSED_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Data source URLs (given in the question paper) ----------------------
CSV_URL = (
    "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/"
    "main/Ghana_Election_Result.csv"
)
PDF_URL = (
    "https://mofep.gov.gh/sites/default/files/budget-statements/"
    "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
)

CSV_PATH = RAW_DIR / "Ghana_Election_Result.csv"
PDF_PATH = RAW_DIR / "2025_Budget_Statement.pdf"

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
META_PATH = PROCESSED_DIR / "meta.jsonl"
FEEDBACK_PATH = PROCESSED_DIR / "feedback.jsonl"


# --- LLM + embedding settings -------------------------------------------
def _read_secret(name: str, default=None):
    """Read a secret from environment variables.

    On Streamlit Cloud, add the secret in the 'Secrets' panel; Streamlit exposes
    it as an env var at runtime. Locally, load_dotenv() (above) pulls it from .env.
    Importing streamlit.secrets here is deliberately avoided because accessing
    st.secrets counts as a Streamlit command and must not happen before
    st.set_page_config() in app.py.
    """
    val = os.environ.get(name)
    return val if val else default


# Groq handles chat completion. Embeddings run locally (no API) via
# sentence-transformers, so we don't need a separate embeddings API key.
GROQ_API_KEY = _read_secret("GROQ_API_KEY")
CHAT_MODEL = _read_secret("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL = _read_secret("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


# --- Tunable RAG parameters ---------------------------------------------
@dataclass(frozen=True)
class RAGConfig:
    # Chunking
    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 60
    max_chunks_per_doc: int = 2000

    # Retrieval
    top_k: int = 6
    hybrid_alpha: float = 0.6
    min_score_threshold: float = 0.15

    # Generation
    max_context_tokens: int = 3000
    temperature: float = 0.2

    # Feedback loop (Part G innovation)
    feedback_boost: float = 0.05
    feedback_cap: float = 0.25


CONFIG = RAGConfig()
