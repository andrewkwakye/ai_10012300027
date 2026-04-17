# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""
Tiny project-wide logger. One file log + console.

We keep this custom rather than using Python's logging.config so the
student (me) can see exactly what is happening at each stage of the RAG
pipeline, which the exam rubric (Part D) asks for.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from .config import LOGS_DIR

_LOG_FILE = LOGS_DIR / "pipeline.log"


def get_logger(name: str = "rag") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def log_stage(logger: logging.Logger, stage: str, payload: dict) -> None:
    """Structured one-liner per pipeline stage (Part D requirement)."""
    try:
        body = json.dumps(payload, default=str, ensure_ascii=False)[:2000]
    except Exception:
        body = str(payload)[:2000]
    logger.info(f"[{stage}] {body}")
