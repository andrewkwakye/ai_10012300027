# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""
Feedback-loop re-ranker  (Part G — Innovation component)

Users in the Streamlit UI can mark each retrieved chunk as
thumbs-up / thumbs-down for a given query. We persist those signals
on disk and feed them back into retrieval as a small additive score
boost (or penalty) per chunk.

Design notes:
    * We do NOT retrain embeddings. The boost is a cheap, interpretable
      mechanism that preserves the base ranking but nudges it toward
      chunks humans have confirmed are useful.
    * The boost is clamped (CONFIG.feedback_cap) so feedback cannot
      completely override semantic similarity — otherwise a single
      malicious thumbs-up could permanently pin an irrelevant chunk.
    * Signals are keyed by chunk_id only (aggregate over queries).
      A more sophisticated version would key by (query_cluster, chunk_id);
      left as future work and noted in docs/architecture.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from .config import CONFIG, FEEDBACK_PATH
from .logger import get_logger

log = get_logger("feedback")


class FeedbackStore:
    def __init__(self, path: Path = FEEDBACK_PATH):
        self.path = path
        self._lock = Lock()
        self._scores: dict[str, float] = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        n = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = row.get("chunk_id")
                delta = float(row.get("delta", 0))
                if not cid:
                    continue
                self._scores[cid] = self._scores.get(cid, 0.0) + delta
                n += 1
        log.info(f"FeedbackStore loaded {n} signals for {len(self._scores)} chunks")

    # ------------------------------------------------------------------
    def record(self, chunk_id: str, query: str, vote: int) -> None:
        """
        vote:  +1 for thumbs-up, -1 for thumbs-down, 0 to clear.
        Writes an append-only JSONL row for auditability.
        """
        delta = CONFIG.feedback_boost * vote
        with self._lock:
            self._scores[chunk_id] = self._scores.get(chunk_id, 0.0) + delta
            with self.path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"chunk_id": chunk_id, "query": query, "vote": vote, "delta": delta},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        log.info(f"feedback recorded chunk={chunk_id[:24]}... vote={vote} delta={delta:+.3f}")

    # ------------------------------------------------------------------
    def boost_for(self, chunk_id: str) -> float:
        return self._scores.get(chunk_id, 0.0)

    def top_positive(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self._scores.items(), key=lambda kv: -kv[1])[:n]
