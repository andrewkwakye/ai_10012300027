# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""
End-to-end RAG pipeline (Part D).

Flow:
    User Query -> Retrieval -> Context Selection -> Prompt -> LLM -> Response

The pipeline logs structured JSON at each stage (via logger.log_stage) so
reviewers can inspect what the retriever returned, what prompt was sent to
the LLM, and what came back. The returned RAGResponse carries the same
data so the Streamlit UI can render it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .config import CONFIG
from .embedder import Embedder
from .feedback import FeedbackStore
from .llm import ChatLLM
from .logger import get_logger, log_stage
from .prompt_builder import DEFAULT_TEMPLATE, TEMPLATES, pack_context
from .retriever import Retriever, RetrievalResult
from .vector_store import VectorStore

log = get_logger("pipeline")


@dataclass
class RAGResponse:
    query: str
    answer: str
    retrieved: list[RetrievalResult]
    used: list[RetrievalResult]
    prompt_system: str
    prompt_user: str
    prompt_name: str
    low_confidence: bool = False
    note: str = ""
    extras: dict = field(default_factory=dict)


class RAGPipeline:
    """
    Build once, call .ask() many times.

    If you want a fresh feedback store in tests, pass your own.
    """

    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        llm: ChatLLM | None = None,
        feedback: FeedbackStore | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore.load()
        self.feedback = feedback or FeedbackStore()
        self.retriever = Retriever(self.store, self.embedder, self.feedback)
        self.llm = llm or ChatLLM()

    # ------------------------------------------------------------------
    def ask(
        self,
        query: str,
        *,
        top_k: int = CONFIG.top_k,
        alpha: float = CONFIG.hybrid_alpha,
        expand: bool = False,
        prompt_name: str = DEFAULT_TEMPLATE,
        use_retrieval: bool = True,
    ) -> RAGResponse:
        """
        use_retrieval=False runs the pure-LLM baseline for Part E's
        comparison (RAG vs no-retrieval).
        """
        log_stage(log, "query", {"query": query, "top_k": top_k, "alpha": alpha,
                                  "expand": expand, "prompt": prompt_name,
                                  "use_retrieval": use_retrieval})

        # --- Retrieval ------------------------------------------------
        retrieved: list[RetrievalResult] = []
        if use_retrieval:
            retrieved = self.retriever.retrieve(query, top_k=top_k, alpha=alpha, expand=expand)
            log_stage(
                log,
                "retrieval",
                {
                    "count": len(retrieved),
                    "top_scores": [round(r.score, 3) for r in retrieved[:3]],
                    "sources": [r.meta.get("source") for r in retrieved],
                },
            )

        # --- Confidence gate -----------------------------------------
        low_conf = False
        note = ""
        if use_retrieval and (not retrieved or retrieved[0].score < CONFIG.min_score_threshold):
            low_conf = True
            top_score = retrieved[0].score if retrieved else 0.0
            note = (
                f"Top combined score {top_score:.3f} is below threshold "
                f"{CONFIG.min_score_threshold}. Model told to refuse if unsupported."
            )
            log_stage(log, "confidence", {"low_confidence": True, "note": note})

        # --- Context packing -----------------------------------------
        if use_retrieval:
            context, used = pack_context(retrieved)
        else:
            context, used = "(no context — pure-LLM baseline)", []

        # --- Prompt construction -------------------------------------
        builder = TEMPLATES[prompt_name]
        prompt = builder(query, context)
        log_stage(
            log,
            "prompt",
            {
                "prompt_name": prompt_name,
                "system_preview": prompt.system[:120] + "...",
                "user_tokens_approx": len(prompt.user) // 4,
            },
        )

        # --- Generation ----------------------------------------------
        answer = self.llm.complete(prompt.system, prompt.user)
        log_stage(log, "answer", {"chars": len(answer), "preview": answer[:160]})

        return RAGResponse(
            query=query,
            answer=answer,
            retrieved=retrieved,
            used=used,
            prompt_system=prompt.system,
            prompt_user=prompt.user,
            prompt_name=prompt_name,
            low_confidence=low_conf,
            note=note,
        )

    # Convenience: give feedback on a chunk after the user has seen the answer
    def give_feedback(self, chunk_id: str, query: str, vote: int) -> None:
        self.feedback.record(chunk_id, query, vote)
