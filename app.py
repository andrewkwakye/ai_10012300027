# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""
Streamlit UI for the Academic City RAG chatbot.

Features (mapped to exam rubric):
    * Query input                              (Final Deliverables i)
    * Display retrieved chunks + scores        (Part D)
    * Show final response                      (Final Deliverables i)
    * Debug panel toggles:
        - show final prompt sent to LLM         (Part D)
        - switch prompt template                 (Part C)
        - toggle pure-LLM baseline               (Part E)
        - tune alpha / top_k                     (Part B)
    * Thumbs-up / thumbs-down per chunk         (Part G — innovation)
"""

from __future__ import annotations

import os

import streamlit as st


# --- Page config ----------------------------------------------------------
# MUST be the first Streamlit call on the page. We deliberately do this
# before importing from src.config so no internal st.* call (e.g. st.secrets
# access) races ahead of it.
st.set_page_config(
    page_title="Academic City RAG Chatbot",
    page_icon="🎓",
    layout="wide",
)

# On Streamlit Cloud, secrets live in st.secrets (set via the Secrets panel).
# Copy them into environment variables so src.config._read_secret() picks
# them up uniformly across local and cloud runs.
try:
    for _k in ("GROQ_API_KEY", "GROQ_CHAT_MODEL", "EMBED_MODEL"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    # No secrets.toml locally — totally fine; .env handles it.
    pass

from src.config import CHAT_MODEL, CONFIG, EMBED_MODEL  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.prompt_builder import TEMPLATES  # noqa: E402

st.title("Academic City RAG Chatbot")
st.caption("Ghana Election Results + 2025 Budget Statement — grounded answers only")


# --- Pipeline (cached) ----------------------------------------------------
@st.cache_resource(show_spinner="Loading vector store and BM25 index...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


try:
    pipe = get_pipeline()
except FileNotFoundError as e:
    st.error(
        "Index not built yet. From a terminal run:\n\n"
        "```\npython scripts/download_data.py\npython scripts/build_index.py\n```"
    )
    st.stop()
except RuntimeError as e:
    st.error(f"Startup failed: {e}")
    st.stop()


# --- Sidebar controls -----------------------------------------------------
with st.sidebar:
    st.header("Retrieval")
    top_k = st.slider("top_k", 1, 15, CONFIG.top_k)
    alpha = st.slider(
        "alpha  (1.0 = pure dense, 0.0 = pure BM25)",
        0.0,
        1.0,
        CONFIG.hybrid_alpha,
        step=0.05,
    )
    expand = st.checkbox("Query expansion (synonyms)", value=False)

    st.header("Prompt")
    prompt_name = st.selectbox(
        "Prompt template",
        options=list(TEMPLATES.keys()),
        index=list(TEMPLATES.keys()).index("v3_structured"),
    )

    st.header("Baseline")
    pure_llm = st.checkbox(
        "Pure LLM (no retrieval)",
        value=False,
        help="Part E: RAG vs. no-retrieval comparison",
    )

    st.header("Debug")
    show_prompt = st.checkbox("Show final prompt sent to LLM", value=True)
    show_scores = st.checkbox("Show retrieved chunks & scores", value=True)


# --- Query input ----------------------------------------------------------
query = st.text_area(
    "Ask a question about Ghana's 2025 budget or past election results:",
    height=90,
    placeholder="e.g. What is the projected fiscal deficit for 2025?",
)

col_a, col_b = st.columns([1, 5])
run = col_a.button("Ask", type="primary", use_container_width=True)
col_b.write("")

# --- Handle query ---------------------------------------------------------
if run and query.strip():
    with st.spinner("Retrieving + generating..."):
        try:
            resp = pipe.ask(
                query.strip(),
                top_k=top_k,
                alpha=alpha,
                expand=expand,
                prompt_name=prompt_name,
                use_retrieval=not pure_llm,
            )
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

    # Answer
    st.subheader("Answer")
    if resp.low_confidence:
        st.warning(resp.note)
    st.write(resp.answer)

    # Retrieved chunks
    if show_scores and resp.retrieved:
        st.subheader("Retrieved chunks")
        for i, r in enumerate(resp.retrieved, start=1):
            with st.expander(
                f"#{i}  score={r.score:.3f}   dense={r.dense_score:.3f}   "
                f"bm25={r.bm25_score:.3f}   boost={r.boost:+.3f}   "
                f"source={r.meta.get('source')}"
            ):
                st.markdown(
                    f"**chunk_id:** `{r.meta.get('chunk_id')}`  \n"
                    f"**strategy:** `{r.meta.get('strategy')}`"
                )
                st.write(r.meta.get("text", ""))
                # --- Feedback buttons (Part G innovation) --------------
                up_key = f"up_{i}_{r.meta['chunk_id']}"
                down_key = f"dn_{i}_{r.meta['chunk_id']}"
                b1, b2, _ = st.columns([1, 1, 6])
                if b1.button("👍 helpful", key=up_key):
                    pipe.give_feedback(r.meta["chunk_id"], resp.query, +1)
                    st.toast("Recorded 👍 — future retrieval will boost this chunk")
                if b2.button("👎 not relevant", key=down_key):
                    pipe.give_feedback(r.meta["chunk_id"], resp.query, -1)
                    st.toast("Recorded 👎 — future retrieval will deprioritise it")

    # Final prompt (debug)
    if show_prompt:
        with st.expander("Final prompt sent to LLM"):
            st.markdown("**system**")
            st.code(resp.prompt_system, language="markdown")
            st.markdown("**user**")
            st.code(resp.prompt_user, language="markdown")

elif run:
    st.info("Please enter a question.")

# --- Footer ---------------------------------------------------------------
st.caption(
    f"chat={CHAT_MODEL} · embed={EMBED_MODEL} · top_k={top_k} · alpha={alpha:.2f} · "
    f"Built by Andrew Kofi Kwakye (10012300027) for CS4241."
)
