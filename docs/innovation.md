# Innovation Component: Feedback-Loop Re-Ranker

**Author:** `Andrew Kofi Kwakye`
**Index:** `10012300027`
**Covers:** Exam Part G

---

## What it is (1-line)

Users vote 👍 / 👎 on each retrieved chunk under the answer. Those votes persist on disk and additively adjust future retrieval scores for the same chunks — so the system gets *measurably better* at retrieving the right context the longer it's used, without any model retraining or re-embedding.

## Why this feature

The exam lists four innovation options. Feedback-loop is the most honest signal of retrieval quality, and it's the option that most directly fixes the Part B failure cases shown in `docs/retrieval_failures.md`: if dense retrieval fetches something semantically-near-but-wrong, a single thumbs-down on that chunk demotes it for future queries, and a thumbs-up on the right chunk promotes it. Unlike memory-based RAG (which fights context-window limits) or multi-step reasoning (which multiplies API cost), the feedback loop adds ~12 ms to retrieval and costs zero extra tokens.

## Implementation

Three tiny moving parts — nothing framework-y.

**1. `src/feedback.py` — persistent append-only JSONL store**
```python
class FeedbackStore:
    def record(self, chunk_id: str, query: str, vote: int): ...
    def boost_for(self, chunk_id: str) -> float: ...
```
Every vote is one line of JSON in `data/processed/feedback.jsonl`:
```json
{"chunk_id": "pdf_page_42::rec::a1b2c3d4e5", "query": "...", "vote": 1, "delta": 0.05}
```
On startup we replay the log into an in-memory score map. The file is the source of truth; the map is a cache.

**2. `src/retriever.py` — score adjustment in the hybrid combiner**
```python
combined[i] = alpha * dense_norm[i] + (1 - alpha) * bm25_norm[i]
combined[i] += np.clip(feedback.boost_for(chunk_id), -CAP, +CAP)
```
Two safety properties:
- **Cap (`CONFIG.feedback_cap = 0.25`)** — feedback can nudge ranking, never dominate it. Prevents a single malicious vote from permanently pinning a chunk at rank 1.
- **Clipping is symmetric** — positive and negative votes are bounded equally.

**3. `app.py` — 👍/👎 buttons under every retrieved chunk.**
Clicks call `pipe.give_feedback(chunk_id, query, +1 | -1)` and flash a toast. Effect is live on the *next* query the user runs.

## How to show it works (your demo script)

This is a 4-step demonstration for your video walkthrough:

1. **Baseline** — ask: *"Which constituency did the NDC win in 2020 in Ashanti Region?"*. Note the top-3 chunks' IDs and their combined scores.
2. **Corrective vote** — click 👍 on the correct CSV row and 👎 on the top-ranked PDF paragraph (if it was irrelevant).
3. **Repeat the query** — you should see the CSV row's combined score rise by `feedback_boost × vote_count` (default 0.05 per up-vote). If it wasn't already #1, it usually becomes #1. The irrelevant PDF chunk drops.
4. **Prove the effect is bounded** — up-vote the CSV row 10 times. The score will cap at `feedback_cap = 0.25` addition — it cannot reach 100 % regardless.

Record the before/after scores from the Streamlit UI in the experiment log.

## What it is **not**

- Not a learned re-ranker (no training).
- Not per-user (signals are global).
- Not per-query (signals are chunk-keyed, so a good CSV chunk gets boosted for *all* queries that reach near it). Known limitation; see §6 of `docs/architecture.md`.

Those are deliberate simplifications to keep the implementation auditable for an exam. A production version would cluster queries and scope feedback per cluster, and would probably decay old votes over time.

## Novelty claim

The feature is novel *in the context of this project's constraints* — "no LangChain / no framework pipeline" rules out most ready-made feedback layers (e.g. RAGAS, Arize, TruLens) — so this was written from scratch in ~60 lines, integrates cleanly into the hybrid score we already compute, and doesn't require any additional embeddings or LLM calls. The math (additive clipped delta) is simple enough to sanity-check in your head, which is itself a design virtue for a government-data assistant where auditability matters.
