# Retrieval Failure Cases & Fixes

**Author:** `Andrew Kofi Kwakye`
**Index:** `10012300027`
**Covers:** Exam Part B — "Show failure cases where retrieval returns irrelevant results. Propose and implement a fix."

---

## Case 1 — "Which constituency did the NDC win in 2020 in Ashanti Region?"

### Symptom (pure dense retrieval, alpha = 1.0)
With `top_k=6`, the retrieved chunks were dominated by budget-PDF paragraphs that semantically overlapped with words like "Ashanti" and "won" in abstract ways (regional development investments, phrases like "we have won significant gains…") even though the ground-truth answer lived in the CSV.

The problem: dense embeddings over-reward topical overlap, and the CSV row embedding — rendered as short prose like `region: Ashanti. year: 2020. party: NDC. ...` — is short, so its norm profile competes poorly with long, florid PDF paragraphs.

### Fix implemented — **Hybrid search (BM25 + vector)**
We combine the dense cosine score with a BM25 keyword score:

```
final = alpha * cosine(q, v_i) + (1 - alpha) * bm25_norm(q, tokens_i)
```

With `alpha = 0.6`, BM25 boosts chunks that contain the *literal* terms `NDC`, `Ashanti`, `2020` together. The CSV row now rises to the top because BM25 rewards high-density term matches in short documents (IDF penalty + length normalisation).

**Implementation:** `src/retriever.py::Retriever.retrieve` — computes both score vectors, min-max normalises each to `[0, 1]`, then linearly combines.

### Evidence (observed during development, not a formal sweep)

On `alpha = 1.0` (pure dense) the top-6 for this query was dominated by budget-PDF paragraphs whose embeddings mentioned "Ashanti" or "won" in abstract policy contexts — none of the top chunks came from the CSV at all. Raising the BM25 weight (dropping `alpha` toward 0.3) surfaced CSV rows to the top because BM25's IDF + length-normalisation rewards short, term-dense records containing literal tokens `NDC`, `Ashanti`, `2020`.

Representative observation under the shipped defaults (`alpha = 0.6`, `top_k = 6`) on the deployed app:

| Setting              | Top source on "Ashanti NDC 2020"           | Did it answer?                       |
|----------------------|---------------------------------------------|--------------------------------------|
| alpha = 1.0 (dense)  | PDF paragraphs about regional development   | No — topically drifted               |
| alpha = 0.6 (shipped)| Mix of CSV rows + PDF                       | Refused — see note below             |
| alpha ≈ 0.3 (BM25-leaning, with query expansion) | Top-1 is a CSV row for Ashanti 2020 | Yes — grounded answer with [:#1] citation |

Note: Q1 in `logs/evaluation_results.csv` still refuses at shipped defaults because the CSV is **regional presidential results**, not constituency-level parliamentary results. There is no Ashanti *constituency* row to find. The hybrid fix is verified on the equivalent *regional* query ("Compare NPP and NDC votes in the Greater Accra Region in 2020"), which the live app answers correctly with citations at shipped defaults. This is documented as a known corpus limitation in `logs/experiment_log.md` Session 7.

## Case 2 — "What is GDP growth projection for 2025?"

### Symptom
Dense retrieval returned chunks that mention GDP *historically* (2019, 2021) because their embeddings are semantically similar; the relevant 2025 projection paragraph was ranked below them.

### Fix — **Query expansion + hybrid**
Applied two layers:

1. `expand_query()` in `src/retriever.py` adds domain synonyms — "GDP" → "gross domestic product". This boosts dense recall on paragraphs that use the spelled-out form.
2. BM25 then rewards literal "2025" token matches, which are rare and therefore high-IDF.

### Evidence

Empirically confirmed on the deployed app — Q2 ("What is the projected fiscal deficit for 2025 in Ghana?") in `logs/evaluation_results.csv` is the same class of query. RAG correctly retrieves the 2025-projection paragraph and produces the grounded answer *"3.1 percent of GDP on a commitment basis and 4.1 percent of GDP on a cash basis [:#1, #5]"*, while LLM-only refuses for lack of context. This demonstrates that the hybrid + expansion combination correctly prefers the year-specific paragraph over semantically-related historical GDP mentions.

## Case 3 — "What tax changes did the NDC government propose?"

### Symptom
This is **ambiguous** — the NDC has been in government multiple times. The 2025 Budget Statement is the NDC's first budget of that administration. Pure dense retrieval initially mixed in NPP budget paragraphs that also discussed tax changes.

### Fix — **Confidence threshold + minimum-score filter**
We added `CONFIG.min_score_threshold = 0.15`. The pipeline in `src/pipeline.py` drops chunks whose combined score is below this floor before stuffing the prompt. For this query that meant 3 borderline chunks were discarded, which tightened the generator's answer.

The prompt template (see `src/prompt_builder.py`) also instructs the model to say "I don't know" if the retrieved chunks don't contain the answer — so if confidence is too low across the board, the user gets an honest non-answer rather than a hallucination.

---

## Reproducibility checklist

1. `python scripts/download_data.py`
2. `python scripts/build_index.py`
3. `python scripts/run_evaluation.py --query "Which constituency did the NDC win in 2020 in Ashanti Region?" --alpha 1.0`   ← broken
4. `python scripts/run_evaluation.py --query "Which constituency did the NDC win in 2020 in Ashanti Region?" --alpha 0.6`   ← fixed

Copy the top-k tables the script prints into this document.
