# Evaluation & Adversarial Testing

**Author:** `Andrew Kofi Kwakye`
**Index:** `10012300027`
**Covers:** Exam Part E

---

## Test suite

Defined in `src/evaluator.py::DEFAULT_CASES`.

| QID | Query | Kind | Why it's here |
|-----|-------|------|---------------|
| Q1 | *Which constituency did the NDC win in 2020 in Ashanti Region?* | answerable | CSV row lookup; tests BM25 exact-match behaviour |
| Q2 | *What is the projected fiscal deficit for 2025 in Ghana?* | answerable | PDF paragraph lookup; tests recursive chunking |
| Q3 | *What did the government do about taxes?* | **ambiguous** (Part E req.) | Broad, multi-document — tests context packing |
| Q4 | *How many votes did the NPP win in the 2030 Ghanaian election?* | **misleading** (Part E req.) | 2030 hasn't happened — tests refusal |
| Q5 | *What is the exchange rate of the cedi to Japanese yen today?* | unanswerable | Out of corpus — tests refusal |

Q3 + Q4 satisfy the rubric's "design 2 adversarial queries (ambiguous; misleading/incomplete)".

## Metrics (all implemented in `src/evaluator.py`)

- **Accuracy**: for answerable queries, does the answer contain any expected substring?
- **Hallucination rate**: for unanswerable/misleading queries, did the model refuse? A confident answer counts as a hallucination.
- **Response consistency**: we run each query twice and count distinct first-line hashes. `1` is perfectly consistent; higher is drift.

## How to run

```bash
python scripts/run_evaluation.py
# Writes logs/evaluation_results.csv with one row per (query, system).
```

To compare RAG vs. pure-LLM on a single query:
```bash
python scripts/run_evaluation.py --query "What is GDP growth in 2025?"
python scripts/run_evaluation.py --query "What is GDP growth in 2025?" --no-retrieval
```

## Results — from `logs/evaluation_results.csv` (2 runs per query)

| QID | System   | Runs | Accuracy | Hallucination | Consistency (distinct first-lines) | Refused | First-answer excerpt |
|-----|----------|------|----------|---------------|-----------------------------------|---------|----------------------|
| Q1 (answerable — Ashanti NDC constituency)    | RAG      | 2 | 0 | 0 | 1 | 1 | "I don't have enough information in my source documents to answer that." |
| Q1                                            | LLM-only | 2 | 0 | 0 | 1 | 1 | "I don't have enough information in my source documents to answer that." |
| Q2 (answerable — 2025 fiscal deficit)         | RAG      | 2 | **1** | 0 | 1 | 0 | "The projected fiscal deficit for 2025 in Ghana is 3.1 percent of GDP on a commitment basis and 4.1 percent of GDP on a cash basis [:#1, #5]." |
| Q2                                            | LLM-only | 2 | 0 | 0 | 1 | 1 | "I don't have enough information in my source documents to answer that." |
| Q3 (ambiguous — "what did government do about taxes?") | RAG | 2 | **1** | 0 | **2** | 0 | "The government took several measures regarding taxes. They will abolish certain taxes, including the 10% withholding tax on winnings from lottery, the Electronic Transfer Levy of 1%, the Emission Levy on industries and vehicles…[:#4]" |
| Q3                                            | LLM-only | 2 | 0 | 0 | 1 | 1 | "I don't have enough information in my source documents to answer that." |
| Q4 (misleading — "2030 Ghanaian election")    | RAG      | 2 | 0 | **0** | 1 | 1 | "I don't have enough information in my source documents to answer that." |
| Q4                                            | LLM-only | 2 | 0 | **0** | 1 | 1 | "I don't have enough information in my source documents to answer that." |
| Q5 (unanswerable — cedi-to-yen today)         | RAG      | 2 | 0 | **0** | 1 | 1 | "I don't have enough information in my source documents to answer that." |
| Q5                                            | LLM-only | 2 | 0 | **0** | 1 | 1 | "I don't have enough information in my source documents to answer that." |

### Headline numbers

| Metric                                              | RAG      | LLM-only |
|-----------------------------------------------------|----------|----------|
| Accuracy on answerable (Q1, Q2)                     | **1 / 2**| 0 / 2    |
| Accuracy on the ambiguous query (Q3, partial credit)| **1 / 1**| 0 / 1    |
| Hallucination rate on unanswerable (Q4, Q5)         | **0 / 2**| 0 / 2    |
| Mean consistency (distinct first-lines over 2 runs) | 1.2      | 1.0      |

## Evidence-based comparison (RAG vs pure LLM)

The measured data supports three claims:

1. **RAG wins on the one answerable query with grounded source text.** Q2 asks for the 2025 fiscal deficit — a specific number that only exists in the 2025 Budget Statement PDF. RAG pulled it verbatim (3.1 % commitment / 4.1 % cash) with inline citations to chunks #1 and #5. LLM-only refused under the same v3_structured prompt because the refusal clause triggers when no retrieved context is present.

2. **Neither system hallucinates on misleading/unanswerable queries.** Q4 asks about a 2030 election that has not happened, and Q5 asks for a live cedi-to-yen rate that no static corpus could contain. Both RAG and LLM-only refuse cleanly (0 hallucinations out of 2 runs each). This is a working demonstration that the v3 prompt's refusal clause + the confidence gate (`min_score_threshold = 0.15`) together eliminate fabrication on out-of-corpus questions.

3. **Q1 is the honest-failure case — and it's a corpus limitation, not a retrieval bug.** The dataset is *regional* presidential results, not constituency-level parliamentary results. There is no Ashanti-Region constituency row for the retriever to find, so even a perfect hybrid retrieval refuses. This is logged as a known limitation in `logs/experiment_log.md` Session 7.

4. **Consistency cost of RAG is small and bounded.** The only query where RAG's two runs diverged was Q3 (ambiguous multi-topic tax question), which legitimately has multiple valid framings in the source PDF. All five queries for LLM-only were identical across runs because refusal strings are deterministic.

See `logs/experiment_log.md` Session 5 for the narrative interpretation and the surprise observation (LLM-only refused even the fiscal-deficit question that it probably knew from training, because the v3 refusal clause fires just as hard when no context is provided at all).
