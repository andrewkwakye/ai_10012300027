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

## Results — fill in from your CSV

Copy the values straight from `logs/evaluation_results.csv`.

| QID | System   | Accuracy | Hallucination | Consistency | First-answer excerpt |
|-----|----------|----------|---------------|-------------|----------------------|
| Q1  | RAG      | _fill_   | _fill_        | _fill_      | _fill_               |
| Q1  | LLM-only | _fill_   | _fill_        | _fill_      | _fill_               |
| Q2  | RAG      | _fill_   | _fill_        | _fill_      | _fill_               |
| Q2  | LLM-only | _fill_   | _fill_        | _fill_      | _fill_               |
| Q3  | RAG      | _fill_   | _fill_        | _fill_      | _fill_               |
| Q3  | LLM-only | _fill_   | _fill_        | _fill_      | _fill_               |
| Q4  | RAG      | _fill_   | _fill_        | _fill_      | _fill_               |
| Q4  | LLM-only | _fill_   | _fill_        | _fill_      | _fill_               |
| Q5  | RAG      | _fill_   | _fill_        | _fill_      | _fill_               |
| Q5  | LLM-only | _fill_   | _fill_        | _fill_      | _fill_               |

## Evidence-based comparison (RAG vs pure LLM)

Summarise once you have your numbers. Expected direction (to verify):

- RAG should **out-accuracy** LLM-only on Q1 and Q2 because both require Ghana-specific facts from the 2025 budget (post-training-cutoff for many models).
- RAG should have **lower hallucination rate** on Q4 and Q5 because the prompt instructs refusal and the retriever returns low-score chunks that the confidence gate flags.
- RAG should have **equal or better consistency** because temperature is the same but RAG conditions on deterministic retrieved text.

If your measurements don't match this direction — report the unexpected result in your experiment log and reason about it. The exam explicitly asks for evidence, not opinion.
