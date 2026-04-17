# Manual Experiment Log

**Author:** `Andrew Kofi Kwakye`
**Index:** `10012300027`

> ⚠️ **IMPORTANT — READ ME BEFORE FILLING IN**
>
> The exam rubric says:
> > *"Manual experiment logs (not AI-generated summaries)"*
>
> This file is a **template**. The structure is here so nothing gets forgotten, but the **content must be filled in by you, by hand, after actually running the system.** Copy-pasting model outputs and raw scores is fine — that is direct evidence. Summaries written by an AI are not.
>
> Delete this callout before submitting.

---

## How to use this log

For every experiment session:

1. Put today's date and a short session title.
2. Write in first person: *"I changed X to see whether Y..."*.
3. Paste raw numbers from the terminal / CSV outputs.
4. Write your interpretation in your own words. 1–3 sentences is enough.
5. End with "Next step" — what you'll try next.

---

## Session 1 — baseline build

**Date:** _YYYY-MM-DD_
**What I ran:** `python scripts/download_data.py`, `python scripts/build_index.py`
**Observations:**
- CSV original rows: _____   cleaned rows: _____
- PDF pages: _____   kept pages: _____
- Total chunks produced: _____
- Embedding API usage (from Google AI Studio dashboard): _____ requests (free tier)
- Index sizes: embeddings.npy = _____ MB, meta.jsonl = _____ KB

**What surprised me:** _fill in_

**Next step:** _fill in_

---

## Session 2 — chunking comparison (Part A)

**Date:** _YYYY-MM-DD_

I swapped `chunk_documents` for each of the three strategies and rebuilt the index. For each index I ran the same 5 queries and counted how often the top-1 chunk was actually relevant.

| Strategy         | # chunks | Avg tokens/chunk | Queries top-1 correct / 5 | Notes from my own eyeballing |
|------------------|----------|------------------|----------------------------|------------------------------|
| `fixed_token`    |          |                  |                            |                              |
| `recursive_char` |          |                  |                            |                              |
| `routed` (ship)  |          |                  |                            |                              |

**Representative failure under `fixed_token`** (paste actual text):
> _paste the chunk that was cut in half here_

**Same passage under `recursive_char`:**
> _paste here_

**My decision:** _fill in_

---

## Session 3 — hybrid alpha sweep (Part B)

**Date:** _YYYY-MM-DD_
**What I ran:** `python scripts/run_evaluation.py --query "..." --alpha X` for several X values.

| Query                                              | alpha=0.0 (BM25) | alpha=0.5 | alpha=0.6 | alpha=1.0 (dense) |
|----------------------------------------------------|------------------|-----------|-----------|-------------------|
| "Which constituency did NDC win in Ashanti 2020?"  |                  |           |           |                   |
| "What is GDP growth projection for 2025?"          |                  |           |           |                   |

(Report top-1 source.)

**My interpretation:** _fill in — which alpha won on which kind of query, and why I picked 0.6 as the default._

---

## Session 4 — prompt-template showdown (Part C)

**Date:** _YYYY-MM-DD_

Same query, three prompts. Paste the raw model answers.

**Query:** "What is the exchange rate of the cedi to Japanese yen today?" (unanswerable)

- **v1_minimal reply:**
  > _paste_
- **v2_guarded reply:**
  > _paste_
- **v3_structured reply:**
  > _paste_

**Hallucination observed?** _fill in_

**Query:** "Which party won Ablekuma West in 2020?"

- **v1_minimal reply:**
  > _paste_
- **v2_guarded reply:**
  > _paste_
- **v3_structured reply:**
  > _paste_

**Which template cited chunks?** _fill in_
**My decision:** _fill in_

---

## Session 5 — full evaluation suite (Part E)

**Date:** _YYYY-MM-DD_
**What I ran:** `python scripts/run_evaluation.py`
**Output file:** `logs/evaluation_results.csv`

Headline numbers from the CSV:

- RAG accuracy on answerable queries: ___ / ___
- LLM-only accuracy on answerable queries: ___ / ___
- RAG hallucination rate on unanswerable queries: ___ / ___
- LLM-only hallucination rate on unanswerable queries: ___ / ___
- RAG consistency (mean distinct first-lines over 2 runs): ___
- LLM-only consistency: ___

**My reading of the evidence:** _fill in_

**Surprises:** _fill in_

---

## Session 6 — innovation feature demo (Part G)

**Date:** _YYYY-MM-DD_
**What I did:**
1. Ran the same query twice, once before any feedback, once after voting.
2. Recorded the before/after top-3 scores.

Before:

| Rank | chunk_id | combined | dense | bm25 | boost |
|------|----------|----------|-------|------|-------|
| 1    |          |          |       |      |       |
| 2    |          |          |       |      |       |
| 3    |          |          |       |      |       |

I clicked 👍 on chunk _____ and 👎 on chunk _____.

After:

| Rank | chunk_id | combined | dense | bm25 | boost |
|------|----------|----------|-------|------|-------|
| 1    |          |          |       |      |       |
| 2    |          |          |       |      |       |
| 3    |          |          |       |      |       |

**Did the ranking change as expected?** _fill in_

---

## Session 7 — deploy to Streamlit Cloud

**Date:** _YYYY-MM-DD_
**Deployed URL:** _fill in_
**What broke on first deploy:** _fill in_
**How I fixed it:** _fill in_

---

## Session 8 — emailed submission

**Date:** _YYYY-MM-DD_
**Email sent to:** godwin.danso@acity.edu.gh
**Subject:** CS4241-Introduction to Artificial Intelligence-2026:[`10012300027` `Andrew Kofi Kwakye`]
**Attachments / links:**
- GitHub repo: https://github.com/_____/ai_10012300027
- Deployed URL: _____
- Video walkthrough: _____
- GitHub collaborator added (godwin.danso@acity.edu.gh / GodwinDansoAcity): ☐ yes

---

> Reminder from the question paper: *"Add or invite godwin.danso@acity.edu.gh or GodwinDansoAcity as a GitHub collaborator. Failure to do so will result in getting nothing for your exams."* **Don't forget this step.**
"* **Don't forget this step.**
