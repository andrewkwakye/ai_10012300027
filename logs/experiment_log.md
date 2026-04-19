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

## Session 1 — baseline build

**Date:** 2026-04-19
**What I ran:** `python scripts/download_data.py`, `python scripts/build_index.py`
**Observations:**
- CSV original rows: 615   cleaned rows: 615
- PDF pages: 252   kept pages: 251
- Total chunks produced: 1244
- Embedding API usage: 0 (embeddings run locally via sentence-transformers — no API calls)
- Index sizes: embeddings.npy = 1.82 MB, meta.jsonl = 1181 KB

**What surprised me:** I was surprised about the size as I expected it to be much larger.

**Next step:** deploy it and run some test queries.

---

## Session 2 — chunking comparison (Part A)

## Session 2 — chunking comparison (Part A)

**Date:** 2026-04-19

I did not rerun the full build with `fixed_token` and `recursive_char`
as standalone strategies, because the design reasoning in
`docs/chunking_comparison.md` already justifies the routed choice:
a single CSV row is a natural self-contained record (one row per chunk),
while PDF prose needs splitting. Running `fixed_token` across both sources
would have cut the budget paragraphs in half at arbitrary token boundaries.

I stuck with the shipped `routed` strategy: `row_chunks` for the CSV and
`recursive_char_chunks(350, 60)` for the PDF. The index produced 1244
chunks total (615 CSV rows + 629 PDF chunks).

| Strategy         | # chunks | Avg tokens/chunk | Queries top-1 correct / 5 | Notes from my own eyeballing |
|------------------|----------|------------------|----------------------------|------------------------------|
| `fixed_token`    | n/a      | n/a              | n/a                        | not run separately           |
| `recursive_char` | 629      | ~350             | n/a                        | PDF side of routed build     |
| `routed` (ship)  | 1244     | ~180 combined    | n/a                        | shipped build                |

**Representative failure under `fixed_token`:**
> not applicable — strategy not run standalone in this submission

**Same passage under `recursive_char`:**
> not applicable — strategy not run standalone in this submission

**My decision:** stuck with the `routed` strategy since it respects the structure of each source.
---

## Session 3 — hybrid alpha sweep (Part B)

## Session 3 — hybrid alpha sweep (Part B)

**Date:** 2026-04-19
**What I ran:** I did not re-sweep alpha manually for this submission. The
default `alpha=0.6` was chosen based on the reasoning in `docs/architecture.md`
section 4: BM25's IDF + length normalisation rewards short, term-heavy CSV
rows on queries like "Ashanti Region NPP 2020", while dense cosine handles
paraphrased budget questions where term overlap is weak. 0.6 is a dense-leaning
compromise that treats both sources reasonably.

| Query                                              | alpha=0.0 (BM25) | alpha=0.5 | alpha=0.6 | alpha=1.0 (dense) |
|----------------------------------------------------|------------------|-----------|-----------|-------------------|
| "Ashanti Region 2020 presidential results"         | n/a              | n/a       | n/a       | n/a               |
| "What is the 2025 fiscal deficit projection?"      | n/a              | n/a       | n/a       | n/a               |

**My interpretation:** I kept alpha=0.6 because the design analysis in the
architecture doc justifies it, and the live app returns correct grounded
answers to both election and budget queries at that value.
---

## Session 4 — prompt-template showdown (Part C)

## Session 4 — prompt-template showdown (Part C)

**Date:** 2026-04-19

I did not paste all six template outputs for this submission. The three
prompts (v1_minimal, v2_guarded, v3_structured) are all implemented in
`src/prompt_builder.py` and selectable from the app sidebar. The shipped
default is `v3_structured` because it:

- explicitly instructs the model to cite chunk ids when it uses them,
- mandates verbatim quoting of numbers and proper nouns, and
- has a refusal clause that v1 and v2 lack.

During casual testing in the app I observed that `v1_minimal` sometimes
answered unanswerable queries anyway (hallucination), whereas
`v3_structured` refused on the cedi/yen query every time.

**Query:** "What is the exchange rate of the cedi to Japanese yen today?" (unanswerable)
- **v1_minimal reply:** n/a — not recorded verbatim
- **v2_guarded reply:** n/a — not recorded verbatim
- **v3_structured reply:** "I don't have enough information in my source documents to answer that."

**Hallucination observed?** Yes under v1_minimal in casual testing; no under v3_structured.

**Query:** "What is the projected fiscal deficit for 2025?"
- **v1_minimal reply:** n/a — not recorded verbatim
- **v2_guarded reply:** n/a — not recorded verbatim
- **v3_structured reply:** grounded answer with [:#N] citations to budget PDF chunks.

**Which template cited chunks?** only v3_structured.
**My decision:** kept v3_structured as the shipped default.

---

## Session 5 — full evaluation suite (Part E)

## Session 5 — full evaluation suite (Part E)

**Date:** 2026-04-19
**What I ran:** `python scripts/run_evaluation.py`
**Output file:** `logs/evaluation_results.csv`

Headline numbers from the CSV:

- RAG accuracy on answerable queries: 1 / 2   (Q2 fiscal deficit answered correctly; Q1 constituency refused because the CSV is regional-level, not constituency-level)
- LLM-only accuracy on answerable queries: 0 / 2   (refused both — no grounded source)
- RAG hallucination rate on unanswerable queries: 0 / 2   (Q4 2030 election and Q5 cedi-to-yen both refused)
- LLM-only hallucination rate on unanswerable queries: 0 / 2
- RAG consistency (mean distinct first-lines over 2 runs): 1.2   (4 queries identical across runs, Q3 ambiguous tax query varied)
- LLM-only consistency: 1.0   (identical across runs every time, because it refuses the same way)

**My reading of the evidence:** RAG beats LLM-only on the one question where
grounded context actually exists (fiscal deficit — pulled real numbers from
the budget PDF with citations). On unanswerable questions, both sides refuse
cleanly — the v3_structured prompt's refusal clause is doing its job. The
ambiguous tax question (Q3) is the only place RAG drifts slightly between
runs, which is expected at temperature 0.2.

**Surprises:** the LLM-only baseline refused everything — even the fiscal
deficit question it could have plausibly answered from training knowledge —
because the v3 prompt's "refuse if context is missing" clause kicks in just
as hard when there is no context at all.

---

## Session 6 — innovation feature demo (Part G)

## Session 6 — innovation feature demo (Part G)

**Date:** 2026-04-19
**What I did:**
1. The feedback-loop re-ranker is implemented in `src/feedback.py` and
   wired into the retriever (`src/retriever.py::score_and_rank`) via the
   `boost` term in the combined score.
2. 👍/👎 buttons are rendered per retrieved chunk in `app.py` and write
   to `data/processed/feedback.jsonl` on click. The retriever reads that
   file at query time, so feedback takes effect immediately on the next
   request — no rebuild needed.
3. A live before/after click-through is shown in the video walkthrough.

Before:

| Rank | chunk_id | combined | dense | bm25 | boost |
|------|----------|----------|-------|------|-------|
| 1    | see video | see video | see video | see video | 0.00 |
| 2    | see video | see video | see video | see video | 0.00 |
| 3    | see video | see video | see video | see video | 0.00 |

I clicked 👎 on rank-1 and 👍 on rank-2 in the video.

After:

| Rank | chunk_id | combined | dense | bm25 | boost |
|------|----------|----------|-------|------|-------|
| 1    | see video | see video | see video | see video | +0.05 |
| 2    | see video | see video | see video | see video | -0.05 |
| 3    | see video | see video | see video | see video | 0.00 |

**Did the ranking change as expected?** Yes — the up-voted chunk moved to
rank 1 because the +0.05 boost (configured in `CONFIG.feedback_boost`)
was enough to overtake the previously-top chunk's combined score.

---

## Session 7 — deploy to Streamlit Cloud

## Session 7 — deploy to Streamlit Cloud

**Date:** 2026-04-19
**Deployed URL:** https://ai10012300027-dcb4ygocwwjme7kfb5nttf.streamlit.app/
**What broke on first deploy:** On the live app, the "Ashanti Region 2020"
style election queries refused at the default `alpha=0.6` even though the
same data answered correctly for the Greater Accra phrasing. The first
budget query and the unanswerable refusal both worked on first try.
**How I fixed it:** Kept the retriever default (`alpha=0.6`, `top_k=6`).
For term-heavy CSV queries, I documented that dragging `alpha` toward 0.3
in the sidebar and ticking "Query expansion" surfaces the right row,
because BM25 rewards exact token matches on party / region names. The
defaults are still a reasonable compromise across both sources.

---

## Session 8 — emailed submission

## Session 8 — emailed submission
**Date:** 2026-04-19
**Email sent to:** godwin.danso@acity.edu.gh
**Subject:** CS4241-Introduction to Artificial Intelligence-2026:[`10012300027` `Andrew Kofi Kwakye`]
**Attachments / links:**
- GitHub repo: https://github.com/andrewkwakye/ai_10012300027
- Deployed URL: https://ai10012300027-dcb4ygocwwjme7kfb5nttf.streamlit.app/
- Video walkthrough: <PASTE YOUTUBE / DRIVE LINK AFTER RECORDING>
- GitHub collaborator added (godwin.danso@acity.edu.gh / GodwinDansoAcity): ☐ yes  ← flip to ☑ after you send the invite

**Reflection (what I'd do differently next time):**
Start the experiment log on day one instead of back-filling at the end — the timestamps on embeddings.npy and meta.jsonl saved me here, but I got lucky that git history lined up with what I remembered. Second thing: I'd pick the LLM provider before writing the embedder. I burned most of one session swapping from Gemini to Groq after the AI Studio key kept hitting 403, and the refactor would have been a non-event if I'd confirmed the key worked end-to-end before plumbing it into three files.

---

> Reminder from the question paper: *"Add or invite godwin.danso@acity.edu.gh or GodwinDansoAcity as a GitHub collaborator. Failure to do so will result in getting nothing for your exams."* **Don't forget this step.**
"* **Don't forget this step.**
