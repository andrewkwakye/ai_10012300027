# Prompt Design Iterations

**Author:** `Andrew Kofi Kwakye`
**Index:** `10012300027`
**Covers:** Exam Part C

---

## Design goals

1. **Inject retrieved context** — the model must be grounded in the CSV + PDF, not its pretraining memory.
2. **Hallucination control** — explicit refusal instruction, no speculation, citations required.
3. **Context-window management** — we never exceed `CONFIG.max_context_tokens = 3000` in the user turn; overflow chunks are dropped (lowest ranked first).

All three templates live in `src/prompt_builder.py`.

## v1 — minimal baseline

```
SYSTEM: You are a helpful assistant for Academic City University. Answer the user's question using the provided context.
USER:   Context:
        <chunks>
        Question: <q>
        Answer:
```

**Why it's in the repo:** it's the control group for the "same query / different prompts" experiment the rubric asks for. Expected to hallucinate on edge cases.

## v2 — guarded + citations

- Requires explicit refusal on insufficient context: a fixed phrase so it's detectable by our evaluator.
- Adds citation requirement: `[#2]` style, tied to chunk IDs emitted by `pack_context`.

## v3 — structured reasoning (shipped default)

- Procedural system prompt with 5 numbered steps.
- Explicitly bans speculation about causes / motivations / future events.
- Requires numbers and proper nouns to come **verbatim** from the context.

This is what the Streamlit app uses by default.

## Context-window management

`src/prompt_builder.py::pack_context` accepts a list of `RetrievalResult` in rank order, tokenises each framed block with tiktoken, and stops when the cumulative budget exceeds `CONFIG.max_context_tokens`. Lowest-ranked chunks are dropped first (since they contribute least confidence). This keeps the prompt well under the generator's 128k context window while letting us experiment with chunk count and size independently.

## Experiments — observed behaviour on the three shipped templates

The three templates are selectable in the Streamlit sidebar (`Prompt template` dropdown). During iteration I ran each template against the same five-query evaluation suite used by `scripts/run_evaluation.py`. The shipped evaluation CSV (`logs/evaluation_results.csv`) uses `v3_structured` throughout; v1 and v2 were spot-checked in the app rather than re-run as full two-pass evaluations. The qualitative differences are consistent across every spot-check.

| Query                                                          | v1_minimal                         | v2_guarded                        | v3_structured                                                                                         |
|----------------------------------------------------------------|------------------------------------|-----------------------------------|-------------------------------------------------------------------------------------------------------|
| "What is the projected fiscal deficit for 2025?" (answerable)  | Answers, **no citations**          | Answers, cites inconsistently     | **Grounded answer with [:#1, #5] citations** — numbers verbatim ("3.1 % commitment, 4.1 % cash")       |
| "Compare NPP and NDC votes in Greater Accra Region in 2020" (answerable) | Answers, sometimes adds commentary | Answers + citations               | Grounded answer, tight, cites the Greater Accra CSV row                                                |
| "What did the government do about taxes?" (ambiguous)          | Picks one framing confidently      | Lists two framings                | Lists measures verbatim from the budget, cites [:#4] — see Q3 in `logs/evaluation_results.csv`        |
| "How many votes did NPP win in 2030 Ghana election?" (misleading) | **Fabricates a plausible-sounding answer** in casual testing | Refuses but sometimes softens the refusal | **Refuses cleanly** — exact string: *"I don't have enough information in my source documents to answer that."* |
| "Exchange rate of cedi to Japanese yen today?" (unanswerable)  | Sometimes guesses a stale rate     | Refuses                           | Refuses cleanly, same exact string as above                                                           |

## Evidence of improvement

1. **Hallucination control works only in v3.** v1_minimal has no refusal clause at all — on the 2030-election query it produces a confident, invented number (a classic hallucination failure). v2_guarded adds a refusal instruction but in casual testing occasionally drifted into partial answers ("The 2030 election has not occurred, but based on trends…"). v3_structured refuses verbatim because the system prompt's step 5 is a single-sentence refusal that doesn't permit follow-on speculation.

2. **Citation consistency tracks prompt structure.** v1 never cites. v2 cites when the context is short but forgets on long context packs because the instruction is stated once, early. v3 frames citations as a procedural step — "for each claim, append the chunk id(s) that support it" — and the model complies on every answerable query in the evaluation CSV (Q2 cites `[:#1, #5]`, Q3 cites `[:#4]`).

3. **Verbatim quoting prevents number drift.** v3's step 3 ("numbers and proper nouns must appear verbatim from context") is what produces exact strings like "3.1 percent of GDP on a commitment basis" rather than a paraphrase like "around three percent." On a government-data assistant this distinction matters — paraphrased numbers are how fake statistics enter the wild.

These differences are the reason `v3_structured` is the shipped default and the one used throughout `logs/evaluation_results.csv`.

## Chosen default

`v3_structured`. Rationale: in pilot runs, v3 refused unanswerable queries cleanly while still giving confident, cited answers on answerable ones. v2 was close but occasionally omitted citations.
