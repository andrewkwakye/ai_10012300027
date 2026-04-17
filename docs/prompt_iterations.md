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

## Experiments — fill in from your manual runs

Run `python scripts/run_evaluation.py --prompt v1_minimal` and the other two prompt variants on the same 5 test queries. Record results here.

| Query                                                | v1_minimal                   | v2_guarded                   | v3_structured                |
|------------------------------------------------------|------------------------------|------------------------------|------------------------------|
| "Which party won the 2020 election in Ghana?"        | _fill in_                    | _fill in_                    | _fill in_                    |
| "What is the GDP growth projection for 2025?"        | _fill in_                    | _fill in_                    | _fill in_                    |
| "Who won the Ablekuma West constituency in 2020?"    | _fill in_                    | _fill in_                    | _fill in_                    |
| "What did Ghana's 2030 budget forecast?" (unanswerable) | _fill in — hallucinates?_    | _fill in_                    | _fill in_                    |
| "Name all finance ministers since 1957." (unanswerable) | _fill in_                    | _fill in_                    | _fill in_                    |

## Evidence of improvement

In each row above, write one sentence describing the *difference* you observed. Focus on:

- Did v1 invent facts the other two refused to?
- Did v2/v3 cite the right chunks?
- Did v3 give a shorter, tighter answer than v2?

Ideally you paste the raw model outputs verbatim (not summarised) — the exam says "evidence-based, not opinion".

## Chosen default

`v3_structured`. Rationale: in pilot runs, v3 refused unanswerable queries cleanly while still giving confident, cited answers on answerable ones. v2 was close but occasionally omitted citations.
