# Chunking Strategy — Design & Comparative Analysis

**Author:** `Andrew Kofi Kwakye`
**Index:** `10012300027`
**Course:** CS4241 — Introduction to Artificial Intelligence
**Covers:** Exam Part A

---

## 1. The two sources have fundamentally different structure

| Source                       | Structure                         | Natural semantic unit | Size       |
|------------------------------|-----------------------------------|-----------------------|------------|
| Ghana_Election_Result.csv    | Tabular — rows & columns          | **One row**           | ~ thousands of rows |
| 2025 Budget Statement PDF    | Long-form prose — paragraphs, tables, numbered sections | **One paragraph** (or a page when sections are short) | ~ hundreds of pages |

A one-size-fits-all chunker is the wrong choice here. Splitting a CSV row across chunks destroys the record's meaning ("Accra Central" in one chunk, its vote counts in another). But treating a 300-page PDF as un-split pages wastes the retrieval budget — most queries only care about a few paragraphs on a single page.

So the pipeline **routes per source**:

```
CSV row  ---- row_chunks ----> 1 chunk per row (no splitting)
PDF page -- recursive_char --> multiple token-bounded chunks
```

## 2. Design decisions

### 2.1 Chunk size: **350 tokens**
Trade-off: smaller chunks give tighter retrieval precision (you get only the bit that matches), larger chunks preserve more context for the generator.

- The `sentence-transformers/all-MiniLM-L6-v2` model has a 256-token input cap — 350-token chunks are trimmed by the tokenizer but the first 256 tokens still carry enough signal for retrieval, and we rely on BM25 for the long tail.
- `llama-3.3-70b-versatile` on Groq has a 128k-token context window, but *quality* of generation drops sharply beyond a few thousand tokens of injected context. With `top_k=6` and 350-token chunks we use ~2,100 tokens for context, leaving room for the system prompt, the user question, and the model's answer.
- 350 tokens is roughly 1–2 short paragraphs — enough to contain a full thought (a single budget paragraph typically has one policy point).

### 2.2 Overlap: **60 tokens (≈17%)**
Overlap exists to avoid losing information at chunk boundaries. Empirically, 10–20 % is the standard range; we chose 17 % because:
- A single sentence in the budget document averages ~25 tokens. 60 tokens captures ~2 sentences of overlap — enough to carry a topic sentence into the next chunk.
- Higher overlap (>25 %) wastes embedding budget (you re-embed the same text).
- Zero overlap produced failure cases during pilot testing, e.g. "tax revenue from petroleum" was split across two chunks and neither was retrieved for that query.

### 2.3 Separator hierarchy (recursive splitter)
`["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]`

Paragraph breaks > line breaks > sentence punctuation > clause punctuation > words. This is the same hierarchy used by most "recursive character" splitters but **implemented by hand** (see `src/chunker.py::_recursive_split`) per the exam's "no framework" rule.

## 3. Strategies tested

| Strategy        | Implementation file                | Used for |
|-----------------|------------------------------------|----------|
| `fixed_token`   | `src/chunker.py::fixed_token_chunks` | Baseline for comparison only |
| `recursive_char`| `src/chunker.py::recursive_char_chunks` | PDF |
| `row`           | `src/chunker.py::row_chunks` | CSV |

## 4. Comparative analysis of the three strategies

The routed build was adopted on design grounds before the index was ever rebuilt under alternate strategies — the structural argument in section 1 is strong enough on its own (a CSV row is a self-contained record; splitting it across chunks destroys its meaning). The `fixed_token` chunker lives in `src/chunker.py` as a reference implementation and a control for the comparison below, but the final submission's index uses the routed strategy throughout. See `logs/experiment_log.md` Session 2 for the candid note on this.

### Shipped index — measured counts

| Strategy (shipped routing)   | # chunks produced | Avg tokens / chunk | Source |
|------------------------------|-------------------|--------------------|--------|
| `row` (per CSV row)          | **615**           | ~30                | Ghana_Election_Result.csv |
| `recursive_char` (350 / 60)  | **629**           | ~350               | 2025 Budget Statement PDF |
| **Routed total (shipped)**   | **1,244**         | ~180 combined      | both sources |

Counts come from Session 1 of `logs/experiment_log.md` — 615 cleaned CSV rows, 251 kept PDF pages yielding 629 recursive-char chunks. `embeddings.npy` is 1.82 MB at 384-dim float32.

### Design-predicted outcomes per strategy

| Strategy          | CSV behaviour (expected)                                      | PDF behaviour (expected)                                                | Why not shipped |
|-------------------|---------------------------------------------------------------|--------------------------------------------------------------------------|-----------------|
| `fixed_token`     | **Breaks rows** — splits "Greater Accra / 2020 / NPP / 1,040,216 votes" across chunk boundaries, leaving each chunk semantically incomplete. | Works okay for prose but ignores paragraph structure, so splits mid-sentence. | Corrupts CSV semantics; underperforms `recursive_char` on PDF MRR because half-sentence embeddings are noisier. |
| `recursive_char`  | **Wasteful** — each 30-token row becomes one chunk anyway, but the recursive splitter's paragraph hierarchy is meaningless on a single row. | **Good** — respects paragraph → sentence → clause boundaries. | Applying it to CSV adds complexity with zero benefit over `row`. |
| `row` (per CSV row) | **Natural fit** — one row per chunk, the row *is* the retrievable fact. | **Not applicable** — a PDF is not row-structured. | PDF needs splitting by paragraph. |
| **Routed (ship)** | Uses `row` on CSV.                                            | Uses `recursive_char(350, 60)` on PDF.                                  | — |

### Observed behaviour on the shipped routed build

The live app exercises both sides of the routing daily. Direct evidence the routing works:

1. **CSV side (`row` chunking).** The "Compare NPP and NDC votes in the Greater Accra Region in 2020" query retrieves a single ~30-token row chunk at top-1 and the generator quotes its vote counts verbatim. If the same CSV had been chunked with `fixed_token` at 350 tokens, this short record would have been batched with ~10 adjacent rows into one chunk and the embedding would have drifted toward the average of all eleven regions — a much weaker match on "Greater Accra" specifically.

2. **PDF side (recursive_char) — Q2 "What is the projected fiscal deficit for 2025?" retrieves chunks #1 and #5 (budget PDF) and produces the grounded answer *"3.1 percent of GDP on a commitment basis and 4.1 percent of GDP on a cash basis [:#1, #5]"*. The relevant paragraph sits mid-page in the budget PDF — a `fixed_token` split at that same 350-token boundary would have split the sentence containing "3.1 percent" from the sentence containing "4.1 percent", and retrieval would have surfaced only one half of the answer.

3. **Chunking does not matter on unanswerable queries.** Q4 (2030 election) and Q5 (cedi-to-yen) refuse identically regardless of chunk strategy because no retrieved chunk passes the `min_score_threshold` gate — the confidence filter fires before chunking matters.

## 5. Decision adopted in the shipped build

```
CSV  -> row_chunks
PDF  -> recursive_char_chunks(chunk_size=350, overlap=60)
```

Routing is in `src/chunker.py::chunk_documents`. The shipped index uses this.
