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

## 4. Comparative experiment — how to read the results table below

We ran the same set of 10 evaluation queries through three index builds that differ **only** in chunker choice, then scored:

- **Recall@6** — of the queries with a clearly-correct answer in the corpus, fraction whose ground-truth span appears in the top-6 retrieved chunks.
- **MRR@6** — Mean Reciprocal Rank of the first relevant chunk.
- **Avg chunk size (tokens)** — characterises how much context is dumped into the prompt.

You will fill this table **manually** during your experiment runs (see `logs/experiment_log.md`). An example row is pre-filled so you know the format:

| Strategy                    | # chunks produced | Avg tokens / chunk | Recall@6 | MRR@6 | Notes |
|-----------------------------|-------------------|--------------------|----------|-------|-------|
| `fixed_token` (350/60)      | _fill in_         | _fill in_          | _fill in_ | _fill in_ | _fill in_ |
| `recursive_char` (350/60)   | _fill in_         | _fill in_          | _fill in_ | _fill in_ | _fill in_ |
| `row` (CSV rows only)       | _fill in_         | _fill in_          | _fill in_ | _fill in_ | Only makes sense for the CSV source |

### Expected direction (what we predicted before running)

1. On the **PDF**, `recursive_char` should beat `fixed_token` on MRR because fixed-token chunking frequently cuts sentences in half, so the embedding of a half-sentence is noisier.
2. On the **CSV**, `row` should trivially dominate — a single row *is* the answer for most factual election queries.
3. `fixed_token` with small chunk sizes (<150 tokens) over-shards the PDF and **recall drops** because the same topic is now spread over too many chunks, none of which individually contains enough context to match semantically rich queries.

### Observed behaviour
Fill in from your experiment logs. Include:
- At least one concrete query where `fixed_token` retrieved a cut-off chunk that `recursive_char` handled correctly.
- At least one query where chunking didn't matter (both strategies retrieved an identical top-1).

## 5. Decision adopted in the shipped build

```
CSV  -> row_chunks
PDF  -> recursive_char_chunks(chunk_size=350, overlap=60)
```

Routing is in `src/chunker.py::chunk_documents`. The shipped index uses this.
