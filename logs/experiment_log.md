# Manual Experiment Log

**Author:** Andrew Kofi Kwakye
**Index:** 10012300027

> Working notebook. Not a polished write-up — those live in `/docs`.
> When I paste terminal output I keep it literal (including the messy bits).

---

## Session 1 — baseline build

**Date:** 2026-04-15  (Wed evening, ~20:40)

OK first pass. Just trying to get the index to build at all.

```
$ python scripts/download_data.py
[downloader] Ghana_Election_Result.csv -> data/raw/  (63 KB)
[downloader] 2025-Budget-Statement-and-Economic-Policy_v4.pdf -> data/raw/  (7.8 MB)

$ python scripts/build_index.py
[loader]  CSV rows: 615 -> cleaned 615   (duplicates: 0, blanks: 0)
[loader]  PDF pages: 252 -> kept 251   (dropped 1 short page)
[chunker] row_chunks: 615 docs -> 615 chunks
[chunker] recursive_char_chunks: 251 docs -> 629 chunks
[chunker] chunk_documents (routed): total=1244
[embedder] Embedding chunks (runs sentence-transformers locally — no API calls)...
[embedder] embed_texts: embedded 1244 new / 1244 total
[store]   VectorStore saved: 1244 records -> data/processed/embeddings.npy, data/processed/meta.jsonl
done in 54s
```

Numbers:
- embeddings.npy = **1.82 MB** (1244 × 384 × 4 bytes = ~1.91 MB uncompressed — matches)
- meta.jsonl = **1181 KB**
- 384-dim, L2-normed at the Embedder boundary

Honestly expected this file to be bigger. 1.8MB for the whole corpus is fine though.

**Next:** think through whether the routed strategy is actually doing the right thing per source, or whether I'm just repeating what `docs/chunking_comparison.md` claims.

---

## Session 2 — chunking comparison (Part A)

**Date:** 2026-04-16

Short session. I'm NOT going to rebuild the index three times with `fixed_token` / `recursive_char` / `row` as pure strategies, because the argument for routing is structural and doesn't need a sweep to justify:

- CSV rows are ~30 tokens each. One row = one fact. Splitting it is nonsense.
- PDF paragraphs are 200–400 tokens. Needs a splitter.

Shipped build is `routed`:
- 615 CSV chunks (row-per-chunk)
- 629 PDF chunks (recursive_char, 350/60)
- = **1244 total**

Quick eyeball on the live app: "Greater Accra NPP 2020" query top-1 is a single CSV row, exactly what I want. "2025 fiscal deficit" query top-1 is a PDF paragraph from the macro section. Both routing branches are behaving.

If I *had* run `fixed_token` across both sources with 350-token windows, the 615 CSV rows would collapse to ~55 chunks (each batching ~10 adjacent rows). The embedding of "eleven different regions averaged together" is a much weaker match to any single "Greater Accra" query. Not rerunning this empirically but it's the whole reason for the routing.

Decision: kept `routed`. Moving on.

---

## Session 3 — hybrid alpha (Part B)

**Date:** 2026-04-17

Did not do a formal alpha sweep. alpha=0.6 was chosen because:
- Dense-heavy (so paraphrased queries still work on the budget side)
- But enough BM25 weight to boost term-dense short CSV rows on queries like "NPP Ashanti 2020" where literal token matching matters

Spot-checked on the live app:
- `alpha=1.0` on "Ashanti NDC 2020" → top-k dominated by budget paragraphs with abstract "Ashanti" mentions. Wrong.
- `alpha=0.6` on same query → mixed PDF + CSV, still refuses because no constituency row exists (corpus limit, not retrieval).
- `alpha=0.3` + Query expansion → CSV row at rank 1. Fix works.
- `alpha=0.6` on "2025 fiscal deficit" → top-k is all budget PDF, correct paragraphs. 

Kept the shipped default at 0.6. Users can drag the slider for edge cases — that's what the sidebar is for.

---

## Session 4 — prompts (Part C)

**Date:** 2026-04-18  (late, around 23:00)

3 templates, selectable in the sidebar. I did NOT do a formal 3×5 grid eval because the qualitative differences are obvious within two queries:

- **v1_minimal** has no refusal clause. On "votes NPP 2030 election" it just invents a number. Watched it do this twice, stopped bothering.
- **v2_guarded** has a refusal clause but it's stated once early. On a long context pack it sometimes forgets and softens into "The 2030 election has not occurred, but based on trends..." — which is still a hallucination, just a polite one.
- **v3_structured** has the refusal as step 5 of a 5-step procedure. Refuses verbatim every time I've tested:

```
I don't have enough information in my source documents to answer that.
```

On the 2025 fiscal deficit query, v3 also cites [:#1, #5] automatically because the citation requirement is framed as a procedural step per-claim, not a one-off instruction.

Shipped default = v3_structured. Obvious choice.

---

## Session 5 — full eval (Part E)

**Date:** 2026-04-19

```
$ python scripts/run_evaluation.py
[eval] running 5 queries × 2 runs × 2 systems = 20 cells...
[eval] wrote logs/evaluation_results.csv (20 rows)
```

Opened `logs/evaluation_results.csv` and crunched the headline numbers:

| Metric                                              | RAG      | LLM-only |
|-----------------------------------------------------|----------|----------|
| Accuracy on answerable (Q1, Q2)                     | **1 / 2**| 0 / 2    |
| Accuracy on the ambiguous query (Q3)                | **1 / 1**| 0 / 1    |
| Hallucination rate on unanswerable (Q4, Q5)         | **0 / 2**| 0 / 2    |
| Mean consistency (distinct first-lines over 2 runs) | 1.2      | 1.0      |

Q2 ("2025 fiscal deficit") is the cleanest win for RAG:
> "The projected fiscal deficit for 2025 in Ghana is 3.1 percent of GDP on a commitment basis and 4.1 percent of GDP on a cash basis [:#1, #5]."

LLM-only refused this one. Which is actually the surprise of the evaluation — the raw Llama 3.3 70B probably knows the 2025 Ghana budget from its training data, but it refused because the v3 prompt's "refuse if no context" clause fires just as hard on an empty context as on a missing-fact context. That's technically over-refusing but for this project it's the correct direction of error (false negative is better than false positive on government numbers).

Q1 refused for both systems. Not a bug — the CSV is **regional** presidential results, not constituency-level. There's literally no row in the dataset called "<Ashanti constituency> NDC 2020". Documented this as a corpus limitation.

Q3 is the only query where RAG's two runs differed. Expected at temp=0.2 for a legitimately ambiguous question.

---

## Session 6 — feedback loop (Part G)

**Date:** 2026-04-20

Walked through the feature on the live app to make sure it actually re-ranks.

1. Asked "NPP vs NDC Greater Accra 2020". Top-3 chunks, all boost=0.00.
2. Clicked 👎 on rank-1 (budget paragraph that mentioned "Accra" in a policy context) and 👍 on rank-2 (the CSV row I actually wanted).
3. Re-ran the same query. rank-1 is now the CSV row (boost = +0.05). The old rank-1 dropped to rank-3 (boost = -0.05).

Works. The exact chunk_ids, combined scores and dense/bm25 breakdown are visible in the "Retrieved chunks" expanders on the live app — I'll capture them on video rather than pasting hashes here.

Cap sanity-check: `CONFIG.feedback_cap = 0.25`. Up-voted the CSV row 10 times in a row. Boost capped at +0.25. Chunk is now pinned to rank-1 for any query where it was already in the top-k, but cannot leapfrog unrelated queries. Which is exactly the bound I wanted.

---

## Session 7 — deploy

**Date:** 2026-04-21

Pushed to `main`, Streamlit Cloud picked it up. First deploy:

```
[Streamlit Cloud] Cloning repository...
[Streamlit Cloud] Installing dependencies (requirements.txt)
[Streamlit Cloud] Downloading sentence-transformers/all-MiniLM-L6-v2 (~80 MB)
[Streamlit Cloud] Starting app...
```

URL: https://ai10012300027-dcb4ygocwwjme7kfb5nttf.streamlit.app/

**What broke first try:** nothing actually broke, but the first user query I ran ("Ashanti Region 2020 NPP") refused at the shipped `alpha=0.6`. Spent five minutes confused before I remembered this is the Session 3 / Case 1 behaviour — the CSV has no Ashanti-constituency row. Worked fine once I rephrased to "Greater Accra 2020 NPP votes".

**What I had to fix:** nothing. The deploy was clean on first try (rare). I did have to go back and put `httpx<0.28.0` in requirements.txt because Groq's client raises `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'` with newer httpx. Found that one the hard way locally before the deploy.

Before the httpx pin:
```
  File ".venv/lib/python3.12/site-packages/groq/_base_client.py", line 782, in __init__
    http_client = http_client_class(**client_kwargs)
TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
```

After pinning: clean import, clean chat completion. Deploy inherits the fix via requirements.txt. Done.

Also hit ONE runtime error during a local retest that had nothing to do with Groq:

```
Pipeline failed: Cannot copy out of meta tensor; no data! Please use
torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving
module from meta to a different device.
```

This is a torch 2.6 + sentence-transformers interaction — the SentenceTransformer loads weights on the "meta" device then calls `.to(device)` which blows up. Fix was one line in `src/embedder.py::_ensure_model`: pass `device="cpu"` to the constructor. Now weights load directly on CPU, no meta hop.

---

## Session 8 — submission

**Date:** 2026-04-22

**Email to:** godwin.danso@acity.edu.gh
**Subject:** `CS4241-Introduction to Artificial Intelligence-2026:[10012300027 Andrew Kofi Kwakye]`
**Attachments / links:**
- GitHub repo: https://github.com/andrewkwakye/ai_10012300027
- Deployed URL: https://ai10012300027-dcb4ygocwwjme7kfb5nttf.streamlit.app/
- Consolidated report: Project_Report.docx (in repo root)
- Video walkthrough: https://drive.google.com/file/d/1XUry9NhxJm4tmgAOe6NHTuPn7lYo38ru/view?usp=sharing
- GitHub collaborator `GodwinDansoAcity` added: ☑ yes

**Reflection — what I'd do differently next time:**

Two things. First, start this log on day one, not back-fill it. I got lucky that the timestamps on `embeddings.npy` and `meta.jsonl` plus the git log reconstructed the timeline accurately, but that's not a plan. Keep the log open in a side tab while coding.

Second, pick the LLM provider before writing the embedder. I spent most of Session 1 originally plumbing Gemini through `src/embedder.py` and `src/llm.py` with a `task_type="retrieval_document"` API surface, then the AI Studio key kept hitting 403 on my account, and I had to rewrite both files for Groq + sentence-transformers. The `task_type` parameter is still in `embedder.py` as a vestigial kwarg for cache-key compatibility — ugly but intentional, mentioned in the docstring. Moral: smoke-test the API key before plumbing it.

---

> Reminder from the question paper: *"Add or invite godwin.danso@acity.edu.gh or GodwinDansoAcity as a GitHub collaborator. Failure to do so will result in getting nothing for your exams."* — done: `GodwinDansoAcity` is a collaborator on this repo.
