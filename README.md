# Academic City RAG Chatbot — CS4241 Introduction to Artificial Intelligence

**Author:** `Andrew Kofi Kwakye`
**Index number:** `10012300027`
**Course:** CS4241 — Introduction to Artificial Intelligence
**Lecturer:** Godwin N. Danso
**Semester:** End of Second Semester, 2026

---

## What this is

A Retrieval-Augmented Generation (RAG) chat assistant for Academic City, built from first principles. It answers questions over two public Ghanaian data sources:

1. **Ghana Election Results** — tabular data (CSV)
2. **Ghana 2025 Budget Statement & Economic Policy** — long-form PDF (Ministry of Finance)

Per the exam brief, **no end-to-end RAG framework is used**. LangChain, LlamaIndex, Haystack, and similar were deliberately avoided. Every core component — chunking, embedding, vector store, retrieval, prompt construction, and the evaluation harness — is written by hand.

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/andrewkwakye/ai_10012300027.git
cd ai_10012300027
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Add your Groq API key (free at https://console.groq.com/keys)
cp .env.example .env
# then edit .env and paste your GROQ_API_KEY
# embeddings run locally via sentence-transformers — no embedding API key needed

# 3. Build the index (downloads data, chunks, embeds)
python scripts/download_data.py
python scripts/build_index.py

# 4. Run the app
streamlit run app.py
```

## Repository map

| Path                          | Purpose                                                        |
| ----------------------------- | -------------------------------------------------------------- |
| `app.py`                      | Streamlit UI (the deployed surface)                            |
| `src/config.py`               | All tunable constants + secret loading                         |
| `src/logger.py`               | Structured per-stage logger (Part D)                           |
| `src/data_loader.py`          | CSV + PDF loading and cleaning                                 |
| `src/chunker.py`              | Three chunking strategies (Part A)                             |
| `src/embedder.py`             | Local sentence-transformers embedding wrapper with caching (Part B) |
| `src/vector_store.py`         | Custom NumPy vector store (Part B)                             |
| `src/retriever.py`            | Top-k, hybrid (BM25+vector), query expansion (Part B)          |
| `src/prompt_builder.py`       | Prompt templates + context-window management (Part C)          |
| `src/llm.py`                  | Groq chat-completion wrapper (`llama-3.3-70b-versatile`)       |
| `src/pipeline.py`             | End-to-end orchestrator (Part D)                               |
| `src/feedback.py`             | Feedback-loop re-ranker (Part G — innovation)                  |
| `src/evaluator.py`            | Adversarial tests, RAG-vs-LLM comparison (Part E)              |
| `scripts/download_data.py`    | Downloads the CSV + PDF from source URLs                       |
| `scripts/build_index.py`      | Runs full ingestion → chunks → embeddings → on-disk index      |
| `scripts/run_evaluation.py`   | Runs the evaluation suite, writes CSV                          |
| `docs/architecture.md`        | System diagram + design justification (Part F)                 |
| `docs/chunking_comparison.md` | Comparative chunking analysis (Part A)                         |
| `docs/prompt_iterations.md`   | Prompt design history (Part C)                                 |
| `logs/experiment_log.md`      | **Manual** experiment logs (filled by hand, not AI-generated)  |
| `tests/test_smoke.py`         | Import + sanity checks                                         |

## Deployment

See `docs/deployment.md`. Short version: Streamlit Community Cloud, point at this repo, set `GROQ_API_KEY` in the Secrets panel, done.

## Contact

Deployed URL: https://ai10012300027-dcb4ygocwwjme7kfb5nttf.streamlit.app/
GitHub: https://github.com/andrewkwakye/ai_10012300027
Author email: kwakyeandrewkofi@gmail.com

---

> All files include the author name and index number at the top, per the exam instructions.
