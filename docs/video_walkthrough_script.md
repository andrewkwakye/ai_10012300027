# Video Walkthrough Script (≤ 2 minutes)

**Author:** `Andrew Kofi Kwakye`
**Index:** `10012300027`

The exam caps the video at **2 minutes**. Below is a script you can read in ~1m50s — leaves 10 seconds of buffer. Record it with Loom / OBS / Zoom (screen + voice). Upload to YouTube Unlisted or Google Drive with link-sharing, and paste the link in your submission email.

---

## 0:00 – 0:15 — Intro

> "Hi, I'm `Andrew Kofi Kwakye`, index `10012300027`. This is my CS4241 RAG chatbot for Academic City. It answers questions over Ghana's 2025 Budget Statement and historical Ghana election results, and I built every component from scratch — no LangChain or LlamaIndex."

## 0:15 – 0:40 — Architecture (show `docs/architecture.md` diagram on screen)

> "The pipeline has two phases. Offline: I clean the CSV and PDF, split them with a per-source chunker — rows for the CSV, a recursive paragraph-to-sentence splitter for the PDF — embed every chunk locally with sentence-transformers (all-MiniLM-L6-v2, 384-dim), and save a NumPy matrix plus a JSONL metadata file. Online: the query is embedded, scored against every vector, and against a BM25 keyword index I also built on the same tokens. I combine them with an alpha weight, that's my hybrid search. Chat inference runs on Groq's LPU using Llama 3.3 70B."

## 0:40 – 1:05 — Demo an answerable query (share Streamlit window)

> "Here's the deployed app. I'll ask *'What is the projected fiscal deficit for 2025 in Ghana?'* [wait] — it returns a grounded answer with chunk citations, and below I can see the exact chunks that were retrieved, each with its dense score, its BM25 score, and the combined score."

## 1:05 – 1:25 — Show the refusal behaviour (adversarial query)

> "Now the interesting part — I'll ask *'What is the cedi-to-yen exchange rate today?'*. That's not in my corpus. The confidence gate sees a low top score, and the v3 prompt instructs the model to refuse. No hallucination — it says it doesn't have enough information."

## 1:25 – 1:50 — Demonstrate the innovation feature

> "My innovation component is a feedback loop. I'll click 👍 on the correct chunk under my last answer. That signal is appended to a JSONL file, and — watch what happens when I re-run a related query — the same chunk now has a positive boost and rises in the ranking. The boost is clamped so feedback can never override semantic similarity, which keeps the system honest."

## 1:50 – 2:00 — Close

> "The repo is on GitHub at `ai_10012300027`, the full docs for chunking, retrieval, prompts, evaluation, architecture, and the innovation write-up are all in the `docs/` folder. Thanks."

---

## Filming tips

- **Pre-record** a terminal already showing the index built and the Streamlit app already loaded. The 2-minute cap is brutal; don't waste seconds on boot.
- **Record at 1080p** so the retrieved-chunk text is readable.
- **Have two browser tabs open**: the deployed URL and `docs/architecture.md` previewed on GitHub.
- **Don't read line-by-line** — skim the script, then record. Natural delivery beats perfect wording.

## What to NOT show in the video

- Your API key (Streamlit secrets are fine, just don't scroll through them).
- Any other student's work.
- Internet dead-ends — pre-test the demo queries so they work on the first try.
