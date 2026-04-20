# Deployment Guide — Streamlit Community Cloud

**Author:** `Andrew Kofi Kwakye`
**Index:** `10012300027`

---

## Prerequisites

- GitHub account
- Streamlit Community Cloud account (free) — sign in at https://share.streamlit.io/ with the same GitHub account
- A Groq API key (https://console.groq.com/keys). **Free** — no credit card required. Sign in with any Google/GitHub account and click "Create API Key".
- Embeddings run locally with `sentence-transformers` (`all-MiniLM-L6-v2`), so no embedding API key is required.

## Step-by-step

### 1. Create the repository

```bash
# from the project root
git init
git add .
git commit -m "Initial commit: RAG chatbot — Andrew Kofi Kwakye 10012300027"
git branch -M main
git remote add origin https://github.com/andrewkwakye/ai_10012300027.git
git push -u origin main
```

> Repo name **must** be `ai_10012300027` — this is in the question paper.

### 2. Invite the lecturer as a collaborator

Per the exam paper: *"Add or invite godwin.danso@acity.edu.gh or GodwinDansoAcity as a GitHub collaborator. Failure to do so will result in getting nothing for your exams."*

GitHub → your repo → Settings → Collaborators → **Add people** → search `GodwinDansoAcity` or paste `godwin.danso@acity.edu.gh` → Add collaborator.

### 3. Build the index locally, then commit it

Streamlit Cloud's free tier doesn't give you a shell to run scripts on first boot, so we pre-build the index and commit it:

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env, paste your GROQ_API_KEY

python scripts/download_data.py   # ~1-2 min
python scripts/build_index.py     # ~2-3 min, runs sentence-transformers locally (no API calls)
```

After build, commit the processed index:

```bash
# Allow the built index to be committed (normally gitignored)
git add -f data/processed/embeddings.npy data/processed/meta.jsonl data/raw/
git commit -m "Add built index and raw source data"
git push
```

If `embeddings.npy` is > 100 MB, enable Git LFS:
```bash
git lfs install
git lfs track "*.npy"
git add .gitattributes
git add data/processed/embeddings.npy
git commit -m "LFS-track embeddings"
git push
```

### 4. Create the Streamlit app

1. Go to https://share.streamlit.io/ → **Deploy an app** → **From existing repo**.
2. Pick `ai_10012300027`, branch `main`, main file `app.py`.
3. **Advanced settings** → Python version: 3.11.
4. **Secrets** → paste:
   ```toml
   GROQ_API_KEY = "your-key-here"
   GROQ_CHAT_MODEL = "llama-3.3-70b-versatile"
   EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
   ```
   (TOML format — the quotes are required.)
5. Click **Deploy**. First build takes ~2 min. You'll get a URL like `https://<something>.streamlit.app`.

### 5. Smoke-test the deploy

- Open the URL.
- Ask: "What is the projected fiscal deficit for 2025?" — should return a grounded answer with chunk citations.
- Ask: "What is the exchange rate of the cedi to Japanese yen today?" — should refuse.
- Click a 👍 button — you should see the toast *"Recorded 👍 — future retrieval will boost this chunk"*.

### 6. Email the lecturer

Per the exam:

> Subject: `CS4241-Introduction to Artificial Intelligence-2026:[10012300027 Andrew Kofi Kwakye]`
> To: `godwin.danso@acity.edu.gh`
> Body: links to the GitHub repo, the deployed URL, and the video walkthrough; mention that the collaborator invite has been sent.

## Troubleshooting

- **ModuleNotFoundError: tiktoken** → you're on Python 3.12+ and hit the pre-built-wheel gap; pin Python 3.11 in Streamlit Cloud's Advanced settings.
- **"Index not built yet"** in the app → you forgot step 3; the index files aren't in the repo.
- **Groq 401 / 403** → secret `GROQ_API_KEY` not set, has a typo/trailing space, or was revoked in the Groq console.
- **Groq 429 (rate limit)** → the free tier enforces per-minute token limits. Wait a minute and retry; for bulk evaluation runs, add `time.sleep(2)` between queries.
- **`TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`** → httpx 0.28 dropped the `proxies` kwarg that `groq==0.11.0` still passes. `requirements.txt` already pins `httpx<0.28.0`; rerun `pip install -r requirements.txt` in the venv.
- **App sleeps after inactivity** → free tier; it auto-wakes on next request. Don't worry about it for grading.
- **"embeddings.npy is too large"** → use Git LFS as shown above, or regenerate with fewer chunks.

## Alternative deployments

The code is plain Python — it will also run on Render, Hugging Face Spaces (Streamlit SDK), Railway, or any VM. Streamlit Cloud is just the zero-effort option.
