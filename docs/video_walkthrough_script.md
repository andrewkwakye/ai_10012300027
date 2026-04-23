# Video Walkthrough Script (≤ 2 minutes)

**Author:** Andrew Kofi Kwakye
**Index:** 10012300027

Plain-English script. Read it once, then record in your own voice — don't read line-by-line.

---

## 0:00 – 0:15 — Intro

> "Hi, I'm Andrew Kofi Kwakye, index 10012300027. This is my RAG chatbot for CS4241. It answers questions about Ghana's 2025 Budget and Ghana's past election results. I built every piece of it myself — no ready-made frameworks like LangChain."

## 0:15 – 0:40 — How it works (show the architecture diagram)

> "Here's how it works. First, I take the CSV and the PDF and break them into small pieces I call chunks. Each row of the CSV is one chunk. The PDF I split into short paragraphs.
>
> Then I save all the chunks with a tag that represents their meaning, so I can match them later.
>
> When someone asks a question, I find the chunks that match best — partly by meaning, partly by the exact words in the question. I take the top few, give them to the AI model, and the model uses those chunks to write the answer."

## 0:40 – 1:05 — Ask a question about the budget

> "Let me show you. I'll type *'What is the projected fiscal deficit for 2025 in Ghana?'* … It gives me the number straight from the budget document, and it cites the chunks it used. Below the answer I can see which chunks were retrieved and their scores."

## 1:05 – 1:25 — Ask a question about the election data

> "Now something from the CSV — *'How many votes did the NPP receive in Greater Accra in 2020?'* … Top result is the exact row from the CSV. This shows the chunker handled the two data sources differently — rows for the CSV, paragraphs for the PDF."

## 1:25 – 1:45 — Ask something it should refuse

> "Now I'll try something that isn't in my data — *'What is the cedi-to-yen exchange rate today?'* … It refuses instead of making up an answer. That's the whole point — on government data, a wrong answer is worse than no answer."

## 1:45 – 2:00 — Show the feedback feature + close

> "My custom feature is a feedback loop. If I click thumbs-up on a good chunk, it gets a small boost the next time I ask a similar question. Thumbs-down does the opposite. That way the system gets better the more I use it.
>
> Code is on GitHub under `ai_10012300027`. Thanks."

---

## Filming tips

- Have the Streamlit app already open and the index already built before you hit record. Two minutes goes fast.
- Pre-test every query once so you know it works. Don't debug on camera.
- Record at 1080p so the text on screen is readable.
- Speak a bit slower than feels natural — nerves make people speed up.
- If you fumble a line, don't stop. Keep going. Record a second take and use the cleaner one.

## What NOT to show

- Your API key (if you have to click around settings, skip that bit).
- Any error messages from earlier testing. Close the terminal if it's showing old output.
- The inside of `.env` or Streamlit secrets.

## Words you might not want to use on camera

If any of these trip you up, here are simpler swaps you can say instead:

- "embedding" → "turn into a vector" or just "convert it"
- "vector store" → "the saved index" or "the saved chunks"
- "hybrid search / BM25" → "I combine a meaning-based match with a keyword match"
- "confidence gate" → "if the best match isn't strong enough, it refuses"
- "Groq LPU" → just say "Groq" — it's the company name
- "llama-3.3-70b-versatile" → just "a large language model" is fine
