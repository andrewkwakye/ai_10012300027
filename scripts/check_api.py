# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""Diagnostic — pings Groq with the configured key to confirm access."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from groq import Groq  # noqa: E402

from src.config import CHAT_MODEL, GROQ_API_KEY  # noqa: E402


def main() -> None:
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY is not loaded. Check your .env file.")
        return

    print(f"Key prefix: {GROQ_API_KEY[:6]}... (length={len(GROQ_API_KEY)})")
    print(f"Chat model: {CHAT_MODEL}")

    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a terse diagnostic agent."},
                {"role": "user", "content": "Reply with exactly: OK"},
            ],
            temperature=0.0,
            max_tokens=4,
        )
        answer = resp.choices[0].message.content.strip()
        print(f"\nResponse: {answer!r}")
        if "OK" in answer.upper():
            print("\n[PASS] Groq API key is valid and the chat model is reachable.")
        else:
            print("\n[WARN] Unexpected reply — key is valid but model output is odd.")
    except Exception as e:
        print(f"\nERROR from Groq: {e}")
        print(
            "\nThis usually means:\n"
            "  (a) the API key is invalid / revoked, or\n"
            "  (b) the key is not from https://console.groq.com/keys\n"
            "A valid Groq key normally starts with 'gsk_'."
        )


if __name__ == "__main__":
    main()
