# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""Diagnostic — asks Gemini what models my API key can access."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import google.generativeai as genai  # noqa: E402

from src.config import GEMINI_API_KEY  # noqa: E402


def main() -> None:
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is not loaded. Check your .env file.")
        return

    print(f"Key prefix: {GEMINI_API_KEY[:6]}... (length={len(GEMINI_API_KEY)})")
    genai.configure(api_key=GEMINI_API_KEY)

    try:
        print("\n--- Models that support embedContent ---")
        for m in genai.list_models():
            if "embedContent" in m.supported_generation_methods:
                print(f"  {m.name}")

        print("\n--- Models that support generateContent (chat) ---")
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                print(f"  {m.name}")
    except Exception as e:
        print(f"\nERROR from Gemini: {e}")
        print(
            "\nThis usually means:\n"
            "  (a) the API key is invalid / revoked, or\n"
            "  (b) the key is not from https://aistudio.google.com/app/apikey\n"
            "A valid AI Studio key normally starts with 'AIzaSy'."
        )


if __name__ == "__main__":
    main()
