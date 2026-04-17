# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""
Thin Groq chat-completion wrapper.

We use the official `groq` Python SDK strictly as a transport — no chains,
no agents, no abstractions. This keeps the pipeline in pipeline.py explicit
and auditable, which matters for the exam's "no RAG framework" rule.

Groq exposes an OpenAI-compatible chat-completions endpoint, hosted on their
LPU inference stack. Free tier limits (as of writing): 30 requests/min and
6,000 tokens/min on the default chat model, which is plenty for this project.
"""

from __future__ import annotations

from groq import Groq

from .config import CHAT_MODEL, CONFIG, GROQ_API_KEY
from .logger import get_logger

log = get_logger("llm")


class ChatLLM:
    def __init__(self, model: str = CHAT_MODEL):
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to .env or Streamlit secrets.")
        self._client = Groq(api_key=GROQ_API_KEY)
        self.model_name = model

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = CONFIG.temperature,
        max_output_tokens: int = 600,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
        )

        choice = resp.choices[0] if resp.choices else None
        text = (choice.message.content if choice and choice.message else "") or ""
        finish = getattr(choice, "finish_reason", None) if choice else None

        usage = getattr(resp, "usage", None)
        usage_dict = (
            {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            if usage is not None
            else "?"
        )

        log.info(
            f"LLM returned {len(text)} chars, finish_reason={finish}, usage={usage_dict}"
        )
        return text
