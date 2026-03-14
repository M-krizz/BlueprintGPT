"""Gemini LLM adapter for guarded explanations."""

from __future__ import annotations

from typing import Callable, Optional


def build_gemini_llm_fn(api_key: str, model: str = "gemini-pro") -> Callable[[str], str]:
    try:
        import google.generativeai as genai
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError("google-generativeai is not installed") from exc

    if not api_key:
        raise ValueError("Gemini API key is required")

    genai.configure(api_key=api_key)

    def _llm_fn(prompt: str) -> str:
        resp = genai.GenerativeModel(model).generate_content(prompt)
        text = resp.text if hasattr(resp, "text") else str(resp)
        return text

    return _llm_fn
