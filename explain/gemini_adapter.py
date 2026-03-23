"""Gemini LLM adapter for generating explanations."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional

DEPRECATED_MODEL_ALIASES = {
    "gemini-1.5-flash": "gemini-2.5-flash",
    "gemini-1.5-pro": "gemini-2.5-pro",
    "gemini-pro": "gemini-2.5-pro",
}


def build_gemini_llm_fn(
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-pro",
) -> Callable[[Dict[str, Any]], str]:
    """Build a Gemini LLM function for explanation generation.

    Args:
        api_key: Gemini API key (falls back to GEMINI_API_KEY env var)
        model: Gemini model name

    Returns:
        Function that takes evidence dict and returns explanation string
    """
    resolved_key = api_key or os.getenv("GEMINI_API_KEY")

    if not resolved_key:
        raise ValueError("Gemini API key not provided and GEMINI_API_KEY not set")

    resolved_model = DEPRECATED_MODEL_ALIASES.get(model.lower(), model)

    try:
        from google import genai as google_genai
        client = google_genai.Client(api_key=resolved_key)

        def _generate(prompt: str) -> str:
            response = client.models.generate_content(
                model=resolved_model,
                contents=prompt,
            )
            return getattr(response, "text", "") or ""

    except ImportError:
        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=resolved_key)
            gemini_model = legacy_genai.GenerativeModel(resolved_model)

            def _generate(prompt: str) -> str:
                response = gemini_model.generate_content(prompt)
                return getattr(response, "text", "") or ""

        except ImportError:
            raise ImportError("Neither google-genai nor google-generativeai is installed")

    def llm_fn(evidence: Dict[str, Any]) -> str:
        prompt = _build_prompt(evidence)
        try:
            return _generate(prompt)
        except Exception as e:
            return f"[LLM Error: {e}]"

    return llm_fn


def _build_prompt(evidence: Dict[str, Any]) -> str:
    """Build a prompt for the LLM from evidence."""
    metrics = evidence.get("metrics", {})
    status = evidence.get("report_status", "UNKNOWN")
    design_score = evidence.get("design_score", 0)

    prompt = f"""You are an architectural assistant explaining a generated floor plan layout.

Layout Status: {status}
Design Score: {design_score:.2f if design_score else 'N/A'}

Metrics:
- Fully Connected: {metrics.get('fully_connected', 'N/A')}
- Adjacency Satisfaction: {metrics.get('adjacency_satisfaction', 'N/A')}
- Alignment Score: {metrics.get('alignment_score', 'N/A')}

Please provide a brief, helpful explanation of this layout in 2-3 sentences.
Focus on the key strengths and any areas for improvement."""

    return prompt
