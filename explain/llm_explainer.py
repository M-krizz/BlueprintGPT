"""LLM explanation layer with strict schema and guardrails.

This module is intentionally simple: it accepts evidence JSON and returns a
structured explanation. By default it uses a deterministic template (no model)
to avoid external dependencies; a pluggable `llm_fn` can be supplied for richer
language if available.
"""

from __future__ import annotations

import json
import time
import re
from typing import Any, Callable, Dict, List, Optional, Set

from .validator import validate_explanation


def explain(
    evidence: Dict,
    *,
    ontology_room_types: Set[str],
    status: str,
    llm_fn: Optional[Callable[[str], str]] = None,
    max_retries: int = 2,
) -> Dict:
    """Produce a guarded explanation JSON from evidence.

    - If llm_fn is None, emits a deterministic template that never hallucinates.
    - Otherwise, llm_fn receives a strict prompt and must return JSON text.
    - The validator enforces evidence-only claims; failures cause a repair prompt.
    """

    base = _deterministic_response(evidence)

    prompt = _build_prompt(evidence, base)

    if llm_fn is None:
        return base

    last_error: List[str] = []
    for _ in range(max_retries + 1):
        raw = llm_fn(prompt if not last_error else _repair_prompt(prompt, last_error))
        try:
            parsed = json.loads(raw)
        except Exception:
            last_error = ["Invalid JSON from model"]
            continue

        ok, errors = validate_explanation(parsed, evidence, ontology_room_types=ontology_room_types, status=status)
        if ok:
            return parsed
        last_error = errors

    # Fallback to deterministic if the model cannot satisfy guardrails.
    return base if not last_error else _deterministic_response(evidence, errors=last_error)


def _build_prompt(evidence: Dict, deterministic: Dict) -> str:
    schema = {
        "summary": "string",
        "why_this_layout": ["string"],
        "constraint_justification": [{"claim": "string", "evidence_path": "string", "value": "number|string|boolean"}],
        "tradeoffs": ["string"],
        "suggested_edits": ["string"],
        "open_questions": ["string"],
    }
    return (
        "You are an explanation engine. Use ONLY the evidence JSON provided. "
        "If a fact or number is missing, respond with 'Not available.' "
        "Return JSON matching the schema exactly. Do not change numeric values from evidence. "
        "Start from the deterministic draft and only rephrase for clarity without altering facts.\n"
        f"Evidence:\n{json.dumps(evidence, indent=2)}\n"
        f"Deterministic draft:\n{json.dumps(deterministic, indent=2)}\n"
        f"Schema:\n{json.dumps(schema, indent=2)}"
    )


def _repair_prompt(original_prompt: str, errors: List[str]) -> str:
    return original_prompt + "\nThe previous response failed validation: " + "; ".join(errors) + ". Fix the JSON; remove unsupported claims."


def _deterministic_response(evidence: Dict, errors: Optional[List[str]] = None) -> Dict:
    status = evidence.get("hard_compliance", {}).get("status", "UNKNOWN")
    metrics = evidence.get("metrics", {})
    ranking = evidence.get("scores", {})
    score = ranking.get("score")
    breakdown = ranking.get("breakdown", {}) or {}

    travel = metrics.get("max_travel_distance")
    allowed = metrics.get("max_allowed_travel_distance")
    adj = metrics.get("adjacency_satisfaction")
    corridor_ratio = metrics.get("circulation_ratio")

    margin = None
    if travel is not None and allowed is not None:
        margin = round(allowed - travel, 3)

    why: List[str] = []
    if score is not None:
        why.append(f"Ranked top with score {score} (hard={breakdown.get('hard_compliance')}, compact={breakdown.get('compactness')}, adjacency={breakdown.get('adjacency')}, travel={breakdown.get('travel_margin')}, circulation={breakdown.get('circulation_ratio')}, alignment={breakdown.get('alignment')}).")
    if margin is not None:
        why.append(f"Travel distance {travel}m within limit {allowed}m (margin {margin}m).")
    if adj is not None:
        why.append(f"Adjacency satisfaction {adj} (higher is better).")
    if corridor_ratio is not None:
        why.append(f"Circulation ratio {round(corridor_ratio,4)} relative to total area.")

    constraint_justification = []
    if travel is not None:
        constraint_justification.append(
            {
                "claim": "Travel distance within limit",
                "evidence_path": "metrics.max_travel_distance",
                "value": travel,
            }
        )
    if allowed is not None:
        constraint_justification.append(
            {
                "claim": "Allowed travel distance",
                "evidence_path": "metrics.max_allowed_travel_distance",
                "value": allowed,
            }
        )
    if metrics.get("corridor_served_ratio") is not None:
        constraint_justification.append(
            {
                "claim": "Corridor serves rooms",
                "evidence_path": "metrics.corridor_served_ratio",
                "value": metrics.get("corridor_served_ratio"),
            }
        )
    if metrics.get("fully_connected") is not None:
        constraint_justification.append(
            {
                "claim": "Graph connectivity",
                "evidence_path": "metrics.fully_connected",
                "value": metrics.get("fully_connected"),
            }
        )

    tradeoffs: List[str] = []
    if adj is not None:
        tradeoffs.append(f"Adjacency satisfaction is {adj}; consider improving priority pairs if low.")
    if corridor_ratio is not None:
        tradeoffs.append(f"Circulation ratio {round(corridor_ratio,4)} balances compactness vs corridor access.")

    response = {
        "summary": f"Layout status: {status}.",
        "why_this_layout": why or ["Selected as top-ranked variant based on compliance evidence."],
        "constraint_justification": constraint_justification,
        "tradeoffs": tradeoffs,
        "suggested_edits": [],
        "open_questions": errors or [],
    }
    if errors:
        response["_llm_validation_errors"] = errors
    return response
