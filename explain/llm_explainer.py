"""Generate layout explanations."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Set


def explain(
    evidence: Dict[str, Any],
    *,
    ontology_room_types: Optional[Set[str]] = None,
    status: str = "UNKNOWN",
    llm_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Generate an explanation for the layout.

    Args:
        evidence: Evidence dictionary from build_evidence
        ontology_room_types: Set of valid room types from ontology
        status: Compliance status (COMPLIANT, NON_COMPLIANT, etc.)
        llm_fn: Optional LLM function for generating natural language explanations

    Returns:
        Explanation dictionary
    """
    metrics = evidence.get("metrics", {})
    design_score = evidence.get("design_score", 0)
    design_reasons = evidence.get("design_reasons", [])
    violations = evidence.get("violations", [])

    # Build deterministic explanation
    summary_parts = []

    if status == "COMPLIANT":
        summary_parts.append("The generated layout meets all compliance requirements.")
    else:
        summary_parts.append(f"Layout status: {status}.")

    if design_score is not None:
        summary_parts.append(f"Design quality score: {design_score:.2f}.")

    if metrics.get("fully_connected"):
        summary_parts.append("All rooms are connected.")

    adj_sat = metrics.get("adjacency_satisfaction")
    if adj_sat is not None:
        summary_parts.append(f"Adjacency satisfaction: {adj_sat:.1%}.")

    align_score = metrics.get("alignment_score")
    if align_score is not None:
        summary_parts.append(f"Alignment score: {align_score:.1%}.")

    explanation = {
        "summary": " ".join(summary_parts),
        "status": status,
        "design_score": design_score,
        "design_reasons": design_reasons,
        "violations_count": len(violations),
        "metrics_snapshot": metrics,
    }

    # If LLM function provided, try to generate enhanced explanation
    if llm_fn is not None:
        try:
            llm_response = llm_fn(evidence)
            if isinstance(llm_response, str):
                explanation["llm_summary"] = llm_response
            elif isinstance(llm_response, dict):
                explanation["llm_summary"] = llm_response.get("summary", str(llm_response))
        except Exception as e:
            explanation["_llm_error"] = str(e)

    return explanation
