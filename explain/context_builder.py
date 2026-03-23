"""Build evidence context for layout explanations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_evidence(
    chosen: Dict[str, Any],
    report: Dict[str, Any],
    *,
    variant_id: Optional[str] = None,
    design_score: Optional[float] = None,
    design_reasons: Optional[List[str]] = None,
    alternatives: Optional[List[Dict[str, Any]]] = None,
    design_filter_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build evidence dictionary for explanation generation.

    Args:
        chosen: The selected layout variant
        report: Compliance report for the layout
        variant_id: Identifier for the variant
        design_score: Design quality score
        design_reasons: Reasons for design quality assessment
        alternatives: Alternative layout options
        design_filter_stats: Statistics from design filtering

    Returns:
        Evidence dictionary containing layout context
    """
    metrics = chosen.get("metrics", {}) or {}

    return {
        "variant_id": variant_id or chosen.get("strategy_name", "unknown"),
        "source": chosen.get("source", "unknown"),
        "design_score": design_score,
        "design_reasons": design_reasons or [],
        "metrics": {
            "fully_connected": metrics.get("fully_connected"),
            "adjacency_satisfaction": metrics.get("adjacency_satisfaction"),
            "alignment_score": metrics.get("alignment_score"),
            "max_travel_distance": metrics.get("max_travel_distance"),
            "max_allowed_travel_distance": metrics.get("max_allowed_travel_distance"),
        },
        "report_status": report.get("status", "UNKNOWN"),
        "violations": report.get("violations", []),
        "alternatives_count": len(alternatives) if alternatives else 0,
        "alternatives": alternatives or [],
        "design_filter_stats": design_filter_stats or {},
    }
