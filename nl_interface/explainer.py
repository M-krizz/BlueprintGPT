"""
explainer.py – Design explanation and ranking justification module.

Generates user-friendly explanations for:
1. Why each design is ranked at its position
2. What works well in each design
3. What could be improved
4. How to request corrections
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from nl_interface.gemini_adapter import explain_design, is_available as gemini_available


def explain_ranked_designs(
    designs: List[Dict[str, Any]],
    spec: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate explanations for a list of ranked designs.

    Parameters
    ----------
    designs : list
        List of design results from the generation pipeline, already sorted by score.
    spec : dict, optional
        The original user specification for context.

    Returns
    -------
    list of dict
        Each dict contains the original design data plus 'explanation' key.
    """
    explained = []
    total = len(designs)

    for rank_0indexed, design in enumerate(designs):
        rank = rank_0indexed + 1
        metrics = _extract_metrics(design)

        explanation = explain_design(
            design_data=design,
            rank=rank,
            total_designs=total,
            metrics=metrics,
        )

        # Add explanation to design
        design_with_explanation = dict(design)
        design_with_explanation["explanation"] = explanation
        design_with_explanation["rank"] = rank
        design_with_explanation["ranking_summary"] = _generate_ranking_summary(design, rank, total, metrics)

        explained.append(design_with_explanation)

    return explained


def _extract_metrics(design: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from design data."""
    metrics = design.get("metrics", {})

    return {
        "design_score": design.get("design_score", metrics.get("design_score", 0)),
        "compliance_status": design.get("report_status", "Unknown"),
        "travel_distance": metrics.get("max_travel_distance", "N/A"),
        "max_travel_distance": metrics.get("max_allowed_travel_distance", "N/A"),
        "room_coverage": design.get("room_coverage", {}),
        "violations": design.get("violations", []),
        "adjacency_satisfaction": metrics.get("adjacency_satisfaction", 0),
        "fully_connected": metrics.get("fully_connected", False),
    }


def _generate_ranking_summary(
    design: Dict[str, Any],
    rank: int,
    total: int,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a structured ranking summary."""
    score = metrics.get("design_score", 0)
    violations = metrics.get("violations", [])
    compliance = metrics.get("compliance_status", "Unknown")
    adj_score = metrics.get("adjacency_satisfaction", 0)

    # Determine primary ranking factors
    strengths = []
    weaknesses = []

    if compliance == "COMPLIANT":
        strengths.append("Meets all regulatory requirements")
    else:
        weaknesses.append("Has compliance violations")

    if adj_score > 0.7:
        strengths.append("Good room adjacency arrangement")
    elif adj_score < 0.3:
        weaknesses.append("Room adjacency could be improved")

    if metrics.get("fully_connected"):
        strengths.append("All rooms are accessible")
    else:
        weaknesses.append("Some rooms may not be accessible")

    if not violations:
        strengths.append("No rule violations")
    elif len(violations) <= 2:
        weaknesses.append(f"{len(violations)} minor violation(s)")
    else:
        weaknesses.append(f"{len(violations)} violations need attention")

    # Generate ranking reason
    if rank == 1:
        reason = "This design scored highest overall, with the best balance of compliance and layout quality."
    elif rank == total:
        reason = "This design has the most room for improvement but may offer alternative arrangements worth considering."
    else:
        reason = f"This design ranked #{rank} due to trade-offs between compliance, adjacency, and layout efficiency."

    return {
        "rank": rank,
        "total": total,
        "score": round(score, 3),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "reason": reason,
        "can_improve": len(weaknesses) > 0,
        "improvement_hints": _generate_improvement_hints(design, metrics),
    }


def _generate_improvement_hints(design: Dict[str, Any], metrics: Dict[str, Any]) -> List[str]:
    """Generate specific hints for how to improve this design."""
    hints = []
    violations = metrics.get("violations", [])
    coverage = design.get("room_coverage", {})

    # Check for missing rooms
    missing = coverage.get("missing", [])
    if missing:
        hints.append(f"Request adding: {', '.join(missing)}")

    # Check for travel distance issues
    travel = metrics.get("travel_distance", 0)
    max_travel = metrics.get("max_travel_distance", 0)
    if travel and max_travel and travel > max_travel:
        hints.append("Ask to 'reduce travel distance' or 'make corridor shorter'")

    # Check for connectivity issues
    if not metrics.get("fully_connected"):
        hints.append("Ask to 'ensure all rooms are connected' or 'add doors'")

    # Generic improvement suggestions
    if not hints:
        hints.append("Try: 'make kitchen larger', 'move bedroom away from entrance', 'swap living room and dining room'")

    return hints


def generate_comparison_explanation(
    designs: List[Dict[str, Any]],
    user_query: Optional[str] = None,
) -> str:
    """
    Generate a natural language comparison of multiple designs.

    Useful for explaining why one design is preferred over others.
    """
    if not designs:
        return "No designs to compare."

    if len(designs) == 1:
        best = designs[0]
        summary = best.get("ranking_summary", {})
        parts = [
            "## Ranking Note",
            "Only one design cleared the current generation and quality filters.",
        ]
        if summary.get("strengths"):
            parts.append("Strengths: " + ", ".join(summary["strengths"]) + ".")
        if summary.get("weaknesses"):
            parts.append("Remaining weak spots: " + ", ".join(summary["weaknesses"]) + ".")
        parts.append("You can keep refining this layout with targeted edits.")
        return "\n\n".join(parts)

    # Build comparison
    best = designs[0]
    others = designs[1:]

    comparison = f"## Design Comparison\n\n"
    comparison += f"I generated **{len(designs)} design options** for you, ranked by quality.\n\n"

    comparison += f"### Best Design (#{best.get('rank', 1)})\n"
    comparison += f"**Score:** {best.get('design_score', 0):.2f}\n"

    summary = best.get("ranking_summary", {})
    if summary.get("strengths"):
        comparison += f"**Strengths:** {', '.join(summary['strengths'])}\n"

    comparison += "\n### Alternative Designs\n"
    for design in others[:3]:  # Show top 3 alternatives
        rank = design.get("rank", "?")
        score = design.get("design_score", 0)
        summary = design.get("ranking_summary", {})
        weaknesses = summary.get("weaknesses", [])

        comparison += f"- **Design #{rank}** (score: {score:.2f})"
        if weaknesses:
            comparison += f" — {weaknesses[0]}"
        comparison += "\n"

    comparison += "\n### Want to modify a design?\n"
    comparison += "Just tell me what you'd like to change, e.g.:\n"
    comparison += '- "Make the kitchen in design #1 larger"\n'
    comparison += '- "Move the bedroom away from the entrance"\n'
    comparison += '- "Swap the positions of living room and dining room"\n'

    return comparison


def explain_correction_result(
    original_design: Dict[str, Any],
    corrected_design: Dict[str, Any],
    changes_applied: List[Dict[str, Any]],
) -> str:
    """
    Explain what changed after applying corrections.
    """
    explanation = "## Correction Applied\n\n"

    # List changes made
    explanation += "**Changes made:**\n"
    for change in changes_applied:
        change_type = change.get("type", "unknown")
        if change_type == "move_room":
            explanation += f"- Moved {change.get('room')} {change.get('direction')}\n"
        elif change_type == "resize_room":
            explanation += f"- Made {change.get('room')} {change.get('size_change')}\n"
        elif change_type == "add_room":
            explanation += f"- Added {change.get('room_type')}\n"
        elif change_type == "remove_room":
            explanation += f"- Removed {change.get('room')}\n"
        elif change_type == "swap_rooms":
            explanation += f"- Swapped {change.get('room_a')} with {change.get('room_b')}\n"
        else:
            explanation += f"- {change_type}: {change}\n"

    # Compare scores
    orig_score = original_design.get("design_score", 0)
    new_score = corrected_design.get("design_score", 0)

    if new_score > orig_score:
        explanation += f"\n**Result:** Score improved from {orig_score:.2f} to {new_score:.2f} (improved)\n"
    elif new_score < orig_score:
        explanation += f"\n**Note:** Score changed from {orig_score:.2f} to {new_score:.2f}. The change affected overall optimization.\n"
    else:
        explanation += f"\n**Result:** Score unchanged at {new_score:.2f}\n"

    explanation += "\nWould you like to make additional changes?"

    return explanation
