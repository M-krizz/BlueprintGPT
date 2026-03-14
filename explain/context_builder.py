"""Build evidence JSON for the explanation layer.

The evidence is a strict, self-contained JSON document derived from the
selected variant and the already computed compliance report. This module does
not call any external services; it only reshapes existing data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_evidence(
    selected_variant: Dict,
    report: Dict,
    *,
    variant_id: Optional[str] = None,
    design_score: Optional[float] = None,
    design_reasons: Optional[List[str]] = None,
    alternatives: Optional[List[Dict]] = None,
    design_filter_stats: Optional[Dict] = None,
) -> Dict:
    """Construct a bounded evidence JSON used by the explainer.

    Only data present in the variant/report is surfaced. Missing fields are
    omitted to keep the evidence minimal and non-speculative.
    """

    metrics = selected_variant.get("metrics", {}) or {}
    circulation = report.get("circulation_space", {}) or {}
    wall_pipeline = selected_variant.get("wall_pipeline", {}) or report.get("wall_pipeline", {}) or {}
    ranking = selected_variant.get("ranking", {}) or {}
    breakdown = ranking.get("breakdown", {}) or {}
    input_spec = selected_variant.get("input_spec", {}) or {}

    evidence: Dict = {
        "selected_variant_id": variant_id or selected_variant.get("strategy_name", "variant"),
        "status": report.get("status", "UNKNOWN"),
        "source": selected_variant.get("source", report.get("source", "algorithmic")),
        "scores": {
            "score": ranking.get("score"),
            "breakdown": breakdown,
        },
        "design": {
            "design_score": design_score,
            "design_reasons": design_reasons or [],
            "alternatives": alternatives or [],
            "filter_stats": design_filter_stats or {},
        },
        "hard_compliance": {
            "status": report.get("status", "UNKNOWN"),
            "violations_hard": report.get("violations", []),
            "violations_soft": _soft_warnings(report),
        },
        "metrics": {
            "max_travel_distance": metrics.get("max_travel_distance"),
            "max_allowed_travel_distance": metrics.get("max_allowed_travel_distance"),
            "travel_distance_compliant": metrics.get("travel_distance_compliant"),
            "corridor_width": metrics.get("corridor_width") or circulation.get("corridor_width"),
            "circulation_walkable_area": metrics.get("circulation_walkable_area") or circulation.get("walkable_area"),
            "compactness": _compactness_from_rank(selected_variant),
            "adjacency_satisfaction": metrics.get("adjacency_satisfaction"),
            "alignment_score": metrics.get("alignment_score"),
            "door_path_travel_distance": metrics.get("door_path_travel_distance"),
            "circulation_ratio": _safe_ratio(
                metrics.get("circulation_walkable_area"), metrics.get("total_area"), default=None
            ),
            "corridor_served_ratio": metrics.get("corridor_served_ratio"),
            "total_area": metrics.get("total_area"),
            "occupant_load": metrics.get("occupant_load"),
        },
        "room_summary": _room_summary(selected_variant),
        "door_summary": {
            "door_count": len(selected_variant.get("building", {}).doors)
            if hasattr(selected_variant.get("building", {}), "doors")
            else len(selected_variant.get("building", {}).get("doors", [])),
            "fully_connected": metrics.get("fully_connected"),
        },
        "corridor_summary": _corridor_summary(selected_variant, circulation),
        "constraints": _constraints(report, metrics),
        "kg_intents_used": _kg_intents(selected_variant),
        "reasoner": _reasoner_info(report),
        "wall_pipeline": wall_pipeline,
        "user_preferences": input_spec.get("preference_weights", {}),
    }

    if selected_variant.get("source") == "learned":
        evidence["repair_trace"] = selected_variant.get("repair_trace", [])
        evidence["generation_summary"] = selected_variant.get("generation_summary", {}) or report.get(
            "generation_summary", {}
        )
    if report.get("truth_table"):
        evidence["truth_table"] = report.get("truth_table")

    # Derived margins for validator-friendly evidence (prevent false positives on legit numbers).
    max_travel = evidence["metrics"].get("max_travel_distance")
    max_allowed = evidence["metrics"].get("max_allowed_travel_distance")
    if max_travel is not None and max_allowed is not None:
        evidence["metrics"]["travel_margin"] = max_allowed - max_travel

    return evidence


def _soft_warnings(report: Dict) -> List[str]:
    warnings: List[str] = []
    grounding = report.get("grounding", {})
    rule_preflight = grounding.get("rule_preflight", {})
    kg_precheck = grounding.get("kg_precheck", {})
    warnings.extend(rule_preflight.get("warnings", []))
    warnings.extend(kg_precheck.get("warnings", []))
    return warnings


def _room_summary(selected_variant: Dict) -> List[Dict]:
    rooms_out: List[Dict] = []
    building = selected_variant.get("building")
    if building and getattr(building, "rooms", None):
        for room in building.rooms:
            entry = {
                "name": getattr(room, "name", None),
                "type": getattr(room, "room_type", None),
                "area": getattr(room, "final_area", None),
            }
            if getattr(room, "polygon", None):
                dims = _room_dims(room.polygon)
                entry.update(dims)
            rooms_out.append(entry)
    return rooms_out


def _room_dims(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return {
        "width": round(max(xs) - min(xs), 4),
        "height": round(max(ys) - min(ys), 4),
    }


def _corridor_summary(selected_variant: Dict, circulation: Dict) -> Dict:
    building = selected_variant.get("building")
    corridors = getattr(building, "corridors", []) if building else []
    return {
        "count": len(corridors),
        "min_width": min((getattr(c, "width", 0) for c in corridors), default=None),
        "walkable_area": circulation.get("walkable_area"),
        "connectivity_to_exit": circulation.get("connectivity_to_exit"),
    }


def _constraints(report: Dict, metrics: Dict) -> List[Dict[str, Any]]:
    checks = report.get("checks", {}) or {}
    constraints: List[Dict[str, Any]] = []
    # Hard checks from report
    for key, passed in checks.items():
        constraints.append({"name": key, "passed": bool(passed)})

    # Derived numeric expectations
    if metrics.get("max_travel_distance") is not None:
        constraints.append(
            {
                "name": "travel_distance",
                "passed": bool(metrics.get("travel_distance_compliant")),
                "required": metrics.get("max_allowed_travel_distance"),
                "actual": metrics.get("max_travel_distance"),
            }
        )
    if metrics.get("corridor_width") is not None:
        constraints.append({"name": "corridor_width", "actual": metrics.get("corridor_width")})
    if metrics.get("required_exit_width") is not None:
        constraints.append({"name": "exit_width", "actual": metrics.get("required_exit_width")})

    return constraints


def _kg_intents(selected_variant: Dict) -> List[Dict]:
    intents = selected_variant.get("kg_intents") or []
    out = []
    for edge in intents:
        if len(edge) == 3:
            a, b, w = edge
            out.append({"a": a, "b": b, "weight": w})
    return out


def _reasoner_info(report: Dict) -> Dict:
    ontology = report.get("ontology", {})
    return {
        "mode": ontology.get("reasoner", "off"),
        "success": ontology.get("reasoner_success"),
        "valid": ontology.get("reasoner_success") if ontology else None,
    }


def _compactness_from_rank(selected_variant: Dict) -> Optional[float]:
    ranking = selected_variant.get("ranking", {})
    breakdown = ranking.get("breakdown", {})
    return breakdown.get("compactness")


def _safe_ratio(numerator: Optional[float], denominator: Optional[float], *, default: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return default
    if denominator == 0:
        return default
    return numerator / denominator
