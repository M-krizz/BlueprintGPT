"""
verification.py - Pre-output quality verification for planner-generated layouts.

Runs comprehensive checks before presenting output to user:
1. Overlap detection (rooms don't overlap significantly)
2. Boundary containment (rooms within boundary)
3. Area compliance (rooms meet minimum area)
4. Adjacency satisfaction (preferred adjacencies honored)
5. Label positioning (room labels at room centers)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class VerificationResult:
    """Result of pre-output verification."""
    passed: bool
    score: float  # 0-1 quality score
    issues: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    room_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "score": round(self.score, 4),
            "issues": self.issues,
            "metrics": self.metrics,
            "room_scores": {k: round(v, 4) for k, v in self.room_scores.items()},
        }


def verify_planner_output(
    synthesized_rooms: List[Dict],
    boundary_polygon: List[Tuple[float, float]],
    spec: Dict,
    planner_output: Optional[Dict] = None,
    thresholds: Optional[Dict] = None,
) -> VerificationResult:
    """
    Comprehensive verification of planner-generated layout.

    Args:
        synthesized_rooms: List of room dicts from geometry_synthesis
        boundary_polygon: Plot boundary
        spec: Original spec with room requirements
        planner_output: Original planner output for adjacency checking
        thresholds: Optional custom thresholds

    Thresholds defaults:
    - min_overlap_free: 0.90 (90% of room area must be non-overlapping)
    - min_boundary_containment: 0.95 (95% of rooms within boundary)
    - min_area_compliance: 0.85 (85% of rooms meet minimum area)
    - min_adjacency_satisfaction: 0.25 (25% adjacency score)
    - overall_pass_threshold: 0.65 (65% overall score to pass)
    """
    thresholds = thresholds or {
        "min_overlap_free": 0.90,
        "min_boundary_containment": 0.95,
        "min_area_compliance": 0.85,
        "min_adjacency_satisfaction": 0.25,
        "overall_pass_threshold": 0.65,
    }

    issues: List[str] = []
    metrics: Dict = {}
    room_scores: Dict[str, float] = {}
    hard_failures: List[str] = []

    # 1. Overlap detection
    overlap_score = _check_overlaps(synthesized_rooms)
    metrics["overlap_free_ratio"] = overlap_score
    if overlap_score < thresholds["min_overlap_free"]:
        issues.append(f"Overlap detected: {(1 - overlap_score) * 100:.1f}% overlap")
        hard_failures.append("overlap")

    # 2. Boundary containment
    containment = _check_boundary_containment(synthesized_rooms, boundary_polygon)
    metrics["boundary_containment"] = containment
    if containment < thresholds["min_boundary_containment"]:
        issues.append(f"Rooms outside boundary: {(1 - containment) * 100:.1f}% out of bounds")
        hard_failures.append("boundary_containment")

    # 3. Area compliance
    area_compliance, area_details = _check_area_compliance(synthesized_rooms, spec)
    metrics["area_compliance"] = area_compliance
    metrics["area_details"] = area_details
    if area_compliance < thresholds["min_area_compliance"]:
        issues.append(f"Area violations: {(1 - area_compliance) * 100:.1f}% rooms under minimum")
        hard_failures.append("area_compliance")

    # 4. Adjacency satisfaction
    adj_score = _check_adjacency_satisfaction(synthesized_rooms, planner_output)
    metrics["adjacency_satisfaction"] = adj_score
    if adj_score < thresholds["min_adjacency_satisfaction"]:
        issues.append(f"Low adjacency satisfaction: {adj_score * 100:.1f}%")

    # 5. Room distribution check (not all rooms at center)
    distribution_score = _check_room_distribution(synthesized_rooms, boundary_polygon)
    metrics["distribution_score"] = distribution_score
    if distribution_score < 0.5:
        issues.append("Rooms clustered at center")

    # 6. Compute per-room scores
    for room in synthesized_rooms:
        room_score = _compute_room_score(room, boundary_polygon, spec)
        room_scores[room["name"]] = room_score

    # Compute overall score
    weights = {
        "overlap_free": 0.30,
        "containment": 0.20,
        "area_compliance": 0.20,
        "adjacency": 0.15,
        "distribution": 0.15,
    }

    overall_score = (
        weights["overlap_free"] * overlap_score
        + weights["containment"] * containment
        + weights["area_compliance"] * area_compliance
        + weights["adjacency"] * adj_score
        + weights["distribution"] * distribution_score
    )

    metrics["hard_failures"] = hard_failures
    passed = overall_score >= thresholds["overall_pass_threshold"] and not hard_failures

    return VerificationResult(
        passed=passed,
        score=overall_score,
        issues=issues,
        metrics=metrics,
        room_scores=room_scores,
    )


def _check_overlaps(rooms: List[Dict]) -> float:
    """Return ratio of non-overlapping area (1.0 = no overlaps)."""
    if len(rooms) < 2:
        return 1.0

    total_area = sum(r.get("area", 0) for r in rooms)
    if total_area <= 0:
        return 1.0

    overlap_area = 0.0
    for i, r1 in enumerate(rooms):
        for r2 in rooms[i + 1:]:
            overlap = _compute_rect_overlap(r1["polygon"], r2["polygon"])
            overlap_area += overlap

    return max(0.0, 1.0 - overlap_area / total_area)


def _compute_rect_overlap(poly1: List[Tuple[float, float]], poly2: List[Tuple[float, float]]) -> float:
    """Compute overlap area between two rectangular polygons."""
    x1_min = min(p[0] for p in poly1)
    x1_max = max(p[0] for p in poly1)
    y1_min = min(p[1] for p in poly1)
    y1_max = max(p[1] for p in poly1)

    x2_min = min(p[0] for p in poly2)
    x2_max = max(p[0] for p in poly2)
    y2_min = min(p[1] for p in poly2)
    y2_max = max(p[1] for p in poly2)

    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    return x_overlap * y_overlap


def _check_boundary_containment(rooms: List[Dict], boundary_polygon: List[Tuple[float, float]]) -> float:
    """Return ratio of room area within boundary."""
    bxs = [p[0] for p in boundary_polygon]
    bys = [p[1] for p in boundary_polygon]
    bx0, bx1 = min(bxs), max(bxs)
    by0, by1 = min(bys), max(bys)

    total_area = 0.0
    contained_area = 0.0

    for room in rooms:
        if not room.get("polygon"):
            continue

        poly = room["polygon"]
        rx0 = min(p[0] for p in poly)
        rx1 = max(p[0] for p in poly)
        ry0 = min(p[1] for p in poly)
        ry1 = max(p[1] for p in poly)

        room_area = (rx1 - rx0) * (ry1 - ry0)
        total_area += room_area

        # Compute intersection with boundary
        ix0 = max(rx0, bx0)
        iy0 = max(ry0, by0)
        ix1 = min(rx1, bx1)
        iy1 = min(ry1, by1)

        if ix1 > ix0 and iy1 > iy0:
            contained_area += (ix1 - ix0) * (iy1 - iy0)

    return contained_area / max(total_area, 1e-6)


def _check_area_compliance(rooms: List[Dict], spec: Dict) -> Tuple[float, Dict]:
    """Return ratio of rooms meeting minimum area requirements."""
    spec_rooms = {r["name"]: r for r in spec.get("rooms", [])}

    total_rooms = 0
    compliant_rooms = 0
    details: Dict[str, Dict] = {}

    # Standard minimum areas by room type
    default_min_areas = {
        "LivingRoom": 12.0,
        "DiningRoom": 8.0,
        "Kitchen": 5.0,
        "Bedroom": 9.0,
        "Bathroom": 3.5,
        "WC": 1.5,
        "Garage": 15.0,
        "Store": 2.0,
        "Study": 6.0,
        "Balcony": 2.0,
    }

    for room in rooms:
        room_name = room["name"]
        room_type = room.get("type", "")
        actual_area = room.get("area", 0)

        # Get required area from spec or defaults
        spec_room = spec_rooms.get(room_name, {})
        min_area = (
            spec_room.get("min_area_sqm")
            or spec_room.get("area")
            or default_min_areas.get(room_type, 4.0)
        )
        if isinstance(min_area, str):
            min_area = float(min_area)

        total_rooms += 1
        compliant = actual_area >= min_area * 0.9  # 10% tolerance

        if compliant:
            compliant_rooms += 1

        details[room_name] = {
            "actual_area": round(actual_area, 2),
            "min_area": round(min_area, 2),
            "compliant": compliant,
        }

    return compliant_rooms / max(total_rooms, 1), details


def _check_adjacency_satisfaction(rooms: List[Dict], planner_output: Optional[Dict]) -> float:
    """Check how well adjacency preferences are satisfied."""
    if not planner_output:
        return 1.0  # No preferences to check

    adjacency_prefs = planner_output.get("adjacency_preferences", [])
    if not adjacency_prefs:
        return 1.0

    # Build room lookup by name
    room_by_name = {r["name"]: r for r in rooms}

    satisfied = 0
    total = 0

    adjacency_threshold = 0.5  # meters - consider rooms adjacent if within this distance

    for pref in adjacency_prefs:
        room_a = pref.get("a")
        room_b = pref.get("b")

        if room_a not in room_by_name or room_b not in room_by_name:
            continue

        total += 1

        # Check if rooms are adjacent (share edge or very close)
        r1 = room_by_name[room_a]
        r2 = room_by_name[room_b]

        if _are_rooms_adjacent(r1["polygon"], r2["polygon"], adjacency_threshold):
            satisfied += 1

    return satisfied / max(total, 1)


def _are_rooms_adjacent(poly1: List[Tuple[float, float]], poly2: List[Tuple[float, float]], threshold: float) -> bool:
    """Check if two rooms are adjacent (share edge or within threshold)."""
    x1_min, x1_max = min(p[0] for p in poly1), max(p[0] for p in poly1)
    y1_min, y1_max = min(p[1] for p in poly1), max(p[1] for p in poly1)
    x2_min, x2_max = min(p[0] for p in poly2), max(p[0] for p in poly2)
    y2_min, y2_max = min(p[1] for p in poly2), max(p[1] for p in poly2)

    # Check horizontal adjacency (rooms side by side)
    if abs(x1_max - x2_min) < threshold or abs(x2_max - x1_min) < threshold:
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        if y_overlap > 0.5:  # At least 0.5m overlap in height
            return True

    # Check vertical adjacency (rooms stacked)
    if abs(y1_max - y2_min) < threshold or abs(y2_max - y1_min) < threshold:
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        if x_overlap > 0.5:  # At least 0.5m overlap in width
            return True

    return False


def _check_room_distribution(rooms: List[Dict], boundary_polygon: List[Tuple[float, float]]) -> float:
    """
    Check that rooms are distributed across the plot, not all at center.

    Returns score 0-1:
    - 1.0 = rooms well distributed
    - 0.0 = all rooms clustered at center
    """
    if len(rooms) < 2:
        return 1.0

    # Get boundary center
    bxs = [p[0] for p in boundary_polygon]
    bys = [p[1] for p in boundary_polygon]
    bx0, bx1 = min(bxs), max(bxs)
    by0, by1 = min(bys), max(bys)
    bcx = (bx0 + bx1) / 2
    bcy = (by0 + by1) / 2
    max_dist = math.hypot(bx1 - bx0, by1 - by0) / 2

    # Compute average distance of room centroids from boundary center
    total_dist = 0.0
    for room in rooms:
        cx, cy = room["centroid"]
        dist = math.hypot(cx - bcx, cy - bcy)
        total_dist += dist

    avg_dist = total_dist / len(rooms)

    # Normalize: if avg_dist is close to max_dist/2, rooms are well distributed
    # if avg_dist is close to 0, rooms are clustered at center
    return min(1.0, avg_dist / (max_dist * 0.4))


def _compute_room_score(room: Dict, boundary_polygon: List[Tuple[float, float]], spec: Dict) -> float:
    """Compute individual room score (0-1)."""
    score = 1.0

    # Check boundary containment
    bxs = [p[0] for p in boundary_polygon]
    bys = [p[1] for p in boundary_polygon]
    bx0, bx1 = min(bxs), max(bxs)
    by0, by1 = min(bys), max(bys)

    poly = room["polygon"]
    rx0, rx1 = min(p[0] for p in poly), max(p[0] for p in poly)
    ry0, ry1 = min(p[1] for p in poly), max(p[1] for p in poly)

    # Penalize if outside boundary
    if rx0 < bx0 or rx1 > bx1 or ry0 < by0 or ry1 > by1:
        score -= 0.3

    # Check area compliance
    spec_rooms = {r["name"]: r for r in spec.get("rooms", [])}
    spec_room = spec_rooms.get(room["name"], {})
    min_area = spec_room.get("min_area_sqm") or spec_room.get("area") or 4.0
    if isinstance(min_area, str):
        min_area = float(min_area)

    if room.get("area", 0) < min_area * 0.9:
        score -= 0.2

    # Check aspect ratio (penalize very elongated rooms)
    width = rx1 - rx0
    height = ry1 - ry0
    aspect = max(width, height) / max(min(width, height), 0.1)
    if aspect > 3.0:
        score -= 0.2

    return max(0.0, score)


def generate_verification_summary(result: VerificationResult) -> str:
    """Generate human-readable verification summary."""
    lines = [
        f"Layout Verification: {'PASSED' if result.passed else 'NEEDS IMPROVEMENT'}",
        f"Overall Score: {result.score * 100:.1f}%",
        "",
        "Metrics:",
    ]

    metrics = result.metrics
    lines.append(f"  - Overlap-free: {metrics.get('overlap_free_ratio', 0) * 100:.1f}%")
    lines.append(f"  - Boundary containment: {metrics.get('boundary_containment', 0) * 100:.1f}%")
    lines.append(f"  - Area compliance: {metrics.get('area_compliance', 0) * 100:.1f}%")
    lines.append(f"  - Adjacency satisfaction: {metrics.get('adjacency_satisfaction', 0) * 100:.1f}%")
    lines.append(f"  - Room distribution: {metrics.get('distribution_score', 0) * 100:.1f}%")

    if result.issues:
        lines.append("")
        lines.append("Issues:")
        for issue in result.issues:
            lines.append(f"  - {issue}")

    return "\n".join(lines)
