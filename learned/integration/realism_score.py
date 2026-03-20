"""
realism_score.py — Pre-rank realism scoring for generated layouts.

Evaluates layouts against Chapter-4 ground truth before repair to identify
high-quality generations that need minimal fixing.

Scoring Components
------------------
1. **Min-dim violations**: Count of rooms below Chapter-4 minimums
2. **Aspect ratio**: Penalty for extreme aspect ratios (slivers)
3. **Corridor continuity**: Connected corridor topology score
4. **Zoning**: Proper room placement (wet areas grouped, etc.)
5. **Travel feasibility**: Estimated travel distance

Returns a realism score [0-1] where higher is better.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from core.building import Building
from constraints.rule_engine import RuleEngine
from constraints.chapter4_helpers import get_min_room_dims, plot_bucket


@dataclass
class RealismScore:
    """Realism scoring result."""
    overall: float  # [0-1], higher is better
    min_dim_violations: int
    min_dim_score: float
    aspect_ratio_score: float
    corridor_score: float
    zoning_score: float
    travel_score: float
    details: Dict[str, Any]


def _bbox(room) -> Tuple[float, float, float, float]:
    """Get bounding box of a room."""
    if not room.polygon or len(room.polygon) < 3:
        return 0, 0, 0, 0
    xs = [p[0] for p in room.polygon]
    ys = [p[1] for p in room.polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _aspect_ratio(room) -> float:
    """Compute aspect ratio (max_dim / min_dim)."""
    x1, y1, x2, y2 = _bbox(room)
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return 10.0  # degenerate
    return max(w, h) / min(w, h)


def score_min_dim_compliance(
    building: Building,
    plot_area_sqm: float,
    regulation_file: str = "ontology/regulation_data.json",
) -> Tuple[int, float, List[str]]:
    """Score room dimension compliance against Chapter-4 minimums.

    Returns
    -------
    violation_count : int
        Number of rooms below minimum dimensions.
    score : float
        [0-1], 1.0 if all rooms meet minimums.
    violations : list[str]
        Descriptions of violations.
    """
    engine = RuleEngine(regulation_file)
    engine.set_plot_area(plot_area_sqm)

    violations = []
    violation_count = 0

    for room in building.rooms:
        dims = get_min_room_dims(room.room_type, plot_area_sqm, engine.data)
        if dims["min_area"] == 0:
            continue  # Not regulated

        actual_area = room.final_area or 0
        x1, y1, x2, y2 = _bbox(room)
        actual_width = min(x2 - x1, y2 - y1)

        # Area check
        if actual_area < dims["min_area"] - 0.01:
            violation_count += 1
            violations.append(
                f"{room.name} ({room.room_type}): area {actual_area:.2f} < min {dims['min_area']}"
            )

        # Width check
        if actual_width > 0 and actual_width < dims["min_width"] - 0.01:
            violation_count += 1
            violations.append(
                f"{room.name} ({room.room_type}): width {actual_width:.2f} < min {dims['min_width']}"
            )

    # Score: exponential decay with violations
    # 0 violations → 1.0
    # 1 violation → 0.85
    # 2 violations → 0.72
    # 5 violations → 0.44
    score = 0.85 ** violation_count if violation_count > 0 else 1.0

    return violation_count, score, violations


def score_aspect_ratios(building: Building) -> Tuple[float, Dict[str, float]]:
    """Score aspect ratios - penalize slivers.

    Returns
    -------
    score : float
        [0-1], 1.0 if all rooms have reasonable aspect ratios.
    details : dict
        Per-room aspect ratios.
    """
    details = {}
    penalties = []

    for room in building.rooms:
        ar = _aspect_ratio(room)
        details[room.name] = round(ar, 2)

        # Penalty: aspect ratio > 3.0 is sliver-like
        # 1.0-2.0: no penalty
        # 2.0-3.0: small penalty
        # 3.0+: large penalty
        if ar > 3.0:
            penalty = min(1.0, (ar - 3.0) / 5.0)  # max penalty at ar=8.0
            penalties.append(penalty)
        elif ar > 2.0:
            penalty = (ar - 2.0) / 10.0
            penalties.append(penalty)

    # Average penalty
    avg_penalty = sum(penalties) / len(penalties) if penalties else 0.0
    score = max(0.0, 1.0 - avg_penalty)

    return score, details


def score_corridor_continuity(building: Building) -> float:
    """Score corridor continuity - stub for now.

    Returns 1.0 if corridors exist and connect rooms, 0.5 otherwise.
    """
    if not hasattr(building, "corridors") or not building.corridors:
        return 0.5

    corr = building.corridors[0]
    if hasattr(corr, "connects") and corr.connects:
        # Ratio of rooms connected by corridor
        connected_count = len(corr.connects)
        total_rooms = len(building.rooms)
        return min(1.0, connected_count / max(total_rooms, 1))

    return 0.7  # Has corridor but no connection info


def score_zoning(building: Building) -> Tuple[float, Dict[str, Any]]:
    """Score zoning based on functional room placement.

    Checks:
    - Wet areas (Bathroom, Kitchen) proximity (should be grouped for plumbing)
    - Bedroom privacy (should not be at entrance)
    - Functional adjacency (Kitchen near Living/Dining)
    - Service areas have exterior adjacency (for ventilation)

    Returns
    -------
    score : float
        [0-1], 1.0 if all zoning rules satisfied.
    details : dict
        Breakdown of zoning metrics.
    """
    if not building.rooms or len(building.rooms) < 2:
        return 0.75, {"reason": "insufficient_rooms"}

    # Categorize rooms
    wet_rooms = []
    bedrooms = []
    living_areas = []  # Living, Dining, Drawing
    service_rooms = []  # Kitchen, Bathroom, WC, BathWC

    for room in building.rooms:
        rt = room.room_type
        if rt in ("Bathroom", "WC", "BathWC"):
            wet_rooms.append(room)
            service_rooms.append(room)
        elif rt == "Kitchen":
            wet_rooms.append(room)
            service_rooms.append(room)
        elif rt == "Bedroom":
            bedrooms.append(room)
        elif rt in ("LivingRoom", "DiningRoom", "DrawingRoom"):
            living_areas.append(room)

    penalties = []
    details = {}

    # ── 1. Wet area clustering (plumbing efficiency) ──────────────────────
    if len(wet_rooms) >= 2:
        # Check if wet areas are close to each other
        # Compute pairwise distances between wet room centroids
        wet_distances = []
        for i, room1 in enumerate(wet_rooms):
            c1 = _centroid(room1)
            if c1 is None:
                continue
            for room2 in wet_rooms[i+1:]:
                c2 = _centroid(room2)
                if c2 is None:
                    continue
                dist = ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5
                wet_distances.append(dist)

        if wet_distances:
            avg_wet_dist = sum(wet_distances) / len(wet_distances)
            details["avg_wet_area_distance"] = round(avg_wet_dist, 3)

            # Penalty if wet areas too far apart (> 0.4 norm space ≈ >6m)
            if avg_wet_dist > 0.4:
                penalty = min(0.3, (avg_wet_dist - 0.4) / 0.6)  # max 0.3
                penalties.append(("wet_clustering", penalty))

    # ── 2. Kitchen-Living proximity (functional adjacency) ─────────────────
    if wet_rooms and living_areas:
        # Find Kitchen
        kitchens = [r for r in wet_rooms if r.room_type == "Kitchen"]
        if kitchens and living_areas:
            k_centroid = _centroid(kitchens[0])
            if k_centroid:
                min_dist = float("inf")
                for living in living_areas:
                    lc = _centroid(living)
                    if lc:
                        dist = ((k_centroid[0] - lc[0])**2 + (k_centroid[1] - lc[1])**2)**0.5
                        min_dist = min(min_dist, dist)

                details["kitchen_living_distance"] = round(min_dist, 3)

                # Penalty if Kitchen far from living areas (> 0.5 norm)
                if min_dist > 0.5:
                    penalty = min(0.2, (min_dist - 0.5) / 0.5)
                    penalties.append(("kitchen_living", penalty))

    # ── 3. Bedroom privacy (not at entrance) ───────────────────────────────
    # Heuristic: Bedrooms should not be in bottom-left quadrant (entrance area)
    # This is a rough proxy; real entrance detection would need boundary info
    if bedrooms:
        entrance_bedrooms = 0
        for bedroom in bedrooms:
            c = _centroid(bedroom)
            if c and c[0] < 0.3 and c[1] < 0.3:  # Bottom-left quadrant
                entrance_bedrooms += 1

        if entrance_bedrooms > 0:
            penalty = min(0.2, entrance_bedrooms * 0.1)
            penalties.append(("bedroom_privacy", penalty))
            details["bedrooms_near_entrance"] = entrance_bedrooms

    # ── Overall score ──────────────────────────────────────────────────────
    total_penalty = sum(p[1] for p in penalties)
    score = max(0.0, 1.0 - total_penalty)

    details["penalties"] = [{"rule": name, "penalty": round(p, 3)} for name, p in penalties]
    details["total_penalty"] = round(total_penalty, 3)

    return score, details


def _centroid(room) -> Optional[Tuple[float, float]]:
    """Compute centroid of a room's polygon."""
    if not room.polygon or len(room.polygon) < 3:
        return None
    xs = [p[0] for p in room.polygon]
    ys = [p[1] for p in room.polygon]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def score_travel_feasibility(building: Building, max_travel: float = 22.5) -> float:
    """Score travel distance feasibility - approximate.

    Returns
    -------
    score : float
        [0-1], 1.0 if estimated travel is well below max_travel.
    """
    # Quick heuristic: max distance between any two rooms
    centroids = []
    for room in building.rooms:
        if not room.polygon or len(room.polygon) < 3:
            continue
        x1, y1, x2, y2 = _bbox(room)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centroids.append((cx, cy))

    if len(centroids) < 2:
        return 1.0

    max_dist = 0.0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dx = centroids[i][0] - centroids[j][0]
            dy = centroids[i][1] - centroids[j][1]
            dist = (dx**2 + dy**2) ** 0.5
            max_dist = max(max_dist, dist)

    # max_dist is in building coordinates
    # Rough estimate: max allowable is ~0.7 of boundary diagonal
    # If max_dist > 0.7 → likely exceeds travel limit
    if max_dist > 0.7:
        excess = (max_dist - 0.7) / 0.3
        score = max(0.0, 1.0 - excess)
    else:
        score = 1.0

    return score


def compute_realism_score(
    building: Building,
    plot_area_sqm: float = 100.0,
    max_travel: float = 22.5,
    regulation_file: str = "ontology/regulation_data.json",
) -> RealismScore:
    """Compute overall realism score for a generated layout.

    Parameters
    ----------
    building : Building
        Generated building to score.
    plot_area_sqm : float
        Plot area for Chapter-4 bucket selection.
    max_travel : float
        Maximum allowed travel distance (m).
    regulation_file : str
        Path to regulation_data.json.

    Returns
    -------
    RealismScore
        Overall score and component scores.
    """
    # Component scores
    violation_count, min_dim_score, min_dim_violations = score_min_dim_compliance(
        building, plot_area_sqm, regulation_file
    )
    aspect_score, aspect_details = score_aspect_ratios(building)
    corridor_score = score_corridor_continuity(building)
    zoning_score, zoning_details = score_zoning(building)
    travel_score = score_travel_feasibility(building, max_travel)

    # Weighted overall score
    # Prioritize: min_dim (40%), aspect (25%), travel (20%), corridor (10%), zoning (5%)
    overall = (
        0.40 * min_dim_score
        + 0.25 * aspect_score
        + 0.20 * travel_score
        + 0.10 * corridor_score
        + 0.05 * zoning_score
    )

    return RealismScore(
        overall=round(overall, 4),
        min_dim_violations=violation_count,
        min_dim_score=round(min_dim_score, 4),
        aspect_ratio_score=round(aspect_score, 4),
        corridor_score=round(corridor_score, 4),
        zoning_score=round(zoning_score, 4),
        travel_score=round(travel_score, 4),
        details={
            "min_dim_violations_list": min_dim_violations,
            "aspect_ratios": aspect_details,
            "zoning": zoning_details,
        },
    )
