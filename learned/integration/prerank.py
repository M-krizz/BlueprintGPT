"""
prerank.py - Lightweight adjacency-aware pre-ranking for raw model samples.

Used before repair to reduce repair load and keep only stronger candidates.
Includes Chapter-4 compliance proxy scoring for improved realism.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from learned.data.tokenizer_layout import RoomBox
from geometry.adjacency_intent import build_adjacency_intent
from constraints.chapter4_helpers import (
    get_min_room_dims,
    is_habitable,
    is_service,
)


def _center(box: RoomBox) -> Tuple[float, float]:
    return ((box.x_min + box.x_max) / 2.0, (box.y_min + box.y_max) / 2.0)


def _box_area(box: RoomBox) -> float:
    return max(0.0, (box.x_max - box.x_min) * (box.y_max - box.y_min))


def _box_width(box: RoomBox) -> float:
    return max(0.0, box.x_max - box.x_min)


def _box_height(box: RoomBox) -> float:
    return max(0.0, box.y_max - box.y_min)


def _aspect_ratio(box: RoomBox) -> float:
    """Return aspect ratio (always >= 1.0)."""
    w = _box_width(box)
    h = _box_height(box)
    if w < 1e-9 or h < 1e-9:
        return float("inf")
    return max(w / h, h / w)


def _axis_gap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    """Gap between 1D intervals (0 if overlapping)."""
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0.0


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _are_adjacent_proxy(
    a: RoomBox,
    b: RoomBox,
    *,
    gap_tolerance: float = 0.05,
    center_distance_threshold: float = 0.25,
) -> bool:
    """Fast adjacency proxy for normalized [0,1] boxes.

    Adjacent if:
    - near in x with y-overlap, or near in y with x-overlap, or
    - center distance below threshold.
    """
    x_gap = _axis_gap_1d(a.x_min, a.x_max, b.x_min, b.x_max)
    y_gap = _axis_gap_1d(a.y_min, a.y_max, b.y_min, b.y_max)
    x_ov = _overlap_1d(a.x_min, a.x_max, b.x_min, b.x_max)
    y_ov = _overlap_1d(a.y_min, a.y_max, b.y_min, b.y_max)

    if (x_gap <= gap_tolerance and y_ov > 0.0) or (y_gap <= gap_tolerance and x_ov > 0.0):
        return True

    ax, ay = _center(a)
    bx, by = _center(b)
    d2 = (ax - bx) ** 2 + (ay - by) ** 2
    return d2 <= center_distance_threshold ** 2


def score_adjacency_proxy(
    room_boxes: Sequence[RoomBox],
    intent_graph: Sequence[Tuple[str, str, float]],
    *,
    gap_tolerance: float = 0.05,
    center_distance_threshold: float = 0.25,
) -> float:
    """Return weighted adjacency satisfaction in [0,1]."""
    typed: Dict[str, List[RoomBox]] = {}
    for rb in room_boxes:
        typed.setdefault(rb.room_type, []).append(rb)

    total_weight = sum(w for _, _, w in intent_graph) or 1.0
    satisfied = 0.0

    for ta, tb, w in intent_graph:
        ok = False
        for a in typed.get(ta, []):
            for b in typed.get(tb, []):
                if a is b:
                    continue
                if _are_adjacent_proxy(
                    a,
                    b,
                    gap_tolerance=gap_tolerance,
                    center_distance_threshold=center_distance_threshold,
                ):
                    ok = True
                    break
            if ok:
                break
        if ok:
            satisfied += w

    return round(satisfied / total_weight, 4)


def prerank_samples(
    samples: Sequence[Dict],
    spec: Dict,
    *,
    top_m: int = 3,
    use_kg: bool = True,
    gap_tolerance: float = 0.05,
    center_distance_threshold: float = 0.25,
) -> List[Dict]:
    """Rank raw sample dicts by adjacency proxy and return top_m items.

    Each sample dict must include key ``raw_rooms`` (list[RoomBox]).
    """
    room_types = [r.get("type") for r in spec.get("rooms", []) if r.get("type")]
    intent = build_adjacency_intent(room_types=room_types or None, use_kg=use_kg)

    scored: List[Dict] = []
    for sample in samples:
        rooms = sample.get("raw_rooms", [])
        adj = score_adjacency_proxy(
            rooms,
            intent,
            gap_tolerance=gap_tolerance,
            center_distance_threshold=center_distance_threshold,
        )
        entry = dict(sample)
        entry["adjacency_proxy"] = adj
        scored.append(entry)

    scored.sort(key=lambda s: s.get("adjacency_proxy", 0.0), reverse=True)
    return scored[: max(1, top_m)]


# ─── Chapter-4 Compliance Proxy Scoring ────────────────────────────────────

# Aspect ratio thresholds: penalize slivers
MAX_ACCEPTABLE_ASPECT_RATIO = 4.0  # rooms with aspect > 4:1 are slivers
SLIVER_PENALTY_WEIGHT = 0.15

# Minimum dimension thresholds (in normalized [0,1] space, scaled by boundary)
MIN_DIM_PENALTY_WEIGHT = 0.20


def score_aspect_ratio_quality(room_boxes: Sequence[RoomBox]) -> float:
    """Score in [0,1] based on aspect ratios. 1.0 = no slivers, 0.0 = all slivers."""
    if not room_boxes:
        return 1.0

    good_count = 0
    for box in room_boxes:
        ar = _aspect_ratio(box)
        if ar <= MAX_ACCEPTABLE_ASPECT_RATIO:
            good_count += 1

    return round(good_count / len(room_boxes), 4)


def score_min_dims_compliance(
    room_boxes: Sequence[RoomBox],
    boundary_width: float,
    boundary_height: float,
    plot_area_sqm: float = 100.0,
    reg: Optional[Dict] = None,
) -> float:
    """Score in [0,1] based on Chapter-4 minimum dimension compliance.

    Converts normalized boxes to real dimensions and checks against
    Chapter-4 room minimums.
    """
    if not room_boxes or boundary_width <= 0 or boundary_height <= 0:
        return 0.0

    compliant_count = 0

    for box in room_boxes:
        # Convert normalized dims to real meters
        real_w = _box_width(box) * boundary_width
        real_h = _box_height(box) * boundary_height
        real_area = real_w * real_h

        # Get Chapter-4 minimums
        dims = get_min_room_dims(box.room_type, plot_area_sqm, reg)
        min_area = dims.get("min_area", 0)
        min_width = dims.get("min_width", 0)

        # Allow 20% tolerance for pre-repair check
        tolerance = 0.80

        area_ok = real_area >= min_area * tolerance
        width_ok = min(real_w, real_h) >= min_width * tolerance

        if area_ok and width_ok:
            compliant_count += 1

    return round(compliant_count / len(room_boxes), 4)


def score_corridor_simplicity(room_boxes: Sequence[RoomBox]) -> float:
    """Score corridor layout simplicity.

    Prefers layouts where Corridor/Passage boxes form a simple connected shape.
    Returns 1.0 for simple corridors, lower for fragmented or missing corridors.
    """
    corridors = [b for b in room_boxes if b.room_type in {"Corridor", "Passage"}]
    if not corridors:
        # No explicit corridor - neutral score
        return 0.5

    if len(corridors) == 1:
        # Single corridor is ideal
        return 1.0

    # Multiple corridors - check if they're adjacent/connected
    connected_pairs = 0
    for i, c1 in enumerate(corridors):
        for c2 in corridors[i + 1:]:
            x_gap = _axis_gap_1d(c1.x_min, c1.x_max, c2.x_min, c2.x_max)
            y_gap = _axis_gap_1d(c1.y_min, c1.y_max, c2.y_min, c2.y_max)
            x_ov = _overlap_1d(c1.x_min, c1.x_max, c2.x_min, c2.x_max)
            y_ov = _overlap_1d(c1.y_min, c1.y_max, c2.y_min, c2.y_max)
            if (x_gap < 0.02 and y_ov > 0.02) or (y_gap < 0.02 and x_ov > 0.02):
                connected_pairs += 1

    # Score based on connectivity ratio
    max_pairs = len(corridors) * (len(corridors) - 1) // 2
    connectivity = connected_pairs / max_pairs if max_pairs > 0 else 0
    return round(0.5 + 0.5 * connectivity, 4)


def compute_realism_score(
    room_boxes: Sequence[RoomBox],
    boundary_width: float = 10.0,
    boundary_height: float = 12.0,
    plot_area_sqm: float = 100.0,
    reg: Optional[Dict] = None,
    *,
    aspect_weight: float = 0.25,
    dims_weight: float = 0.35,
    corridor_weight: float = 0.15,
    adjacency_score: float = 0.0,
    adjacency_weight: float = 0.25,
) -> Dict:
    """Compute composite realism score for pre-ranking.

    Returns dict with component scores and weighted total.
    """
    aspect = score_aspect_ratio_quality(room_boxes)
    dims = score_min_dims_compliance(
        room_boxes, boundary_width, boundary_height, plot_area_sqm, reg
    )
    corridor = score_corridor_simplicity(room_boxes)

    total = (
        aspect * aspect_weight
        + dims * dims_weight
        + corridor * corridor_weight
        + adjacency_score * adjacency_weight
    )

    return {
        "aspect_ratio_score": aspect,
        "min_dims_score": dims,
        "corridor_score": corridor,
        "adjacency_score": adjacency_score,
        "realism_total": round(total, 4),
    }


def estimate_repair_severity(
    room_boxes: Sequence[RoomBox],
    boundary_width: float = 10.0,
    boundary_height: float = 12.0,
    plot_area_sqm: float = 100.0,
) -> Dict:
    """Estimate how much repair will be needed.

    Lower severity = better candidate (less repair work needed).
    """
    issues = {
        "sliver_rooms": 0,
        "undersized_rooms": 0,
        "overlapping_pairs": 0,
    }

    for box in room_boxes:
        ar = _aspect_ratio(box)
        if ar > MAX_ACCEPTABLE_ASPECT_RATIO:
            issues["sliver_rooms"] += 1

        real_w = _box_width(box) * boundary_width
        real_h = _box_height(box) * boundary_height
        real_area = real_w * real_h
        dims = get_min_room_dims(box.room_type, plot_area_sqm)
        if real_area < dims.get("min_area", 0) * 0.7:
            issues["undersized_rooms"] += 1

    # Check for overlaps
    for i, b1 in enumerate(room_boxes):
        for b2 in room_boxes[i + 1:]:
            x_ov = _overlap_1d(b1.x_min, b1.x_max, b2.x_min, b2.x_max)
            y_ov = _overlap_1d(b1.y_min, b1.y_max, b2.y_min, b2.y_max)
            if x_ov > 0.01 and y_ov > 0.01:
                issues["overlapping_pairs"] += 1

    severity = (
        issues["sliver_rooms"] * 2
        + issues["undersized_rooms"] * 3
        + issues["overlapping_pairs"] * 4
    )

    return {
        "issues": issues,
        "severity": severity,
    }


def prerank_samples_v2(
    samples: Sequence[Dict],
    spec: Dict,
    *,
    boundary_width: float = 10.0,
    boundary_height: float = 12.0,
    plot_area_sqm: float = 100.0,
    top_m: int = 3,
    use_kg: bool = True,
    gap_tolerance: float = 0.05,
    center_distance_threshold: float = 0.25,
) -> List[Dict]:
    """Enhanced pre-ranking using Chapter-4 compliance proxy + adjacency.

    Combines adjacency, aspect ratio, min dimensions, and corridor simplicity
    into a composite realism score for ranking.
    """
    room_types = [r.get("type") for r in spec.get("rooms", []) if r.get("type")]
    intent = build_adjacency_intent(room_types=room_types or None, use_kg=use_kg)

    scored: List[Dict] = []
    for sample in samples:
        rooms = sample.get("raw_rooms", [])
        if not rooms:
            continue

        # Compute adjacency score
        adj = score_adjacency_proxy(
            rooms,
            intent,
            gap_tolerance=gap_tolerance,
            center_distance_threshold=center_distance_threshold,
        )

        # Compute realism score
        realism = compute_realism_score(
            rooms,
            boundary_width=boundary_width,
            boundary_height=boundary_height,
            plot_area_sqm=plot_area_sqm,
            adjacency_score=adj,
        )

        # Estimate repair severity
        repair = estimate_repair_severity(
            rooms,
            boundary_width=boundary_width,
            boundary_height=boundary_height,
            plot_area_sqm=plot_area_sqm,
        )

        entry = dict(sample)
        entry["adjacency_proxy"] = adj
        entry["realism_scores"] = realism
        entry["repair_estimate"] = repair
        # Composite ranking: high realism, low repair severity
        entry["rank_score"] = realism["realism_total"] - repair["severity"] * 0.02
        scored.append(entry)

    scored.sort(key=lambda s: s.get("rank_score", 0.0), reverse=True)
    return scored[: max(1, top_m)]
