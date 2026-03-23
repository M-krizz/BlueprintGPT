"""
geometry_synthesis.py - Convert planner model outputs to room geometry.

The PlannerTransformer outputs:
- centroid: normalized (x, y) in [0, 1] for each room
- area_ratio: fraction of total area for each room
- adjacency_logits: room-to-room adjacency preferences

This module synthesizes actual room polygons from these outputs,
bypassing the algorithmic PolygonPacker bisection.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

MIN_DIRECT_DOOR_SPAN = 0.8
MIN_BRIDGE_DOOR_SPAN = 0.5
BRIDGE_GAP_TOLERANCE = 0.35
MINOR_OVERLAP_TOLERANCE = 0.35
ADJACENCY_TARGET_GAP = 0.08
ADJACENCY_REPAIR_ITERATIONS = 6
PLACEMENT_GRID_STEP = 0.35
PRIVATE_ENTRANCE_BUFFER_RATIO = 0.22


def synthesize_room_geometry(
    planner_output: Dict,
    boundary_polygon: List[Tuple[float, float]],
    spec: Dict,
    entrance_point: Optional[Tuple[float, float]] = None,
) -> List[Dict]:
    """
    Convert planner model outputs to room bounding boxes.

    Strategy:
    1. Use area_ratios to compute room areas from boundary area
    2. Use centroids (spatial_hints) as room center positions
    3. Compute room dimensions from area with aspect ratio constraints
    4. Repair overlaps and pull preferred adjacencies into contact
    5. Clamp rooms to boundary
    """
    boundary_xs = [p[0] for p in boundary_polygon]
    boundary_ys = [p[1] for p in boundary_polygon]
    bx0, bx1 = min(boundary_xs), max(boundary_xs)
    by0, by1 = min(boundary_ys), max(boundary_ys)
    boundary_width = bx1 - bx0
    boundary_height = by1 - by0
    boundary_area = boundary_width * boundary_height

    spatial_hints = planner_output.get("spatial_hints", {})
    area_ratios = planner_output.get("area_ratios", {})
    room_order = planner_output.get("room_order", [])
    room_zones = planner_output.get("room_zones", {})

    spec_rooms = {r["name"]: r for r in spec.get("rooms", [])}
    rooms: List[Dict] = []

    for room_name in room_order:
        if room_name not in spatial_hints:
            continue

        cx_norm, cy_norm = spatial_hints[room_name]
        cx = bx0 + cx_norm * boundary_width
        cy = by0 + cy_norm * boundary_height

        area_ratio = area_ratios.get(room_name, 0.1)
        room_area = boundary_area * area_ratio

        spec_room = spec_rooms.get(room_name, {})
        if spec_room.get("area"):
            room_area = max(room_area, float(spec_room["area"]))
        elif spec_room.get("min_area_sqm"):
            room_area = max(room_area, float(spec_room["min_area_sqm"]))
        elif spec_room.get("recommended_area_sqm"):
            room_area = max(room_area, float(spec_room["recommended_area_sqm"]))

        room_type = spec_room.get("type", room_name.split("_")[0])
        aspect = _get_room_aspect_ratio(room_type)

        width = math.sqrt(room_area * aspect)
        height = room_area / width if width > 0 else 1.0

        x1 = cx - width / 2
        y1 = cy - height / 2
        x2 = cx + width / 2
        y2 = cy + height / 2

        x1, y1 = max(bx0, x1), max(by0, y1)
        x2, y2 = min(bx1, x2), min(by1, y2)

        if x2 - x1 < 2.0:
            x2 = x1 + 2.0
        if y2 - y1 < 2.0:
            y2 = y1 + 2.0

        if x2 > bx1:
            x1 = max(bx0, bx1 - (x2 - x1))
            x2 = bx1
        if y2 > by1:
            y1 = max(by0, by1 - (y2 - y1))
            y2 = by1

        rooms.append(
            {
                "name": room_name,
                "type": room_type,
                "polygon": [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                "area": (x2 - x1) * (y2 - y1),
                "centroid": ((x1 + x2) / 2, (y1 + y2) / 2),
                "desired_centroid": (cx, cy),
                "planner_centroid": (cx_norm, cy_norm),
                "zone": room_zones.get(room_name, "private"),
            }
        )

    rooms = _repack_rooms_without_overlap(rooms, planner_output, bx0, by0, bx1, by1, entrance_point)
    rooms = _resolve_overlaps(rooms, bx0, by0, bx1, by1)
    rooms = _apply_adjacency_contact_repair(rooms, planner_output, bx0, by0, bx1, by1)
    rooms = _resolve_overlaps(rooms, bx0, by0, bx1, by1)
    rooms = _trim_residual_overlaps(rooms, spec, bx0, by0, bx1, by1)
    rooms = _resolve_overlaps(rooms, bx0, by0, bx1, by1, max_iterations=30)
    if _room_overlap_free_ratio(rooms) < 0.9:
        rooms = _repack_rooms_without_overlap(rooms, planner_output, bx0, by0, bx1, by1, entrance_point)
        rooms = _resolve_overlaps(rooms, bx0, by0, bx1, by1)
        rooms = _trim_residual_overlaps(rooms, spec, bx0, by0, bx1, by1)
        rooms = _resolve_overlaps(rooms, bx0, by0, bx1, by1, max_iterations=20)
    return rooms


def _get_room_aspect_ratio(room_type: str) -> float:
    aspect_ratios = {
        "LivingRoom": 1.4,
        "DiningRoom": 1.2,
        "Kitchen": 1.3,
        "Bedroom": 1.25,
        "Bathroom": 0.8,
        "WC": 0.7,
        "Garage": 1.5,
        "Store": 1.0,
        "Study": 1.1,
        "Balcony": 2.0,
    }
    return aspect_ratios.get(room_type, 1.2)


def _room_sort_key(room: Dict) -> Tuple[int, int, float]:
    zone_priority = {"public": 0, "service": 1, "private": 2}
    type_priority = {
        "LivingRoom": 0,
        "DiningRoom": 1,
        "Kitchen": 2,
        "Study": 3,
        "Bedroom": 4,
        "Bathroom": 5,
        "WC": 6,
    }
    return (
        zone_priority.get(room.get("zone", "private"), 2),
        type_priority.get(room.get("type", ""), 7),
        -float(room.get("area", 0.0)),
    )


def _candidate_positions(start_min: float, start_max: float, desired_start: float, step: float) -> List[float]:
    if start_max <= start_min:
        return [round(start_min, 4)]

    values = [round(start_min, 4), round(start_max, 4), round(min(max(desired_start, start_min), start_max), 4)]
    current = start_min
    while current <= start_max:
        values.append(round(current, 4))
        current += step
    unique = sorted(set(values), key=lambda value: abs(value - desired_start))
    return unique


def _room_gap(poly1: List[Tuple[float, float]], poly2: List[Tuple[float, float]]) -> float:
    metrics = _pair_metrics(poly1, poly2)
    if metrics["overlap_x"] > 0 and metrics["overlap_y"] > 0:
        return 0.0
    if metrics["overlap_x"] > 0:
        return metrics["gap_y"]
    if metrics["overlap_y"] > 0:
        return metrics["gap_x"]
    return math.hypot(metrics["gap_x"], metrics["gap_y"])


def _room_overlap_free_ratio(rooms: List[Dict]) -> float:
    total_area = sum(float(room.get("area", 0.0) or 0.0) for room in rooms)
    if total_area <= 0:
        return 1.0
    overlap_area = 0.0
    for index, room in enumerate(rooms):
        for other in rooms[index + 1 :]:
            overlap_area += _compute_overlap(room["polygon"], other["polygon"])
    return max(0.0, 1.0 - overlap_area / total_area)


def _build_preference_scores(planner_output: Dict) -> Dict[Tuple[str, str], float]:
    return {
        tuple(sorted((room_a, room_b))): score
        for room_a, room_b, score in _sorted_adjacency_preferences(planner_output)
    }


def _placement_score(
    candidate_room: Dict,
    placed_rooms: List[Dict],
    preference_scores: Dict[Tuple[str, str], float],
    entrance_point: Optional[Tuple[float, float]],
    diag: float,
) -> Tuple[float, float]:
    overlap_area = 0.0
    for other in placed_rooms:
        overlap_area += _compute_overlap(candidate_room["polygon"], other["polygon"])

    cx, cy = candidate_room["centroid"]
    desired_cx, desired_cy = candidate_room.get("desired_centroid", candidate_room["centroid"])
    score = math.hypot(cx - desired_cx, cy - desired_cy) / diag
    score += overlap_area * 20.0

    if entrance_point is not None:
        ex, ey = entrance_point
        entrance_distance = math.hypot(cx - ex, cy - ey) / diag
        room_type = candidate_room.get("type")
        if room_type in {"Bathroom", "WC"}:
            score += max(0.0, PRIVATE_ENTRANCE_BUFFER_RATIO - entrance_distance) * 28.0
        elif room_type == "Bedroom":
            score += max(0.0, PRIVATE_ENTRANCE_BUFFER_RATIO * 0.8 - entrance_distance) * 18.0
        elif room_type == "LivingRoom":
            score += entrance_distance * 0.7
        elif room_type in {"Kitchen", "DiningRoom"}:
            score += entrance_distance * 0.4

    for other in placed_rooms:
        key = tuple(sorted((candidate_room["name"], other["name"])))
        pref_score = preference_scores.get(key, 0.0)
        if pref_score <= 0:
            continue
        if _are_room_polygons_adjacent(candidate_room["polygon"], other["polygon"], threshold=0.12):
            score -= 0.75 * pref_score
        else:
            score += (_room_gap(candidate_room["polygon"], other["polygon"]) / diag) * (3.0 * pref_score)

    return score, overlap_area


def _best_room_placement(
    room: Dict,
    placed_rooms: List[Dict],
    preference_scores: Dict[Tuple[str, str], float],
    entrance_point: Optional[Tuple[float, float]],
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
) -> Dict:
    x1, y1, x2, y2 = _bounds(room["polygon"])
    width = x2 - x1
    height = y2 - y1
    desired_cx, desired_cy = room.get("desired_centroid", room["centroid"])
    desired_x1 = desired_cx - width / 2.0
    desired_y1 = desired_cy - height / 2.0
    step = max(PLACEMENT_GRID_STEP, min(width, height) / 5.0)
    diag = max(math.hypot(bx1 - bx0, by1 - by0), 1e-6)

    best_zero_overlap: Optional[Tuple[float, Dict]] = None
    best_any: Optional[Tuple[Tuple[float, float], Dict]] = None

    for candidate_x1 in _candidate_positions(bx0, bx1 - width, desired_x1, step):
        for candidate_y1 in _candidate_positions(by0, by1 - height, desired_y1, step):
            candidate = dict(room)
            candidate_x2 = candidate_x1 + width
            candidate_y2 = candidate_y1 + height
            candidate["polygon"] = [
                (candidate_x1, candidate_y1),
                (candidate_x2, candidate_y1),
                (candidate_x2, candidate_y2),
                (candidate_x1, candidate_y2),
            ]
            candidate["centroid"] = ((candidate_x1 + candidate_x2) / 2.0, (candidate_y1 + candidate_y2) / 2.0)
            candidate["area"] = width * height
            score, overlap_area = _placement_score(candidate, placed_rooms, preference_scores, entrance_point, diag)
            if overlap_area <= 0.01:
                if best_zero_overlap is None or score < best_zero_overlap[0]:
                    best_zero_overlap = (score, candidate)
            else:
                key = (overlap_area, score)
                if best_any is None or key < best_any[0]:
                    best_any = (key, candidate)
    if best_zero_overlap is not None:
        return best_zero_overlap[1]
    if best_any is not None:
        return best_any[1]
    return room


def _repack_rooms_without_overlap(
    rooms: List[Dict],
    planner_output: Dict,
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
    entrance_point: Optional[Tuple[float, float]],
) -> List[Dict]:
    if len(rooms) < 2:
        return rooms

    preference_scores = _build_preference_scores(planner_output)
    placed_rooms: List[Dict] = []
    for room in sorted(rooms, key=_room_sort_key):
        placed_rooms.append(
            _best_room_placement(
                room,
                placed_rooms,
                preference_scores,
                entrance_point,
                bx0,
                by0,
                bx1,
                by1,
            )
        )

    room_by_name = {room["name"]: room for room in placed_rooms}
    return [room_by_name.get(room["name"], room) for room in rooms]


def _resolve_overlaps(
    rooms: List[Dict],
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
    max_iterations: int = 80,
) -> List[Dict]:
    """Push overlapping rooms apart until overlaps become negligible."""
    for _ in range(max_iterations):
        any_overlap = False
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                overlap_area = _compute_overlap(rooms[i]["polygon"], rooms[j]["polygon"])
                if overlap_area > 0.01:
                    any_overlap = True
                    if rooms[i]["area"] >= rooms[j]["area"]:
                        rooms[j] = _push_room(rooms[j], rooms[i], bx0, by0, bx1, by1)
                    else:
                        rooms[i] = _push_room(rooms[i], rooms[j], bx0, by0, bx1, by1)
        if not any_overlap:
            break
    return rooms


def _apply_adjacency_contact_repair(
    rooms: List[Dict],
    planner_output: Dict,
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
) -> List[Dict]:
    room_by_name = {room["name"]: room for room in rooms}
    preferences = _sorted_adjacency_preferences(planner_output)
    if not preferences:
        return rooms

    for _ in range(ADJACENCY_REPAIR_ITERATIONS):
        moved = False
        for pref in preferences:
            room_a = room_by_name.get(pref[0])
            room_b = room_by_name.get(pref[1])
            if room_a is None or room_b is None:
                continue
            if _are_room_polygons_adjacent(room_a["polygon"], room_b["polygon"], threshold=0.18):
                continue

            if room_a["area"] <= room_b["area"]:
                room_to_move, room_fixed = room_a, room_b
            else:
                room_to_move, room_fixed = room_b, room_a

            original = list(room_to_move["polygon"])
            room_to_move = _pull_room_toward_contact(room_to_move, room_fixed, bx0, by0, bx1, by1)
            room_by_name[room_to_move["name"]] = room_to_move
            if room_to_move["polygon"] != original:
                moved = True

        if not moved:
            break
        rooms = _resolve_overlaps(rooms, bx0, by0, bx1, by1, max_iterations=20)
        room_by_name = {room["name"]: room for room in rooms}

    return rooms


def _sorted_adjacency_preferences(planner_output: Dict) -> List[Tuple[str, str, float]]:
    pairs: Dict[Tuple[str, str], float] = {}
    for pref in planner_output.get("adjacency_preferences", []) or []:
        room_a = pref.get("a")
        room_b = pref.get("b")
        if not room_a or not room_b or room_a == room_b:
            continue
        score = pref.get("score", pref.get("weight", 1.0))
        try:
            score = float(score)
        except Exception:
            score = 1.0
        key = tuple(sorted((room_a, room_b)))
        pairs[key] = max(score, pairs.get(key, 0.0))
    return [(a, b, score) for (a, b), score in sorted(pairs.items(), key=lambda item: item[1], reverse=True)]


def _pull_room_toward_contact(
    room_to_move: Dict,
    room_fixed: Dict,
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
) -> Dict:
    fixed_bounds = _bounds(room_fixed["polygon"])
    move_bounds = _bounds(room_to_move["polygon"])
    fixed_cx, fixed_cy = room_fixed["centroid"]
    move_cx, move_cy = room_to_move["centroid"]

    x1, y1 = move_bounds[0], move_bounds[1]
    x2, y2 = move_bounds[2], move_bounds[3]
    width = x2 - x1
    height = y2 - y1

    horizontal_dominant = abs(move_cx - fixed_cx) >= abs(move_cy - fixed_cy)

    if horizontal_dominant:
        if move_cx >= fixed_cx:
            new_x1 = fixed_bounds[2] + ADJACENCY_TARGET_GAP
        else:
            new_x1 = fixed_bounds[0] - ADJACENCY_TARGET_GAP - width
        desired_center_y = fixed_cy
        new_y1 = desired_center_y - height / 2.0
    else:
        if move_cy >= fixed_cy:
            new_y1 = fixed_bounds[3] + ADJACENCY_TARGET_GAP
        else:
            new_y1 = fixed_bounds[1] - ADJACENCY_TARGET_GAP - height
        desired_center_x = fixed_cx
        new_x1 = desired_center_x - width / 2.0

    new_x1 = min(max(new_x1, bx0), bx1 - width)
    new_y1 = min(max(new_y1, by0), by1 - height)
    new_x2 = new_x1 + width
    new_y2 = new_y1 + height

    room_to_move["polygon"] = [(new_x1, new_y1), (new_x2, new_y1), (new_x2, new_y2), (new_x1, new_y2)]
    room_to_move["centroid"] = ((new_x1 + new_x2) / 2.0, (new_y1 + new_y2) / 2.0)
    room_to_move["area"] = width * height
    return room_to_move


def _bounds(poly: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _pair_metrics(poly1: List[Tuple[float, float]], poly2: List[Tuple[float, float]]) -> Dict[str, float]:
    x1_min, y1_min, x1_max, y1_max = _bounds(poly1)
    x2_min, y2_min, x2_max, y2_max = _bounds(poly2)
    return {
        "x1_min": x1_min,
        "x1_max": x1_max,
        "y1_min": y1_min,
        "y1_max": y1_max,
        "x2_min": x2_min,
        "x2_max": x2_max,
        "y2_min": y2_min,
        "y2_max": y2_max,
        "overlap_x": max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min)),
        "overlap_y": max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min)),
        "gap_x": max(0.0, max(x1_min - x2_max, x2_min - x1_max)),
        "gap_y": max(0.0, max(y1_min - y2_max, y2_min - y1_max)),
    }


def _compute_overlap(poly1: List[Tuple[float, float]], poly2: List[Tuple[float, float]]) -> float:
    metrics = _pair_metrics(poly1, poly2)
    return metrics["overlap_x"] * metrics["overlap_y"]


def _push_room(
    room_to_move: Dict,
    room_fixed: Dict,
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
) -> Dict:
    """Push room away from fixed room, preferring the smallest penetration axis."""
    poly_move = room_to_move["polygon"]
    poly_fixed = room_fixed["polygon"]
    metrics = _pair_metrics(poly_move, poly_fixed)

    x1, y1 = poly_move[0]
    x2, y2 = poly_move[2]
    width = x2 - x1
    height = y2 - y1

    overlap_x = metrics["overlap_x"]
    overlap_y = metrics["overlap_y"]
    cx_m, cy_m = room_to_move["centroid"]
    cx_f, cy_f = room_fixed["centroid"]

    if overlap_x > 0 and overlap_y > 0:
        if overlap_x <= overlap_y:
            shift = overlap_x + 0.1
            if cx_m >= cx_f:
                x1 += shift
                x2 += shift
            else:
                x1 -= shift
                x2 -= shift
        else:
            shift = overlap_y + 0.1
            if cy_m >= cy_f:
                y1 += shift
                y2 += shift
            else:
                y1 -= shift
                y2 -= shift
    else:
        dx = cx_m - cx_f
        dy = cy_m - cy_f
        dist = max(math.hypot(dx, dy), 0.01)
        overlap = _compute_overlap(poly_move, poly_fixed)
        push_dist = math.sqrt(overlap) + 0.1
        x1 += (dx / dist) * push_dist
        x2 += (dx / dist) * push_dist
        y1 += (dy / dist) * push_dist
        y2 += (dy / dist) * push_dist

    if x1 < bx0:
        x1, x2 = bx0, bx0 + width
    if y1 < by0:
        y1, y2 = by0, by0 + height
    if x2 > bx1:
        x1, x2 = bx1 - width, bx1
    if y2 > by1:
        y1, y2 = by1 - height, by1

    room_to_move["polygon"] = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    room_to_move["area"] = (x2 - x1) * (y2 - y1)
    room_to_move["centroid"] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return room_to_move



def _minimum_room_area(spec_room: Dict, room_type: str) -> float:
    if spec_room.get("min_area_sqm"):
        return float(spec_room["min_area_sqm"])
    if spec_room.get("area"):
        return float(spec_room["area"])
    defaults = {
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
    return defaults.get(room_type, 4.0)



def _shrink_room_against_overlap(
    room: Dict,
    other: Dict,
    spec_room: Dict,
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
) -> Dict:
    x1, y1, x2, y2 = _bounds(room["polygon"])
    ox1, oy1, ox2, oy2 = _bounds(other["polygon"])
    width = x2 - x1
    height = y2 - y1
    min_area = _minimum_room_area(spec_room, room.get("type", "")) * 0.9
    metrics = _pair_metrics(room["polygon"], other["polygon"])

    if metrics["overlap_x"] <= 0 or metrics["overlap_y"] <= 0:
        return room

    shrink_x = metrics["overlap_x"] + 0.05
    shrink_y = metrics["overlap_y"] + 0.05

    can_shrink_x = (width - shrink_x) * height >= min_area and (width - shrink_x) >= 1.8
    can_shrink_y = width * (height - shrink_y) >= min_area and (height - shrink_y) >= 1.8

    if not can_shrink_x and not can_shrink_y:
        return room

    cx, cy = room["centroid"]
    ocx, ocy = other["centroid"]
    if can_shrink_x and (not can_shrink_y or metrics["overlap_x"] <= metrics["overlap_y"]):
        if cx >= ocx:
            x1 = min(max(x1 + shrink_x, bx0), x2 - 1.8)
        else:
            x2 = max(min(x2 - shrink_x, bx1), x1 + 1.8)
    else:
        if cy >= ocy:
            y1 = min(max(y1 + shrink_y, by0), y2 - 1.8)
        else:
            y2 = max(min(y2 - shrink_y, by1), y1 + 1.8)

    room["polygon"] = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    room["centroid"] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    room["area"] = (x2 - x1) * (y2 - y1)
    return room



def _trim_residual_overlaps(
    rooms: List[Dict],
    spec: Dict,
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
    max_iterations: int = 12,
) -> List[Dict]:
    spec_rooms = {room["name"]: room for room in spec.get("rooms", [])}
    for _ in range(max_iterations):
        adjusted = False
        for index, room in enumerate(rooms):
            for other_index in range(index + 1, len(rooms)):
                other = rooms[other_index]
                if _compute_overlap(room["polygon"], other["polygon"]) <= 0.01:
                    continue
                adjusted = True
                if room.get("area", 0.0) <= other.get("area", 0.0):
                    rooms[index] = _shrink_room_against_overlap(room, other, spec_rooms.get(room["name"], {}), bx0, by0, bx1, by1)
                else:
                    rooms[other_index] = _shrink_room_against_overlap(other, room, spec_rooms.get(other["name"], {}), bx0, by0, bx1, by1)
        if not adjusted:
            break
    return rooms

def synthesize_simple_doors(
    rooms: List[Dict],
    boundary_polygon: List[Tuple[float, float]],
    entrance_point: Optional[Tuple[float, float]] = None,
) -> List[Dict]:
    """Generate direct and bridge door placements between nearby rooms."""
    doors: List[Dict] = []
    seen_pairs = set()

    for i, room1 in enumerate(rooms):
        for room2 in rooms[i + 1 :]:
            pair_key = tuple(sorted((room1["name"], room2["name"])))
            if pair_key in seen_pairs:
                continue

            shared_length = _compute_shared_edge_length(room1["polygon"], room2["polygon"])
            segment = None
            door_kind = "room_to_room"

            if shared_length >= MIN_DIRECT_DOOR_SPAN:
                segment = _find_door_segment(room1["polygon"], room2["polygon"], width=0.9)
            if segment is None:
                segment = _find_bridge_door_segment(room1["polygon"], room2["polygon"], width=0.9)
                if segment is not None:
                    door_kind = "room_to_room_bridge"

            if segment is not None:
                seen_pairs.add(pair_key)
                doors.append(
                    {
                        "from_room": room1["name"],
                        "to_room": room2["name"],
                        "position": _segment_midpoint(segment),
                        "segment": segment,
                        "width": round(_segment_length(segment), 3),
                        "door_type": door_kind,
                    }
                )

    if entrance_point:
        nearest_room = _find_nearest_room_to_entrance(rooms, entrance_point)
        if nearest_room:
            segment = _find_entrance_door_segment(nearest_room["polygon"], entrance_point, width=1.0)
            doors.append(
                {
                    "from_room": nearest_room["name"],
                    "to_room": None,
                    "position": _segment_midpoint(segment) if segment else entrance_point,
                    "segment": segment,
                    "width": round(_segment_length(segment), 3) if segment else 1.0,
                    "door_type": "exit",
                }
            )

    return doors


def _compute_shared_edge_length(poly1: List[Tuple[float, float]], poly2: List[Tuple[float, float]]) -> float:
    metrics = _pair_metrics(poly1, poly2)
    tolerance = 0.3

    if metrics["gap_x"] <= tolerance:
        return metrics["overlap_y"]
    if metrics["gap_y"] <= tolerance:
        return metrics["overlap_x"]
    return 0.0


def _find_door_segment(
    poly1: List[Tuple[float, float]],
    poly2: List[Tuple[float, float]],
    width: float = 0.9,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    metrics = _pair_metrics(poly1, poly2)
    tolerance = 0.3

    if metrics["gap_x"] <= tolerance and metrics["overlap_y"] > 0:
        x = metrics["x1_max"] if metrics["x1_max"] <= metrics["x2_min"] else metrics["x1_min"]
        return _vertical_segment(
            x,
            max(metrics["y1_min"], metrics["y2_min"]),
            min(metrics["y1_max"], metrics["y2_max"]),
            width,
        )

    if metrics["gap_y"] <= tolerance and metrics["overlap_x"] > 0:
        y = metrics["y1_max"] if metrics["y1_max"] <= metrics["y2_min"] else metrics["y1_min"]
        return _horizontal_segment(
            y,
            max(metrics["x1_min"], metrics["x2_min"]),
            min(metrics["x1_max"], metrics["x2_max"]),
            width,
        )

    return None


def _find_bridge_door_segment(
    poly1: List[Tuple[float, float]],
    poly2: List[Tuple[float, float]],
    width: float = 0.9,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    metrics = _pair_metrics(poly1, poly2)

    if metrics["overlap_y"] >= MIN_BRIDGE_DOOR_SPAN and metrics["gap_x"] <= BRIDGE_GAP_TOLERANCE:
        if metrics["x1_max"] <= metrics["x2_min"]:
            x = (metrics["x1_max"] + metrics["x2_min"]) / 2.0
        elif metrics["x2_max"] <= metrics["x1_min"]:
            x = (metrics["x2_max"] + metrics["x1_min"]) / 2.0
        elif 0 < metrics["overlap_x"] <= MINOR_OVERLAP_TOLERANCE:
            x = (max(metrics["x1_min"], metrics["x2_min"]) + min(metrics["x1_max"], metrics["x2_max"])) / 2.0
        else:
            x = None
        if x is not None:
            return _vertical_segment(
                x,
                max(metrics["y1_min"], metrics["y2_min"]),
                min(metrics["y1_max"], metrics["y2_max"]),
                width,
            )

    if metrics["overlap_x"] >= MIN_BRIDGE_DOOR_SPAN and metrics["gap_y"] <= BRIDGE_GAP_TOLERANCE:
        if metrics["y1_max"] <= metrics["y2_min"]:
            y = (metrics["y1_max"] + metrics["y2_min"]) / 2.0
        elif metrics["y2_max"] <= metrics["y1_min"]:
            y = (metrics["y2_max"] + metrics["y1_min"]) / 2.0
        elif 0 < metrics["overlap_y"] <= MINOR_OVERLAP_TOLERANCE:
            y = (max(metrics["y1_min"], metrics["y2_min"]) + min(metrics["y1_max"], metrics["y2_max"])) / 2.0
        else:
            y = None
        if y is not None:
            return _horizontal_segment(
                y,
                max(metrics["x1_min"], metrics["x2_min"]),
                min(metrics["x1_max"], metrics["x2_max"]),
                width,
            )

    return None


def _are_room_polygons_adjacent(
    poly1: List[Tuple[float, float]],
    poly2: List[Tuple[float, float]],
    threshold: float,
) -> bool:
    metrics = _pair_metrics(poly1, poly2)
    if metrics["gap_x"] <= threshold and metrics["overlap_y"] > MIN_BRIDGE_DOOR_SPAN:
        return True
    if metrics["gap_y"] <= threshold and metrics["overlap_x"] > MIN_BRIDGE_DOOR_SPAN:
        return True
    return False


def _find_entrance_door_segment(
    polygon: List[Tuple[float, float]],
    entrance_point: Tuple[float, float],
    width: float = 1.0,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    x_min, y_min, x_max, y_max = _bounds(polygon)
    ex, ey = entrance_point
    edge_distances = {
        "left": abs(ex - x_min),
        "right": abs(ex - x_max),
        "bottom": abs(ey - y_min),
        "top": abs(ey - y_max),
    }
    side = min(edge_distances, key=edge_distances.get)

    if side == "left":
        return _vertical_segment(x_min, y_min, y_max, width, preferred_center=ey)
    if side == "right":
        return _vertical_segment(x_max, y_min, y_max, width, preferred_center=ey)
    if side == "bottom":
        return _horizontal_segment(y_min, x_min, x_max, width, preferred_center=ex)
    return _horizontal_segment(y_max, x_min, x_max, width, preferred_center=ex)


def _vertical_segment(
    x: float,
    y_min: float,
    y_max: float,
    width: float,
    preferred_center: Optional[float] = None,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    length = max(0.0, y_max - y_min)
    if length <= 0:
        return None
    actual_width = min(width, length)
    half = actual_width / 2.0
    center = preferred_center if preferred_center is not None else (y_min + y_max) / 2.0
    if length > actual_width:
        center = min(max(center, y_min + half), y_max - half)
    else:
        center = (y_min + y_max) / 2.0
    return _round_segment(((x, center - half), (x, center + half)))


def _horizontal_segment(
    y: float,
    x_min: float,
    x_max: float,
    width: float,
    preferred_center: Optional[float] = None,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    length = max(0.0, x_max - x_min)
    if length <= 0:
        return None
    actual_width = min(width, length)
    half = actual_width / 2.0
    center = preferred_center if preferred_center is not None else (x_min + x_max) / 2.0
    if length > actual_width:
        center = min(max(center, x_min + half), x_max - half)
    else:
        center = (x_min + x_max) / 2.0
    return _round_segment(((center - half, y), (center + half, y)))


def _round_segment(
    segment: Tuple[Tuple[float, float], Tuple[float, float]]
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    (x1, y1), (x2, y2) = segment
    return ((round(x1, 4), round(y1, 4)), (round(x2, 4), round(y2, 4)))


def _segment_midpoint(segment: Tuple[Tuple[float, float], Tuple[float, float]]) -> Tuple[float, float]:
    (x1, y1), (x2, y2) = segment
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _segment_length(segment: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]) -> float:
    if segment is None:
        return 0.0
    (x1, y1), (x2, y2) = segment
    return math.hypot(x2 - x1, y2 - y1)


def _find_nearest_room_to_entrance(
    rooms: List[Dict],
    entrance_point: Tuple[float, float],
) -> Optional[Dict]:
    min_dist = float("inf")
    nearest = None
    ex, ey = entrance_point

    for room in rooms:
        cx, cy = room["centroid"]
        dist = math.hypot(cx - ex, cy - ey)
        if dist < min_dist:
            min_dist = dist
            nearest = room

    return nearest




