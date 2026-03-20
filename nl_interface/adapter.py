"""Adapter and routing helpers for translating NL specs into backend specs."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from nl_interface.constants import (
    ALLOWED_BUILDING_TYPE,
    CORE_ALGORITHMIC_ROOM_TYPES,
    EXTENDED_LEARNED_ROOM_TYPES,
    EXTERNAL_TO_INTERNAL_ROOM,
)


def make_room(name: str, room_type: str) -> Dict[str, str]:
    return {"name": name, "type": room_type}


def route_backend(current_spec: Dict) -> Optional[str]:
    room_types = {room.get("type") for room in current_spec.get("rooms", []) if room.get("type")}
    print(f"[ROUTE_BACKEND] Room types found: {room_types}")
    if not room_types:
        print(f"[ROUTE_BACKEND] No room types -> returning None (no backend target)")
        return None
    # We now default to hybrid routing to run both backends
    print(f"[ROUTE_BACKEND] Routing to: hybrid")
    return "hybrid"


def validate_resolution(resolution: Optional[Dict]) -> Tuple[Optional[Dict], List[str]]:
    if not resolution:
        return None, ["boundary_polygon"]

    normalized = deepcopy(resolution)
    boundary = normalized.get("boundary_polygon")
    boundary_size = normalized.get("boundary_size")
    missing = []

    if not boundary:
        if boundary_size and len(boundary_size) == 2:
            width, height = float(boundary_size[0]), float(boundary_size[1])
            normalized["boundary_polygon"] = [
                (0.0, 0.0),
                (width, 0.0),
                (width, height),
                (0.0, height),
            ]
        else:
            missing.append("boundary_polygon")

    boundary = normalized.get("boundary_polygon")
    if boundary:
        normalized["boundary_polygon"] = [tuple(map(float, point)) for point in boundary]
        if len(normalized["boundary_polygon"]) < 3:
            missing.append("boundary_polygon")

    entrance_point = normalized.get("entrance_point")
    if entrance_point:
        normalized["entrance_point"] = tuple(map(float, entrance_point))

    area_unit = normalized.get("area_unit", "sq.m")
    normalized["area_unit"] = area_unit

    total_area = normalized.get("total_area")
    if total_area is not None:
        normalized["total_area"] = float(total_area)

    return normalized, missing


def build_backend_spec(current_spec: Dict, resolution: Optional[Dict] = None) -> Tuple[Optional[Dict], List[str]]:
    backend_target = route_backend(current_spec)
    warnings: List[str] = []
    if backend_target is None:
        return None, ["No backend target could be chosen from the current room set."]

    normalized_resolution, resolution_missing = validate_resolution(resolution)
    if resolution_missing:
        return None, ["Backend translation needs actual boundary geometry before execution."]

    boundary_polygon = normalized_resolution["boundary_polygon"]
    entrance_side = current_spec.get("entrance_side")
    entrance_point = normalized_resolution.get("entrance_point") or _entrance_point_from_side(
        boundary_polygon,
        entrance_side,
    )
    total_area = normalized_resolution.get("total_area")
    area_unit = normalized_resolution.get("area_unit", "sq.m")
    if total_area is None:
        total_area = _polygon_area(boundary_polygon)
        area_unit = "sq.m"

    room_sequence = _expand_room_types(current_spec)
    translated_adjacency, adjacency_warnings = _translate_adjacency(current_spec)
    warnings.extend(adjacency_warnings)

    base = {
        "occupancy": ALLOWED_BUILDING_TYPE,
        "building_type": ALLOWED_BUILDING_TYPE,
        "plot_type": current_spec.get("plot_type"),
        "boundary_polygon": boundary_polygon,
        "entrance_point": entrance_point,
        "entrance_side": entrance_side,
        "total_area": round(float(total_area), 3),
        "area_unit": area_unit,
        "allocation_strategy": "priority_weights",
        "adjacency": translated_adjacency,
        "preferences": _translate_preferences(current_spec),
        "_nl_spec": deepcopy(current_spec),
        "_nl_constraints": {
            "adjacency": deepcopy(current_spec.get("preferences", {}).get("adjacency", [])),
            "privacy": deepcopy(current_spec.get("preferences", {}).get("privacy", {})),
            "weights": deepcopy(current_spec.get("weights", {})),
        },
    }

    # All backends need rooms with name and type fields for spec validation
    type_counts = {}
    expanded_rooms = []
    for room_type in room_sequence:
        type_counts[room_type] = type_counts.get(room_type, 0) + 1
        expanded_rooms.append(make_room(f"{room_type}_{type_counts[room_type]}", room_type))
    base["rooms"] = expanded_rooms

    return base, warnings


def _expand_room_types(current_spec: Dict) -> List[str]:
    expanded = []
    for room in current_spec.get("rooms", []):
        external_type = room.get("type", "")
        internal_type = EXTERNAL_TO_INTERNAL_ROOM.get(external_type)
        if internal_type is None:
            continue  # Skip unknown room types
        count = int(room.get("count", 1))
        for _ in range(max(count, 0)):
            expanded.append(internal_type)
    return expanded


def _translate_adjacency(current_spec: Dict) -> Tuple[List[Dict], List[str]]:
    translated = []
    warnings: List[str] = []
    for source_type, target_type, relation in current_spec.get("preferences", {}).get("adjacency", []):
        a_internal = EXTERNAL_TO_INTERNAL_ROOM.get(source_type, source_type)
        b_internal = EXTERNAL_TO_INTERNAL_ROOM.get(target_type, target_type)
        if relation == "adjacent_to":
            translated.append({"a": a_internal, "b": b_internal, "type": "must"})
        elif relation == "near_to":
            translated.append({"a": a_internal, "b": b_internal, "type": "prefer"})
        else:
            warnings.append(
                f"The current backend does not enforce far_from directly; preserved as metadata for {source_type} and {target_type}."
            )
    return translated, warnings


def _translate_preferences(current_spec: Dict) -> Dict:
    weights = current_spec.get("weights", {})
    compactness_value = weights.get("compactness", 0.0)
    if compactness_value >= 0.45:
        compactness = "high"
    elif compactness_value >= 0.25:
        compactness = "medium"
    else:
        compactness = "low"

    corridor_weight = weights.get("corridor", 0.0)
    return {
        "compactness": compactness,
        "minimize_travel_distance": bool(
            current_spec.get("preferences", {}).get("minimize_corridor", False)
            or corridor_weight >= 0.4
        ),
    }


def _entrance_point_from_side(boundary_polygon: List[Tuple[float, float]], entrance_side: str) -> Tuple[float, float]:
    xs = [point[0] for point in boundary_polygon]
    ys = [point[1] for point in boundary_polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_mid = round((x_min + x_max) / 2.0, 4)
    y_mid = round((y_min + y_max) / 2.0, 4)

    if entrance_side == "North":
        return (x_mid, y_max)
    if entrance_side == "South":
        return (x_mid, y_min)
    if entrance_side == "East":
        return (x_max, y_mid)
    return (x_min, y_mid)


def _polygon_area(points: List[Tuple[float, float]]) -> float:
    area = 0.0
    count = len(points)
    for idx in range(count):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % count]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


