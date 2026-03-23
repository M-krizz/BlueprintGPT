"""Adapter and routing helpers for translating NL specs into backend specs."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.processing_logger import ProcessingLogger
from nl_interface.constants import (
    ALLOWED_BUILDING_TYPE,
    CORE_ALGORITHMIC_ROOM_TYPES,
    EXTENDED_LEARNED_ROOM_TYPES,
    EXTERNAL_TO_INTERNAL_ROOM,
)

# Constraint Analyzer for proper room specifications
try:
    from nl_interface.constraint_analyzer import get_layout_requirements, calculate_optimal_dimensions
    _HAS_CONSTRAINT_ANALYZER = True
except ImportError:
    _HAS_CONSTRAINT_ANALYZER = False
    get_layout_requirements = None
    calculate_optimal_dimensions = None

# Backend routing configuration
# Options: "auto" (route by room program), "planner", "learned", "algorithmic", "hybrid"
DEFAULT_BACKEND = os.getenv("BLUEPRINT_BACKEND_MODE", "auto").lower()
AUTO_CORE_BACKEND_ENV = "BLUEPRINT_AUTO_CORE_BACKEND"
PLANNER_CHECKPOINT_ENV = "BLUEPRINTGPT_PLANNER_CHECKPOINT"
DEFAULT_PLANNER_CHECKPOINT = "learned/planner/checkpoints/room_planner.pt"


def make_room(name: str, room_type: str) -> Dict[str, str]:
    return {"name": name, "type": room_type}


def _configured_backend_mode() -> str:
    return os.getenv("BLUEPRINT_BACKEND_MODE", DEFAULT_BACKEND).lower()


def _planner_checkpoint_exists() -> bool:
    checkpoint_path = Path(os.getenv(PLANNER_CHECKPOINT_ENV, DEFAULT_PLANNER_CHECKPOINT))
    return checkpoint_path.exists()


def _resolve_core_backend() -> Tuple[str, str, bool]:
    policy = os.getenv(AUTO_CORE_BACKEND_ENV, "algorithmic").lower()
    checkpoint_ready = _planner_checkpoint_exists()

    if policy == "planner":
        return "planner", policy, checkpoint_ready
    if policy == "planner_direct":
        return "planner_direct", policy, checkpoint_ready
    if policy in {"planner_if_available", "planner-if-available"} and checkpoint_ready:
        return "planner", policy, checkpoint_ready
    if policy in {"planner_direct_if_available", "planner-direct-if-available"} and checkpoint_ready:
        return "planner_direct", policy, checkpoint_ready
    return "algorithmic", policy, checkpoint_ready


def route_backend(current_spec: Dict) -> Optional[str]:
    room_types = {room.get("type") for room in current_spec.get("rooms", []) if room.get("type")}
    ProcessingLogger.logger.debug(f"route_backend: room_types={room_types}")
    if not room_types:
        ProcessingLogger.logger.debug("route_backend: no room types, returning None")
        return None

    backend_mode = _configured_backend_mode()

    # Explicit override takes precedence.
    if backend_mode in {"algorithmic", "planner", "planner_direct", "learned", "hybrid"}:
        ProcessingLogger.logger.info(f"Routing to {backend_mode} backend (BLUEPRINT_BACKEND_MODE={backend_mode})")
        return backend_mode

    extended_room_types = room_types & EXTENDED_LEARNED_ROOM_TYPES
    auto_core_policy = os.getenv(AUTO_CORE_BACKEND_ENV, "algorithmic").lower()
    planner_checkpoint_ready = _planner_checkpoint_exists()
    if extended_room_types:
        backend = "learned"
    elif room_types <= CORE_ALGORITHMIC_ROOM_TYPES:
        backend, auto_core_policy, planner_checkpoint_ready = _resolve_core_backend()
    else:
        backend = "hybrid"

    ProcessingLogger.logger.info(
        f"Routing to {backend} backend (BLUEPRINT_BACKEND_MODE={backend_mode}, "
        f"extended_room_types={sorted(extended_room_types)}, "
        f"auto_core_policy={auto_core_policy}, "
        f"planner_checkpoint_ready={planner_checkpoint_ready})"
    )
    return backend


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

    # Determine layout type from rooms
    bedroom_count = sum(
        r.get("count", 1) for r in current_spec.get("rooms", [])
        if r.get("type") == "Bedroom"
    )
    layout_type = current_spec.get("layout_type") or (f"{bedroom_count}BHK" if bedroom_count > 0 else "2BHK")

    # Get constraint requirements for this layout type
    constraint_requirements = None
    if _HAS_CONSTRAINT_ANALYZER and get_layout_requirements:
        try:
            constraint_requirements = get_layout_requirements(layout_type)
            ProcessingLogger.logger.info(
                f"CONSTRAINT_REQUIREMENTS - Layout: {layout_type}, "
                f"Min area: {constraint_requirements.get('min_total_area_sqm')} sqm, "
                f"Rooms: {constraint_requirements.get('room_count')}"
            )
        except Exception as e:
            ProcessingLogger.logger.warning(f"Failed to get constraint requirements: {e}")

    # If no boundary provided, calculate optimal dimensions
    if resolution is None:
        resolution = {}

    if not resolution.get("boundary_polygon") and not resolution.get("boundary_size"):
        if _HAS_CONSTRAINT_ANALYZER and calculate_optimal_dimensions:
            try:
                optimal = calculate_optimal_dimensions(layout_type)
                resolution["boundary_size"] = [optimal["width_m"], optimal["height_m"]]
                ProcessingLogger.logger.info(
                    f"AUTO_DIMENSIONS - {optimal['width_m']}m x {optimal['height_m']}m "
                    f"({optimal['area_sqm']} sqm, {optimal['efficiency_percent']}% efficient)"
                )
                warnings.append(
                    f"Auto-calculated dimensions: {optimal['width_m']}m x {optimal['height_m']}m "
                    f"for {layout_type} layout"
                )
            except Exception as e:
                ProcessingLogger.logger.warning(f"Failed to calculate optimal dimensions: {e}")

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

    room_sequence, inferred_room_types = _expand_room_types(current_spec, constraint_requirements)
    translated_adjacency, adjacency_warnings = _translate_adjacency(current_spec)
    warnings.extend(adjacency_warnings)

    if inferred_room_types:
        warnings.append(
            "Inferred standard layout rooms from "
            f"{layout_type}: {', '.join(inferred_room_types)}."
        )

    # Add constraint-based adjacency if not provided
    if not translated_adjacency and constraint_requirements:
        for adj in constraint_requirements.get("adjacency_requirements", []):
            translated_adjacency.append({
                "a": adj["source"],
                "b": adj["target"],
                "type": "prefer" if adj["relation"] == "near_to" else "must",
            })
        if translated_adjacency:
            ProcessingLogger.logger.info(
                f"AUTO_ADJACENCY - Added {len(translated_adjacency)} constraint-based adjacency rules"
            )

    base = {
        "occupancy": ALLOWED_BUILDING_TYPE,
        "building_type": ALLOWED_BUILDING_TYPE,
        "plot_type": current_spec.get("plot_type"),
        "layout_type": layout_type,
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
        "semantic_spec": deepcopy(current_spec.get("semantic_spec")),
        "room_program": deepcopy(current_spec.get("room_program")),
        "zoning_plan": deepcopy(current_spec.get("zoning_plan")),
    }

    # Include constraint metadata for model guidance
    if constraint_requirements:
        base["_constraint_metadata"] = {
            "min_total_area_sqm": constraint_requirements.get("min_total_area_sqm"),
            "recommended_area_sqm": constraint_requirements.get("recommended_area_sqm"),
            "plot_bucket": constraint_requirements.get("plot_bucket"),
            "spacing": constraint_requirements.get("spacing"),
        }

    # All backends need rooms with name and type fields for spec validation
    # Include minimum area requirements from constraints
    type_counts = {}
    expanded_rooms = []
    constraint_rooms = {r["type"]: r for r in (constraint_requirements or {}).get("rooms", [])}
    zoning_size_priors = (current_spec.get("zoning_plan") or {}).get("size_priors") or {}

    for room_type in room_sequence:
        type_counts[room_type] = type_counts.get(room_type, 0) + 1
        room_name = f"{room_type}_{type_counts[room_type]}"

        room_data = make_room(room_name, room_type)

        # Add minimum area from constraints
        if room_type in constraint_rooms:
            cr = constraint_rooms[room_type]
            room_data["min_area_sqm"] = cr.get("min_area_sqm", 9.5)
            room_data["min_width_m"] = cr.get("min_width_m", 2.4)
            room_data["recommended_area_sqm"] = cr.get("recommended_area_sqm", room_data["min_area_sqm"] * 1.3)

        size_prior = zoning_size_priors.get(room_name) or {}
        if size_prior.get("ideal_area_sqm") is not None:
            room_data["area"] = round(float(size_prior["ideal_area_sqm"]), 3)

        expanded_rooms.append(room_data)

    base["rooms"] = expanded_rooms
    zoning_plan = current_spec.get("zoning_plan") or {}
    if zoning_plan:
        planner_guidance = _planner_guidance_from_zoning(zoning_plan, expanded_rooms)
        if planner_guidance.get("room_order") or planner_guidance.get("spatial_hints"):
            base["planner_guidance"] = planner_guidance
            base["learned_spatial_hints"] = planner_guidance.get("spatial_hints", {})
            warnings.append(
                f"Applied zoning-driven placement order for {layout_type} to stabilize residential composition."
            )

    ProcessingLogger.logger.info(
        f"BACKEND_SPEC_BUILT - Layout: {layout_type}, Rooms: {len(expanded_rooms)}, "
        f"Area: {total_area:.1f} sqm, Boundary: {len(boundary_polygon)} points"
    )

    return base, warnings


def _expand_room_types(
    current_spec: Dict,
    constraint_requirements: Optional[Dict] = None,
) -> Tuple[List[str], List[str]]:
    room_program = current_spec.get("room_program") or {}
    program_rooms = room_program.get("rooms", []) if isinstance(room_program, dict) else []
    if program_rooms:
        expanded = [room.get("type") for room in program_rooms if room.get("type")]
        inferred_room_types: List[str] = []
        explicit_counts = {
            room.get("type"): int(room.get("count", 1))
            for room in current_spec.get("rooms", [])
            if room.get("type")
        }
        for room in program_rooms:
            room_type = room.get("type")
            if room_type and explicit_counts.get(room_type, 0) <= 0:
                inferred_room_types.append(room_type)
        return expanded, sorted(set(inferred_room_types))

    explicit_counts: Dict[str, int] = {}
    ordered_explicit_types: List[str] = []

    for room in current_spec.get("rooms", []):
        external_type = room.get("type", "")
        internal_type = EXTERNAL_TO_INTERNAL_ROOM.get(external_type)
        if internal_type is None:
            continue
        if internal_type not in ordered_explicit_types:
            ordered_explicit_types.append(internal_type)
        explicit_counts[internal_type] = explicit_counts.get(internal_type, 0) + max(int(room.get("count", 1)), 0)

    required_counts: Dict[str, int] = {}
    required_order: List[str] = []
    for room in (constraint_requirements or {}).get("rooms", []):
        internal_type = EXTERNAL_TO_INTERNAL_ROOM.get(room.get("type", ""), room.get("type"))
        if not internal_type:
            continue
        if internal_type not in required_order:
            required_order.append(internal_type)
        required_counts[internal_type] = required_counts.get(internal_type, 0) + 1

    final_order = required_order + [room_type for room_type in ordered_explicit_types if room_type not in required_order]
    final_counts: Dict[str, int] = {}
    inferred_room_types: List[str] = []

    for room_type in final_order:
        explicit = explicit_counts.get(room_type, 0)
        required = required_counts.get(room_type, 0)
        final_count = max(explicit, required)
        if final_count <= 0:
            continue
        final_counts[room_type] = final_count
        if required > explicit:
            inferred_room_types.append(room_type)

    expanded: List[str] = []
    for room_type in final_order:
        for _ in range(final_counts.get(room_type, 0)):
            expanded.append(room_type)

    return expanded, inferred_room_types


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
        "zoning_pattern": (current_spec.get("zoning_plan") or {}).get("layout_pattern"),
    }


def _planner_guidance_from_zoning(zoning_plan: Dict, expanded_rooms: List[Dict]) -> Dict:
    room_names = [room.get("name") for room in expanded_rooms if room.get("name")]
    zone_map = dict(zoning_plan.get("zone_map") or {})
    spatial_hints = {
        room_name: [round(float(point[0]), 4), round(float(point[1]), 4)]
        for room_name, point in (zoning_plan.get("spatial_hints") or {}).items()
        if room_name in room_names and isinstance(point, (list, tuple)) and len(point) == 2
    }
    room_order = [room_name for room_name in (zoning_plan.get("room_order") or []) if room_name in room_names]
    named_adjacency = [
        {
            "a": item.get("a"),
            "b": item.get("b"),
            "type": item.get("type", "prefer"),
            "score": float(item.get("score", 0.9)),
        }
        for item in (zoning_plan.get("named_adjacency") or [])
        if item.get("a") in room_names and item.get("b") in room_names
    ]
    size_priors = {
        room_name: dict(priors)
        for room_name, priors in (zoning_plan.get("size_priors") or {}).items()
        if room_name in room_names and isinstance(priors, dict)
    }
    return {
        "source": "zoning-plan",
        "layout_pattern": zoning_plan.get("layout_pattern"),
        "frontage_room": zoning_plan.get("frontage_room"),
        "spatial_hints": spatial_hints,
        "room_order": room_order or room_names,
        "room_zones": {name: zone_map.get(name, "private") for name in room_names},
        "adjacency_preferences": named_adjacency,
        "size_priors": size_priors,
    }


def _entrance_point_from_side(boundary_polygon: List[Tuple[float, float]], entrance_side: str) -> Tuple[float, float]:
    xs = [point[0] for point in boundary_polygon]
    ys = [point[1] for point in boundary_polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_mid = round((x_min + x_max) / 2.0, 4)
    y_mid = round((y_min + y_max) / 2.0, 4)

    if entrance_side == "North":
        return (x_mid, y_min)
    if entrance_side == "South":
        return (x_mid, y_max)
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
