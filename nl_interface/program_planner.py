from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any, Dict, List, Optional, Tuple

from config.constants import LayoutStandards
from nl_interface.constants import DEFAULT_PRIVACY_BY_ROOM
from nl_interface.contracts import ProgramRoom, RoomProgram, SemanticSpec, ZoningPlan


CANONICAL_BHK_COUNTS: Dict[str, Dict[str, int]] = {
    "1BHK": {"Bedroom": 1, "LivingRoom": 1, "Kitchen": 1, "Bathroom": 1},
    "2BHK": {"Bedroom": 2, "LivingRoom": 1, "Kitchen": 1, "Bathroom": 2},
    "3BHK": {"Bedroom": 3, "LivingRoom": 1, "Kitchen": 1, "Bathroom": 2},
    "4BHK": {"Bedroom": 4, "LivingRoom": 1, "Kitchen": 1, "Bathroom": 2},
}


ROLE_ANCHORS_NORTH: Dict[str, List[Tuple[float, float]]] = {
    "public_anchor": [(0.50, 0.22), (0.36, 0.24), (0.64, 0.24)],
    "service_anchor": [(0.24, 0.24), (0.24, 0.54), (0.76, 0.24)],
    "master_bedroom": [(0.72, 0.76)],
    "secondary_bedroom": [(0.30, 0.76), (0.50, 0.88), (0.18, 0.72), (0.82, 0.72)],
    "attached_bathroom": [(0.78, 0.58)],
    "common_bathroom": [(0.24, 0.58), (0.50, 0.60)],
    "service_support": [(0.78, 0.54), (0.22, 0.40)],
}

ROOM_ORDER_PRIORITY: Dict[str, int] = {
    "public_anchor": 0,
    "service_anchor": 1,
    "master_bedroom": 2,
    "secondary_bedroom": 3,
    "attached_bathroom": 4,
    "common_bathroom": 5,
    "service_support": 6,
}


def infer_layout_type_from_counts(room_counts: Dict[str, int]) -> Optional[str]:
    bedrooms = int(room_counts.get("Bedroom", 0) or 0)
    if bedrooms <= 0:
        return None
    if bedrooms == 1:
        return "1BHK"
    if bedrooms == 2:
        return "2BHK"
    if bedrooms == 3:
        return "3BHK"
    if bedrooms >= 4:
        return "4BHK"
    return None


def _detect_bhk_shorthand(user_prompt: Optional[str], explicit_layout_type: Optional[str]) -> Optional[str]:
    if explicit_layout_type in CANONICAL_BHK_COUNTS and user_prompt is None:
        return explicit_layout_type
    prompt = str(user_prompt or "")
    match = re.search(r"\b([1-4])\s*bhk\b", prompt, flags=re.IGNORECASE)
    if match:
        return f"{int(match.group(1))}BHK"
    if explicit_layout_type in CANONICAL_BHK_COUNTS and explicit_layout_type.lower() in prompt.lower():
        return explicit_layout_type
    return None


def build_semantic_spec(
    spec: Dict[str, Any],
    resolution: Optional[Dict[str, Any]] = None,
    user_prompt: Optional[str] = None,
) -> SemanticSpec:
    resolution = resolution or {}
    room_counts = {
        str(room.get("type")): int(room.get("count", 1))
        for room in spec.get("rooms", [])
        if room.get("type")
    }
    explicit_layout_type = spec.get("layout_type")
    layout_type = explicit_layout_type or infer_layout_type_from_counts(room_counts)
    shorthand = _detect_bhk_shorthand(user_prompt, explicit_layout_type)
    boundary_size = None
    if resolution.get("boundary_size"):
        boundary_size = [round(float(resolution["boundary_size"][0]), 3), round(float(resolution["boundary_size"][1]), 3)]

    unsupported_requests = []
    for err in spec.get("_validation_errors", []) or []:
        if "Unsupported room type" in str(err) or "Unsupported room type" in str(err):
            unsupported_requests.append(str(err))

    unresolved_fields: List[str] = []
    if not layout_type and not room_counts:
        unresolved_fields.append("room_program")

    assumptions: List[str] = []
    if shorthand:
        assumptions.append(f"Interpreted '{shorthand}' as a canonical residential shorthand program.")
    elif layout_type and room_counts:
        assumptions.append(f"Inferred a {layout_type}-compatible residential program from your explicit room counts.")
    if not resolution.get("total_area") and boundary_size:
        assumptions.append("Working plot size is being used as the main sizing envelope.")

    adjacency_prefs = []
    for source, target, relation in spec.get("preferences", {}).get("adjacency", []):
        adjacency_prefs.append({"source": source, "target": target, "relation": relation})

    semantic = SemanticSpec(
        building_type=str(spec.get("building_type") or "Residential"),
        layout_type=layout_type,
        shorthand=shorthand,
        requested_rooms=room_counts,
        plot_type=spec.get("plot_type"),
        entrance_side=spec.get("entrance_side"),
        total_area_sqm=(round(float(resolution["total_area"]), 3) if resolution.get("total_area") is not None else None),
        boundary_size_m=boundary_size,
        style_hints=list(spec.get("style_hints", []) or []),
        adjacency_preferences=adjacency_prefs,
        privacy_preferences=dict(spec.get("preferences", {}).get("privacy", {}) or {}),
        unsupported_requests=unsupported_requests,
        unresolved_fields=unresolved_fields,
        assumptions_used=assumptions,
        user_prompt=user_prompt,
    )
    return semantic


def build_room_program(semantic_spec: SemanticSpec) -> RoomProgram:
    requested_counts = dict(semantic_spec.requested_rooms)
    room_counts = dict(requested_counts)
    assumptions = list(semantic_spec.assumptions_used)
    deferred_semantics: List[str] = []
    canonical = bool(semantic_spec.shorthand in CANONICAL_BHK_COUNTS)
    supported_scope = "custom_residential"

    if canonical:
        canonical_counts = CANONICAL_BHK_COUNTS[semantic_spec.shorthand]
        room_counts = dict(canonical_counts)
        assumptions.append(
            f"Expanded {semantic_spec.shorthand} into the canonical room program used by the generation core."
        )
        for room_type, count in requested_counts.items():
            canonical_count = canonical_counts.get(room_type, 0)
            if canonical_count <= 0:
                room_counts[room_type] = count
                assumptions.append(f"Kept your explicit {room_type} request in addition to the shorthand program.")
                continue
            if count > canonical_count:
                room_counts[room_type] = count
                assumptions.append(f"Increased {room_type} count to {count} based on your explicit request.")
            elif count < canonical_count:
                assumptions.append(
                    f"Kept the standard {canonical_count} {room_type}(s) for {semantic_spec.shorthand}."
                )
        supported_scope = "canonical_plus_custom" if any(
            requested_counts.get(room_type, 0) != canonical_counts.get(room_type, 0)
            for room_type in set(requested_counts) | set(canonical_counts)
        ) else "canonical_residential"

    rooms: List[ProgramRoom] = []
    type_counts = defaultdict(int)

    bedroom_count = room_counts.get("Bedroom", 0)
    bathroom_count = room_counts.get("Bathroom", 0)

    for room_type, count in room_counts.items():
        for _ in range(int(count)):
            type_counts[room_type] += 1
            room_name = f"{room_type}_{type_counts[room_type]}"
            zone = DEFAULT_PRIVACY_BY_ROOM.get(room_type, "service")

            if room_type == "Bedroom":
                role = "master_bedroom" if type_counts[room_type] == 1 and bedroom_count >= 2 else "secondary_bedroom"
                if role == "master_bedroom":
                    deferred_semantics.append("Bedroom_1 is treated as the preferred master bedroom.")
            elif room_type == "Bathroom":
                role = "attached_bathroom" if type_counts[room_type] == 1 and bathroom_count >= 2 else "common_bathroom"
                if role == "attached_bathroom":
                    deferred_semantics.append("Bathroom_1 is treated as the preferred attached bathroom.")
            elif room_type in {"LivingRoom", "DrawingRoom"}:
                role = "public_anchor"
            elif room_type == "DiningRoom":
                role = "public_support"
            elif room_type == "Kitchen":
                role = "service_anchor"
            elif room_type in {"Garage", "Store"}:
                role = "service_support"
            else:
                role = f"{zone}_room"

            rooms.append(
                ProgramRoom(
                    name=room_name,
                    type=room_type,
                    zone=zone,
                    semantic_role=role,
                    required=True,
                    metadata={"canonical_layout": canonical},
                )
            )

    return RoomProgram(
        layout_type=semantic_spec.layout_type,
        canonical=canonical,
        rooms=rooms,
        required_counts=room_counts,
        assumptions_used=assumptions,
        deferred_semantics=deferred_semantics,
        supported_scope=supported_scope,
    )


def build_zoning_plan(
    room_program: RoomProgram,
    semantic_spec: SemanticSpec,
    resolution: Optional[Dict[str, Any]] = None,
    source_spec: Optional[Dict[str, Any]] = None,
) -> ZoningPlan:
    resolution = resolution or {}
    source_spec = source_spec or {}
    layout_pattern = _layout_pattern(room_program)
    entrance_zone = "public"
    frontage_room = next((room.name for room in room_program.rooms if room.semantic_role == "public_anchor"), None)

    spatial_hints: Dict[str, List[float]] = {}
    zone_map: Dict[str, str] = {}
    role_counts: Dict[str, int] = defaultdict(int)
    heuristics = [
        "Public rooms should stay close to the entrance frontage.",
        "Bedrooms should remain deeper in the private zone when possible.",
        "Kitchen should remain close to the living room or shared public space.",
        "Bathrooms should stay close to bedrooms and accessible without dominating the entrance view.",
    ]

    size_priors: Dict[str, Dict[str, float]] = {}
    privacy_overrides = dict(semantic_spec.privacy_preferences or {})
    adjacency = _named_adjacency(room_program.rooms, semantic_spec.adjacency_preferences)

    for room in room_program.rooms:
        zone_map[room.name] = privacy_overrides.get(room.type, room.zone)
        role_counts[room.semantic_role] += 1
        base_point = _role_anchor(room.semantic_role, role_counts[room.semantic_role] - 1)
        rotated = _rotate_point(base_point, semantic_spec.entrance_side)
        spatial_hints[room.name] = [round(rotated[0], 4), round(rotated[1], 4)]

        area_standard = LayoutStandards.ROOM_AREA_STANDARDS.get(room.type, {"ideal": 8.0, "min": 4.0, "max": 12.0})
        size_priors[room.name] = {
            "ideal_area_sqm": float(area_standard.get("ideal", area_standard.get("min", 8.0))),
            "min_area_sqm": float(area_standard.get("min", 4.0)),
            "max_area_sqm": float(area_standard.get("max", area_standard.get("ideal", 12.0))),
        }

    _apply_room_position_preferences(
        spatial_hints,
        source_spec.get("room_position_preferences") or {},
        entrance_side=semantic_spec.entrance_side,
    )
    _apply_room_swaps(
        spatial_hints,
        source_spec.get("room_swaps") or [],
    )
    _apply_room_size_preferences(
        size_priors,
        source_spec.get("room_size_preferences") or {},
    )

    room_lookup = {room.name: room for room in room_program.rooms}
    zone_priority = {"public": 0, "service": 1, "private": 2}
    room_order = [
        room_name
        for room_name, _ in sorted(
            spatial_hints.items(),
            key=lambda item: (
                zone_priority.get(zone_map.get(item[0], "private"), 3),
                ROOM_ORDER_PRIORITY.get(room_lookup[item[0]].semantic_role, 9),
                _front_score(item[1], semantic_spec.entrance_side),
                item[0],
            ),
        )
    ]

    assumptions = list(room_program.assumptions_used)
    assumptions.append(f"Using a '{layout_pattern}' zoning pattern for the initial composition.")

    return ZoningPlan(
        layout_pattern=layout_pattern,
        entrance_frontage_zone=entrance_zone,
        frontage_room=frontage_room,
        named_adjacency=adjacency,
        zone_map=zone_map,
        spatial_hints=spatial_hints,
        room_order=room_order,
        size_priors=size_priors,
        heuristics=heuristics,
        assumptions_used=assumptions,
    )


def enrich_spec_with_planning(
    spec: Dict[str, Any],
    resolution: Optional[Dict[str, Any]] = None,
    user_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    enriched = dict(spec)
    semantic_spec = build_semantic_spec(spec, resolution=resolution, user_prompt=user_prompt)
    room_program = build_room_program(semantic_spec)
    zoning_plan = build_zoning_plan(room_program, semantic_spec, resolution=resolution, source_spec=spec)
    enriched["semantic_spec"] = semantic_spec.to_dict()
    enriched["room_program"] = room_program.to_dict()
    enriched["zoning_plan"] = zoning_plan.to_dict()
    return enriched


def summarize_room_program(room_program: Optional[Dict[str, Any]]) -> Optional[str]:
    if not room_program:
        return None
    counts = Counter(room.get("type") for room in room_program.get("rooms", []))
    if not counts:
        return None
    ordered = ", ".join(f"{count} {room_type}" for room_type, count in sorted(counts.items()))
    layout_type = room_program.get("layout_type") or "custom program"
    return f"{layout_type} program: {ordered}."


def summarize_zoning_plan(zoning_plan: Optional[Dict[str, Any]]) -> Optional[str]:
    if not zoning_plan:
        return None
    frontage = zoning_plan.get("frontage_room") or "public room"
    pattern = zoning_plan.get("layout_pattern") or "balanced"
    heuristics = zoning_plan.get("heuristics") or []
    summary = f"Pattern: {pattern}. Frontage anchor: {frontage}."
    if heuristics:
        summary += " " + " ".join(heuristics[:2])
    return summary


def _layout_pattern(room_program: RoomProgram) -> str:
    bedroom_count = sum(1 for room in room_program.rooms if room.type == "Bedroom")
    if bedroom_count >= 3:
        return "zonal_split"
    if bedroom_count == 2:
        return "public_front_private_rear"
    return "compact_frontage"


def _named_adjacency(rooms: List[ProgramRoom], adjacency_preferences: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    bedrooms = [room.name for room in rooms if room.type == "Bedroom"]
    bathrooms = [room.name for room in rooms if room.type == "Bathroom"]
    living = [room.name for room in rooms if room.type in {"LivingRoom", "DrawingRoom"}]
    kitchens = [room.name for room in rooms if room.type == "Kitchen"]
    dining = [room.name for room in rooms if room.type == "DiningRoom"]
    garage = [room.name for room in rooms if room.type == "Garage"]
    store = [room.name for room in rooms if room.type == "Store"]
    adjacency: List[Dict[str, Any]] = []

    if living and kitchens:
        adjacency.append({"a": kitchens[0], "b": living[0], "type": "prefer", "score": 1.0})
    if dining and living:
        adjacency.append({"a": dining[0], "b": living[0], "type": "prefer", "score": 0.95})
    if dining and kitchens:
        adjacency.append({"a": dining[0], "b": kitchens[0], "type": "prefer", "score": 0.9})
    if garage and store:
        adjacency.append({"a": garage[0], "b": store[0], "type": "prefer", "score": 0.8})
    for idx, bath_name in enumerate(bathrooms):
        target = bedrooms[min(idx, len(bedrooms) - 1)] if bedrooms else None
        if target:
            adjacency.append({"a": bath_name, "b": target, "type": "prefer", "score": 0.95 if idx == 0 else 0.85})

    name_by_type: Dict[str, List[str]] = defaultdict(list)
    room_name_map: Dict[str, str] = {}
    for room in rooms:
        name_by_type[room.type].append(room.name)
        room_name_map[room.name] = room.name

    for pref in adjacency_preferences or []:
        source_type = str(pref.get("source") or "").strip()
        target_type = str(pref.get("target") or "").strip()
        relation = str(pref.get("relation") or "near_to").strip().lower()
        score = 1.0 if relation in {"near_to", "adjacent_to"} else 0.75
        source_names = _resolve_room_names(room_name_map, source_type) or name_by_type.get(source_type, [])
        target_names = _resolve_room_names(room_name_map, target_type) or name_by_type.get(target_type, [])
        for source_name in source_names[:1]:
            for target_name in target_names[:1]:
                if source_name == target_name:
                    continue
                relation_type = "prefer" if relation in {"near_to", "adjacent_to"} else "avoid"
                candidate = {"a": source_name, "b": target_name, "type": relation_type, "score": score}
                if candidate not in adjacency:
                    adjacency.append(candidate)

    return adjacency


def _resolve_room_names(mapping: Dict[str, Any], room_key: str) -> List[str]:
    if room_key in mapping:
        return [room_key]
    prefix = f"{room_key}_"
    return [name for name in mapping if name.startswith(prefix)]


def _apply_room_position_preferences(
    spatial_hints: Dict[str, List[float]],
    room_position_preferences: Dict[str, Dict[str, Any]],
    *,
    entrance_side: Optional[str],
) -> None:
    del entrance_side  # Reserved for future directional interpretation.
    direction_deltas = {
        "left": (-0.12, 0.0),
        "right": (0.12, 0.0),
        "up": (0.0, -0.12),
        "down": (0.0, 0.12),
        "north": (0.0, -0.12),
        "south": (0.0, 0.12),
        "east": (0.12, 0.0),
        "west": (-0.12, 0.0),
    }
    for room_key, pref in room_position_preferences.items():
        direction = str((pref or {}).get("direction") or "").strip().lower()
        dx, dy = direction_deltas.get(direction, (0.0, 0.0))
        if dx == 0.0 and dy == 0.0:
            continue
        for room_name in _resolve_room_names(spatial_hints, room_key):
            point = spatial_hints.get(room_name)
            if not point:
                continue
            spatial_hints[room_name] = [
                round(min(0.92, max(0.08, float(point[0]) + dx)), 4),
                round(min(0.92, max(0.08, float(point[1]) + dy)), 4),
            ]


def _apply_room_swaps(
    spatial_hints: Dict[str, List[float]],
    room_swaps: List[Tuple[str, str]],
) -> None:
    for room_a, room_b in room_swaps:
        names_a = _resolve_room_names(spatial_hints, str(room_a))
        names_b = _resolve_room_names(spatial_hints, str(room_b))
        if not names_a or not names_b:
            continue
        spatial_hints[names_a[0]], spatial_hints[names_b[0]] = spatial_hints[names_b[0]], spatial_hints[names_a[0]]


def _apply_room_size_preferences(
    size_priors: Dict[str, Dict[str, float]],
    room_size_preferences: Dict[str, str],
) -> None:
    scale_map = {
        "larger": 1.18,
        "bigger": 1.18,
        "wider": 1.12,
        "smaller": 0.88,
        "narrower": 0.9,
    }
    for room_key, size_change in room_size_preferences.items():
        scale = scale_map.get(str(size_change).strip().lower())
        if scale is None:
            continue
        for room_name in _resolve_room_names(size_priors, room_key):
            priors = size_priors.get(room_name)
            if not priors:
                continue
            ideal = float(priors.get("ideal_area_sqm", 8.0))
            min_area = float(priors.get("min_area_sqm", 4.0))
            max_area = float(priors.get("max_area_sqm", max(ideal, min_area)))
            adjusted = max(min_area, min(max_area, ideal * scale))
            priors["ideal_area_sqm"] = round(adjusted, 3)


def _role_anchor(role: str, index: int) -> Tuple[float, float]:
    if role in ROLE_ANCHORS_NORTH:
        anchors = ROLE_ANCHORS_NORTH[role]
    elif role.startswith("public"):
        anchors = ROLE_ANCHORS_NORTH["public_anchor"]
    elif role.startswith("service"):
        anchors = ROLE_ANCHORS_NORTH["service_support"]
    else:
        anchors = ROLE_ANCHORS_NORTH["secondary_bedroom"]
    return anchors[index % len(anchors)]


def _rotate_point(point: Tuple[float, float], entrance_side: Optional[str]) -> Tuple[float, float]:
    x, y = point
    side = (entrance_side or "North").lower()
    if side == "north":
        return x, y
    if side == "south":
        return x, 1.0 - y
    if side == "east":
        return 1.0 - y, x
    if side == "west":
        return y, 1.0 - x
    return x, y


def _front_score(point: List[float], entrance_side: Optional[str]) -> float:
    x, y = float(point[0]), float(point[1])
    side = (entrance_side or "North").lower()
    if side == "north":
        return y
    if side == "south":
        return 1.0 - y
    if side == "east":
        return 1.0 - x
    if side == "west":
        return x
    return y
