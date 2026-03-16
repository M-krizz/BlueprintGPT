"""Service layer for the additive natural-language residential spec interface."""

from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Dict, List, Optional

from nl_interface.adapter import build_backend_spec, route_backend, validate_resolution
from nl_interface.constants import (
    ALLOWED_BUILDING_TYPE,
    ALLOWED_ENTRANCE_SIDES,
    ALLOWED_PLOT_TYPES,
    ALLOWED_PRIVACY_ZONES,
    ALLOWED_RELATIONSHIPS,
    ALLOWED_ROOM_TYPES,
    DEFAULT_PRIVACY_BY_ROOM,
    DEFAULT_WEIGHTS,
    NUMBER_WORDS,
    RELATION_PHRASES,
    ROOM_LABELS,
    STYLE_HINTS,
    UNSUPPORTED_RELATION_PHRASES,
    UNSUPPORTED_ROOM_LABELS,
    load_plot_capacity_config,
)


def blank_current_spec() -> Dict:
    return {
        "building_type": ALLOWED_BUILDING_TYPE,
        "plot_type": None,
        "entrance_side": None,
        "rooms": [],
        "preferences": {
            "adjacency": [],
            "privacy": {},
            "minimize_corridor": False,
        },
        "weights": dict(DEFAULT_WEIGHTS),
    }


def process_user_request(
    user_text: str,
    current_spec: Optional[Dict] = None,
    resolution: Optional[Dict] = None,
) -> Dict:
    working = normalize_current_spec(current_spec or {})
    extracted = _extract_from_text(user_text)
    merged = _apply_extracted(working, extracted)
    normalized = normalize_current_spec(merged)

    backend_target = route_backend(normalized)
    validation_errors = list(normalized.pop("_validation_errors", []))
    feasibility_warnings = list(normalized.pop("_feasibility_warnings", []))
    missing_fields = list(normalized.pop("_missing_fields", []))
    interpretation_notes = list(normalized.pop("_interpretation_notes", []))

    backend_spec = None
    backend_translation_warnings: List[str] = []
    if resolution:
        # Canvas geometry fulfills plot and entrance if provided
        if "plot_type" in missing_fields and (resolution.get("boundary_polygon") or resolution.get("boundary_size")):
            normalized["plot_type"] = "Custom"
            missing_fields.remove("plot_type")
            
        if "entrance_side" in missing_fields and resolution.get("entrance_point"):
            normalized["entrance_side"] = "Custom"
            missing_fields.remove("entrance_side")

        normalized_resolution, resolution_missing = validate_resolution(resolution)
        if resolution_missing:
            missing_fields.extend(field for field in resolution_missing if field not in missing_fields)
        else:
            backend_spec, backend_translation_warnings = build_backend_spec(normalized, normalized_resolution)
    else:
        if "boundary_polygon" not in missing_fields:
            missing_fields.append("boundary_polygon")

    backend_ready = bool(
        backend_target
        and not missing_fields
        and not validation_errors
        and backend_spec is not None
    )

    assistant_text = _build_assistant_text(
        current_spec=normalized,
        missing_fields=missing_fields,
        validation_errors=validation_errors,
        feasibility_warnings=feasibility_warnings,
        interpretation_notes=interpretation_notes,
        backend_target=backend_target,
        backend_ready=backend_ready,
        backend_translation_warnings=backend_translation_warnings,
    )

    return {
        "assistant_text": assistant_text,
        "current_spec": normalized,
        "missing_fields": missing_fields,
        "validation_errors": validation_errors,
        "feasibility_warnings": feasibility_warnings,
        "backend_ready": backend_ready,
        "backend_target": backend_target,
        "backend_spec": backend_spec if backend_ready else None,
        "backend_translation_warnings": backend_translation_warnings,
    }


def normalize_current_spec(spec: Dict) -> Dict:
    normalized = blank_current_spec()
    normalized.update(
        {
            "plot_type": spec.get("plot_type"),
            "entrance_side": spec.get("entrance_side"),
        }
    )

    normalized["preferences"] = copy.deepcopy(normalized["preferences"])
    normalized["preferences"].update(copy.deepcopy(spec.get("preferences", {})))
    normalized["weights"] = copy.deepcopy(spec.get("weights", DEFAULT_WEIGHTS))

    errors = list(spec.get("_validation_errors", []))
    missing_fields: List[str] = []

    plot_type = _canonical_plot_type(normalized.get("plot_type"))
    if normalized.get("plot_type") is not None and plot_type is None:
        errors.append(f"Unsupported plot_type '{normalized.get('plot_type')}'.")
    normalized["plot_type"] = plot_type
    if plot_type is None:
        missing_fields.append("plot_type")

    entrance_side = _canonical_entrance_side(normalized.get("entrance_side"))
    if normalized.get("entrance_side") is not None and entrance_side is None:
        errors.append(f"Unsupported entrance_side '{normalized.get('entrance_side')}'.")
    normalized["entrance_side"] = entrance_side
    if entrance_side is None:
        missing_fields.append("entrance_side")

    normalized["rooms"] = _normalize_rooms(spec.get("rooms", []), errors)
    normalized["preferences"]["adjacency"] = _normalize_adjacency(
        normalized["preferences"].get("adjacency", []),
        errors,
    )
    normalized["preferences"]["privacy"] = _normalize_privacy(
        normalized["preferences"].get("privacy", {}),
        normalized["rooms"],
        errors,
    )
    normalized["preferences"]["minimize_corridor"] = bool(
        normalized["preferences"].get("minimize_corridor", False)
    )
    normalized["weights"] = _normalize_weights(normalized["weights"])

    feasibility_warnings = _build_feasibility_warnings(normalized)
    normalized["_missing_fields"] = missing_fields
    normalized["_validation_errors"] = errors
    normalized["_feasibility_warnings"] = feasibility_warnings
    normalized["_interpretation_notes"] = list(spec.get("_interpretation_notes", []))
    return normalized


def _extract_from_text(user_text: str) -> Dict:
    text = " ".join((user_text or "").strip().split())
    lowered = text.lower()

    extracted = {
        "plot_type": None,
        "entrance_side": None,
        "rooms": [],
        "preferences": {
            "adjacency": [],
            "privacy": {},
            "minimize_corridor": None,
        },
        "weights": None,
        "validation_errors": [],
        "interpretation_notes": [],
    }

    extracted["plot_type"] = _extract_plot_type(lowered)
    extracted["entrance_side"] = _extract_entrance_side(lowered)
    extracted["rooms"] = _extract_rooms(lowered)
    extracted["preferences"]["adjacency"] = _extract_adjacency(lowered)
    extracted["preferences"]["privacy"] = _extract_privacy(lowered)
    extracted["preferences"]["minimize_corridor"] = _extract_minimize_corridor(lowered)

    weight_result = _extract_weight_preferences(lowered)
    extracted["weights"] = weight_result["weights"]
    extracted["interpretation_notes"].extend(weight_result["notes"])

    extracted["validation_errors"].extend(_find_unsupported_room_labels(lowered))
    extracted["validation_errors"].extend(_find_unsupported_relations(lowered))
    return extracted


def _apply_extracted(current_spec: Dict, extracted: Dict) -> Dict:
    merged = copy.deepcopy(current_spec)
    room_counter = Counter({room["type"]: int(room.get("count", 1)) for room in merged.get("rooms", [])})

    if extracted.get("plot_type"):
        merged["plot_type"] = extracted["plot_type"]
    if extracted.get("entrance_side"):
        merged["entrance_side"] = extracted["entrance_side"]

    for room in extracted.get("rooms", []):
        room_counter[room["type"]] = int(room["count"])

    ordered_rooms = []
    for room_type in ALLOWED_ROOM_TYPES:
        count = room_counter.get(room_type, 0)
        if count > 0:
            ordered_rooms.append({"type": room_type, "count": count})
    merged["rooms"] = ordered_rooms

    merged.setdefault("preferences", {})

    adjacency = list(merged.get("preferences", {}).get("adjacency", []))
    for triple in extracted.get("preferences", {}).get("adjacency", []):
        if triple not in adjacency:
            adjacency.append(triple)
    merged["preferences"]["adjacency"] = adjacency

    privacy = dict(merged.get("preferences", {}).get("privacy", {}))
    privacy.update(extracted.get("preferences", {}).get("privacy", {}))
    merged["preferences"]["privacy"] = privacy

    minimize_corridor = extracted.get("preferences", {}).get("minimize_corridor")
    if minimize_corridor is not None:
        merged["preferences"]["minimize_corridor"] = minimize_corridor

    if extracted.get("weights"):
        merged["weights"] = extracted["weights"]

    merged.setdefault("_validation_errors", [])
    merged["_validation_errors"] = list(merged.get("_validation_errors", [])) + extracted.get("validation_errors", [])
    merged.setdefault("_interpretation_notes", [])
    merged["_interpretation_notes"] = list(merged.get("_interpretation_notes", [])) + extracted.get("interpretation_notes", [])
    return merged


def _extract_plot_type(text: str) -> Optional[str]:
    if "5 marla" in text or "5marla" in text:
        return "5Marla"
    if "10 marla" in text or "10marla" in text:
        return "10Marla"
    if "20 marla" in text or "20marla" in text:
        return "20Marla"
    if "custom plot" in text or "custom lot" in text:
        return "Custom"
    return None


def _extract_entrance_side(text: str) -> Optional[str]:
    for side in ALLOWED_ENTRANCE_SIDES:
        side_lower = side.lower()
        patterns = (
            f"{side_lower} entrance",
            f"entrance on the {side_lower}",
            f"entrance from the {side_lower}",
            f"{side_lower}-facing entrance",
            f"{side_lower} side entrance",
        )
        if any(pattern in text for pattern in patterns):
            return side
    return None


def _extract_rooms(text: str) -> List[Dict]:
    room_counts = Counter()
    room_pattern = "|".join(
        sorted(
            {re.escape(label) for labels in ROOM_LABELS.values() for label in labels},
            key=len,
            reverse=True,
        )
    )
    count_pattern = "|".join(sorted(NUMBER_WORDS.keys(), key=len, reverse=True))
    regex = re.compile(rf"\b(?P<count>\d+|{count_pattern})\s+(?P<room>{room_pattern})\b")

    for match in regex.finditer(text):
        room_type = _canonical_room_label(match.group("room"))
        if room_type is None:
            continue
        room_counts[room_type] += _to_count(match.group("count"))

    article_regex = re.compile(rf"\b(?:a|an)\s+(?P<room>{room_pattern})\b")
    for match in article_regex.finditer(text):
        room_type = _canonical_room_label(match.group("room"))
        if room_type is None or room_type in room_counts:
            continue
        room_counts[room_type] += 1

    return [
        {"type": room_type, "count": room_counts[room_type]}
        for room_type in ALLOWED_ROOM_TYPES
        if room_counts.get(room_type, 0) > 0
    ]


def _extract_adjacency(text: str) -> List[List[str]]:
    room_pattern = "|".join(
        sorted(
            {re.escape(label) for labels in ROOM_LABELS.values() for label in labels},
            key=len,
            reverse=True,
        )
    )
    triples: List[List[str]] = []
    for relation, phrases in RELATION_PHRASES.items():
        phrase_pattern = "|".join(sorted((re.escape(phrase) for phrase in phrases), key=len, reverse=True))
        regex = re.compile(
            rf"\b(?P<a>{room_pattern})\b\s+(?:should be\s+|should stay\s+|is\s+|be\s+)?"
            rf"(?P<relation>{phrase_pattern})\s+\b(?P<b>{room_pattern})\b"
        )
        for match in regex.finditer(text):
            source = _canonical_room_label(match.group("a"))
            target = _canonical_room_label(match.group("b"))
            if source and target:
                triple = [source, target, relation]
                if triple not in triples:
                    triples.append(triple)
    return triples


def _extract_privacy(text: str) -> Dict[str, str]:
    room_pattern = "|".join(
        sorted(
            {re.escape(label) for labels in ROOM_LABELS.values() for label in labels},
            key=len,
            reverse=True,
        )
    )
    privacy_map: Dict[str, str] = {}
    regex = re.compile(rf"\b(?:keep|make)\s+(?P<room>{room_pattern})\s+(?P<zone>public|service|private)\b")
    for match in regex.finditer(text):
        room_type = _canonical_room_label(match.group("room"))
        zone = match.group("zone")
        if room_type:
            privacy_map[room_type] = zone
    return privacy_map


def _extract_minimize_corridor(text: str) -> Optional[bool]:
    if any(phrase in text for phrase in ("minimize corridor", "reduce corridor", "less corridor", "fewer corridors", "minimal corridor")):
        return True
    return None


def _extract_weight_preferences(text: str) -> Dict:
    weights = dict(DEFAULT_WEIGHTS)
    notes: List[str] = []
    matched = False
    for hint in STYLE_HINTS:
        if any(phrase in text for phrase in hint["phrases"]):
            matched = True
            for key, delta in hint["delta"].items():
                weights[key] = max(0.0, weights.get(key, 0.0) + delta)
            notes.append(hint["note"])
    if not matched:
        return {"weights": None, "notes": notes}
    return {"weights": _normalize_weights(weights), "notes": notes}


def _find_unsupported_room_labels(text: str) -> List[str]:
    errors = []
    for phrase, canonical in UNSUPPORTED_ROOM_LABELS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            errors.append(
                f"Unsupported room type '{canonical}' in user-facing ontology. Use one of: {', '.join(ALLOWED_ROOM_TYPES)}."
            )
    return errors


def _find_unsupported_relations(text: str) -> List[str]:
    errors = []
    for phrase in UNSUPPORTED_RELATION_PHRASES:
        if phrase in text:
            errors.append(
                f"Unsupported relationship phrase '{phrase}'. Use only: {', '.join(ALLOWED_RELATIONSHIPS)}."
            )
    return errors


def _normalize_rooms(rooms: List[Dict], errors: List[str]) -> List[Dict]:
    counts = Counter()
    for room in rooms:
        room_type = room.get("type")
        if room_type not in ALLOWED_ROOM_TYPES:
            errors.append(f"Unsupported room type '{room_type}'.")
            continue
        try:
            count = int(room.get("count", 1))
        except Exception:
            errors.append(f"Room count for '{room_type}' must be numeric.")
            continue
        if count <= 0:
            errors.append(f"Room count for '{room_type}' must be greater than zero.")
            continue
        counts[room_type] += count
    return [
        {"type": room_type, "count": counts[room_type]}
        for room_type in ALLOWED_ROOM_TYPES
        if counts.get(room_type, 0) > 0
    ]


def _normalize_adjacency(adjacency: List, errors: List[str]) -> List[List[str]]:
    normalized = []
    for triple in adjacency:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            errors.append("Adjacency entries must be typed triples [RoomA, RoomB, relationship].")
            continue
        source, target, relation = triple
        if source not in ALLOWED_ROOM_TYPES or target not in ALLOWED_ROOM_TYPES:
            errors.append("Adjacency room types must use the allowed ontology vocabulary.")
            continue
        if relation not in ALLOWED_RELATIONSHIPS:
            errors.append(f"Unsupported relationship '{relation}'.")
            continue
        entry = [source, target, relation]
        if entry not in normalized:
            normalized.append(entry)
    return normalized


def _normalize_privacy(privacy: Dict, rooms: List[Dict], errors: List[str]) -> Dict[str, str]:
    normalized = {}
    present_types = {room["type"] for room in rooms}
    for room_type, zone in privacy.items():
        if room_type not in ALLOWED_ROOM_TYPES:
            errors.append(f"Unsupported privacy room type '{room_type}'.")
            continue
        if zone not in ALLOWED_PRIVACY_ZONES:
            errors.append(f"Unsupported privacy zone '{zone}' for '{room_type}'.")
            continue
        normalized[room_type] = zone

    for room_type in present_types:
        normalized.setdefault(room_type, DEFAULT_PRIVACY_BY_ROOM[room_type])
    return normalized


def _normalize_weights(weights: Dict) -> Dict[str, float]:
    normalized = {}
    for key in ("privacy", "compactness", "corridor"):
        try:
            normalized[key] = max(0.0, float(weights.get(key, DEFAULT_WEIGHTS[key])))
        except Exception:
            normalized[key] = DEFAULT_WEIGHTS[key]

    total = sum(normalized.values())
    if total <= 0:
        normalized = dict(DEFAULT_WEIGHTS)
        total = sum(normalized.values())

    return {
        key: round(value / total, 4)
        for key, value in normalized.items()
    }


def _build_feasibility_warnings(current_spec: Dict) -> List[str]:
    plot_type = current_spec.get("plot_type")
    if plot_type not in ALLOWED_PLOT_TYPES:
        return []

    config = load_plot_capacity_config().get(plot_type, {})
    if not config:
        return []

    total_rooms = sum(int(room.get("count", 1)) for room in current_spec.get("rooms", []))
    bedrooms = sum(int(room.get("count", 1)) for room in current_spec.get("rooms", []) if room.get("type") == "Bedroom")
    warnings = []

    max_total_rooms = config.get("max_total_rooms")
    if max_total_rooms and total_rooms > max_total_rooms:
        warnings.append(
            f"Feasibility Warning: {plot_type} requests {total_rooms} rooms, above the heuristic limit of {max_total_rooms}."
        )

    max_bedrooms = config.get("max_bedrooms")
    if max_bedrooms and bedrooms > max_bedrooms:
        warnings.append(
            f"Feasibility Warning: {plot_type} requests {bedrooms} bedrooms, above the heuristic limit of {max_bedrooms}."
        )

    return warnings


def _build_assistant_text(
    current_spec: Dict,
    missing_fields: List[str],
    validation_errors: List[str],
    feasibility_warnings: List[str],
    interpretation_notes: List[str],
    backend_target: Optional[str],
    backend_ready: bool,
    backend_translation_warnings: List[str],
) -> str:
    room_summary = ", ".join(
        f"{room['count']} {room['type']}"
        for room in current_spec.get("rooms", [])
    ) or "no rooms yet"

    dominant_weight = max(current_spec.get("weights", {}), key=current_spec.get("weights", {}).get)
    parts = [
        f"Current Spec captures {room_summary} for a {current_spec.get('building_type')} plan."
    ]

    if interpretation_notes:
        parts.extend(interpretation_notes)

    parts.append(
        f"The current trade-off leans toward {dominant_weight}, with corridor minimization set to "
        f"{current_spec.get('preferences', {}).get('minimize_corridor', False)}."
    )

    if validation_errors:
        parts.append("Validation issues: " + " ".join(validation_errors))

    if missing_fields:
        parts.append(
            "I still need "
            + ", ".join(missing_fields)
            + " before the request can be treated as execution-ready."
        )

    if feasibility_warnings:
        parts.append(" ".join(feasibility_warnings))

    if backend_target:
        parts.append(f"The current room mix routes to the {backend_target} backend.")
    else:
        parts.append("The backend target is not selected until the room program is clear.")

    if backend_translation_warnings:
        parts.append(" ".join(backend_translation_warnings))

    if backend_ready:
        parts.append("The request is ready for backend translation with resolved geometry.")
    else:
        parts.append("Backend translation remains blocked until actual boundary geometry is supplied.")

    return " ".join(parts)


def _canonical_plot_type(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower().replace(" ", "")
    mapping = {
        "5marla": "5Marla",
        "10marla": "10Marla",
        "20marla": "20Marla",
        "custom": "Custom",
    }
    return mapping.get(text)


def _canonical_entrance_side(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    mapping = {
        "north": "North",
        "south": "South",
        "east": "East",
        "west": "West",
    }
    return mapping.get(text)


def _canonical_room_label(label: str) -> Optional[str]:
    cleaned = str(label).strip().lower().replace("  ", " ")
    for room_type, labels in ROOM_LABELS.items():
        if cleaned in labels:
            return room_type
    return None


def _to_count(raw_count: str) -> int:
    text = str(raw_count).strip().lower()
    if text.isdigit():
        return int(text)
    return NUMBER_WORDS.get(text, 1)


