import json
from pathlib import Path


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_spec(
    spec,
    schema_path="ontology/building_spec.schema.json",
    regulation_path="ontology/regulation_data.json",
):
    schema = _load_json(schema_path)
    regulation = _load_json(regulation_path)

    errors = []
    normalized = dict(spec or {})

    occupancy = normalized.get("occupancy", "Residential")
    allowed_occupancies = schema.get("properties", {}).get("occupancy", {}).get("enum", ["Residential"])
    if occupancy not in allowed_occupancies:
        errors.append(f"Unsupported occupancy '{occupancy}'")

    normalized.setdefault("occupancy", "Residential")
    normalized.setdefault("adjacency", [])
    normalized.setdefault("preferences", {})
    normalized.setdefault("area_unit", "sq.ft")
    normalized.setdefault("allocation_strategy", "priority_weights")

    room_specs = normalized.get("rooms", [])
    if not isinstance(room_specs, list) or not room_specs:
        errors.append("rooms must be a non-empty list")
        room_specs = []

    schema_room_types = set(
        schema.get("properties", {})
        .get("rooms", {})
        .get("items", {})
        .get("properties", {})
        .get("type", {})
        .get("enum", [])
    )
    ontology_room_types = set(regulation.get("Residential", {}).get("rooms", {}).keys())

    room_errors = []
    cleaned_rooms = []
    for idx, room in enumerate(room_specs):
        if not isinstance(room, dict):
            room_errors.append(f"rooms[{idx}] must be an object")
            continue

        name = str(room.get("name", "")).strip()
        room_type = str(room.get("type", "")).strip()
        if not name:
            room_errors.append(f"rooms[{idx}] missing name")
            continue
        if not room_type:
            room_errors.append(f"rooms[{idx}] missing type")
            continue

        if room_type not in schema_room_types:
            room_errors.append(f"rooms[{idx}] uses unsupported schema room type '{room_type}'")
        if room_type not in ontology_room_types:
            room_errors.append(f"rooms[{idx}] uses unsupported ontology room type '{room_type}'")

        room_out = {"name": name, "type": room_type}
        if "area" in room:
            try:
                area = float(room["area"])
                if area <= 0:
                    room_errors.append(f"rooms[{idx}] area must be > 0")
                else:
                    room_out["area"] = area
            except Exception:
                room_errors.append(f"rooms[{idx}] area is not numeric")

        cleaned_rooms.append(room_out)

    normalized["rooms"] = cleaned_rooms
    errors.extend(room_errors)

    schema_valid = len([e for e in errors if "schema" in e or "rooms" in e or "occupancy" in e]) == 0
    kg_valid = len([e for e in errors if "ontology" in e or "unsupported" in e]) == 0

    return {
        "schema_valid": schema_valid,
        "kg_valid": kg_valid,
        "valid": schema_valid and kg_valid and len(errors) == 0,
        "errors": errors,
        "normalized_spec": normalized,
    }
