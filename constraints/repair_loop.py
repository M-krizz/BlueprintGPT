from copy import deepcopy


ROOM_TYPE_ALIASES = {
    "bedroom": "Bedroom",
    "living": "LivingRoom",
    "livingroom": "LivingRoom",
    "hall": "LivingRoom",
    "kitchen": "Kitchen",
    "bathroom": "Bathroom",
    "toilet": "WC",
    "wc": "WC",
}


def _repair_once(spec):
    fixed = deepcopy(spec)
    fixed.setdefault("occupancy", "Residential")
    fixed.setdefault("area_unit", "sq.ft")
    fixed.setdefault("allocation_strategy", "priority_weights")
    fixed.setdefault("adjacency", [])
    fixed.setdefault("preferences", {})

    repaired_rooms = []
    for idx, room in enumerate(fixed.get("rooms", []), start=1):
        if not isinstance(room, dict):
            continue
        room_name = str(room.get("name", "")).strip() or f"Room_{idx}"
        raw_type = str(room.get("type", "")).strip()
        canonical_type = ROOM_TYPE_ALIASES.get(raw_type.lower(), raw_type)
        room_out = {"name": room_name, "type": canonical_type}
        if "area" in room:
            try:
                area = float(room["area"])
                if area > 0:
                    room_out["area"] = area
            except Exception:
                pass
        repaired_rooms.append(room_out)

    fixed["rooms"] = repaired_rooms
    return fixed


def validate_and_repair_spec(spec, validator_fn, max_attempts=3):
    attempts = 0
    current = deepcopy(spec)
    last_validation = validator_fn(current)

    while not last_validation.get("valid", False) and attempts < max_attempts:
        attempts += 1
        current = _repair_once(last_validation.get("normalized_spec", current))
        last_validation = validator_fn(current)

    validation_summary = dict(last_validation)
    validation_summary.pop("normalized_spec", None)

    return {
        "spec": last_validation.get("normalized_spec", current),
        "validation": validation_summary,
        "repair_attempts": attempts,
    }
