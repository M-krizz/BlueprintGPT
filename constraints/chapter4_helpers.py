"""
chapter4_helpers.py — Plot-bucket selection and Chapter-4 regulation lookups.

Single-source helpers consumed by rule_engine, compliance_report,
window_placer, and learned-pipeline constraint masks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_DEFAULT_REG_PATH = Path(__file__).resolve().parent.parent / "ontology" / "regulation_data.json"

_HABITABLE_ROOM_TYPES = frozenset({
    "Bedroom", "LivingRoom", "DrawingRoom", "DiningRoom", "Study",
    "Habitable",
})

_SERVICE_ROOM_TYPES = frozenset({
    "Kitchen", "Bathroom", "WC", "BathWC", "Pantry",
})


def load_regulation_data(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the canonical regulation JSON."""
    p = Path(path) if path else _DEFAULT_REG_PATH
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Plot bucket selection ----------

def polygon_area(points: list) -> float:
    """Shoelace formula for a list of (x, y) tuples."""
    if not points or len(points) < 3:
        return 0.0
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def rect_area(width: float, height: float) -> float:
    return abs(width * height)


def plot_bucket(plot_area_sqm: float) -> str:
    """Return 'upto_50sqm' or 'above_50sqm' based on Chapter-4 Table 4.2."""
    return "upto_50sqm" if plot_area_sqm <= 50.0 else "above_50sqm"


def get_bucket_rules(
    plot_area_sqm: float,
    reg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the Chapter-4 room/door/stair rules for the correct plot bucket."""
    if reg is None:
        reg = load_regulation_data()
    bucket = plot_bucket(plot_area_sqm)
    return reg["Residential"]["chapter4_residential"]["plot_buckets"][bucket]


# ---------- Room classification ----------

def is_habitable(room_type: str) -> bool:
    return room_type in _HABITABLE_ROOM_TYPES


def is_service(room_type: str) -> bool:
    return room_type in _SERVICE_ROOM_TYPES


def chapter4_room_category(room_type: str) -> str:
    """Map a concrete room type to the Chapter-4 category key used in the
    plot-bucket rule tables.
    """
    if room_type in ("Kitchen", "Bathroom", "WC", "BathWC", "Pantry", "Garage"):
        return room_type
    # All habitable rooms map to "Habitable"
    if room_type in _HABITABLE_ROOM_TYPES:
        return "Habitable"
    return room_type


# ---------- Chapter-4 numeric lookups ----------

def get_min_room_dims(
    room_type: str,
    plot_area_sqm: float,
    reg: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Return {'min_area', 'min_width', 'min_height'} for a room type under
    the correct plot bucket.
    Falls back to the top-level Residential rooms if room_type is not in the
    bucket table (e.g. room types not explicitly listed in Chapter-4).
    """
    bucket_rules = get_bucket_rules(plot_area_sqm, reg)
    cat = chapter4_room_category(room_type)
    dims = bucket_rules.get("rooms", {}).get(cat)
    if dims:
        return dict(dims)
    # Fallback to top-level Residential rooms (which use >50 sq.m values)
    if reg is None:
        reg = load_regulation_data()
    top = reg.get("Residential", {}).get("rooms", {}).get(room_type)
    if top:
        return {
            "min_area": top.get("min_area", 0),
            "min_width": top.get("min_width", 0),
            "min_height": top.get("min_height", 0),
        }
    return {"min_area": 0, "min_width": 0, "min_height": 0}


def get_door_dims(
    room_type: str,
    plot_area_sqm: float,
    reg: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Return door min_width and min_height for the room type under the
    correct plot bucket.
    """
    bucket_rules = get_bucket_rules(plot_area_sqm, reg)
    if is_habitable(room_type):
        return dict(bucket_rules["doors"]["habitable"])
    return dict(bucket_rules["doors"]["service"])


def get_stair_width(
    plot_area_sqm: float,
    reg: Optional[Dict[str, Any]] = None,
) -> float:
    bucket_rules = get_bucket_rules(plot_area_sqm, reg)
    return bucket_rules.get("stair_width", 0.9)


# ---------- Egress / corridor lookups ----------

def get_travel_distance_limit(occupancy: str, reg: Optional[Dict[str, Any]] = None) -> float:
    """Return max allowed travel distance (m) for an occupancy type."""
    if reg is None:
        reg = load_regulation_data()
    table = reg.get("chapter4_egress", {}).get("travel_distance", {})
    return float(table.get(occupancy, table.get("Residential", 22.5)))


def get_corridor_min_width(occupancy: str, sub_type: str = "",
                           reg: Optional[Dict[str, Any]] = None) -> float:
    """Return minimum corridor width (m) by occupancy + sub-type."""
    if reg is None:
        reg = load_regulation_data()
    table = reg.get("chapter4_egress", {}).get("corridor_min_width", {})

    # Explicit occupancy+sub_type combos
    key_map = {
        ("Residential", "dwelling_unit"): "residential_dwelling_unit",
        ("Residential", "hostel"):        "residential_hostel",
        ("Institutional", "hospital"):    "hospital",
    }
    key = key_map.get((occupancy, sub_type))

    # Fallback: occupancy-level defaults
    if key is None:
        occ_defaults = {
            "Residential": "residential_dwelling_unit",
            "Assembly":    "assembly",
            "Institutional": "hospital",
        }
        key = occ_defaults.get(occupancy, "others")

    return float(table.get(key, 1.5))


def get_exit_capacity(occupancy: str, element: str = "stair",
                      reg: Optional[Dict[str, Any]] = None) -> int:
    """Return persons per 50 cm unit width for stair/ramp/door (Table 4.3)."""
    if reg is None:
        reg = load_regulation_data()
    table = reg.get("chapter4_egress", {}).get("exit_capacity_per_50cm_unit_width", {})
    occ_data = table.get(occupancy, table.get("Residential", {}))
    return int(occ_data.get(element, 25))


def get_stair_min_width_by_occupancy(occupancy: str,
                                     sub_type: str = "",
                                     reg: Optional[Dict[str, Any]] = None) -> float:
    """Return minimum stairway width (m) by occupancy (Section 4.8.6)."""
    if reg is None:
        reg = load_regulation_data()
    table = reg.get("chapter4_egress", {}).get("stair_min_width", {})
    key_map = {
        ("Residential", "low_rise"): "residential_low_rise",
        ("Residential", ""):         "residential_other",
        ("Assembly", ""):            "assembly",
        ("Institutional", ""):       "institutional",
        ("Educational", ""):         "educational",
    }
    key = key_map.get((occupancy, sub_type), "others")
    return float(table.get(key, 1.5))


def get_exit_door_dims(occupancy: str,
                       reg: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Return exit door min_width and min_height."""
    if reg is None:
        reg = load_regulation_data()
    door = reg.get("chapter4_egress", {}).get("exit_door", {})
    w = float(door.get("min_width", 1.0))
    if occupancy == "Institutional":
        w = max(w, float(door.get("min_width_hospital", 1.5)))
    return {"min_width": w, "min_height": float(door.get("min_height", 2.0))}


# ---------- Lighting & ventilation lookups ----------

def get_opening_ratio(reg: Optional[Dict[str, Any]] = None) -> float:
    """Min opening area / floor area for habitable rooms and kitchens."""
    if reg is None:
        reg = load_regulation_data()
    return float(
        reg.get("chapter4_lighting_ventilation", {})
        .get("opening_area_ratio", {})
        .get("min_ratio", 0.10)
    )


def get_max_lighting_depth(reg: Optional[Dict[str, Any]] = None) -> float:
    if reg is None:
        reg = load_regulation_data()
    return float(
        reg.get("chapter4_lighting_ventilation", {})
        .get("max_lighting_depth", {})
        .get("max_distance_m", 7.5)
    )


def get_kitchen_window_min(reg: Optional[Dict[str, Any]] = None) -> float:
    if reg is None:
        reg = load_regulation_data()
    return float(
        reg.get("chapter4_lighting_ventilation", {})
        .get("kitchen_window", {})
        .get("min_opening_area_sqm", 1.0)
    )


def get_bathroom_vent_min(reg: Optional[Dict[str, Any]] = None) -> float:
    if reg is None:
        reg = load_regulation_data()
    return float(
        reg.get("chapter4_lighting_ventilation", {})
        .get("bathroom_wc_ventilation", {})
        .get("min_opening_area_sqm", 0.37)
    )


def get_shaft_min(building_height_m: float,
                  reg: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Return shaft min_area_sqm and min_dimension_m for a building height."""
    if reg is None:
        reg = load_regulation_data()
    table = reg.get("chapter4_lighting_ventilation", {}).get("ventilation_shaft_min_sizes", {})
    if building_height_m <= 9.0:
        return dict(table.get("upto_9m", {"min_area_sqm": 1.5, "min_dimension_m": 1.0}))
    if building_height_m <= 12.5:
        return dict(table.get("upto_12p5m", {"min_area_sqm": 3.0, "min_dimension_m": 1.2}))
    return dict(table.get("upto_15m", {"min_area_sqm": 4.0, "min_dimension_m": 1.5}))
