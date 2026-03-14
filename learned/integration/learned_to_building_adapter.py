"""
learned_to_building_adapter.py – Convert Transformer-sampled token sequences
(list of RoomBox) into a fully-wired ``Building`` object that can enter the
deterministic repair / corridor / compliance pipeline.

Responsibilities
~~~~~~~~~~~~~~~~
1. **Room-type canonicalization** – map dataset labels (Kaggle, etc.) to the
   internal regulation-compatible types.
2. **Scale normalised bbox** into real boundary coordinates.
3. **BBox → polygon** – axis-aligned rectangle from (x1, y1, x2, y2).
4. **Create Room objects** with polygon + area + provenance metadata.
5. **Ensure required room counts** – add placeholders for missing rooms,
   drop surplus rooms beyond the spec's requirements.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from shapely.geometry import Polygon

from core.building import Building
from core.room import Room
from core.exit import Exit
from learned.data.tokenizer_layout import RoomBox

# ═══════════════════════════════════════════════════════════════════════════════
#  Room-type canonicalization table
# ═══════════════════════════════════════════════════════════════════════════════

_CANONICAL_MAP: Dict[str, str] = {
    # Kaggle / model output labels → internal canonical types
    "DrawingRoom":    "LivingRoom",
    "Lounge":         "LivingRoom",
    "Lobby":          "LivingRoom",
    "Dining":         "DiningRoom",
    "DiningRoom":     "DiningRoom",
    "Store":          "Storage",
    "Storage":        "Storage",
    "Stairs":         "Staircase",
    "Staircase":      "Staircase",
    "Passage":        "Corridor",
    "Corridor":       "Corridor",
    "Toilet":         "WC",
    "WC":             "WC",
    # Identity mappings for regulated types
    "Bedroom":        "Bedroom",
    "LivingRoom":     "LivingRoom",
    "Kitchen":        "Kitchen",
    "Bathroom":       "Bathroom",
    # Other valid but non-regulated types – kept as-is
    "Study":          "Study",
    "Balcony":        "Balcony",
    "Garage":         "Garage",
    "OpenSpace":      "OpenSpace",
    "SideGarden":     "SideGarden",
    "DressingArea":   "DressingArea",
    "PrayerRoom":     "PrayerRoom",
    "ServantQuarter": "ServantQuarter",
    "Backyard":       "Backyard",
    "Laundry":        "Laundry",
    "Lawn":           "Lawn",
}

# Types that are circulation or nonsensical – always dropped
_SKIP_TYPES = {"Corridor", "Passage", "Unknown"}


def _canonicalize(raw_type: str) -> str:
    return _CANONICAL_MAP.get(raw_type, raw_type)


# ═══════════════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _boundary_bbox(boundary) -> Tuple[float, float, float, float]:
    if isinstance(boundary, Polygon):
        return boundary.bounds
    xs = [p[0] for p in boundary]
    ys = [p[1] for p in boundary]
    return min(xs), min(ys), max(xs), max(ys)


def _scale_room(rbox: RoomBox, bx0, by0, bx1, by1):
    w, h = bx1 - bx0, by1 - by0
    return (
        bx0 + rbox.x_min * w,
        by0 + rbox.y_min * h,
        bx0 + rbox.x_max * w,
        by0 + rbox.y_max * h,
    )


def _rect_polygon(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _load_regulation(regulation_data) -> dict:
    if isinstance(regulation_data, dict):
        return regulation_data
    if regulation_data is None:
        return {}
    p = Path(regulation_data)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
#  Main adapter
# ═══════════════════════════════════════════════════════════════════════════════

def adapt_generated_layout_to_building(
    decoded_rooms: List[RoomBox],
    boundary_poly: Union[List[Tuple[float, float]], Polygon],
    entrance: Optional[Tuple[float, float]] = None,
    spec: Optional[Dict[str, Any]] = None,
    regulation_data: Union[str, Dict, None] = "ontology/regulation_data.json",
    *,
    sample_id: Optional[int] = None,
    exit_width: float = 1.0,
) -> Building:
    """
    Convert a list of decoded ``RoomBox`` into a ``Building`` with real geometry.

    Parameters
    ----------
    decoded_rooms : list[RoomBox]
        Output of ``tokenizer.decode_rooms(tokens)`` – normalised [0, 1] boxes.
    boundary_poly : list[(x,y)] | shapely.Polygon
        User-supplied boundary polygon in metres.
    entrance : (x, y) | None
        Entrance location on the boundary.
    spec : dict | None
        Building spec (rooms list, occupancy, etc.).  When provided the adapter
        enforces required room counts: adds placeholder rooms for any missing
        required types, trims surplus rooms beyond what the spec requests.
    regulation_data : str | dict | None
        Path to ``regulation_data.json`` or pre-loaded dict.
    sample_id : int | None
        Identifier for this sample (stored in provenance).
    exit_width : float
        Minimum exit width in metres.

    Returns
    -------
    Building
        ``.rooms`` populated with polygons + areas + provenance.
        No corridors / doors yet – those come from the repair gate.
    """
    spec = spec or {}
    occupancy = spec.get("occupancy", "Residential")
    building = Building(occupancy_type=occupancy)

    regs = _load_regulation(regulation_data)
    occ_room_regs = regs.get(occupancy, {}).get("rooms", {})

    bx0, by0, bx1, by1 = _boundary_bbox(boundary_poly)

    # ── 1. Canonicalize, scale, create candidate Room objects ─────────────
    type_counter: Dict[str, int] = {}
    candidates: List[Tuple[Room, str, float]] = []  # (room, canonical, area)

    for idx, rbox in enumerate(decoded_rooms):
        canonical = _canonicalize(rbox.room_type)
        if canonical in _SKIP_TYPES:
            continue

        x1, y1, x2, y2 = _scale_room(rbox, bx0, by0, bx1, by1)

        # Clamp to boundary bbox
        x1, y1 = max(bx0, min(x1, bx1)), max(by0, min(y1, by1))
        x2, y2 = max(bx0, min(x2, bx1)), max(by0, min(y2, by1))

        w, h = x2 - x1, y2 - y1
        if w < 0.3 or h < 0.3:
            continue

        area = w * h
        n = type_counter.get(canonical, 0) + 1
        type_counter[canonical] = n
        name = f"{canonical}_{n}"

        room = Room(name=name, room_type=canonical, requested_area=area)
        room.final_area = area
        room.polygon = _rect_polygon(x1, y1, x2, y2)
        room.provenance = {
            "source": "learned",
            "bbox_norm": [rbox.x_min, rbox.y_min, rbox.x_max, rbox.y_max],
            "sample_id": sample_id,
            "raw_type": rbox.room_type,
            "decoded_index": idx,
        }
        candidates.append((room, canonical, area))

    # ── 2. Ensure required room counts from spec ─────────────────────────
    spec_rooms = spec.get("rooms", [])
    if spec_rooms:
        required_counts: Counter = Counter(r["type"] for r in spec_rooms)

        # Group candidates by canonical type
        by_type: Dict[str, List[Tuple[Room, str, float]]] = {}
        for tup in candidates:
            by_type.setdefault(tup[1], []).append(tup)

        final_rooms: List[Room] = []

        # For each required type: keep exactly the needed count (largest first)
        for rtype, needed in required_counts.items():
            available = by_type.pop(rtype, [])
            available.sort(key=lambda t: t[2], reverse=True)
            for room, _, _ in available[:needed]:
                final_rooms.append(room)

            # Create placeholders for any deficit
            deficit = needed - len(available)
            for j in range(max(0, deficit)):
                min_area = occ_room_regs.get(rtype, {}).get("min_area", 4.0)
                ph_name = f"{rtype}_placeholder_{j + 1}"
                ph = Room(name=ph_name, room_type=rtype, requested_area=min_area)
                ph.final_area = min_area
                ph.polygon = None  # will be packed in repair
                ph.provenance = {
                    "source": "placeholder",
                    "reason": "missing_from_model",
                    "sample_id": sample_id,
                }
                final_rooms.append(ph)

        # Keep a limited number of extra generated rooms not in spec
        for rtype, extras in by_type.items():
            extras.sort(key=lambda t: t[2], reverse=True)
            for room, _, _ in extras[:2]:
                final_rooms.append(room)
    else:
        final_rooms = [room for room, _, _ in candidates]

    for room in final_rooms:
        building.add_room(room)

    # ── 3. Exit ──────────────────────────────────────────────────────────
    ex = Exit(width=exit_width)
    if entrance:
        ex.segment = ((entrance[0], entrance[1]),
                       (entrance[0] + exit_width, entrance[1]))
    else:
        ex.segment = ((bx0, by0), (bx0 + exit_width, by0))
    building.set_exit(ex)

    # ── 4. Metrics ───────────────────────────────────────────────────────
    building.total_area = sum(r.final_area for r in building.rooms if r.final_area)
    building.occupant_load = max(1, int(building.total_area / 100 * 8))

    return building


# ═══════════════════════════════════════════════════════════════════════════════
#  Backward-compatible alias (old API)
# ═══════════════════════════════════════════════════════════════════════════════

def adapt_generated_layout(
    decoded_rooms: List[RoomBox],
    boundary_polygon: List[Tuple[float, float]],
    entrance_point=None,
    occupancy_type: str = "Residential",
    exit_width: float = 1.0,
    skip_passage: bool = True,
) -> Building:
    """Legacy wrapper – delegates to ``adapt_generated_layout_to_building``."""
    return adapt_generated_layout_to_building(
        decoded_rooms,
        boundary_poly=boundary_polygon,
        entrance=entrance_point,
        spec={"occupancy": occupancy_type},
        regulation_data=None,
        exit_width=exit_width,
    )
