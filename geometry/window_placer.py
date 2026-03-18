"""
window_placer.py - Heuristic window placement on exterior walls.

Generates axis-aligned window segments on building boundary-facing room edges.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

Point = Tuple[float, float]
Seg = Tuple[Point, Point]

_EPS = 1e-6

# Ratio of chosen exterior wall span used for glazing by room type.
WINDOW_RATIO_BY_TYPE = {
    "LivingRoom": 0.60,
    "DrawingRoom": 0.60,
    "DiningRoom": 0.50,
    "Bedroom": 0.35,
    "Study": 0.35,
    "Kitchen": 0.35,
    "Bathroom": 0.20,
    "WC": 0.20,
    "Stairs": 0.18,
    "Lobby": 0.15,
    "Passage": 0.12,
    "Corridor": 0.12,
}

ROOM_TYPE_SCORE_BOOST = {
    "LivingRoom": 1.00,
    "DrawingRoom": 1.00,
    "DiningRoom": 0.85,
    "Bedroom": 0.80,
    "Study": 0.75,
    "Kitchen": 0.70,
    "Bathroom": 0.45,
    "WC": 0.40,
    "Stairs": 0.25,
    "Lobby": 0.22,
    "Passage": 0.18,
    "Corridor": 0.18,
}


def _bbox(polygon: Sequence[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _seg_len(seg: Seg) -> float:
    (x1, y1), (x2, y2) = seg
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _canonical(seg: Seg) -> Seg:
    a, b = seg
    return (a, b) if a <= b else (b, a)


def _is_h(seg: Seg, tol: float = _EPS) -> bool:
    return abs(seg[0][1] - seg[1][1]) <= tol


def _is_v(seg: Seg, tol: float = _EPS) -> bool:
    return abs(seg[0][0] - seg[1][0]) <= tol


def _midpoint(seg: Seg) -> Point:
    (x1, y1), (x2, y2) = seg
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _point_to_seg_dist(p: Point, seg: Seg) -> float:
    (ax, ay), (bx, by) = seg
    px, py = p
    dx = bx - ax
    dy = by - ay
    if abs(dx) < _EPS and abs(dy) < _EPS:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    qx = ax + t * dx
    qy = ay + t * dy
    return ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5


def _iter_edges(polygon: Sequence[Point]) -> List[Seg]:
    edges: List[Seg] = []
    if not polygon or len(polygon) < 3:
        return edges
    pts = list(polygon)
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        if a != b:
            edges.append((a, b))
    return edges


def _boundary_edges(boundary_polygon: Sequence[Point]) -> List[Seg]:
    return _iter_edges(boundary_polygon)


def _edge_side(seg: Seg, boundary_bbox: Tuple[float, float, float, float], tol: float) -> str:
    bx0, by0, bx1, by1 = boundary_bbox
    if _is_v(seg):
        x = seg[0][0]
        if abs(x - bx0) <= tol:
            return "W"
        if abs(x - bx1) <= tol:
            return "E"
    if _is_h(seg):
        y = seg[0][1]
        if abs(y - by0) <= tol:
            return "N"
        if abs(y - by1) <= tol:
            return "S"
    return ""


def _same_line_overlap(a: Seg, b: Seg, tol: float) -> float:
    (ax1, ay1), (ax2, ay2) = a
    (bx1, by1), (bx2, by2) = b
    if _is_v(a, tol) and _is_v(b, tol) and abs(ax1 - bx1) <= tol:
        a0, a1 = sorted([ay1, ay2])
        b0, b1 = sorted([by1, by2])
        return _overlap_1d(a0, a1, b0, b1)
    if _is_h(a, tol) and _is_h(b, tol) and abs(ay1 - by1) <= tol:
        a0, a1 = sorted([ax1, ax2])
        b0, b1 = sorted([bx1, bx2])
        return _overlap_1d(a0, a1, b0, b1)
    return 0.0


def _extract_exterior_room_edges(
    room_polygon: Sequence[Point],
    boundary_polygon: Sequence[Point],
    *,
    touch_tol: float,
    min_span: float,
) -> List[Tuple[Seg, str]]:
    out: List[Tuple[Seg, str]] = []
    rb = _iter_edges(room_polygon)
    bb = _boundary_edges(boundary_polygon)
    bbox = _bbox(boundary_polygon)

    for re in rb:
        if not (_is_h(re, touch_tol) or _is_v(re, touch_tol)):
            continue
        side = _edge_side(re, bbox, touch_tol)
        if not side:
            continue

        best = 0.0
        for be in bb:
            best = max(best, _same_line_overlap(re, be, touch_tol))
        if best >= min_span:
            out.append((_canonical(re), side))
    return out


def _facade_preference_order(entrance_side: str) -> Tuple[str, str, str, str]:
    if entrance_side == "N":
        return ("S", "E", "W", "N")
    if entrance_side == "S":
        return ("N", "E", "W", "S")
    if entrance_side == "E":
        return ("W", "N", "S", "E")
    if entrance_side == "W":
        return ("E", "N", "S", "W")
    return ("N", "S", "E", "W")


def _entrance_side(
    entrance_point: Point | None,
    boundary_bbox: Tuple[float, float, float, float],
    tol: float,
) -> str:
    if entrance_point is None:
        return ""
    ex, ey = entrance_point
    bx0, by0, bx1, by1 = boundary_bbox
    d = {
        "W": abs(ex - bx0),
        "E": abs(ex - bx1),
        "N": abs(ey - by0),
        "S": abs(ey - by1),
    }
    side = min(d, key=d.get)
    return side if d[side] <= max(tol, 0.35) else ""


def _window_segment_on_edge(seg: Seg, length_ratio: float, corner_margin: float, min_window_len: float) -> Seg | None:
    (x1, y1), (x2, y2) = seg
    span = _seg_len(seg)
    usable = span - 2.0 * corner_margin
    if usable < min_window_len:
        return None

    target = max(min_window_len, usable * length_ratio)
    target = min(target, usable)

    if _is_v(seg):
        lo, hi = sorted([y1, y2])
        center = (lo + hi) / 2.0
        s = max(lo + corner_margin, center - target / 2.0)
        e = min(hi - corner_margin, center + target / 2.0)
        if e - s < min_window_len:
            return None
        return ((x1, s), (x1, e))

    lo, hi = sorted([x1, x2])
    center = (lo + hi) / 2.0
    s = max(lo + corner_margin, center - target / 2.0)
    e = min(hi - corner_margin, center + target / 2.0)
    if e - s < min_window_len:
        return None
    return ((s, y1), (e, y1))


def _same_axis_and_overlaps(a: Seg, b: Seg, tol: float = 1e-6) -> bool:
    (ax1, ay1), (ax2, ay2) = a
    (bx1, by1), (bx2, by2) = b

    if abs(ax1 - ax2) < tol and abs(bx1 - bx2) < tol and abs(ax1 - bx1) < tol:
        a0, a1 = sorted([ay1, ay2])
        b0, b1 = sorted([by1, by2])
        return _overlap_1d(a0, a1, b0, b1) > tol

    if abs(ay1 - ay2) < tol and abs(by1 - by2) < tol and abs(ay1 - by1) < tol:
        a0, a1 = sorted([ax1, ax2])
        b0, b1 = sorted([bx1, bx2])
        return _overlap_1d(a0, a1, b0, b1) > tol

    return False


def _room_associated_doors(
    room_polygon: Sequence[Point],
    door_segments: Sequence[Seg],
    *,
    attach_tol: float,
) -> List[Seg]:
    edges = _iter_edges(room_polygon)
    result: List[Seg] = []
    for d in door_segments:
        dm = _midpoint(d)
        if any(_point_to_seg_dist(dm, e) <= attach_tol for e in edges):
            result.append(d)
    return result


def suggest_window_segments(
    rooms: Sequence,
    boundary_polygon: Sequence[Point] | None,
    *,
    door_segments: Iterable[Seg] | None = None,
    entrance_point: Point | None = None,
    touch_tol: float = 0.05,
    min_span: float = 1.2,
    corner_margin: float = 0.25,
    min_window_len: float = 0.55,
    default_length_ratio: float = 0.30,
    clearance_from_door: float = 0.2,
    room_door_attach_tol: float = 0.35,
) -> List[Seg]:
    """Place one tuned window per exterior-facing room edge.

    Heuristics:
    - picks true exterior edges (overlap with boundary edges)
    - scores by room type + edge length
    - biases facade opposite/adjacent to entrance side
    - prefers edge farther from the room's nearest door
    - enforces corner margins and min opening length
    """
    if not boundary_polygon or len(boundary_polygon) < 3:
        return []

    door_list = [d for d in (door_segments or []) if d is not None]
    bb = _bbox(boundary_polygon)
    ent_side = _entrance_side(entrance_point, bb, touch_tol)
    facade_order = _facade_preference_order(ent_side)
    facade_weight = {side: 0.55 - i * 0.18 for i, side in enumerate(facade_order)}
    facade_weight = {k: max(0.0, v) for k, v in facade_weight.items()}

    windows: List[Seg] = []
    for room in rooms:
        polygon = getattr(room, "polygon", None)
        if not polygon or len(polygon) < 3:
            continue

        rtype = getattr(room, "room_type", "")
        length_ratio = WINDOW_RATIO_BY_TYPE.get(rtype, default_length_ratio)
        type_boost = ROOM_TYPE_SCORE_BOOST.get(rtype, 0.25)

        if rtype in {"Passage", "Corridor"} and type_boost < 0.2:
            continue

        ext_edges = _extract_exterior_room_edges(
            polygon,
            boundary_polygon,
            touch_tol=touch_tol,
            min_span=min_span,
        )
        if not ext_edges:
            continue

        room_doors = _room_associated_doors(
            polygon,
            door_list,
            attach_tol=room_door_attach_tol,
        )

        candidates: List[Tuple[float, Seg]] = []
        for edge, side in ext_edges:
            window_seg = _window_segment_on_edge(
                edge,
                length_ratio=length_ratio,
                corner_margin=corner_margin,
                min_window_len=min_window_len,
            )
            if window_seg is None:
                continue

            if any(_same_axis_and_overlaps(window_seg, d, tol=clearance_from_door) for d in door_list):
                continue

            score = _seg_len(edge) * (0.8 + type_boost)
            score *= 1.0 + facade_weight.get(side, 0.0)

            if room_doors:
                wmid = _midpoint(window_seg)
                nearest = min(_point_to_seg_dist(wmid, d) for d in room_doors)
                score += 0.5 * nearest

            candidates.append((score, window_seg))

        if not candidates:
            continue

        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0][1]

        windows.append(best)

    # Deduplicate any overlapping selections from adjacent room logic.
    uniq = {}
    for w in windows:
        uniq[_canonical(w)] = w
    return list(uniq.values())


# ─── Lighting & Ventilation Compliance (Chapter-4) ─────────────────────────

# Default window height assumed for opening area estimation (meters)
ASSUMED_WINDOW_HEIGHT = 1.2


def _room_polygon_area(polygon: Sequence[Point]) -> float:
    """Compute polygon area using shoelace formula."""
    if not polygon or len(polygon) < 3:
        return 0.0
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def compute_window_opening_area(window_seg: Seg, height: float = ASSUMED_WINDOW_HEIGHT) -> float:
    """Estimate window opening area from segment length and assumed height."""
    length = _seg_len(window_seg)
    return round(length * height, 3)


def check_lighting_ventilation_compliance(
    rooms: Sequence,
    window_segments: Sequence[Seg],
    *,
    opening_ratio_min: float = 0.10,
    kitchen_window_min_sqm: float = 1.0,
    bath_wc_vent_min_sqm: float = 0.37,
    max_lighting_depth_m: float = 7.5,
    assumed_window_height: float = ASSUMED_WINDOW_HEIGHT,
    room_window_attach_tol: float = 0.35,
) -> dict:
    """Check rooms against Chapter-4 lighting & ventilation requirements.

    Returns a dict with:
      - compliant: bool (True if all requirements met)
      - rooms: list of per-room compliance details
      - violations: list of violation messages
      - warnings: list of warning messages (soft failures)

    Requirements (Chapter-4):
      - Habitable rooms & kitchens: opening area >= 10% of floor area
      - Kitchen: window opening >= 1.0 sq.m (to exterior, not shaft)
      - Bathroom/WC: ventilation opening >= 0.37 sq.m
      - No portion considered lighted if > 7.5m from opening
    """
    violations = []
    warnings = []
    room_details = []

    # Build mapping of windows to rooms
    window_list = list(window_segments or [])

    for room in rooms:
        polygon = getattr(room, "polygon", None)
        room_type = getattr(room, "room_type", "")
        room_name = getattr(room, "name", str(room))

        if not polygon or len(polygon) < 3:
            room_details.append({
                "name": room_name,
                "type": room_type,
                "floor_area": 0,
                "required_opening": 0,
                "achieved_opening": 0,
                "compliant": True,
                "note": "No polygon defined",
            })
            continue

        floor_area = _room_polygon_area(polygon)
        room_edges = _iter_edges(polygon)

        # Find windows attached to this room
        attached_windows = []
        for w in window_list:
            wm = _midpoint(w)
            if any(_point_to_seg_dist(wm, e) <= room_window_attach_tol for e in room_edges):
                attached_windows.append(w)

        # Compute total window opening area
        total_opening = sum(
            compute_window_opening_area(w, assumed_window_height)
            for w in attached_windows
        )

        # Determine requirements based on room type
        is_habitable = room_type in {
            "Bedroom", "LivingRoom", "DrawingRoom", "DiningRoom", "Study", "Habitable"
        }
        is_kitchen = room_type == "Kitchen"
        is_bath_wc = room_type in {"Bathroom", "WC", "BathWC"}

        required_opening = 0.0
        requirement_source = ""

        if is_habitable or is_kitchen:
            required_opening = floor_area * opening_ratio_min
            requirement_source = f"10% of floor area ({floor_area:.2f} sq.m)"
            if is_kitchen:
                required_opening = max(required_opening, kitchen_window_min_sqm)
                requirement_source = f"max(10% floor area, {kitchen_window_min_sqm} sq.m)"
        elif is_bath_wc:
            required_opening = bath_wc_vent_min_sqm
            requirement_source = f"min {bath_wc_vent_min_sqm} sq.m vent"

        compliant = total_opening >= required_opening - 0.01

        detail = {
            "name": room_name,
            "type": room_type,
            "floor_area": round(floor_area, 2),
            "required_opening": round(required_opening, 3),
            "achieved_opening": round(total_opening, 3),
            "window_count": len(attached_windows),
            "requirement_source": requirement_source,
            "compliant": compliant,
        }

        if not compliant and required_opening > 0:
            deficit = required_opening - total_opening
            if is_habitable or is_kitchen:
                violations.append(
                    f"{room_name} ({room_type}): opening area {total_opening:.2f} sq.m < required {required_opening:.2f} sq.m"
                )
            elif is_bath_wc:
                # Bath/WC can use shaft - record as warning
                warnings.append(
                    f"{room_name} ({room_type}): vent opening {total_opening:.2f} sq.m < required {required_opening:.2f} sq.m (may use shaft)"
                )
            detail["deficit"] = round(deficit, 3)

        # Check lighting depth (only for habitable rooms)
        if is_habitable and attached_windows:
            # Simplified check: compute max distance from any room point to nearest window
            # For now, use room bounding box diagonal as proxy
            bbox = _bbox(polygon)
            room_width = bbox[2] - bbox[0]
            room_depth = bbox[3] - bbox[1]
            max_dim = max(room_width, room_depth)
            if max_dim > max_lighting_depth_m:
                warnings.append(
                    f"{room_name}: room dimension {max_dim:.1f}m > max lighting depth {max_lighting_depth_m}m"
                )
                detail["lighting_depth_warning"] = True

        room_details.append(detail)

    return {
        "compliant": len(violations) == 0,
        "rooms": room_details,
        "violations": violations,
        "warnings": warnings,
        "parameters": {
            "opening_ratio_min": opening_ratio_min,
            "kitchen_window_min_sqm": kitchen_window_min_sqm,
            "bath_wc_vent_min_sqm": bath_wc_vent_min_sqm,
            "max_lighting_depth_m": max_lighting_depth_m,
            "assumed_window_height": assumed_window_height,
        },
    }


def summarize_lv_compliance(rooms: Sequence, window_segments: Sequence[Seg], **kwargs) -> dict:
    """Return a summary dict suitable for inclusion in compliance reports."""
    full = check_lighting_ventilation_compliance(rooms, window_segments, **kwargs)

    total_floor = sum(r["floor_area"] for r in full["rooms"])
    total_opening = sum(r["achieved_opening"] for r in full["rooms"])
    rooms_checked = len(full["rooms"])
    rooms_compliant = sum(1 for r in full["rooms"] if r["compliant"])

    return {
        "compliant": full["compliant"],
        "rooms_checked": rooms_checked,
        "rooms_compliant": rooms_compliant,
        "total_floor_area_sqm": round(total_floor, 2),
        "total_opening_area_sqm": round(total_opening, 3),
        "opening_ratio_achieved": round(total_opening / total_floor, 3) if total_floor > 0 else 0,
        "violation_count": len(full["violations"]),
        "warning_count": len(full["warnings"]),
        "violations": full["violations"],
        "warnings": full["warnings"],
    }
