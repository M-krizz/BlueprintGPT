"""
polygon.py – Polygon helper utilities: grid snapping, aspect‐ratio enforcement,
wall extraction, alignment scoring, and simplification.

These are used by the repair gate, ranking module, and SVG exporter.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from shapely.geometry import Polygon as ShapelyPolygon


# ═══════════════════════════════════════════════════════════════════════════════
#  Grid snapping
# ═══════════════════════════════════════════════════════════════════════════════

def snap_to_grid(
    polygon: List[Tuple[float, float]],
    step: float = 0.15,
) -> List[Tuple[float, float]]:
    """Snap every vertex of *polygon* to the nearest grid point.

    Parameters
    ----------
    polygon : list of (x, y) tuples
    step    : grid cell size in metres (default 0.15 m ≈ 6 in ≈ 150 mm)

    Returns
    -------
    list of (x, y) snapped tuples (deduplicated, preserving order).
    """
    if not polygon or step <= 0:
        return polygon

    inv = 1.0 / step
    snapped: list[tuple[float, float]] = []
    for x, y in polygon:
        sx = round(round(x * inv) / inv, 6)
        sy = round(round(y * inv) / inv, 6)
        if not snapped or (sx, sy) != snapped[-1]:
            snapped.append((sx, sy))

    # Remove closing duplicate if first == last after snapping
    if len(snapped) > 1 and snapped[0] == snapped[-1]:
        snapped = snapped[:-1]

    return snapped


def snap_room_to_grid(room, step: float = 0.15) -> bool:
    """Snap a Room object's polygon in-place. Returns True if polygon changed."""
    if room.polygon is None:
        return False
    original = list(room.polygon)
    room.polygon = snap_to_grid(original, step)
    # Recompute area from snapped polygon
    if room.polygon and len(room.polygon) >= 3:
        sp = ShapelyPolygon(room.polygon)
        if sp.is_valid:
            room.final_area = round(sp.area, 4)
    return room.polygon != original


def snap_building_to_grid(building, step: float = 0.15) -> int:
    """Snap all rooms + corridors in a Building to the grid. Returns count of changed polygons."""
    changed = 0
    for room in building.rooms:
        if snap_room_to_grid(room, step):
            changed += 1
    for corridor in getattr(building, "corridors", []):
        if corridor.polygon:
            old = list(corridor.polygon)
            corridor.polygon = snap_to_grid(old, step)
            if corridor.polygon != old:
                changed += 1
    return changed


# ═══════════════════════════════════════════════════════════════════════════════
#  Aspect ratio enforcement
# ═══════════════════════════════════════════════════════════════════════════════

# Per-room-type aspect ratio limits  (min_ratio, max_ratio)  where ratio = w / h
DEFAULT_ASPECT_LIMITS: dict[str, Tuple[float, float]] = {
    "Bedroom":    (0.50, 2.00),
    "LivingRoom": (0.50, 2.50),
    "Kitchen":    (0.55, 2.00),
    "Bathroom":   (0.50, 2.00),
    "WC":         (0.50, 2.00),
    "DiningRoom": (0.50, 2.50),
    "Study":      (0.50, 2.00),
    "Storage":    (0.45, 2.50),
    "_default":   (0.50, 2.50),
}


def aspect_ratio(polygon: List[Tuple[float, float]]) -> float:
    """Return w/h for the bounding box of *polygon*. Returns 1.0 if degenerate."""
    if not polygon or len(polygon) < 3:
        return 1.0
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    if h < 1e-6:
        return 999.0
    return w / h


def enforce_aspect_ratio(
    room,
    limits: Optional[dict[str, Tuple[float, float]]] = None,
) -> bool:
    """Reshape *room* so that its bounding-box aspect ratio falls within limits.

    The room's area is preserved; the shorter dimension is expanded and the
    longer dimension is shrunk proportionally.

    Returns True if the room was reshaped.
    """
    limits = limits or DEFAULT_ASPECT_LIMITS
    lo, hi = limits.get(room.room_type, limits.get("_default", (0.5, 2.5)))

    if room.polygon is None or len(room.polygon) < 3:
        return False

    xs = [p[0] for p in room.polygon]
    ys = [p[1] for p in room.polygon]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    w = x2 - x1
    h = y2 - y1
    if w < 1e-6 or h < 1e-6:
        return False

    ratio = w / h
    if lo <= ratio <= hi:
        return False  # already within limits

    area = w * h
    if ratio < lo:
        # Too tall → widen and shorten
        target_ratio = lo
    else:
        # Too wide → narrow and heighten
        target_ratio = hi

    new_w = math.sqrt(area * target_ratio)
    new_h = area / new_w

    # Centre the new rectangle on the old centre
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nx1 = cx - new_w / 2.0
    nx2 = cx + new_w / 2.0
    ny1 = cy - new_h / 2.0
    ny2 = cy + new_h / 2.0

    room.polygon = [
        (round(nx1, 4), round(ny1, 4)),
        (round(nx2, 4), round(ny1, 4)),
        (round(nx2, 4), round(ny2, 4)),
        (round(nx1, 4), round(ny2, 4)),
    ]
    room.final_area = round(new_w * new_h, 4)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  Wall / edge extraction
# ═══════════════════════════════════════════════════════════════════════════════

def walls(polygon: List[Tuple[float, float]]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Return ordered list of wall segments [(p0, p1), (p1, p2), …]."""
    if not polygon or len(polygon) < 2:
        return []
    edges = []
    n = len(polygon)
    for i in range(n):
        edges.append((polygon[i], polygon[(i + 1) % n]))
    return edges


def horizontal_walls(polygon: List[Tuple[float, float]], tol: float = 0.01) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Return wall segments that are approximately horizontal."""
    return [(a, b) for a, b in walls(polygon) if abs(a[1] - b[1]) < tol]


def vertical_walls(polygon: List[Tuple[float, float]], tol: float = 0.01) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Return wall segments that are approximately vertical."""
    return [(a, b) for a, b in walls(polygon) if abs(a[0] - b[0]) < tol]


# ═══════════════════════════════════════════════════════════════════════════════
#  Alignment scoring
# ═══════════════════════════════════════════════════════════════════════════════

def _collinear_overlap(seg_a, seg_b, tol: float = 0.05) -> float:
    """Return the overlap length between two axis-aligned segments on the same line.

    Returns 0.0 if segments aren't collinear within *tol*.
    """
    (ax1, ay1), (ax2, ay2) = seg_a
    (bx1, by1), (bx2, by2) = seg_b

    # Horizontal case
    if abs(ay1 - ay2) < tol and abs(by1 - by2) < tol and abs(ay1 - by1) < tol:
        lo = max(min(ax1, ax2), min(bx1, bx2))
        hi = min(max(ax1, ax2), max(bx1, bx2))
        return max(0.0, hi - lo)

    # Vertical case
    if abs(ax1 - ax2) < tol and abs(bx1 - bx2) < tol and abs(ax1 - bx1) < tol:
        lo = max(min(ay1, ay2), min(by1, by2))
        hi = min(max(ay1, ay2), max(by1, by2))
        return max(0.0, hi - lo)

    return 0.0


def alignment_score(building, include_corridor: bool = True) -> float:
    """Compute a 0–1 alignment score for a building layout.

    Measures how much total wall length is shared (collinear overlap) between
    adjacent rooms and corridor edges, normalised by the total wall perimeter.

    Higher = more walls are aligned → more architecturally plausible.
    """
    all_segments: list[list[tuple]] = []
    for room in building.rooms:
        if room.polygon and len(room.polygon) >= 3:
            all_segments.append(walls(room.polygon))

    if include_corridor:
        for corridor in getattr(building, "corridors", []):
            if corridor.polygon and len(corridor.polygon) >= 3:
                all_segments.append(walls(corridor.polygon))

    if len(all_segments) < 2:
        return 1.0  # single element is trivially aligned

    total_wall_length = 0.0
    shared_length = 0.0

    for segs in all_segments:
        for a, b in segs:
            total_wall_length += math.hypot(b[0] - a[0], b[1] - a[1])

    if total_wall_length < 1e-6:
        return 1.0

    # Pair-wise collinear overlap
    for i in range(len(all_segments)):
        for j in range(i + 1, len(all_segments)):
            for seg_a in all_segments[i]:
                for seg_b in all_segments[j]:
                    shared_length += _collinear_overlap(seg_a, seg_b)

    # Normalise: shared_length could at most be half of total (each shared wall counted from both sides)
    return min(1.0, round(shared_length * 2.0 / total_wall_length, 4))


# ═══════════════════════════════════════════════════════════════════════════════
#  Polygon simplification
# ═══════════════════════════════════════════════════════════════════════════════

def simplify_polygon(
    polygon: List[Tuple[float, float]],
    tolerance: float = 0.05,
) -> List[Tuple[float, float]]:
    """Simplify polygon by removing nearly-collinear vertices."""
    if not polygon or len(polygon) < 3:
        return polygon
    sp = ShapelyPolygon(polygon)
    if not sp.is_valid:
        sp = sp.buffer(0)
    simplified = sp.simplify(tolerance)
    return [(round(x, 4), round(y, 4)) for x, y in simplified.exterior.coords[:-1]]
