"""
bounding_box.py – Axis-aligned bounding-box utilities for rooms and buildings.

Provides lightweight helpers used by the allocator, polygon packer, and
repair gate — without pulling in Shapely where it's not needed.
"""
from __future__ import annotations

from typing import List, Optional, Tuple


def bbox(polygon: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Return (x_min, y_min, x_max, y_max) for an arbitrary polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_area(polygon: List[Tuple[float, float]]) -> float:
    """Bounding-box area of a polygon (not the polygon area itself)."""
    x0, y0, x1, y1 = bbox(polygon)
    return (x1 - x0) * (y1 - y0)


def bbox_dims(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Return (width, height) of the bounding box."""
    x0, y0, x1, y1 = bbox(polygon)
    return x1 - x0, y1 - y0


def bbox_center(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Centroid of the bounding box."""
    x0, y0, x1, y1 = bbox(polygon)
    return (x0 + x1) / 2, (y0 + y1) / 2


def bbox_overlap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Overlap area between two AABB tuples ``(x0, y0, x1, y1)``."""
    dx = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    dy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return dx * dy


def bbox_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """IoU of two AABB tuples ``(x0, y0, x1, y1)``."""
    ov = bbox_overlap(a, b)
    if ov == 0:
        return 0.0
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-12)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1e-12)
    return ov / (area_a + area_b - ov)


def bbox_contains(
    outer: Tuple[float, float, float, float],
    inner: Tuple[float, float, float, float],
    tolerance: float = 0.0,
) -> bool:
    """True if *inner* is fully inside *outer* (with optional tolerance)."""
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def clamp_to_bbox(
    polygon: List[Tuple[float, float]],
    boundary: Tuple[float, float, float, float],
) -> List[Tuple[float, float]]:
    """Clamp every vertex of *polygon* to lie within *boundary* AABB."""
    bx0, by0, bx1, by1 = boundary
    return [
        (max(bx0, min(bx1, x)), max(by0, min(by1, y)))
        for x, y in polygon
    ]


def merge_bboxes(
    boxes: List[Tuple[float, float, float, float]],
) -> Optional[Tuple[float, float, float, float]]:
    """Return the smallest AABB that encloses all *boxes* (or None if empty)."""
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return x0, y0, x1, y1
