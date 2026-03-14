"""
walls.py - Orthogonal wall extraction, collinear merging, and door-gap carving.

This module converts room polygons + door segments into final wall line segments
ready for drafting-style rendering.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

Point = Tuple[float, float]
Seg = Tuple[Point, Point]


def _snap_round(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(round(value / step) * step, 6)


def snap_point(p: Point, step: float = 0.0, ndigits: int = 3) -> Point:
    if step > 0:
        return (_snap_round(p[0], step), _snap_round(p[1], step))
    return (round(p[0], ndigits), round(p[1], ndigits))


def canonical_edge(a: Point, b: Point) -> Seg:
    return (a, b) if a <= b else (b, a)


def is_h(a: Point, b: Point, eps: float = 1e-6) -> bool:
    return abs(a[1] - b[1]) < eps


def is_v(a: Point, b: Point, eps: float = 1e-6) -> bool:
    return abs(a[0] - b[0]) < eps


def _seg_len(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _door_key(seg: Seg) -> Seg:
    """Canonical key for door deduplication."""
    return canonical_edge(seg[0], seg[1])


def _distance_point_to_segment(p: Point, a: Point, b: Point) -> float:
    """Euclidean distance from point p to segment ab."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx = bx - ax
    dy = by - ay
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    qx = ax + t * dx
    qy = ay + t * dy
    return ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5


def extract_edges(
    polygons: Sequence[Sequence[Point]],
    *,
    snap_step: float = 0.0,
    ndigits: int = 3,
    min_len: float = 1e-4,
) -> Dict[Seg, int]:
    """Extract canonical polygon edges and count duplicates."""
    edge_count: Dict[Seg, int] = defaultdict(int)

    for poly in polygons:
        if not poly or len(poly) < 3:
            continue
        pts = [snap_point(p, step=snap_step, ndigits=ndigits) for p in poly]
        if pts[0] != pts[-1]:
            pts.append(pts[0])

        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if a == b:
                continue
            if _seg_len(a, b) < min_len:
                continue
            edge_count[canonical_edge(a, b)] += 1

    return edge_count


def _boundary_edge_set(
    boundary_polygon: Sequence[Point] | None,
    *,
    snap_step: float = 0.0,
    ndigits: int = 3,
) -> set[Seg]:
    if not boundary_polygon or len(boundary_polygon) < 3:
        return set()

    bpts = [snap_point(p, step=snap_step, ndigits=ndigits) for p in boundary_polygon]
    if bpts[0] != bpts[-1]:
        bpts.append(bpts[0])

    out: set[Seg] = set()
    for i in range(len(bpts) - 1):
        a, b = bpts[i], bpts[i + 1]
        if a == b:
            continue
        out.add(canonical_edge(a, b))
    return out


def _interval_to_seg(kind: str, const: float, s: float, e: float) -> Seg:
    if kind == "H":
        return ((s, const), (e, const))
    return ((const, s), (const, e))


def merge_collinear_edges(
    edge_count: Dict[Seg, int],
    *,
    boundary_edges: set[Seg] | None = None,
    merge_eps: float = 1e-6,
    axis_eps: float = 1e-6,
) -> List[Dict]:
    """Merge axis-aligned collinear edges into long segments.

    Returns list of dicts:
        {"segment": ((x1,y1),(x2,y2)), "wall_type": "outer"|"inner"}
    """
    boundary_edges = boundary_edges or set()
    groups: Dict[Tuple[str, float, str], List[Tuple[float, float]]] = defaultdict(list)

    for (a, b), _cnt in edge_count.items():
        wall_type = "outer" if canonical_edge(a, b) in boundary_edges else "inner"
        if is_h(a, b, axis_eps):
            y = round(a[1], 6)
            x1, x2 = sorted([a[0], b[0]])
            groups[("H", y, wall_type)].append((x1, x2))
        elif is_v(a, b, axis_eps):
            x = round(a[0], 6)
            y1, y2 = sorted([a[1], b[1]])
            groups[("V", x, wall_type)].append((y1, y2))

    merged: List[Dict] = []
    for (kind, const, wall_type), intervals in groups.items():
        if not intervals:
            continue
        intervals.sort(key=lambda t: t[0])
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_e + merge_eps:
                cur_e = max(cur_e, e)
            else:
                merged.append({
                    "segment": _interval_to_seg(kind, const, cur_s, cur_e),
                    "wall_type": wall_type,
                })
                cur_s, cur_e = s, e
        merged.append({
            "segment": _interval_to_seg(kind, const, cur_s, cur_e),
            "wall_type": wall_type,
        })

    return merged


def _subtract_gaps_1d(s: float, e: float, gaps: Iterable[Tuple[float, float]], eps: float) -> List[Tuple[float, float]]:
    gaps_sorted = sorted(gaps)
    if not gaps_sorted:
        return [(s, e)]

    out: List[Tuple[float, float]] = []
    cur = s
    for gs, ge in gaps_sorted:
        if ge <= s or gs >= e:
            continue
        gs = max(gs, s)
        ge = min(ge, e)
        if gs > cur + eps:
            out.append((cur, gs))
        cur = max(cur, ge)
    if cur < e - eps:
        out.append((cur, e))
    return [seg for seg in out if (seg[1] - seg[0]) > 10 * eps]


def cut_doors_from_walls(
    walls: Sequence[Dict],
    doors: Sequence[Seg],
    *,
    eps: float = 0.1,
    axis_eps: float = 1e-6,
    door_pad: float = 0.01,
    door_attach_tol: float = 0.06,
    min_overlap_ratio: float = 0.8,
    stats: Dict | None = None,
    stats_key: str = "door_gaps_cut_count",
) -> List[Dict]:
    """Subtract door openings from wall segments.

    Returns wall list in the same dict format as merge_collinear_edges.
    """
    out: List[Dict] = []
    gaps_cut = 0

    for wall in walls:
        (a, b) = wall["segment"]
        wall_type = wall.get("wall_type", "inner")

        if is_v(a, b, axis_eps):
            x = a[0]
            wmin, wmax = sorted([a[1], b[1]])
            gaps = []
            for d in doors:
                p, q = d
                if not is_v(p, q, axis_eps):
                    continue
                if abs(p[0] - x) <= eps and abs(q[0] - x) <= eps:
                    gmin, gmax = sorted([p[1], q[1]])
                    overlap = max(0.0, min(gmax, wmax) - max(gmin, wmin))
                    door_len = max(1e-9, abs(q[1] - p[1]))
                    if overlap < door_len * min_overlap_ratio:
                        continue
                    mid = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
                    if _distance_point_to_segment(mid, a, b) > door_attach_tol:
                        continue
                    gaps.append((gmin - door_pad, gmax + door_pad))
            remain = _subtract_gaps_1d(wmin, wmax, gaps, eps=max(1e-6, door_pad / 2))
            if not remain:
                if gaps:
                    gaps_cut += len(gaps)
                continue
            for s, e in remain:
                out.append({"segment": ((x, s), (x, e)), "wall_type": wall_type})
            gaps_cut += len(gaps)

        elif is_h(a, b, axis_eps):
            y = a[1]
            wmin, wmax = sorted([a[0], b[0]])
            gaps = []
            for d in doors:
                p, q = d
                if not is_h(p, q, axis_eps):
                    continue
                if abs(p[1] - y) <= eps and abs(q[1] - y) <= eps:
                    gmin, gmax = sorted([p[0], q[0]])
                    overlap = max(0.0, min(gmax, wmax) - max(gmin, wmin))
                    door_len = max(1e-9, abs(q[0] - p[0]))
                    if overlap < door_len * min_overlap_ratio:
                        continue
                    mid = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
                    if _distance_point_to_segment(mid, a, b) > door_attach_tol:
                        continue
                    gaps.append((gmin - door_pad, gmax + door_pad))
            remain = _subtract_gaps_1d(wmin, wmax, gaps, eps=max(1e-6, door_pad / 2))
            if not remain:
                if gaps:
                    gaps_cut += len(gaps)
                continue
            for s, e in remain:
                out.append({"segment": ((s, y), (e, y)), "wall_type": wall_type})
            gaps_cut += len(gaps)

        else:
            out.append({"segment": (a, b), "wall_type": wall_type})

    if stats is not None:
        stats[stats_key] = gaps_cut
    return out


def build_wall_segments(
    room_polygons: Sequence[Sequence[Point]],
    door_segments: Sequence[Seg],
    *,
    window_segments: Sequence[Seg] | None = None,
    boundary_polygon: Sequence[Point] | None = None,
    snap_step: float = 0.15,
    ndigits: int = 3,
    merge_eps: float = 1e-6,
    door_eps: float = 0.1,
    door_pad: float = 0.01,
    door_attach_tol: float = 0.06,
    window_eps: float | None = None,
    window_pad: float = 0.005,
    window_attach_tol: float | None = None,
    min_overlap_ratio: float = 0.8,
    return_stats: bool = False,
) -> List[Dict] | Tuple[List[Dict], Dict]:
    """Full pipeline: extract -> merge -> carve door openings."""
    edge_count = extract_edges(room_polygons, snap_step=snap_step, ndigits=ndigits)
    bset = _boundary_edge_set(boundary_polygon, snap_step=snap_step, ndigits=ndigits)
    merged = merge_collinear_edges(edge_count, boundary_edges=bset, merge_eps=merge_eps)

    snapped_doors: List[Seg] = []
    for seg in door_segments:
        p, q = seg
        snapped_doors.append((snap_point(p, step=snap_step, ndigits=ndigits),
                              snap_point(q, step=snap_step, ndigits=ndigits)))

    snapped_windows: List[Seg] = []
    for seg in (window_segments or []):
        p, q = seg
        snapped_windows.append((snap_point(p, step=snap_step, ndigits=ndigits),
                                snap_point(q, step=snap_step, ndigits=ndigits)))

    # Deduplicate identical / reverse-direction duplicate doors.
    uniq = {}
    for d in snapped_doors:
        uniq[_door_key(d)] = d
    deduped_doors = list(uniq.values())

    uniq_windows = {}
    for w in snapped_windows:
        uniq_windows[_door_key(w)] = w
    deduped_windows = list(uniq_windows.values())

    stats: Dict = {
        "wall_segments_raw_count": len(edge_count),
        "wall_segments_merged_count": len(merged),
        "doors_input_count": len(snapped_doors),
        "doors_deduped_count": len(deduped_doors),
        "windows_input_count": len(snapped_windows),
        "windows_deduped_count": len(deduped_windows),
        "door_gaps_cut_count": 0,
        "window_gaps_cut_count": 0,
    }

    walls = cut_doors_from_walls(
        merged,
        deduped_doors,
        eps=door_eps,
        door_pad=door_pad,
        door_attach_tol=door_attach_tol,
        min_overlap_ratio=min_overlap_ratio,
        stats=stats,
        stats_key="door_gaps_cut_count",
    )

    if deduped_windows:
        walls = cut_doors_from_walls(
            walls,
            deduped_windows,
            eps=window_eps if window_eps is not None else door_eps,
            door_pad=window_pad,
            door_attach_tol=(
                window_attach_tol if window_attach_tol is not None else door_attach_tol
            ),
            min_overlap_ratio=min_overlap_ratio,
            stats=stats,
            stats_key="window_gaps_cut_count",
        )

    if return_stats:
        return walls, stats
    return walls
