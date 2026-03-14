"""
prerank.py - Lightweight adjacency-aware pre-ranking for raw model samples.

Used before repair to reduce repair load and keep only stronger candidates.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from learned.data.tokenizer_layout import RoomBox
from geometry.adjacency_intent import build_adjacency_intent


def _center(box: RoomBox) -> Tuple[float, float]:
    return ((box.x_min + box.x_max) / 2.0, (box.y_min + box.y_max) / 2.0)


def _axis_gap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    """Gap between 1D intervals (0 if overlapping)."""
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0.0


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _are_adjacent_proxy(
    a: RoomBox,
    b: RoomBox,
    *,
    gap_tolerance: float = 0.05,
    center_distance_threshold: float = 0.25,
) -> bool:
    """Fast adjacency proxy for normalized [0,1] boxes.

    Adjacent if:
    - near in x with y-overlap, or near in y with x-overlap, or
    - center distance below threshold.
    """
    x_gap = _axis_gap_1d(a.x_min, a.x_max, b.x_min, b.x_max)
    y_gap = _axis_gap_1d(a.y_min, a.y_max, b.y_min, b.y_max)
    x_ov = _overlap_1d(a.x_min, a.x_max, b.x_min, b.x_max)
    y_ov = _overlap_1d(a.y_min, a.y_max, b.y_min, b.y_max)

    if (x_gap <= gap_tolerance and y_ov > 0.0) or (y_gap <= gap_tolerance and x_ov > 0.0):
        return True

    ax, ay = _center(a)
    bx, by = _center(b)
    d2 = (ax - bx) ** 2 + (ay - by) ** 2
    return d2 <= center_distance_threshold ** 2


def score_adjacency_proxy(
    room_boxes: Sequence[RoomBox],
    intent_graph: Sequence[Tuple[str, str, float]],
    *,
    gap_tolerance: float = 0.05,
    center_distance_threshold: float = 0.25,
) -> float:
    """Return weighted adjacency satisfaction in [0,1]."""
    typed: Dict[str, List[RoomBox]] = {}
    for rb in room_boxes:
        typed.setdefault(rb.room_type, []).append(rb)

    total_weight = sum(w for _, _, w in intent_graph) or 1.0
    satisfied = 0.0

    for ta, tb, w in intent_graph:
        ok = False
        for a in typed.get(ta, []):
            for b in typed.get(tb, []):
                if a is b:
                    continue
                if _are_adjacent_proxy(
                    a,
                    b,
                    gap_tolerance=gap_tolerance,
                    center_distance_threshold=center_distance_threshold,
                ):
                    ok = True
                    break
            if ok:
                break
        if ok:
            satisfied += w

    return round(satisfied / total_weight, 4)


def prerank_samples(
    samples: Sequence[Dict],
    spec: Dict,
    *,
    top_m: int = 3,
    use_kg: bool = True,
    gap_tolerance: float = 0.05,
    center_distance_threshold: float = 0.25,
) -> List[Dict]:
    """Rank raw sample dicts by adjacency proxy and return top_m items.

    Each sample dict must include key ``raw_rooms`` (list[RoomBox]).
    """
    room_types = [r.get("type") for r in spec.get("rooms", []) if r.get("type")]
    intent = build_adjacency_intent(room_types=room_types or None, use_kg=use_kg)

    scored: List[Dict] = []
    for sample in samples:
        rooms = sample.get("raw_rooms", [])
        adj = score_adjacency_proxy(
            rooms,
            intent,
            gap_tolerance=gap_tolerance,
            center_distance_threshold=center_distance_threshold,
        )
        entry = dict(sample)
        entry["adjacency_proxy"] = adj
        scored.append(entry)

    scored.sort(key=lambda s: s.get("adjacency_proxy", 0.0), reverse=True)
    return scored[: max(1, top_m)]
