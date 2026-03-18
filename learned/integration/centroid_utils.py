"""
centroid_utils.py – Lightweight utilities for centroid collapse detection and overlap filtering.

This module contains no heavy dependencies (no torch, no model imports) so it can be
easily imported and tested in isolation.
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration: Centroid Collapse Detection & Jitter
# ═══════════════════════════════════════════════════════════════════════════════

# Enable/disable centroid jitter to break ties in collapsed centroids
LEARNED_JITTER_ENABLED = os.getenv("LEARNED_JITTER_ENABLED", "true").lower() == "true"

# Gaussian noise sigma for jittering centroids (in normalized [0,1] space)
LEARNED_JITTER_SIGMA = float(os.getenv("LEARNED_JITTER_SIGMA", "0.01"))

# Enable adaptive jitter: scale sigma based on collapse severity
ADAPTIVE_JITTER_ENABLED = os.getenv("ADAPTIVE_JITTER_ENABLED", "true").lower() == "true"

# Maximum adaptive jitter multiplier (sigma can scale up to base_sigma * MAX_ADAPTIVE_MULTIPLIER)
MAX_ADAPTIVE_JITTER_MULTIPLIER = float(os.getenv("MAX_ADAPTIVE_JITTER_MULTIPLIER", "3.0"))

# Enable directional jitter: bias away from boundaries, toward interior
DIRECTIONAL_JITTER_ENABLED = os.getenv("DIRECTIONAL_JITTER_ENABLED", "true").lower() == "true"

# Distance from normalized edge [0,1] within which boundary bias activates
BOUNDARY_MARGIN = float(os.getenv("BOUNDARY_MARGIN", "0.12"))

# Minimum number of rooms required to trigger collapse detection
COLLAPSE_MIN_ROOMS = int(os.getenv("COLLAPSE_MIN_ROOMS", "3"))

# Median centroid distance threshold (collapse if below this)
COLLAPSE_MEDIAN_DIST_THRESHOLD = float(os.getenv("COLLAPSE_MEDIAN_DIST_THRESHOLD", "0.02"))

# IoU threshold for detecting bad overlaps
IOU_BAD_THRESH = float(os.getenv("IOU_BAD_THRESH", "0.5"))

# Fraction of pairs with bad IoU to trigger collapse detection
COLLAPSE_IOU_PAIR_RATIO_THRESHOLD = float(os.getenv("COLLAPSE_IOU_PAIR_RATIO_THRESHOLD", "0.30"))


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration: Early Overlap Filtering
# ═══════════════════════════════════════════════════════════════════════════════

# Enable/disable early overlap filtering (drop samples with excessive overlaps)
LEARNED_OVERLAP_FILTER_ENABLED = os.getenv("LEARNED_OVERLAP_FILTER_ENABLED", "true").lower() == "true"

# Threshold: if fraction of pairs with IoU > IOU_BAD_THRESH exceeds this, drop/resample
OVERLAP_DROP_FRAC = float(os.getenv("OVERLAP_DROP_FRAC", "0.4"))

# Maximum resample attempts when overlap filter triggers
MAX_RESAMPLE_ON_OVERLAP = int(os.getenv("MAX_RESAMPLE_ON_OVERLAP", "2"))


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper functions: Centroid & Overlap Detection
# ═══════════════════════════════════════════════════════════════════════════════

def compute_centroid(rbox) -> Tuple[float, float]:
    """Compute centroid (cx, cy) for a RoomBox."""
    cx = (getattr(rbox, "x_min", 0.0) + getattr(rbox, "x_max", 0.0)) / 2.0
    cy = (getattr(rbox, "y_min", 0.0) + getattr(rbox, "y_max", 0.0)) / 2.0
    return (cx, cy)


def compute_iou(box1, box2) -> float:
    """Compute IoU (Intersection over Union) between two RoomBox objects."""
    x1_min = getattr(box1, "x_min", 0.0)
    x1_max = getattr(box1, "x_max", 0.0)
    y1_min = getattr(box1, "y_min", 0.0)
    y1_max = getattr(box1, "y_max", 0.0)

    x2_min = getattr(box2, "x_min", 0.0)
    x2_max = getattr(box2, "x_max", 0.0)
    y2_min = getattr(box2, "y_min", 0.0)
    y2_max = getattr(box2, "y_max", 0.0)

    # Intersection
    ix_min = max(x1_min, x2_min)
    ix_max = min(x1_max, x2_max)
    iy_min = max(y1_min, y2_min)
    iy_max = min(y1_max, y2_max)

    if ix_max <= ix_min or iy_max <= iy_min:
        return 0.0

    inter_area = (ix_max - ix_min) * (iy_max - iy_min)

    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    if union_area <= 1e-9:
        return 0.0

    return inter_area / union_area


def compute_pairwise_iou_fraction(raw_rooms: List, threshold: float = 0.5) -> float:
    """Compute fraction of room pairs with IoU > threshold."""
    if len(raw_rooms) < 2:
        return 0.0

    bad_pairs = 0
    total_pairs = 0

    for i in range(len(raw_rooms)):
        for j in range(i + 1, len(raw_rooms)):
            iou = compute_iou(raw_rooms[i], raw_rooms[j])
            if iou > threshold:
                bad_pairs += 1
            total_pairs += 1

    return bad_pairs / total_pairs if total_pairs > 0 else 0.0


def compute_median_centroid_distance(raw_rooms: List) -> float:
    """Compute median pairwise Euclidean distance between room centroids."""
    if len(raw_rooms) < 2:
        return 1.0  # No collapse if only one room

    centroids = [compute_centroid(rbox) for rbox in raw_rooms]
    distances = []

    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            cx1, cy1 = centroids[i]
            cx2, cy2 = centroids[j]
            dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
            distances.append(dist)

    if not distances:
        return 1.0

    distances.sort()
    n = len(distances)
    if n % 2 == 0:
        return (distances[n // 2 - 1] + distances[n // 2]) / 2.0
    else:
        return distances[n // 2]


def compute_boundary_bias(
    cx: float,
    cy: float,
    margin: float = BOUNDARY_MARGIN,
    sigma: float = LEARNED_JITTER_SIGMA,
) -> Tuple[float, float]:
    """Compute a directional bias nudging a centroid away from the boundary.

    For a centroid near an edge, the bias pushes it inward toward the interior.
    The bias magnitude scales linearly with proximity to the edge, capped at sigma.

    Parameters
    ----------
    cx, cy : float
        Normalized centroid coordinates in [0, 1]
    margin : float
        Distance from edge within which bias activates (default: 0.12)
    sigma : float
        Max bias magnitude (matches jitter noise scale)

    Returns
    -------
    (bias_x, bias_y) directional correction to add before Gaussian noise
    """
    bias_x = 0.0
    bias_y = 0.0

    # Left edge: push right
    if cx < margin:
        strength = (margin - cx) / margin  # 0 at margin, 1 at edge
        bias_x += strength * sigma

    # Right edge: push left
    if cx > 1.0 - margin:
        strength = (cx - (1.0 - margin)) / margin
        bias_x -= strength * sigma

    # Bottom edge: push up
    if cy < margin:
        strength = (margin - cy) / margin
        bias_y += strength * sigma

    # Top edge: push down
    if cy > 1.0 - margin:
        strength = (cy - (1.0 - margin)) / margin
        bias_y -= strength * sigma

    return (bias_x, bias_y)


def detect_centroid_collapse(
    raw_rooms: List,
    min_rooms: int = COLLAPSE_MIN_ROOMS,
    median_dist_thresh: float = COLLAPSE_MEDIAN_DIST_THRESHOLD,
    iou_pair_ratio_thresh: float = COLLAPSE_IOU_PAIR_RATIO_THRESHOLD,
    iou_bad_thresh: float = IOU_BAD_THRESH,
) -> Tuple[bool, Dict[str, float]]:
    """Detect if centroids have collapsed (many rooms at same position).

    Returns
    -------
    (is_collapsed, metrics) where metrics includes:
        - median_centroid_distance
        - pairwise_iou_fraction
        - collapse_severity (0-1, higher = more severe)
    """
    if len(raw_rooms) < min_rooms:
        return False, {
            "median_centroid_distance": 1.0,
            "pairwise_iou_fraction": 0.0,
            "collapse_severity": 0.0,
        }

    median_dist = compute_median_centroid_distance(raw_rooms)
    iou_frac = compute_pairwise_iou_fraction(raw_rooms, threshold=iou_bad_thresh)

    is_collapsed = (
        median_dist < median_dist_thresh or
        iou_frac > iou_pair_ratio_thresh
    )

    # Compute collapse severity score (0-1)
    # Considers both distance and overlap metrics
    severity_dist = max(0.0, 1.0 - (median_dist / median_dist_thresh))
    severity_iou = max(0.0, (iou_frac - iou_pair_ratio_thresh) / (1.0 - iou_pair_ratio_thresh))
    collapse_severity = min(1.0, max(severity_dist, severity_iou))

    metrics = {
        "median_centroid_distance": round(median_dist, 4),
        "pairwise_iou_fraction": round(iou_frac, 4),
        "collapse_severity": round(collapse_severity, 4),
    }

    return is_collapsed, metrics


def jitter_centroids(
    raw_rooms: List,
    sigma: float = LEARNED_JITTER_SIGMA,
    seed: Optional[int] = None,
    adaptive: bool = ADAPTIVE_JITTER_ENABLED,
    collapse_severity: float = 0.0,
    directional: bool = DIRECTIONAL_JITTER_ENABLED,
) -> Dict[str, Tuple[float, float]]:
    """Apply small Gaussian jitter to centroids to break ties.

    Parameters
    ----------
    raw_rooms : List
        List of RoomBox objects with normalized coordinates
    sigma : float
        Base Gaussian noise sigma (default: 0.01)
    seed : int, optional
        Random seed for reproducibility
    adaptive : bool
        If True, scale sigma based on collapse_severity
    collapse_severity : float
        Severity score (0-1) from detect_centroid_collapse
    directional : bool
        If True, apply boundary-aware bias before Gaussian noise so
        centroids near edges are nudged inward rather than clamped hard.

    Returns
    -------
    Dict[str, Tuple[float, float]]
        {room_type: (jittered_cx, jittered_cy)} for spatial hints only.
        Does not modify actual room geometry.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Adaptive jitter: scale sigma based on collapse severity
    if adaptive and collapse_severity > 0:
        # More severe collapse → larger jitter
        # severity=0 → multiplier=1.0
        # severity=1 → multiplier=MAX_ADAPTIVE_JITTER_MULTIPLIER
        multiplier = 1.0 + (collapse_severity * (MAX_ADAPTIVE_JITTER_MULTIPLIER - 1.0))
        effective_sigma = sigma * multiplier
    else:
        effective_sigma = sigma

    # Compute original centroids per room type
    _hint_acc: Dict[str, List[Tuple[float, float]]] = {}
    for rbox in raw_rooms:
        rtype = getattr(rbox, "room_type", None)
        if not rtype:
            continue
        cx, cy = compute_centroid(rbox)
        _hint_acc.setdefault(rtype, []).append((cx, cy))

    # Average centroids per type, apply directional bias + Gaussian noise, clamp
    jittered_hints = {}
    for rtype, pts in _hint_acc.items():
        avg_cx = sum(c[0] for c in pts) / len(pts)
        avg_cy = sum(c[1] for c in pts) / len(pts)

        # Boundary-aware directional bias
        if directional:
            bias_x, bias_y = compute_boundary_bias(avg_cx, avg_cy, sigma=effective_sigma)
        else:
            bias_x, bias_y = 0.0, 0.0

        # Gaussian noise
        jitter_x = np.random.normal(0, effective_sigma)
        jitter_y = np.random.normal(0, effective_sigma)

        # Clamp to [0, 1]
        jittered_cx = max(0.0, min(1.0, avg_cx + bias_x + jitter_x))
        jittered_cy = max(0.0, min(1.0, avg_cy + bias_y + jitter_y))

        jittered_hints[rtype] = (jittered_cx, jittered_cy)

    return jittered_hints
