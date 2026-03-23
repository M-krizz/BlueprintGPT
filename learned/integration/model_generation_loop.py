"""
model_generation_loop.py – Resample loop: generate up to *max_attempts*
candidate layouts from the trained LayoutTransformer, run the repair gate +
compliance on each, keep the *K* best by score, and return the overall winner.

Also collects raw / post-repair validity stats for evaluation evidence.

Main entry point
-----------------
    best_variant, summary = generate_best_layout_from_model(
        spec=spec,
        boundary_poly=boundary,
        entrance=entrance,
        sampler=sample_layout,
        tokenizer=tok,
        K=10,
        max_attempts=30,
    )
"""
from __future__ import annotations

import copy
import os
import logging
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

from learned.model.sample import load_model, sample_layout, constrained_sample_layout
from learned.model.model_cache import cached_load_model, get_cache_stats
from learned.monitoring import log_generation_quality
from learned.data.tokenizer_layout import RoomBox
from learned.templates import find_layout_template, apply_layout_template
from learned.integration.learned_to_building_adapter import adapt_generated_layout_to_building
from learned.integration.prerank import prerank_samples
from learned.integration.repair_gate import (
    validate_and_repair_generated_layout,
    evaluate_variant,
    RepairReport,
)
from generator.ranking import _score_variant

# Import centroid collapse detection and jitter utilities
from learned.integration.centroid_utils import (
    compute_centroid,
    compute_pairwise_iou_fraction,
    detect_centroid_collapse,
    jitter_centroids,
    LEARNED_JITTER_ENABLED,
    LEARNED_JITTER_SIGMA,
    ADAPTIVE_JITTER_ENABLED,
    DIRECTIONAL_JITTER_ENABLED,
    LEARNED_OVERLAP_FILTER_ENABLED,
    OVERLAP_DROP_FRAC,
    MAX_RESAMPLE_ON_OVERLAP,
    IOU_BAD_THRESH,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Quick raw-validity check (no repair)
# ═══════════════════════════════════════════════════════════════════════════════

def _raw_validity(building, regulation_file) -> Tuple[bool, List[str]]:
    """Check whether a (non-repaired) building passes hard constraints."""
    from constraints.rule_engine import RuleEngine
    from graph.connectivity import is_fully_connected
    from graph.door_graph_path import door_graph_travel_distance
    from graph.manhattan_path import max_travel_distance

    if not building.rooms:
        return False, ["no_rooms"]

    # Skip rooms without polygons (placeholders) for geometry checks
    rooms_with_poly = [r for r in building.rooms if r.polygon is not None]
    if not rooms_with_poly:
        return False, ["no_rooms_with_geometry"]

    engine = RuleEngine(regulation_file)
    occ = building.occupancy_type
    regs = engine.data.get(occ, {}).get("rooms", {})
    max_allowed = engine.get_max_travel_distance(occ)
    hard = []

    if not is_fully_connected(building):
        hard.append("not_connected")
    try:
        try:
            travel = door_graph_travel_distance(building)
            if travel is None or travel >= 999.0:
                travel = max_travel_distance(building)
        except Exception:
            travel = max_travel_distance(building)
        if travel > max_allowed:
            hard.append("travel_exceeded")
    except Exception:
        hard.append("travel_compute_error")
    for r in building.rooms:
        rule = regs.get(r.room_type)
        if rule and (r.final_area or 0) < rule["min_area"]:
            hard.append(f"{r.name}_under_area")

    return len(hard) == 0, hard


def _template_room_boxes(template_layout: Dict[str, Any], boundary_polygon) -> List[RoomBox]:
    """Convert template absolute room bounds back into normalized RoomBox objects."""
    bx0, by0, bx1, by1 = boundary_polygon.bounds
    width = max(bx1 - bx0, 1e-6)
    height = max(by1 - by0, 1e-6)
    room_boxes: List[RoomBox] = []

    for room_data in template_layout.get("rooms", []):
        try:
            x1, y1, x2, y2 = room_data["bounds"]
        except Exception:
            continue

        room_boxes.append(
            RoomBox(
                room_type=str(room_data.get("type", "Unknown")),
                x_min=max(0.0, min(1.0, (x1 - bx0) / width)),
                y_min=max(0.0, min(1.0, (y1 - by0) / height)),
                x_max=max(0.0, min(1.0, (x2 - bx0) / width)),
                y_max=max(0.0, min(1.0, (y2 - by0) / height)),
            )
        )

    return room_boxes


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def generate_best_layout_from_model(
    spec: Dict[str, Any],
    boundary_poly: List[Tuple[float, float]],
    entrance: Optional[Tuple[float, float]],
    sampler: Optional[Callable] = None,
    tokenizer=None,
    model=None,
    checkpoint_path: str = "learned/model/checkpoints/improved_v1.pt",
    regulation_file: str = "ontology/regulation_data.json",
    K: int = 10,
    max_attempts: int = 30,
    temperature: float = 0.85,
    top_p: float = 0.95,
    top_k: int = 0,
    device: str = "cpu",
    early_stop_score: float = 0.85,
    pre_rank_top_m: int = 3,
    pre_rank_gap_tolerance: float = 0.05,
    pre_rank_center_distance: float = 0.25,
    use_templates: Optional[bool] = None,
    template_threshold: float = 70.0,
) -> Tuple[Optional[Dict], Dict]:
    """
    Sample up to *max_attempts* layouts, repair each, keep top *K*, return best.

    Parameters
    ----------
    spec : dict
        Building spec (rooms, occupancy, etc.).
    boundary_poly : list[(x,y)]
        Boundary polygon in metres.
    entrance : (x,y) | None
        Entrance point on boundary.
    sampler / tokenizer / model : optional
        Pre-loaded sampler function, tokenizer, and model.  If not given they
        are loaded fresh from *checkpoint_path*.
    K : int
        Number of best candidates to keep.
    max_attempts : int
        Maximum sampling attempts before stopping.
    early_stop_score : float
        Stop early if a compliant candidate exceeds this score.

    Returns
    -------
    best_variant : dict | None
        Dict with keys: building, score, violations, status, strategy_name,
        metrics, breakdown, repair_trace, raw_rooms, candidate_index.
    summary : dict
        Aggregate stats: raw_valid_count, repaired_valid_count, total_attempts,
        top_failure_reasons, all_candidates.
    """
    occupancy = spec.get("occupancy", "Residential")
    building_type = spec.get("building_type", occupancy)
    room_types = [r["type"] for r in spec.get("rooms", [])]

    # ── Load model if needed (using cache for performance) ──────────────
    if model is None or tokenizer is None:
        model, tokenizer = cached_load_model(checkpoint_path, device=device)

    _sampler = sampler or sample_layout

    # Use constrained sampling when spec has explicit room requirements
    use_constrained = (
        _sampler is sample_layout
        and room_types
        and len(room_types) > 0
    )

    candidates: List[Dict] = []
    raw_candidates: List[Dict] = []
    failure_reasons: Counter = Counter()
    raw_valid_count = 0
    repaired_valid_count = 0
    best_so_far: Optional[Dict] = None

    # ── Diagnostics for centroid collapse & overlap filtering ─────────────
    diagnostics = {
        "jittered_count": 0,
        "raw_overlap_dropped": 0,
        "resample_attempts_on_overlap": 0,
        "centroid_collapse_detected": 0,
        "median_centroid_distances": [],
        "pairwise_iou_fractions": [],
        "collapse_severities": [],  # Track severity scores
        "final_best_was_jittered": False,  # Did the final best candidate get jitter?
    }

    # ── Template-based generation (optional fast path) ─────────────────────────
    template_used = None
    if use_templates is None:
        use_templates = os.getenv("LAYOUT_USE_TEMPLATES", "false").lower() == "true"

    if use_templates:
        try:
            # Find best matching template
            template = find_layout_template(spec)
            if template:
                compatibility = template.calculate_compatibility(spec)
                diagnostics["template_compatibility"] = compatibility

                if compatibility >= template_threshold:
                    try:
                        from shapely.geometry import Polygon

                        # Convert boundary to polygon for template engine
                        boundary_polygon = Polygon(boundary_poly)

                        # Generate layout using template
                        template_layout = apply_layout_template(template, boundary_polygon, spec)

                        template_rooms = _template_room_boxes(template_layout, boundary_polygon)

                        if template_rooms:
                            # Try to adapt the template rooms to Building format
                            try:
                                adapted_building = adapt_generated_layout_to_building(
                                    decoded_rooms=template_rooms,
                                    boundary_poly=boundary_poly,
                                    entrance=entrance,
                                    spec=spec,
                                    regulation_data=regulation_file,
                                    sample_id=-1,
                                )

                                # Create template candidate
                                template_candidate = {
                                    "raw_rooms": template_rooms,
                                    "building": adapted_building,
                                    "raw_valid": True,
                                    "index": -1,  # Special index for templates
                                    "adjacency_proxy": 0.8,  # Templates have good adjacency by design
                                    "template_source": template.name,
                                }

                                raw_candidates.append(template_candidate)
                                template_used = template.name
                                raw_valid_count += 1

                                diagnostics["template_used"] = template.name
                                diagnostics["template_quality"] = template.quality_score

                                logger.info(f"Generated template candidate '{template.name}' (quality: {template.quality_score}, compatibility: {compatibility:.1f}%)")

                            except Exception as e:
                                logger.warning(f"Template adaptation failed: {e}")

                    except ImportError:
                        logger.warning("Shapely not available for template generation")
                    except Exception as e:
                        logger.warning(f"Template generation error: {e}")

        except Exception as e:
            logger.warning(f"Template system error: {e}")
            diagnostics["template_error"] = str(e)

    # ── Stage A: sample + raw validity + collect for pre-rank ───────────────
    for attempt in range(max_attempts):
        # ── Sample ────────────────────────────────────────────────────────
        if use_constrained:
            decoded_rooms = constrained_sample_layout(
                model, tokenizer,
                spec=spec,
                building_type=building_type,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                device=device,
            )
        else:
            decoded_rooms = _sampler(
                model, tokenizer,
                building_type=building_type,
                room_types=room_types if room_types else None,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                device=device,
            )

        if not decoded_rooms:
            failure_reasons["empty_generation"] += 1
            continue

        # ── Early overlap filtering ────────────────────────────────────────
        # Drop samples with excessive overlaps before expensive adaptation
        if LEARNED_OVERLAP_FILTER_ENABLED:
            overlap_frac = compute_pairwise_iou_fraction(decoded_rooms, threshold=IOU_BAD_THRESH)
            diagnostics["pairwise_iou_fractions"].append(overlap_frac)

            if overlap_frac > OVERLAP_DROP_FRAC:
                diagnostics["raw_overlap_dropped"] += 1
                failure_reasons["raw_excessive_overlap"] += 1

                # Optionally resample (limited attempts)
                if diagnostics["resample_attempts_on_overlap"] < MAX_RESAMPLE_ON_OVERLAP:
                    diagnostics["resample_attempts_on_overlap"] += 1
                    # Decrement attempt counter to give it another try
                    # (This is a simple retry strategy; more sophisticated approach would use while loop)
                    continue
                else:
                    continue  # Skip this sample and move to next

        # ── Adapt ─────────────────────────────────────────────────────────
        building = adapt_generated_layout_to_building(
            decoded_rooms,
            boundary_poly=boundary_poly,
            entrance=entrance,
            spec=spec,
            regulation_data=regulation_file,
            sample_id=attempt,
        )

        if not building.rooms:
            failure_reasons["no_rooms_after_adapt"] += 1
            continue

        # ── Raw validity check ────────────────────────────────────────────
        raw_ok, raw_issues = _raw_validity(building, regulation_file)
        if raw_ok:
            raw_valid_count += 1
        for issue in raw_issues:
            failure_reasons[f"raw_{issue}"] += 1

        raw_candidates.append({
            "index": attempt,
            "raw_rooms": decoded_rooms,
            "raw_valid": raw_ok,
            "raw_issues": raw_issues,
            "building": building,
        })

    if not raw_candidates:
        summary = {
            "total_attempts": failure_reasons.total(),
            "valid_samples": 0,
            "raw_valid_count": raw_valid_count,
            "repaired_valid_count": 0,
            "pre_ranked_count": 0,
            "top_failure_reasons": dict(failure_reasons.most_common(10)),
            "all_candidates": [],
            "diagnostics": {
                "jittered_count": 0,
                "final_best_was_jittered": False,
                "raw_overlap_dropped": diagnostics["raw_overlap_dropped"],
                "resample_attempts_on_overlap": diagnostics["resample_attempts_on_overlap"],
                "centroid_collapse_detected": 0,
                "avg_median_centroid_distance": 0.0,
                "avg_pairwise_iou_fraction": 0.0,
                "avg_collapse_severity": 0.0,
            },
        }
        return None, summary

    # ── Stage B: adjacency-aware pre-rank; keep top-M for repair ───────────
    shortlisted = prerank_samples(
        raw_candidates,
        spec,
        top_m=min(pre_rank_top_m, len(raw_candidates)),
        gap_tolerance=pre_rank_gap_tolerance,
        center_distance_threshold=pre_rank_center_distance,
    )

    # ── Stage C: repair only shortlisted candidates ─────────────────────────
    for raw in shortlisted:
        attempt = raw["index"]
        decoded_rooms = raw["raw_rooms"]
        building = raw["building"]
        raw_ok = raw.get("raw_valid", False)
        adjacency_proxy = raw.get("adjacency_proxy", 0.0)

        # ── Repair gate ───────────────────────────────────────────────────
        repaired, violations, status, repair_trace, repair_report = validate_and_repair_generated_layout(
            building,
            boundary_polygon=boundary_poly,
            entrance_point=entrance,
            regulation_file=regulation_file,
            spec=spec,
        )

        if status == "COMPLIANT":
            repaired_valid_count += 1

        # ── Score (same ranker as algorithmic variants) ───────────────────
        metrics = evaluate_variant(repaired, regulation_file, entrance)
        pseudo = {"building": repaired, "metrics": metrics, "modifications": violations}
        score, breakdown = _score_variant(pseudo)

        cand = {
            "index": attempt,
            "raw_rooms": decoded_rooms,
            "raw_valid": raw_ok,
            "adjacency_proxy": adjacency_proxy,
            "building": repaired,
            "violations": violations,
            "status": status,
            "score": score,
            "breakdown": breakdown,
            "metrics": metrics,
            "repair_trace": repair_trace,
            "repair_report": repair_report.to_dict() if repair_report else None,
        }
        candidates.append(cand)

        # ── Track best ────────────────────────────────────────────────────
        if best_so_far is None or score > best_so_far["score"]:
            best_so_far = cand

        # ── Early stop ────────────────────────────────────────────────────
        if status == "COMPLIANT" and score >= early_stop_score:
            break

    # ── Sort and trim to top K ────────────────────────────────────────────
    candidates.sort(key=lambda c: c.get("score", -1.0), reverse=True)
    top_k_candidates = candidates[:K]

    summary = {
        "total_attempts": len(raw_candidates) + failure_reasons.total(),
        "valid_samples": len(candidates),
        "raw_valid_count": raw_valid_count,
        "repaired_valid_count": repaired_valid_count,
        "pre_ranked_count": len(shortlisted),
        "top_failure_reasons": dict(failure_reasons.most_common(10)),
        "all_candidates": top_k_candidates,
        # ── New diagnostics ────────────────────────────────────────────────
        "diagnostics": {
            "jittered_count": diagnostics["jittered_count"],
            "final_best_was_jittered": diagnostics["final_best_was_jittered"],
            "raw_overlap_dropped": diagnostics["raw_overlap_dropped"],
            "resample_attempts_on_overlap": diagnostics["resample_attempts_on_overlap"],
            "centroid_collapse_detected": diagnostics["centroid_collapse_detected"],
            "avg_median_centroid_distance": round(
                sum(diagnostics["median_centroid_distances"]) / len(diagnostics["median_centroid_distances"])
                if diagnostics["median_centroid_distances"] else 0.0, 4
            ),
            "avg_pairwise_iou_fraction": round(
                sum(diagnostics["pairwise_iou_fractions"]) / len(diagnostics["pairwise_iou_fractions"])
                if diagnostics["pairwise_iou_fractions"] else 0.0, 4
            ),
            "avg_collapse_severity": round(
                sum(diagnostics["collapse_severities"]) / len(diagnostics["collapse_severities"])
                if diagnostics["collapse_severities"] else 0.0, 4
            ),
        },
    }

    if not best_so_far or best_so_far.get("building") is None:
        return None, summary

    best = top_k_candidates[0] if top_k_candidates else best_so_far

    # ── Extract spatial hints from the transformer output ───────────────────
    # The raw_rooms list contains RoomBox tokens with normalized [0,1] positions.
    # Build a dict {room_type: (cx_norm, cy_norm)} for each unique room type.
    # When multiple rooms of the same type exist, we average their centroids.
    # These hints seed the PolygonPacker's bisection ordering (not room geometry).
    raw_rooms = best.get("raw_rooms", [])

    # ── Centroid collapse detection & jitter ────────────────────────────────
    is_collapsed = False
    collapse_metrics = {}

    if LEARNED_JITTER_ENABLED and raw_rooms:
        is_collapsed, collapse_metrics = detect_centroid_collapse(raw_rooms)
        diagnostics["median_centroid_distances"].append(collapse_metrics.get("median_centroid_distance", 1.0))

        # Track collapse severity for adaptive jitter
        collapse_severity = collapse_metrics.get("collapse_severity", 0.0)
        diagnostics["collapse_severities"].append(collapse_severity)

        if is_collapsed:
            diagnostics["centroid_collapse_detected"] += 1
            diagnostics["jittered_count"] += 1
            diagnostics["final_best_was_jittered"] = True  # Mark that final best got jitter
            # Apply adaptive + directional jitter to break ties
            learned_spatial_hints = jitter_centroids(
                raw_rooms,
                sigma=LEARNED_JITTER_SIGMA,
                adaptive=ADAPTIVE_JITTER_ENABLED,
                collapse_severity=collapse_severity,
                directional=DIRECTIONAL_JITTER_ENABLED,
            )
        else:
            # No collapse: use normal centroid averaging
            _hint_acc: dict = {}   # {room_type: [(cx, cy), ...]}
            for rbox in raw_rooms:
                rtype = getattr(rbox, "room_type", None)
                if not rtype:
                    continue
                cx, cy = compute_centroid(rbox)
                _hint_acc.setdefault(rtype, []).append((cx, cy))
            learned_spatial_hints = {
                rtype: (
                    sum(c[0] for c in pts) / len(pts),
                    sum(c[1] for c in pts) / len(pts),
                )
                for rtype, pts in _hint_acc.items()
            }
    else:
        # Jitter disabled or no rooms: use normal averaging
        _hint_acc: dict = {}   # {room_type: [(cx, cy), ...]}
        for rbox in raw_rooms:
            rtype = getattr(rbox, "room_type", None)
            if not rtype:
                continue
            cx, cy = compute_centroid(rbox)
            _hint_acc.setdefault(rtype, []).append((cx, cy))
        learned_spatial_hints = {
            rtype: (
                sum(c[0] for c in pts) / len(pts),
                sum(c[1] for c in pts) / len(pts),
            )
            for rtype, pts in _hint_acc.items()
        }

    best_variant = {
        "building": best["building"],
        "score": best["score"],
        "violations": best["violations"],
        "status": best["status"],
        "strategy_name": "learned-generation",
        "metrics": best.get("metrics", {}),
        "breakdown": best.get("breakdown", {}),
        "repair_trace": best.get("repair_trace", []),
        "all_candidates": top_k_candidates,
        "raw_rooms": raw_rooms,
        "candidate_index": best["index"],
        "raw_valid": best.get("raw_valid", False),
        # Spatial hints for the PolygonPacker: {room_type: (cx_norm, cy_norm)}
        "learned_spatial_hints": learned_spatial_hints,
    }
    return best_variant, summary


# ═══════════════════════════════════════════════════════════════════════════════
#  Simplified wrapper (backward-compatible with old generate_best_layout API)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_best_layout(
    checkpoint_path: str = "learned/model/checkpoints/improved_v1.pt",
    boundary_polygon: List[Tuple[float, float]] = None,
    entrance_point: Optional[Tuple[float, float]] = None,
    occupancy_type: str = "Residential",
    building_type: str = "Residential",
    room_types: Optional[List[str]] = None,
    regulation_file: str = "ontology/regulation_data.json",
    K: int = 10,
    temperature: float = 0.85,
    top_p: float = 0.95,
    top_k: int = 0,
    device: str = "cpu",
) -> Dict:
    """Backward-compatible wrapper around ``generate_best_layout_from_model``."""
    if boundary_polygon is None:
        boundary_polygon = [(0, 0), (15, 0), (15, 10), (0, 10)]

    spec = {
        "occupancy": occupancy_type,
        "building_type": building_type,
        "rooms": [{"name": f"{rt}_1", "type": rt} for rt in (room_types or [])],
    }

    best, summary = generate_best_layout_from_model(
        spec=spec,
        boundary_poly=boundary_polygon,
        entrance=entrance_point,
        checkpoint_path=checkpoint_path,
        regulation_file=regulation_file,
        K=K,
        max_attempts=max(K * 3, 15),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        device=device,
    )

    if best is None:
        return {
            "building": None,
            "score": -1.0,
            "violations": ["all samples failed"],
            "status": "NON_COMPLIANT",
            "strategy_name": "learned-generation",
            "metrics": {},
            "all_candidates": summary.get("all_candidates", []),
            "raw_rooms": [],
        }

    return best
