"""
generate_best_layout.py – Resample loop: generate K candidate layouts from the
trained Transformer, run repair + compliance on each, keep the best by score.

Main entry point
-----------------
    result = generate_best_layout(
        checkpoint_path="learned/model/checkpoints/kaggle_test.pt",
        boundary_polygon=[(0,0),(15,0),(15,10),(0,10)],
        entrance_point=(0.2, 0.0),
        K=10,
    )
    best_building = result["building"]
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

from learned.model.sample import load_model, sample_layout
from learned.integration.learned_to_building_adapter import adapt_generated_layout
from learned.integration.validate_and_repair import (
    validate_and_repair_generated_layout,
    evaluate_variant,
)
from generator.ranking import _score_variant


def generate_best_layout(
    checkpoint_path: str = "learned/model/checkpoints/kaggle_test.pt",
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
    """
    Sample K layouts, repair each, score, return the best.

    Returns
    -------
    dict with keys:
        building      – best Building object (repaired, with corridors + doors)
        score         – ranking score (float)
        violations    – list[str] repair actions applied
        status        – "COMPLIANT" or "NON_COMPLIANT"
        strategy_name – corridor strategy used
        metrics       – evaluation metrics dict
        all_candidates – list of all K candidate results (for analysis)
        raw_rooms     – list[RoomBox] from the best raw sample (before repair)
    """
    if boundary_polygon is None:
        boundary_polygon = [(0, 0), (15, 0), (15, 10), (0, 10)]

    model, tok = load_model(checkpoint_path, device=device)

    candidates = []

    for i in range(K):
        # ── Sample ────────────────────────────────────────────────────────
        decoded_rooms = sample_layout(
            model, tok,
            building_type=building_type,
            room_types=room_types,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            device=device,
        )

        if not decoded_rooms:
            candidates.append({
                "index": i,
                "raw_rooms": [],
                "building": None,
                "violations": ["empty generation"],
                "status": "NON_COMPLIANT",
                "score": -1.0,
            })
            continue

        # ── Adapt to Building ─────────────────────────────────────────────
        building = adapt_generated_layout(
            decoded_rooms,
            boundary_polygon,
            entrance_point=entrance_point,
            occupancy_type=occupancy_type,
        )

        if not building.rooms:
            candidates.append({
                "index": i,
                "raw_rooms": decoded_rooms,
                "building": building,
                "violations": ["no rooms after adaptation"],
                "status": "NON_COMPLIANT",
                "score": -1.0,
            })
            continue

        # ── Repair ────────────────────────────────────────────────────────
        repaired, violations, status = validate_and_repair_generated_layout(
            building,
            boundary_polygon,
            entrance_point=entrance_point,
            regulation_file=regulation_file,
        )

        # ── Score using the same ranker as algorithmic variants ───────────
        metrics = evaluate_variant(repaired, regulation_file, entrance_point)
        exit_w = repaired.exit.width if repaired.exit else 0

        pseudo_variant = {
            "building": repaired,
            "metrics": metrics,
            "modifications": violations,
        }
        score, breakdown = _score_variant(pseudo_variant)

        candidates.append({
            "index": i,
            "raw_rooms": decoded_rooms,
            "building": repaired,
            "violations": violations,
            "status": status,
            "score": score,
            "breakdown": breakdown,
            "metrics": metrics,
        })

    # ── Pick best ─────────────────────────────────────────────────────────
    candidates.sort(key=lambda c: c.get("score", -1.0), reverse=True)
    best = candidates[0] if candidates else None

    if best is None or best.get("building") is None:
        return {
            "building": None,
            "score": -1.0,
            "violations": ["all K samples failed"],
            "status": "NON_COMPLIANT",
            "strategy_name": "learned-generation",
            "metrics": {},
            "all_candidates": candidates,
            "raw_rooms": [],
        }

    return {
        "building": best["building"],
        "score": best["score"],
        "violations": best["violations"],
        "status": best["status"],
        "strategy_name": "learned-generation",
        "metrics": best.get("metrics", {}),
        "breakdown": best.get("breakdown", {}),
        "all_candidates": candidates,
        "raw_rooms": best.get("raw_rooms", []),
        "candidate_index": best["index"],
    }
