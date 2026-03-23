"""Inference helpers for planner-guided packing."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from geometry.adjacency_intent import build_adjacency_intent
from learned.planner.data import (
    PLANNER_PLOT_TYPE_TO_ID,
    PLANNER_ROOM_TYPE_TO_ID,
)
from learned.planner.model import PlannerTransformer, PlannerTransformerConfig


DEFAULT_PLANNER_CHECKPOINT = "learned/planner/checkpoints/room_planner.pt"
PLANNER_CHECKPOINT_ENV_VAR = "BLUEPRINTGPT_PLANNER_CHECKPOINT"

ROOM_ZONE = {
    "LivingRoom": "public",
    "DiningRoom": "public",
    "Garage": "service",
    "Kitchen": "service",
    "Store": "service",
    "Bathroom": "private",
    "WC": "private",
    "Bedroom": "private",
    "Study": "private",
    "Balcony": "public",
}

BASE_ROOM_ANCHORS = {
    "LivingRoom": [(0.50, 0.16)],
    "DiningRoom": [(0.58, 0.34)],
    "Kitchen": [(0.78, 0.38)],
    "Garage": [(0.16, 0.16)],
    "Store": [(0.88, 0.34)],
    "Bathroom": [(0.82, 0.70), (0.18, 0.70)],
    "WC": [(0.78, 0.60)],
    "Bedroom": [(0.22, 0.72), (0.78, 0.72), (0.22, 0.52), (0.78, 0.52)],
    "Study": [(0.22, 0.38)],
    "Balcony": [(0.92, 0.86)],
}


def _rotate_point(point: Tuple[float, float], entrance_side: Optional[str]) -> Tuple[float, float]:
    x, y = point
    if entrance_side == "South" or entrance_side is None:
        return x, y
    if entrance_side == "North":
        return x, 1.0 - y
    if entrance_side == "East":
        return 1.0 - y, x
    if entrance_side == "West":
        return y, 1.0 - x
    return x, y


def _room_area_ratios(spec: Dict) -> Dict[str, float]:
    weighted = {}
    total = 0.0
    for room in spec.get("rooms", []):
        area = float(room.get("area") or room.get("recommended_area_sqm") or room.get("min_area_sqm") or 1.0)
        weighted[room["name"]] = area
        total += area
    total = max(total, 1e-6)
    return {name: value / total for name, value in weighted.items()}


def _room_front_score(point: Tuple[float, float], entrance_side: Optional[str]) -> float:
    x, y = point
    if entrance_side == "North":
        return y
    if entrance_side == "East":
        return 1.0 - x
    if entrance_side == "West":
        return x
    return 1.0 - y


def _room_names_by_type(spec: Dict) -> Dict[str, List[str]]:
    names: Dict[str, List[str]] = {}
    for room in spec.get("rooms", []):
        room_type = room.get("type")
        room_name = room.get("name")
        if not room_type or not room_name:
            continue
        names.setdefault(room_type, []).append(room_name)
    return names


def _resolve_room_targets(label: str, names_by_type: Dict[str, List[str]]) -> List[str]:
    all_names = {name for names in names_by_type.values() for name in names}
    if label in all_names:
        return [label]
    return list(names_by_type.get(label, []))


def _expand_room_pairs(source_names: List[str], target_names: List[str]) -> List[Tuple[str, str]]:
    if not source_names or not target_names:
        return []
    if len(source_names) == 1:
        return [(source_names[0], target) for target in target_names if source_names[0] != target]
    if len(target_names) == 1:
        return [(source, target_names[0]) for source in source_names if source != target_names[0]]
    limit = min(len(source_names), len(target_names))
    return [
        (source_names[index], target_names[index])
        for index in range(limit)
        if source_names[index] != target_names[index]
    ]


def _normalized_named_adjacency_preferences(spec: Dict) -> List[Dict]:
    names_by_type = _room_names_by_type(spec)
    if not names_by_type:
        return []

    relation_scores = {
        "near_to": 1.0,
        "near": 1.0,
        "adjacent_to": 1.0,
        "adjacent": 1.0,
        "prefer": 0.9,
    }
    pair_scores: Dict[Tuple[str, str], float] = {}

    explicit = deepcopy(spec.get("adjacency") or [])
    if explicit:
        for item in explicit:
            if not isinstance(item, dict):
                continue
            source = item.get("source", item.get("a"))
            target = item.get("target", item.get("b"))
            relation = str(item.get("relation", item.get("type", "near_to"))).strip().lower()
            if relation not in relation_scores:
                continue
            source_names = _resolve_room_targets(str(source), names_by_type)
            target_names = _resolve_room_targets(str(target), names_by_type)
            score = float(item.get("score", item.get("weight", relation_scores[relation])))
            for left_name, right_name in _expand_room_pairs(source_names, target_names):
                key = tuple(sorted((left_name, right_name)))
                pair_scores[key] = max(score, pair_scores.get(key, 0.0))
    else:
        room_types = [room["type"] for room in spec.get("rooms", [])]
        for type_a, type_b, weight in build_adjacency_intent(room_types=room_types, use_kg=False):
            if weight <= 0:
                continue
            source_names = names_by_type.get(type_a, [])
            target_names = names_by_type.get(type_b, [])
            for left_name, right_name in _expand_room_pairs(source_names, target_names):
                key = tuple(sorted((left_name, right_name)))
                pair_scores[key] = max(float(weight), pair_scores.get(key, 0.0))

    return [
        {"a": left_name, "b": right_name, "type": "prefer", "score": round(score, 4)}
        for (left_name, right_name), score in sorted(pair_scores.items(), key=lambda item: item[1], reverse=True)
    ]


def _default_adjacency_preferences(spec: Dict) -> List[Dict]:
    return _normalized_named_adjacency_preferences(spec)


def _merge_adjacency_preferences(predicted: List[Dict], spec: Dict) -> List[Dict]:
    merged: Dict[Tuple[str, str], Dict] = {}

    strong_predicted = []
    for pref in predicted or []:
        room_a = pref.get("a")
        room_b = pref.get("b")
        if not room_a or not room_b or room_a == room_b:
            continue
        score = float(pref.get("score", pref.get("weight", 0.0)))
        if score < 0.72:
            continue
        strong_predicted.append((room_a, room_b, score))

    strong_predicted.sort(key=lambda item: item[2], reverse=True)
    for room_a, room_b, score in strong_predicted[:4]:
        key = tuple(sorted((room_a, room_b)))
        merged[key] = {"a": room_a, "b": room_b, "type": "prefer", "score": score}

    for pref in _default_adjacency_preferences(spec):
        room_a = pref["a"]
        room_b = pref["b"]
        key = tuple(sorted((room_a, room_b)))
        score = float(pref.get("score", pref.get("weight", 0.0)))
        if key in merged:
            merged[key]["score"] = max(score, merged[key]["score"])
        else:
            merged[key] = {"a": room_a, "b": room_b, "type": "prefer", "score": score}

    return sorted(merged.values(), key=lambda item: item["score"], reverse=True)


def _blend_spatial_hints(model_hints: Dict[str, List[float]], heuristic_hints: Dict[str, List[float]]) -> Dict[str, List[float]]:
    blended: Dict[str, List[float]] = {}
    for room_name in set(model_hints) | set(heuristic_hints):
        model_point = model_hints.get(room_name)
        heuristic_point = heuristic_hints.get(room_name)
        if model_point and heuristic_point:
            blended[room_name] = [
                round(0.7 * float(model_point[0]) + 0.3 * float(heuristic_point[0]), 4),
                round(0.7 * float(model_point[1]) + 0.3 * float(heuristic_point[1]), 4),
            ]
        else:
            point = model_point or heuristic_point or [0.5, 0.5]
            blended[room_name] = [round(float(point[0]), 4), round(float(point[1]), 4)]
    return blended


def _build_room_order(spatial_hints: Dict[str, List[float]], entrance_side: Optional[str], room_zones: Dict[str, str]) -> List[str]:
    zone_priority = {"public": 0, "service": 1, "private": 2}
    room_type_priority = {
        "LivingRoom": 0,
        "DrawingRoom": 0,
        "Kitchen": 1,
        "Bedroom": 2,
        "Bathroom": 3,
        "WC": 4,
    }

    def _room_type_priority(room_name: str) -> int:
        room_type = str(room_name).split("_", 1)[0]
        return room_type_priority.get(room_type, 9)

    return [
        room_name
        for room_name, _ in sorted(
            spatial_hints.items(),
            key=lambda item: (
                zone_priority.get(room_zones.get(item[0], "private"), 3),
                _room_type_priority(item[0]),
                _room_front_score(tuple(item[1]), entrance_side),
                item[0],
            ),
        )
    ]


def build_heuristic_planner_guidance(spec: Dict) -> Dict:
    """Deterministic planner fallback used until a trained planner checkpoint exists."""
    entrance_side = spec.get("entrance_side")
    area_ratios = _room_area_ratios(spec)
    anchors_used: Dict[str, int] = {}
    spatial_hints: Dict[str, List[float]] = {}
    room_zones: Dict[str, str] = {}

    for room in spec.get("rooms", []):
        room_type = room["type"]
        room_name = room["name"]
        anchor_list = BASE_ROOM_ANCHORS.get(room_type, [(0.5, 0.5)])
        anchor_index = anchors_used.get(room_type, 0) % len(anchor_list)
        anchors_used[room_type] = anchors_used.get(room_type, 0) + 1
        anchor = _rotate_point(anchor_list[anchor_index], entrance_side)
        spatial_hints[room_name] = [round(anchor[0], 4), round(anchor[1], 4)]
        room_zones[room_name] = ROOM_ZONE.get(room_type, "private")

    room_order = _build_room_order(spatial_hints, entrance_side, room_zones)

    return {
        "source": "heuristic",
        "spatial_hints": spatial_hints,
        "room_order": room_order,
        "area_ratios": {name: round(value, 6) for name, value in area_ratios.items()},
        "room_zones": room_zones,
        "adjacency_preferences": _default_adjacency_preferences(spec),
    }


def _load_planner_model(checkpoint_path: str, device: str = "cpu") -> PlannerTransformer:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = PlannerTransformerConfig(**checkpoint["config"])
    model = PlannerTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model


def predict_planner_guidance(
    spec: Dict,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> Dict:
    """Predict planner guidance or fall back to the heuristic planner."""
    checkpoint_path = checkpoint_path or os.getenv(PLANNER_CHECKPOINT_ENV_VAR, DEFAULT_PLANNER_CHECKPOINT)
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return build_heuristic_planner_guidance(spec)

    try:
        model = _load_planner_model(checkpoint_path, device=device)
    except Exception:
        return build_heuristic_planner_guidance(spec)

    heuristic_guidance = build_heuristic_planner_guidance(spec)
    rooms = spec.get("rooms", [])
    max_rooms = model.cfg.max_rooms
    room_type_ids = torch.zeros(1, max_rooms, dtype=torch.long, device=device)
    room_mask = torch.zeros(1, max_rooms, dtype=torch.bool, device=device)
    room_names: List[str] = []
    for index, room in enumerate(rooms[:max_rooms]):
        room_type_ids[0, index] = PLANNER_ROOM_TYPE_TO_ID.get(room["type"], 0)
        room_mask[0, index] = True
        room_names.append(room["name"])

    plot_type = spec.get("plot_type", "Custom")
    plot_type_id = torch.tensor(
        [PLANNER_PLOT_TYPE_TO_ID.get(plot_type, PLANNER_PLOT_TYPE_TO_ID["Custom"])],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        outputs = model(room_type_ids=room_type_ids, plot_type_ids=plot_type_id, room_mask=room_mask)

    entrance_side = spec.get("entrance_side")
    spatial_hints: Dict[str, List[float]] = {}
    room_zones: Dict[str, str] = {}
    for index, room_name in enumerate(room_names):
        raw_point = outputs["centroid"][0, index].detach().cpu().tolist()
        rotated = _rotate_point((float(raw_point[0]), float(raw_point[1])), entrance_side)
        spatial_hints[room_name] = [round(rotated[0], 4), round(rotated[1], 4)]
        room_zones[room_name] = ROOM_ZONE.get(rooms[index]["type"], "private")

    spatial_hints = _blend_spatial_hints(spatial_hints, heuristic_guidance.get("spatial_hints", {}))
    area_ratios = {
        room_name: round(float(outputs["area_ratio"][0, index].detach().cpu().item()), 6)
        for index, room_name in enumerate(room_names)
    }

    room_order = _build_room_order(spatial_hints, entrance_side, room_zones)

    adjacency_preferences: List[Dict] = []
    adjacency_scores = torch.sigmoid(outputs["adjacency_logits"][0]).detach().cpu()
    for left_index, left_name in enumerate(room_names):
        for right_index in range(left_index + 1, len(room_names)):
            score = float(adjacency_scores[left_index, right_index].item())
            if score < 0.55:
                continue
            adjacency_preferences.append(
                {
                    "a": left_name,
                    "b": room_names[right_index],
                    "type": "prefer",
                    "score": round(score, 4),
                }
            )

    adjacency_preferences = _merge_adjacency_preferences(adjacency_preferences, spec)

    return {
        "source": "planner_model",
        "checkpoint_path": checkpoint_path,
        "spatial_hints": spatial_hints,
        "room_order": room_order,
        "area_ratios": area_ratios,
        "room_zones": room_zones,
        "adjacency_preferences": adjacency_preferences,
    }



