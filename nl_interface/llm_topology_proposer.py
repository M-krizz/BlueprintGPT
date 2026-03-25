from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

from nl_interface.chat_spec_adapter import ChatSpecAdapter, default_chat_spec_adapter
from nl_interface.contracts import RoomProgram, SemanticSpec, ZoningPlan


SIZE_LABEL_SCALE = {
    "small": 0.86,
    "medium": 1.0,
    "large": 1.16,
    "xlarge": 1.28,
}


def propose_topology_hints(
    semantic_spec: SemanticSpec,
    room_program: RoomProgram,
    base_zoning_plan: ZoningPlan,
    *,
    chat_adapter: Optional[ChatSpecAdapter] = None,
) -> ZoningPlan:
    adapter = chat_adapter or default_chat_spec_adapter()
    try:
        hints = adapter.propose_topology(
            semantic_spec.to_dict(),
            room_program.to_dict(),
            base_zoning_plan=base_zoning_plan.to_dict(),
        )
    except Exception:
        hints = None
    if not hints:
        return base_zoning_plan
    provider_name = getattr(adapter, "provider_name", "llm")
    return _merge_topology_hints(base_zoning_plan, hints, provider_name=provider_name)


def _merge_topology_hints(base: ZoningPlan, hints: Dict[str, Any], *, provider_name: str) -> ZoningPlan:
    room_names = set(base.zone_map.keys()) | set(base.spatial_hints.keys()) | set(base.room_order)
    zone_map = dict(base.zone_map)
    for key, zone_name in (
        ("public_zone", "public"),
        ("service_zone", "service"),
        ("private_zone", "private"),
    ):
        for room_name in hints.get(key, []) or []:
            if room_name in room_names:
                zone_map[room_name] = zone_name

    frontage_room = hints.get("frontage_room") if hints.get("frontage_room") in room_names else base.frontage_room

    room_order = []
    for room_name in hints.get("room_order") or []:
        if room_name in room_names and room_name not in room_order:
            room_order.append(room_name)
    for room_name in base.room_order:
        if room_name not in room_order:
            room_order.append(room_name)

    size_priors = {name: dict(priors) for name, priors in base.size_priors.items()}
    for room_name, prior in (hints.get("size_priors") or {}).items():
        if room_name not in room_names:
            continue
        if isinstance(prior, str):
            _apply_size_label(size_priors.setdefault(room_name, {}), prior)
        elif isinstance(prior, dict):
            target = size_priors.setdefault(room_name, {})
            for key in ("ideal_area_sqm", "min_area_sqm", "max_area_sqm"):
                if prior.get(key) is not None:
                    target[key] = float(prior[key])

    named_adjacency = [dict(item) for item in base.named_adjacency]
    existing_edges = {
        (item.get("a"), item.get("b"), item.get("type"))
        for item in named_adjacency
    }
    for item in hints.get("named_adjacency") or []:
        edge = (item.get("a"), item.get("b"), item.get("type"))
        if item.get("a") in room_names and item.get("b") in room_names and edge not in existing_edges:
            named_adjacency.append(dict(item))
            existing_edges.add(edge)

    heuristics = list(base.heuristics)
    for text in hints.get("heuristics") or []:
        if text and text not in heuristics:
            heuristics.append(str(text))
    assumptions_used = list(base.assumptions_used)
    assumptions_note = f"Merged {provider_name} topology hints into the deterministic zoning plan."
    if assumptions_note not in assumptions_used:
        assumptions_used.append(assumptions_note)

    return replace(
        base,
        layout_pattern=str(hints.get("layout_pattern") or base.layout_pattern),
        frontage_room=frontage_room,
        zone_map=zone_map,
        room_order=room_order,
        size_priors=size_priors,
        named_adjacency=named_adjacency,
        heuristics=heuristics,
        assumptions_used=assumptions_used,
        topology_source=provider_name,
        topology_hints_applied=True,
    )


def _apply_size_label(priors: Dict[str, float], label: str) -> None:
    scale = SIZE_LABEL_SCALE.get(str(label).strip().lower())
    if scale is None:
        return
    ideal = float(priors.get("ideal_area_sqm", priors.get("min_area_sqm", 8.0)))
    min_area = float(priors.get("min_area_sqm", min(ideal, 4.0)))
    max_area = float(priors.get("max_area_sqm", max(ideal, min_area)))
    adjusted = max(min_area, min(max_area, ideal * scale))
    priors["ideal_area_sqm"] = round(adjusted, 3)
