from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


VALID_ROOM_TYPES = [
    "Bedroom",
    "Kitchen",
    "Bathroom",
    "LivingRoom",
    "DiningRoom",
    "DrawingRoom",
    "Garage",
    "Store",
    "Pantry",
    "WC",
    "Study",
    "Balcony",
]


def _history_block(conversation_history: Optional[List[Dict[str, Any]]]) -> str:
    if not conversation_history:
        return "[]"
    compact = [
        {
            "role": str(item.get("role") or "user"),
            "content": str(item.get("content") or "")[:400],
        }
        for item in conversation_history[-8:]
    ]
    return json.dumps(compact, ensure_ascii=True)


def build_intent_prompt(
    user_message: str,
    *,
    context: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    context = context or {}
    payload = {
        "message": user_message,
        "state": context.get("state"),
        "has_design": bool(context.get("num_designs", 0) or context.get("latest_design")),
        "history": json.loads(_history_block(conversation_history)),
    }
    return (
        "You are BlueprintGPT's NL intent router. "
        "Classify the user message into exactly one intent: design, correction, question, or conversation. "
        "If the user already has a design and asks to move, resize, swap, add, remove, or change adjacency/relationship, choose correction. "
        "Return JSON only with keys: intent, confidence, reason, design_keywords, question_type.\n"
        + json.dumps(payload, ensure_ascii=True)
    )


def build_spec_prompt(
    user_message: str,
    *,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    schema = {
        "building_type": "Residential",
        "layout_type": "1BHK|2BHK|3BHK|4BHK|custom|null",
        "plot_type": "plot|site|footprint|custom|null",
        "entrance_side": "North|South|East|West|null",
        "boundary_size": {"width": "number|null", "height": "number|null", "unit": "m|ft|null"},
        "rooms": [{"type": "RoomType", "count": 1}],
        "preferences": {
            "adjacency": [["RoomTypeOrName", "RoomTypeOrName", "near_to|adjacent_to|separate|buffer_zone"]],
            "privacy": {"RoomType": "public|private|service"},
        },
        "style_hints": ["compact", "low_corridor", "privacy_focused"],
        "assumptions_used": [],
        "unresolved_fields": [],
    }
    payload = {
        "message": user_message,
        "conversation_history": json.loads(_history_block(conversation_history)),
        "valid_room_types": VALID_ROOM_TYPES,
        "schema": schema,
    }
    return (
        "You extract a normalized residential floor-plan specification from the user's natural language. "
        "Preserve only supported room types. Expand shorthand like 3BHK only if the user does not specify room counts explicitly. "
        "Do not invent geometry, SVG, or polygons. Return strict JSON only.\n"
        + json.dumps(payload, ensure_ascii=True)
    )


def build_correction_prompt(
    user_message: str,
    *,
    design_index: int,
    current_rooms: Optional[List[Dict[str, Any]]] = None,
) -> str:
    room_names = [
        {
            "name": str(room.get("name") or room.get("type") or ""),
            "type": str(room.get("type") or room.get("name") or ""),
        }
        for room in (current_rooms or [])
    ]
    schema = {
        "understood": True,
        "changes": [
            {
                "type": "move_room|resize_room|swap_rooms|add_room|remove_room|change_adjacency",
                "room": "Bedroom_1",
                "room_a": "Bedroom_1",
                "room_b": "Bedroom_2",
                "room_type": "Store",
                "direction": "left|right|up|down|north|south|east|west",
                "size_change": "larger|smaller|wider|narrower",
                "relation": "adjacent_to|near_to|separate|buffer_zone",
            }
        ],
        "clarification_needed": None,
    }
    payload = {
        "message": user_message,
        "design_index": design_index,
        "current_rooms": room_names,
        "schema": schema,
    }
    return (
        "You parse a user's modification request for an existing floor plan. "
        "Prefer named rooms like Bedroom_1 when possible. "
        "If the request is clear, return understood=true and structured changes. "
        "Return JSON only.\n"
        + json.dumps(payload, ensure_ascii=True)
    )


def build_topology_prompt(
    semantic_spec: Dict[str, Any],
    room_program: Dict[str, Any],
    *,
    base_zoning_plan: Optional[Dict[str, Any]] = None,
) -> str:
    schema = {
        "layout_pattern": "balanced|compact_frontage|public_front_private_rear|zonal_split|front_public_back_private|null",
        "frontage_room": "LivingRoom_1|null",
        "public_zone": ["LivingRoom_1"],
        "service_zone": ["Kitchen_1"],
        "private_zone": ["Bedroom_1", "Bedroom_2"],
        "room_order": ["LivingRoom_1", "Kitchen_1", "Bedroom_1"],
        "size_priors": {"LivingRoom_1": "small|medium|large|xlarge"},
        "named_adjacency": [{"a": "Kitchen_1", "b": "LivingRoom_1", "type": "prefer|avoid", "score": 0.9}],
        "heuristics": ["Keep the living room at the frontage."],
    }
    payload = {
        "semantic_spec": semantic_spec,
        "room_program": room_program,
        "base_zoning_plan": base_zoning_plan or {},
        "schema": schema,
    }
    return (
        "You propose soft topology hints for a residential floor plan. "
        "Do not emit coordinates, polygons, or SVG. "
        "Use only room names that already exist in room_program.rooms. "
        "Return JSON only with optional topology hints that can be merged into a deterministic zoning planner.\n"
        + json.dumps(payload, ensure_ascii=True)
    )


def build_chat_prompt(
    user_message: str,
    *,
    context: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    payload = {
        "message": user_message,
        "context": context or {},
        "conversation_history": json.loads(_history_block(conversation_history)),
    }
    return (
        "You are BlueprintGPT's conversational layer. "
        "Answer naturally, stay grounded in the provided context, and do not invent geometry facts. "
        "If the user asks for a change, describe the change clearly and keep the current design context in mind.\n"
        + json.dumps(payload, ensure_ascii=True)
    )
