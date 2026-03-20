"""
gemini_adapter.py – Google Gemini API integration for natural language understanding,
design explanation, and correction parsing.

This module provides:
1. NL → Spec conversion (parse user intent into system arguments)
2. Design → Explanation (explain generated layouts and rankings)
3. Correction → Delta (parse user corrections into actionable changes)
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Gemini API
try:
    import google.generativeai as genai
    _HAS_GENAI = True
except ImportError:
    genai = None
    _HAS_GENAI = False

# ── Configuration ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "true").lower() == "true"


def _init_gemini() -> bool:
    """Initialize Gemini client. Returns True if successful."""
    if not _HAS_GENAI:
        return False
    if not GEMINI_API_KEY:
        return False
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return True
    except Exception:
        return False


_gemini_ready = _init_gemini() if GEMINI_ENABLED else False


# ═══════════════════════════════════════════════════════════════════════════════
#  System Prompts
# ═══════════════════════════════════════════════════════════════════════════════

SPEC_EXTRACTION_PROMPT = """You are an AI assistant for an architectural floor plan generator.
Your task is to extract structured specifications from natural language descriptions.

Extract the following from the user's message:
1. **rooms**: List of room types and counts (e.g., "2 bedrooms" → [{"type": "Bedroom", "count": 2}])
   - **BHK Format**: "3BHK" means 3 Bedrooms + 1 Hall (LivingRoom) + 1 Kitchen + 2 Bathrooms
   - "2BHK" means 2 Bedrooms + 1 Hall (LivingRoom) + 1 Kitchen + 1 Bathroom
2. **adjacency**: Room relationships (e.g., "kitchen near dining" → [{"source": "Kitchen", "target": "DiningRoom", "relation": "near_to"}])
3. **entrance_side**: North, South, East, or West
4. **plot_type**: "5marla", "10marla", "20marla", or "custom"
5. **style_hints**: Any style preferences like "open plan", "compact", "privacy first", "minimize corridor"
6. **corrections**: If user is requesting changes to a previous design

Valid room types: Bedroom, Kitchen, Bathroom, LivingRoom, DiningRoom, DrawingRoom, Garage, Store

Respond ONLY with a valid JSON object. Example:
{
  "rooms": [{"type": "Bedroom", "count": 2}, {"type": "Kitchen", "count": 1}],
  "adjacency": [{"source": "Kitchen", "target": "DiningRoom", "relation": "near_to"}],
  "entrance_side": "North",
  "plot_type": "10marla",
  "style_hints": ["open plan"],
  "corrections": null,
  "intent": "new_design"
}

If user is requesting a correction, set intent to "correction" and populate corrections field:
{
  "intent": "correction",
  "corrections": {
    "target_design_index": 0,
    "changes": [
      {"type": "move_room", "room": "Kitchen", "direction": "left"},
      {"type": "resize_room", "room": "Bedroom_1", "size_change": "larger"},
      {"type": "add_room", "room_type": "Bathroom"},
      {"type": "remove_room", "room": "Store"},
      {"type": "change_adjacency", "room_a": "Kitchen", "room_b": "LivingRoom", "new_relation": "adjacent_to"}
    ]
  }
}
"""

EXPLANATION_PROMPT = """You are an AI assistant explaining architectural floor plan designs to users.

Given the design data below, provide a clear, friendly explanation that covers:
1. **Overview**: Brief summary of what was generated
2. **Room Layout**: How rooms are arranged and why
3. **Strengths**: What works well in this design
4. **Weaknesses**: Any limitations or trade-offs
5. **Ranking Reason**: Why this design is ranked at position {rank} out of {total}
6. **Suggestions**: What could be improved (if the user wants to request changes)

Be conversational and helpful. Use simple language. Mention specific rooms and their positions.
If there are compliance violations, explain them simply.

Design Data:
{design_data}

Metrics:
- Design Score: {score}
- Compliance Status: {compliance_status}
- Travel Distance: {travel_distance}m (max allowed: {max_travel}m)
- Room Coverage: {room_coverage}
- Violations: {violations}
"""

CORRECTION_PROMPT = """You are an AI assistant parsing user corrections for floor plans.

The user wants to modify design #{design_index}. Parse their request into actionable changes.

Current design has these rooms: {current_rooms}

User's correction request: "{user_request}"

Extract the changes as a JSON object:
{
  "understood": true,
  "changes": [
    {"type": "move_room", "room": "Kitchen", "direction": "right", "amount": "2m"},
    {"type": "resize_room", "room": "Bedroom_1", "dimension": "width", "change": "+1m"},
    {"type": "swap_rooms", "room_a": "Kitchen", "room_b": "DiningRoom"},
    {"type": "add_room", "room_type": "Bathroom", "near": "Bedroom_1"},
    {"type": "remove_room", "room": "Store"},
    {"type": "change_adjacency", "room_a": "Kitchen", "room_b": "LivingRoom", "relation": "adjacent_to"}
  ],
  "clarification_needed": null
}

If the request is ambiguous, set "understood" to false and explain in "clarification_needed".
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Core Functions
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract a JSON object from text, handling markdown fences and nested braces."""
    # Strip markdown code fences if present
    stripped = re.sub(r'```(?:json)?\s*', '', text).strip().rstrip('`')

    # Try parsing the whole thing first (common case: response is pure JSON)
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass

    # Find all { positions and try parsing from each one
    for i, ch in enumerate(stripped):
        if ch == '{':
            # Walk forward tracking brace depth
            depth = 0
            for j in range(i, len(stripped)):
                if stripped[j] == '{':
                    depth += 1
                elif stripped[j] == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = stripped[i:j+1]
                        try:
                            return json.loads(candidate)
                        except (json.JSONDecodeError, ValueError):
                            break  # This opening brace didn't work, try next
    return None

def is_available() -> bool:
    """Check if Gemini is available and configured."""
    return _gemini_ready and GEMINI_ENABLED


def extract_spec_from_nl(user_text: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Extract structured specification from natural language using Gemini.

    Falls back to regex-based extraction if Gemini is unavailable.
    """
    print(f"\n[GEMINI_ADAPTER] extract_spec_from_nl called")
    print(f"[GEMINI_ADAPTER] User text: '{user_text}'")
    print(f"[GEMINI_ADAPTER] Gemini available: {is_available()}")

    if not is_available():
        print(f"[GEMINI_ADAPTER] Using fallback extraction (Gemini not available)")
        result = _fallback_extract(user_text)
        print(f"[GEMINI_ADAPTER] Fallback result: {result}")
        return result

    try:
        print(f"[GEMINI_ADAPTER] Calling Gemini API with model: {GEMINI_MODEL}")
        model = genai.GenerativeModel(GEMINI_MODEL)

        # Build context with conversation history
        messages = []
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                messages.append(f"{msg['role']}: {msg['content']}")

        context = "\n".join(messages) if messages else ""
        full_prompt = f"{SPEC_EXTRACTION_PROMPT}\n\nConversation context:\n{context}\n\nUser message: {user_text}"

        response = model.generate_content(full_prompt)
        text = response.text
        if not text:
            print(f"[GEMINI_ADAPTER] Gemini returned empty response (blocked?), using fallback")
            return _fallback_extract(user_text)
        text = text.strip()

        print(f"[GEMINI_ADAPTER] Gemini raw response (first 200 chars): {text[:200]}")

        # Extract JSON from response
        result = _extract_json_from_text(text)
        if result:
            print(f"[GEMINI_ADAPTER] Gemini extracted JSON: {result}")
            return result

        print(f"[GEMINI_ADAPTER] No JSON found in Gemini response, using fallback")
        result = _fallback_extract(user_text)
        print(f"[GEMINI_ADAPTER] Fallback result: {result}")
        return result

    except Exception as e:
        print(f"[GEMINI_ADAPTER] Gemini extraction failed with error: {e}")
        result = _fallback_extract(user_text)
        print(f"[GEMINI_ADAPTER] Fallback result: {result}")
        return result


def explain_design(
    design_data: Dict,
    rank: int,
    total_designs: int,
    metrics: Optional[Dict] = None,
) -> str:
    """
    Generate a natural language explanation for a design.

    Falls back to template-based explanation if Gemini is unavailable.
    """
    if not is_available():
        return _fallback_explain(design_data, rank, total_designs, metrics)

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)

        score = metrics.get("design_score", 0) if metrics else 0
        compliance = metrics.get("compliance_status", "Unknown") if metrics else "Unknown"
        travel = metrics.get("travel_distance", "N/A") if metrics else "N/A"
        max_travel = metrics.get("max_travel_distance", "N/A") if metrics else "N/A"
        coverage = metrics.get("room_coverage", {}) if metrics else {}
        violations = metrics.get("violations", []) if metrics else []

        prompt = EXPLANATION_PROMPT.format(
            rank=rank,
            total=total_designs,
            design_data=json.dumps(design_data, indent=2, default=str),
            score=score,
            compliance_status=compliance,
            travel_distance=travel,
            max_travel=max_travel,
            room_coverage=json.dumps(coverage),
            violations=json.dumps(violations),
        )

        response = model.generate_content(prompt)
        if not response.text:
            return _fallback_explain(design_data, rank, total_designs, metrics)
        return response.text.strip()

    except Exception as e:
        print(f"Gemini explanation failed: {e}")
        return _fallback_explain(design_data, rank, total_designs, metrics)


def parse_correction(
    user_request: str,
    design_index: int,
    current_rooms: List[Dict],
) -> Dict[str, Any]:
    """
    Parse user's correction request into actionable changes.

    Falls back to simple keyword matching if Gemini is unavailable.
    """
    if not is_available():
        return _fallback_parse_correction(user_request, design_index, current_rooms)

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)

        rooms_str = ", ".join([r.get("name", r.get("type", "Unknown")) for r in current_rooms])

        prompt = CORRECTION_PROMPT.format(
            design_index=design_index,
            current_rooms=rooms_str,
            user_request=user_request,
        )

        response = model.generate_content(prompt)
        if not response.text:
            return _fallback_parse_correction(user_request, design_index, current_rooms)
        text = response.text.strip()

        # Extract JSON from response
        parsed = _extract_json_from_text(text)
        if parsed:
            return parsed

        return _fallback_parse_correction(user_request, design_index, current_rooms)

    except Exception as e:
        print(f"Gemini correction parsing failed: {e}")
        return _fallback_parse_correction(user_request, design_index, current_rooms)


def chat_response(
    user_message: str,
    context: Dict[str, Any],
    conversation_history: Optional[List[Dict]] = None,
) -> str:
    """
    Generate a conversational response about the design process.
    """
    if not is_available():
        return _fallback_chat(user_message, context)

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)

        system_context = """You are BlueprintGPT, an AI assistant for architectural floor plan design.
You help users create and refine floor plans through natural conversation.
Be helpful, friendly, and specific about architectural details.
When explaining designs, mention room names, positions, and relationships.
If you need more information, ask clarifying questions."""

        history = ""
        if conversation_history:
            for msg in conversation_history[-10:]:
                history += f"\n{msg['role'].upper()}: {msg['content']}"

        prompt = f"""{system_context}

Current context:
- Design state: {context.get('state', 'initial')}
- Generated designs: {context.get('num_designs', 0)}
- Current spec: {json.dumps(context.get('spec', {}), indent=2)}

Conversation history:{history}

USER: {user_message}

Respond naturally and helpfully."""

        response = model.generate_content(prompt)
        if not response.text:
            return _fallback_chat(user_message, context)
        return response.text.strip()

    except Exception as e:
        print(f"Gemini chat failed: {e}")
        return _fallback_chat(user_message, context)


# ═══════════════════════════════════════════════════════════════════════════════
#  Fallback Functions (when Gemini unavailable)
# ═══════════════════════════════════════════════════════════════════════════════

def _canonical_room_from_label(label: str) -> str:
    """Convert a room label like 'living room' to canonical type like 'LivingRoom'."""
    from nl_interface.constants import ROOM_LABELS
    label_lower = label.lower().strip()
    for canonical, labels in ROOM_LABELS.items():
        if label_lower in labels or label_lower == canonical.lower():
            return canonical
    return label.capitalize()


def _fallback_extract(user_text: str) -> Dict[str, Any]:
    """Simple regex-based extraction when Gemini is unavailable."""
    print(f"\n[FALLBACK_EXTRACT] Starting fallback extraction for: '{user_text}'")
    from nl_interface.service import _extract_from_text

    extracted = _extract_from_text(user_text)
    print(f"[FALLBACK_EXTRACT] Raw extraction - rooms: {extracted.get('rooms', [])}")
    print(f"[FALLBACK_EXTRACT] Raw extraction - plot_type: {extracted.get('plot_type')}, entrance_side: {extracted.get('entrance_side')}")

    # rooms is already a list of dicts [{"type": ..., "count": ...}]
    rooms = extracted.get("rooms", [])

    adjacency = []
    for src, tgt, rel in extracted.get("preferences", {}).get("adjacency", []):
        adjacency.append({"source": src, "target": tgt, "relation": rel})

    # Extract style hints from weights and interpretation notes
    style_hints = []
    weights = extracted.get("weights", {})
    if weights:
        if weights.get("compactness", 0) > 0.4:
            style_hints.append("compact")
        if weights.get("privacy", 0) > 0.4:
            style_hints.append("private")
    notes = extracted.get("interpretation_notes", [])
    for note in notes:
        note_lower = note.lower()
        if "compact" in note_lower:
            style_hints.append("compact")
        if "open" in note_lower:
            style_hints.append("open")
        if "privacy" in note_lower or "private" in note_lower:
            style_hints.append("private")

    result = {
        "rooms": rooms,
        "adjacency": adjacency,
        "entrance_side": extracted.get("entrance_side"),
        "plot_type": extracted.get("plot_type"),
        "style_hints": list(set(style_hints)),  # deduplicate
        "corrections": None,
        "intent": "new_design",
    }
    print(f"[FALLBACK_EXTRACT] Final result: rooms={result['rooms']}, plot_type={result['plot_type']}, entrance_side={result['entrance_side']}")
    return result


def _fallback_explain(
    design_data: Dict,
    rank: int,
    total_designs: int,
    metrics: Optional[Dict] = None,
) -> str:
    """Template-based explanation when Gemini is unavailable."""
    metrics = metrics or {}

    rooms = design_data.get("generated_rooms", {})
    room_list = ", ".join([f"{count} {rtype}" for rtype, count in rooms.items()])

    score = metrics.get("design_score", 0)
    violations = metrics.get("violations", [])
    compliance = "COMPLIANT" if not violations else "NON_COMPLIANT"

    explanation = f"""## Design #{rank} of {total_designs}

**Rooms Generated:** {room_list or 'None'}

**Design Score:** {score:.2f}/1.0

**Compliance Status:** {compliance}

"""

    if rank == 1:
        explanation += "**Why #1:** This design scored highest based on room arrangement, compliance, and adjacency satisfaction.\n\n"
    else:
        explanation += f"**Ranking:** This design ranked #{rank} due to "
        if violations:
            explanation += f"compliance issues: {', '.join(violations[:3])}\n\n"
        else:
            explanation += "lower overall optimization score.\n\n"

    if violations:
        explanation += f"**Issues to Address:**\n"
        for v in violations[:5]:
            explanation += f"- {v}\n"
        explanation += "\n"

    explanation += "**To request changes:** Tell me what you'd like to modify (e.g., 'make the kitchen larger', 'move bedroom away from entrance')."

    return explanation


def _fallback_parse_correction(
    user_request: str,
    design_index: int,
    current_rooms: List[Dict],
) -> Dict[str, Any]:
    """Simple keyword-based correction parsing when Gemini is unavailable."""
    changes = []
    user_lower = user_request.lower()

    # Build room name pattern from known room labels
    from nl_interface.constants import ROOM_LABELS
    all_labels = sorted(
        {label for labels in ROOM_LABELS.values() for label in labels},
        key=len, reverse=True,
    )
    room_pat = "|".join(re.escape(l) for l in all_labels)
    # Also match single-word capitalized names as fallback
    room_or_word = rf"(?:{room_pat}|\w+)"
    article = r"(?:the|a|an)\s+"

    # Detect move requests: "move (the) kitchen left"
    move_patterns = [
        (rf"move\s+{article}?({room_or_word})\s+(left|right|up|down)", "move_room"),
        (rf"shift\s+{article}?({room_or_word})\s+(left|right|up|down)", "move_room"),
    ]
    for pattern, change_type in move_patterns:
        match = re.search(pattern, user_lower)
        if match:
            room_name = _canonical_room_from_label(match.group(1))
            changes.append({
                "type": change_type,
                "room": room_name,
                "direction": match.group(2),
            })

    # Detect resize requests: "make (the) kitchen larger"
    resize_patterns = [
        (rf"make\s+{article}?({room_or_word})\s+(larger|bigger|smaller|bigger|wider|narrower)", "resize_room"),
        (rf"(increase|decrease)\s+{article}?({room_or_word})\s+size", "resize_room"),
    ]
    for pattern, change_type in resize_patterns:
        match = re.search(pattern, user_lower)
        if match:
            if "make" in pattern:
                room_name = _canonical_room_from_label(match.group(1))
            else:
                room_name = _canonical_room_from_label(match.group(2))
            size_dir = "larger" if any(w in user_lower for w in ("larger", "bigger", "increase", "wider")) else "smaller"
            changes.append({
                "type": change_type,
                "room": room_name,
                "size_change": size_dir,
            })

    # Detect swap requests: "swap kitchen and bedroom"
    swap_match = re.search(rf"swap\s+{article}?({room_or_word})\s+(?:and|with)\s+{article}?({room_or_word})", user_lower)
    if swap_match:
        changes.append({
            "type": "swap_rooms",
            "room_a": _canonical_room_from_label(swap_match.group(1)),
            "room_b": _canonical_room_from_label(swap_match.group(2)),
        })

    # Detect add/remove
    add_match = re.search(rf"add\s+{article}?({room_or_word})", user_lower)
    if add_match:
        changes.append({"type": "add_room", "room_type": _canonical_room_from_label(add_match.group(1))})

    remove_match = re.search(rf"remove\s+{article}?({room_or_word})", user_lower)
    if remove_match:
        changes.append({"type": "remove_room", "room": _canonical_room_from_label(remove_match.group(1))})

    return {
        "understood": len(changes) > 0,
        "changes": changes,
        "clarification_needed": None if changes else "I couldn't understand your correction request. Please be more specific about what you'd like to change.",
    }


def _fallback_chat(user_message: str, context: Dict[str, Any]) -> str:
    """Simple template-based chat response when Gemini is unavailable."""
    state = context.get("state", "initial")

    if state == "initial":
        return "I'm ready to help you design a floor plan! Please describe the rooms you need, the plot size, and any layout preferences."

    if state == "designs_generated":
        num_designs = context.get("num_designs", 0)
        return f"I've generated {num_designs} design options for you. You can see them above. Let me know if you'd like to modify any of them!"

    return "I'm here to help with your floor plan. What would you like to do?"
