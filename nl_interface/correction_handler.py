"""
correction_handler.py – Handles user corrections to generated designs.

This module:
1. Parses user correction requests
2. Translates corrections into spec modifications
3. Applies geometric changes to designs
4. Re-runs generation with modified specs
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from nl_interface.gemini_adapter import parse_correction


def handle_correction_request(
    user_request: str,
    design_index: int,
    session,  # ConversationSession
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Handle a user's correction request.

    Parameters
    ----------
    user_request : str
        The user's natural language correction request.
    design_index : int
        Index of the design to correct.
    session : ConversationSession
        The current conversation session.

    Returns
    -------
    tuple of (result_dict, error_message)
        result_dict contains the parsed corrections and modified spec.
        error_message is None if successful, otherwise contains the error.
    """
    if design_index < 0 or design_index >= len(session.designs):
        if not session.designs:
            return {}, "No designs have been generated yet. Please generate a design first."
        return {}, f"Invalid design index: {design_index}. Available designs: 0-{len(session.designs) - 1}"

    design = session.designs[design_index]
    current_rooms = [
        {"name": name, "type": room["type"], "count": 1}
        for room in design.rooms
        for name in [f"{room['type']}_{i+1}" for i in range(room.get("count", 1))]
    ]

    # Parse the correction using Gemini
    parsed = parse_correction(user_request, design_index, current_rooms)

    if not parsed.get("understood"):
        clarification = parsed.get("clarification_needed", "I couldn't understand your correction request.")
        return {"parsed": parsed, "needs_clarification": True}, clarification

    # Apply corrections to spec
    changes = parsed.get("changes", [])
    modified_spec, applied_changes = apply_corrections_to_spec(
        session.current_spec,
        changes,
        session.resolution,
    )

    # Record correction
    session.add_correction({
        "design_index": design_index,
        "user_request": user_request,
        "parsed": parsed,
        "changes": applied_changes,
    })

    return {
        "parsed": parsed,
        "changes": applied_changes,
        "modified_spec": modified_spec,
        "needs_regeneration": True,
    }, None


def apply_corrections_to_spec(
    current_spec: Dict[str, Any],
    changes: List[Dict[str, Any]],
    resolution: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Apply parsed corrections to the current spec.

    Returns the modified spec and list of actually applied changes.
    """
    modified = deepcopy(current_spec)
    applied = []

    for change in changes:
        change_type = change.get("type")

        if change_type == "add_room":
            room_type = change.get("room_type")
            if room_type:
                # Add to rooms list
                existing = next(
                    (r for r in modified.get("rooms", []) if r.get("type") == room_type),
                    None
                )
                if existing:
                    existing["count"] = existing.get("count", 1) + 1
                else:
                    modified.setdefault("rooms", []).append({"type": room_type, "count": 1})

                # Handle adjacency hint
                near = change.get("near")
                if near:
                    modified.setdefault("preferences", {}).setdefault("adjacency", []).append(
                        (room_type, near, "near_to")
                    )

                applied.append(change)

        elif change_type == "remove_room":
            room_name = change.get("room")
            if room_name:
                # Extract type from name (e.g., "Kitchen_1" → "Kitchen")
                room_type = room_name.split("_")[0] if "_" in room_name else room_name

                for room in modified.get("rooms", []):
                    if room.get("type") == room_type and room.get("count", 1) > 0:
                        room["count"] = room.get("count", 1) - 1
                        applied.append(change)
                        break

                # Remove rooms with count 0
                modified["rooms"] = [r for r in modified.get("rooms", []) if r.get("count", 1) > 0]

        elif change_type == "resize_room":
            room_name = change.get("room")
            size_change = change.get("size_change", "larger")

            # Add size preference
            modified.setdefault("room_size_preferences", {})[room_name] = size_change
            applied.append(change)

        elif change_type == "move_room":
            room_name = change.get("room")
            direction = change.get("direction")

            # Add position preference
            modified.setdefault("room_position_preferences", {})[room_name] = {
                "direction": direction,
                "amount": change.get("amount", "1m"),
            }
            applied.append(change)

        elif change_type == "swap_rooms":
            room_a = change.get("room_a")
            room_b = change.get("room_b")

            if room_a and room_b:
                modified.setdefault("room_swaps", []).append((room_a, room_b))
                applied.append(change)

        elif change_type == "change_adjacency":
            room_a = change.get("room_a")
            room_b = change.get("room_b")
            relation = change.get("relation", "adjacent_to")

            if room_a and room_b:
                # Remove existing adjacency between these rooms
                adj_list = modified.setdefault("preferences", {}).setdefault("adjacency", [])
                filtered = []
                for adj in adj_list:
                    # Handle both tuple and dict formats
                    if isinstance(adj, dict):
                        src, tgt = adj.get("source"), adj.get("target")
                    else:
                        src, tgt = adj[0], adj[1]
                    # Keep if not matching the rooms we're modifying
                    if not ((src == room_a and tgt == room_b) or (src == room_b and tgt == room_a)):
                        filtered.append(adj)
                # Add new adjacency
                filtered.append((room_a, room_b, relation))
                modified["preferences"]["adjacency"] = filtered
                applied.append(change)

    return modified, applied


def translate_corrections_to_geometry(
    changes: List[Dict[str, Any]],
    current_layout: Dict[str, Any],
    boundary_polygon: List[Tuple[float, float]],
) -> Dict[str, Any]:
    """
    Translate high-level corrections into geometric modifications.

    This is for fine-grained position/size adjustments that don't require
    full regeneration.
    """
    modified_layout = deepcopy(current_layout)
    rooms = modified_layout.get("rooms", [])

    # Guard against empty boundary
    if not boundary_polygon or len(boundary_polygon) < 3:
        return modified_layout  # Cannot compute geometry changes without valid boundary

    # Calculate boundary bbox
    xs = [p[0] for p in boundary_polygon]
    ys = [p[1] for p in boundary_polygon]
    bx0, bx1 = min(xs), max(xs)
    by0, by1 = min(ys), max(ys)
    b_width = bx1 - bx0
    b_height = by1 - by0

    for change in changes:
        change_type = change.get("type")

        if change_type == "move_room":
            room_name = change.get("room")
            direction = change.get("direction", "").lower()
            amount_str = change.get("amount", "1m")

            # Parse amount
            try:
                amount = float(amount_str.replace("m", "").strip())
            except ValueError:
                amount = 1.0

            # Find room and move it
            for room in rooms:
                if room.get("name", "").startswith(room_name) or room.get("type") == room_name:
                    polygon = room.get("polygon", [])
                    if polygon and len(polygon) >= 4:
                        # Calculate movement
                        dx, dy = 0, 0
                        if direction == "left":
                            dx = -amount
                        elif direction == "right":
                            dx = amount
                        elif direction == "up":
                            dy = amount
                        elif direction == "down":
                            dy = -amount

                        # Apply movement with boundary clamping
                        new_polygon = []
                        for x, y in polygon:
                            new_x = max(bx0, min(bx1, x + dx))
                            new_y = max(by0, min(by1, y + dy))
                            new_polygon.append((new_x, new_y))

                        room["polygon"] = new_polygon
                    break

        elif change_type == "resize_room":
            room_name = change.get("room")
            size_change = change.get("size_change", "larger")
            dimension = change.get("dimension", "both")  # width, height, or both

            scale = 1.2 if size_change == "larger" else 0.8

            for room in rooms:
                if room.get("name", "").startswith(room_name) or room.get("type") == room_name:
                    polygon = room.get("polygon", [])
                    if polygon and len(polygon) >= 4:
                        # Calculate centroid
                        cx = sum(p[0] for p in polygon) / len(polygon)
                        cy = sum(p[1] for p in polygon) / len(polygon)

                        # Scale around centroid
                        new_polygon = []
                        for x, y in polygon:
                            if dimension == "both":
                                new_x = cx + (x - cx) * scale
                                new_y = cy + (y - cy) * scale
                            elif dimension == "width":
                                new_x = cx + (x - cx) * scale
                                new_y = y
                            else:  # height
                                new_x = x
                                new_y = cy + (y - cy) * scale

                            # Clamp to boundary
                            new_x = max(bx0, min(bx1, new_x))
                            new_y = max(by0, min(by1, new_y))
                            new_polygon.append((new_x, new_y))

                        room["polygon"] = new_polygon
                    break

        elif change_type == "swap_rooms":
            room_a_name = change.get("room_a")
            room_b_name = change.get("room_b")

            room_a = None
            room_b = None
            for room in rooms:
                name = room.get("name", "")
                if name.startswith(room_a_name) or room.get("type") == room_a_name:
                    room_a = room
                elif name.startswith(room_b_name) or room.get("type") == room_b_name:
                    room_b = room

            if room_a and room_b:
                # Swap polygons
                room_a["polygon"], room_b["polygon"] = room_b["polygon"], room_a["polygon"]

    return modified_layout


def validate_correction_feasibility(
    changes: List[Dict[str, Any]],
    current_spec: Dict[str, Any],
    resolution: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate if the requested corrections are feasible.

    Returns (is_feasible, list of warnings/errors).
    """
    warnings = []
    is_feasible = True

    for change in changes:
        change_type = change.get("type")

        if change_type == "add_room":
            room_type = change.get("room_type")
            # Check if we're exceeding reasonable room count
            existing_count = sum(
                r.get("count", 1) for r in current_spec.get("rooms", [])
                if r.get("type") == room_type
            )
            if existing_count >= 5:
                warnings.append(f"Already have {existing_count} {room_type}(s). Adding more may crowd the layout.")

        elif change_type == "remove_room":
            room_name = change.get("room")
            room_type = room_name.split("_")[0] if "_" in room_name else room_name
            existing = [r for r in current_spec.get("rooms", []) if r.get("type") == room_type]
            if not existing:
                warnings.append(f"No {room_type} found to remove.")
                is_feasible = False

        elif change_type == "resize_room":
            size_change = change.get("size_change", "larger")
            if size_change == "larger" and resolution:
                # Check if boundary can accommodate larger room
                boundary_area = resolution.get("total_area", 100)
                current_rooms = len(current_spec.get("rooms", []))
                if boundary_area / max(current_rooms, 1) < 8:  # Less than 8 sq m per room
                    warnings.append("Space is tight. Making rooms larger may cause overlaps.")

    return is_feasible, warnings
