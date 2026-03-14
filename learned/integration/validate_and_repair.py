"""
validate_and_repair_generated_layout.py – Deterministic repair pipeline that
takes a ``Building`` produced by the learned generator adapter and makes it
compliant with KG / regulatory constraints.

Pipeline stages
---------------
1. **Clamp to boundary** – ensure every room polygon lies within the boundary.
2. **Drop degenerate rooms** – remove rooms that are too small or zero-area.
3. **Resolve overlaps** – push-apart / shrink overlapping room rectangles.
4. **Enforce minimum area / width / height** from regulation_data.json.
5. **Carve corridor + place doors** using existing corridor-first planner.
6. **Check connectivity** (BFS over doors) and travel distance.
7. **Run ontology validation** (optional).

Main entry point
-----------------
    building_fixed, violations, status = validate_and_repair_generated_layout(
        building, boundary_polygon, entrance_point, regulation_file)
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from shapely.geometry import Polygon, box as shapely_box
from shapely.ops import unary_union

from core.building import Building
from core.room import Room
from core.exit import Exit
from core.corridor import Corridor
from constraints.rule_engine import RuleEngine
from geometry.door_placer import DoorPlacer
from geometry.corridor_first_planner import generate_corridor_first_variants
from geometry.zoning import assign_room_zones
from geometry.adjacency_intent import adjacency_satisfaction_score
from graph.connectivity import is_fully_connected
from graph.manhattan_path import max_travel_distance


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 1: Clamp rooms to boundary
# ═══════════════════════════════════════════════════════════════════════════════

def _boundary_bbox(boundary_polygon):
    xs = [p[0] for p in boundary_polygon]
    ys = [p[1] for p in boundary_polygon]
    return min(xs), min(ys), max(xs), max(ys)


def clamp_rooms_to_boundary(building: Building, boundary_polygon):
    """Clamp every room's rectangle to the boundary bounding box."""
    bx0, by0, bx1, by1 = _boundary_bbox(boundary_polygon)
    violations = []
    for room in building.rooms:
        if room.polygon is None:
            continue
        xs = [p[0] for p in room.polygon]
        ys = [p[1] for p in room.polygon]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        cx1, cy1 = max(bx0, x1), max(by0, y1)
        cx2, cy2 = min(bx1, x2), min(by1, y2)
        if cx2 - cx1 < 0.1 or cy2 - cy1 < 0.1:
            violations.append(f"{room.name}: collapsed after clamping")
            continue
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            violations.append(f"{room.name}: clamped to boundary")
        room.polygon = [(cx1, cy1), (cx2, cy1), (cx2, cy2), (cx1, cy2)]
        room.final_area = (cx2 - cx1) * (cy2 - cy1)
    return violations


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 2: Drop degenerate rooms
# ═══════════════════════════════════════════════════════════════════════════════

def drop_degenerate_rooms(building: Building, min_area: float = 0.5, min_dim: float = 0.3):
    """Remove rooms with area < min_area or any dimension < min_dim."""
    violations = []
    keep = []
    for room in building.rooms:
        if room.polygon is None:
            violations.append(f"{room.name}: no polygon, dropped")
            continue
        xs = [p[0] for p in room.polygon]
        ys = [p[1] for p in room.polygon]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        area = w * h
        if area < min_area or w < min_dim or h < min_dim:
            violations.append(f"{room.name}: degenerate ({w:.2f}×{h:.2f}={area:.2f}), dropped")
        else:
            keep.append(room)
    building.rooms = keep
    return violations


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 3: Resolve overlaps (greedy push-apart)
# ═══════════════════════════════════════════════════════════════════════════════

def _room_bbox(room):
    xs = [p[0] for p in room.polygon]
    ys = [p[1] for p in room.polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap_area(r1, r2):
    x1a, y1a, x2a, y2a = _room_bbox(r1)
    x1b, y1b, x2b, y2b = _room_bbox(r2)
    dx = max(0, min(x2a, x2b) - max(x1a, x1b))
    dy = max(0, min(y2a, y2b) - max(y1a, y1b))
    return dx * dy


def _set_room_rect(room, x1, y1, x2, y2):
    room.polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    room.final_area = (x2 - x1) * (y2 - y1)


def resolve_overlaps(building: Building, boundary_polygon, max_iterations: int = 50):
    """Iteratively push overlapping rooms apart within boundary."""
    bx0, by0, bx1, by1 = _boundary_bbox(boundary_polygon)
    violations = []
    rooms = building.rooms

    for iteration in range(max_iterations):
        moved = False
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                if rooms[i].polygon is None or rooms[j].polygon is None:
                    continue
                ov = _overlap_area(rooms[i], rooms[j])
                if ov < 0.01:
                    continue
                # Push the smaller room away
                a, b = (i, j) if (rooms[i].final_area or 0) >= (rooms[j].final_area or 0) else (j, i)
                ax1, ay1, ax2, ay2 = _room_bbox(rooms[a])
                bx1r, by1r, bx2r, by2r = _room_bbox(rooms[b])

                # Find minimum push direction
                push_right = ax2 - bx1r
                push_left = bx2r - ax1
                push_down = ay2 - by1r
                push_up = by2r - ay1

                candidates = []
                if push_right > 0:
                    candidates.append(("right", push_right))
                if push_left > 0:
                    candidates.append(("left", push_left))
                if push_down > 0:
                    candidates.append(("down", push_down))
                if push_up > 0:
                    candidates.append(("up", push_up))

                if not candidates:
                    continue

                direction, dist = min(candidates, key=lambda c: c[1])
                nudge = dist + 0.05  # small gap

                nx1, ny1, nx2, ny2 = bx1r, by1r, bx2r, by2r
                if direction == "right":
                    nx1 += nudge
                    nx2 += nudge
                elif direction == "left":
                    nx1 -= nudge
                    nx2 -= nudge
                elif direction == "down":
                    ny1 += nudge
                    ny2 += nudge
                elif direction == "up":
                    ny1 -= nudge
                    ny2 -= nudge

                # Clamp to boundary
                if nx1 < bx0:
                    nx2 += bx0 - nx1
                    nx1 = bx0
                if ny1 < by0:
                    ny2 += by0 - ny1
                    ny1 = by0
                if nx2 > bx1:
                    nx1 -= nx2 - bx1
                    nx2 = bx1
                if ny2 > by1:
                    ny1 -= ny2 - by1
                    ny2 = by1

                nx1 = max(bx0, nx1)
                ny1 = max(by0, ny1)
                nx2 = min(bx1, nx2)
                ny2 = min(by1, ny2)

                if nx2 - nx1 > 0.3 and ny2 - ny1 > 0.3:
                    _set_room_rect(rooms[b], nx1, ny1, nx2, ny2)
                    moved = True

        if not moved:
            break
    else:
        violations.append(f"overlap resolution: reached {max_iterations} iterations")

    # Final check for remaining overlaps
    remaining = 0
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            if rooms[i].polygon and rooms[j].polygon:
                ov = _overlap_area(rooms[i], rooms[j])
                if ov > 0.01:
                    remaining += 1
    if remaining > 0:
        violations.append(f"{remaining} room pair(s) still overlapping after repair")

    return violations


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 4: Enforce minimum area / width / height
# ═══════════════════════════════════════════════════════════════════════════════

def enforce_room_minimums(building: Building, regulation_file: str):
    """Grow rooms that are below regulation minimums."""
    engine = RuleEngine(regulation_file)
    regulations = engine.data.get(building.occupancy_type, {}).get("rooms", {})
    violations = []

    for room in building.rooms:
        rule = regulations.get(room.room_type)
        if rule is None:
            continue

        min_area = rule["min_area"]
        min_w = rule["min_width"]
        min_h = rule.get("min_height", 0)

        room.set_regulation_constraints(min_area, min_w, min_h)

        if room.polygon is None:
            continue

        xs = [p[0] for p in room.polygon]
        ys = [p[1] for p in room.polygon]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        w, h = x2 - x1, y2 - y1
        area = w * h

        grown = False
        # Grow width if needed
        if w < min_w:
            diff = min_w - w
            x2 = x1 + min_w
            grown = True
            violations.append(f"{room.name}: width {w:.2f} < {min_w}, expanded")

        # Grow to meet min area
        w = x2 - x1
        h = y2 - y1
        if w * h < min_area:
            needed_h = min_area / max(w, 0.1)
            if needed_h > h:
                y2 = y1 + needed_h
                grown = True
                violations.append(f"{room.name}: area {w*h:.2f} < {min_area}, expanded height")

        if grown:
            room.polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            room.final_area = (x2 - x1) * (y2 - y1)

    return violations


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 5: Corridor carving + doors
# ═══════════════════════════════════════════════════════════════════════════════

def carve_corridors_and_doors(
    building: Building,
    boundary_polygon,
    entrance_point,
    regulation_file: str,
) -> Tuple[List[Tuple[Building, str]], List[str]]:
    """
    Use the existing corridor-first planner to generate corridor variants
    and place doors.  Returns (variants, violations).
    Each variant is (building_copy, strategy_name).
    """
    engine = RuleEngine(regulation_file)
    occ = building.occupancy_type
    min_corridor_width = engine.data[occ].get("corridor", {}).get("min_width", 1.2)
    min_door_width = engine.get_min_door_width(occ)

    violations = []
    variants = generate_corridor_first_variants(
        building,
        boundary_polygon=boundary_polygon,
        entrance_point=entrance_point,
        min_corridor_width=min_corridor_width,
    )

    if not variants:
        violations.append("no corridor variants generated")
        variants = [(building, "fallback-none")]

    for var_building, strategy_name in variants:
        door_placer = DoorPlacer(var_building, min_door_width)
        door_placer.place_doors()

    return variants, violations


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 6: Evaluate connectivity + travel distance
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_variant(building: Building, regulation_file: str, entrance_point=None):
    """Compute all metrics for a single variant. Returns metrics dict."""
    engine = RuleEngine(regulation_file)
    occ = building.occupancy_type
    max_allowed_travel = engine.get_max_travel_distance(occ)
    total_area_val = sum(r.final_area for r in building.rooms if r.final_area)
    building.total_area = total_area_val
    building.occupant_load = max(1, int(total_area_val / 100 * 8))

    connected = is_fully_connected(building)
    travel_dist = max_travel_distance(building)
    zone_map = assign_room_zones(building, entrance_point=entrance_point)
    adj_score, adj_details = adjacency_satisfaction_score(building)

    circ_spaces = getattr(building, "corridors", [])
    walkable = round(sum(getattr(c, "walkable_area", 0.0) for c in circ_spaces), 2)
    corr_width = round(max((c.width for c in circ_spaces), default=0.0), 2)
    conn_exit = all(getattr(c, "connectivity_to_exit", False) for c in circ_spaces) if circ_spaces else False

    exit_width = building.exit.width if building.exit else 0

    return {
        "total_area": total_area_val,
        "occupant_load": building.occupant_load,
        "required_exit_width": exit_width,
        "max_travel_distance": travel_dist,
        "max_allowed_travel_distance": max_allowed_travel,
        "travel_distance_compliant": travel_dist <= max_allowed_travel,
        "fully_connected": connected,
        "zone_map": zone_map,
        "adjacency_satisfaction": adj_score,
        "adjacency_details": adj_details,
        "corridor_width": corr_width,
        "circulation_walkable_area": walkable,
        "connectivity_to_exit": conn_exit,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main: full repair pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def validate_and_repair_generated_layout(
    building: Building,
    boundary_polygon: List[Tuple[float, float]],
    entrance_point: Optional[Tuple[float, float]] = None,
    regulation_file: str = "ontology/regulation_data.json",
) -> Tuple[Building, List[str], str]:
    """
    Full deterministic repair pipeline for a learned-generator Building.

    Returns
    -------
    building_best : Building
        Best repaired variant (with corridor + doors).
    all_violations : list[str]
        All violations / repair actions applied.
    status : str
        "COMPLIANT" or "NON_COMPLIANT"
    """
    all_violations: list[str] = []

    # Stage 1: clamp
    all_violations.extend(clamp_rooms_to_boundary(building, boundary_polygon))

    # Stage 2: drop degenerate
    all_violations.extend(drop_degenerate_rooms(building))

    if not building.rooms:
        return building, all_violations + ["no rooms survived repair"], "NON_COMPLIANT"

    # Stage 3: overlaps
    all_violations.extend(resolve_overlaps(building, boundary_polygon))

    # Stage 4: enforce minimums
    all_violations.extend(enforce_room_minimums(building, regulation_file))

    # Re-clamp after growth
    all_violations.extend(clamp_rooms_to_boundary(building, boundary_polygon))

    # Stage 5: corridor + doors (produces multiple strategy variants)
    variants, corr_violations = carve_corridors_and_doors(
        building, boundary_polygon, entrance_point, regulation_file,
    )
    all_violations.extend(corr_violations)

    # Stage 6: evaluate each variant; pick best
    best_variant = None
    best_score = -1.0

    for var_bldg, strategy in variants:
        metrics = evaluate_variant(var_bldg, regulation_file, entrance_point)
        # Simple scoring: compliance > adjacency > compactness
        score = 0.0
        if metrics["fully_connected"] and metrics["travel_distance_compliant"]:
            score += 1.0
        score += 0.3 * metrics.get("adjacency_satisfaction", 0.0)
        if metrics.get("connectivity_to_exit"):
            score += 0.2
        if best_variant is None or score > best_score:
            best_variant = var_bldg
            best_score = score
            best_metrics = metrics
            best_strategy = strategy

    if best_variant is None:
        return building, all_violations + ["no viable variant"], "NON_COMPLIANT"

    compliant = best_metrics["fully_connected"] and best_metrics["travel_distance_compliant"]
    status = "COMPLIANT" if compliant else "NON_COMPLIANT"

    return best_variant, all_violations, status
