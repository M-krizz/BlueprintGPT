"""
repair_gate.py – Deterministic 8-stage repair pipeline that takes a ``Building``
produced by the learned generator adapter and makes it compliant with the
regulatory / KG constraint system.

Pipeline stages (executed **in order**)
---------------------------------------
1. **Sanitize geometry** – clamp rooms to boundary polygon, drop degenerate
   rooms, merge near-duplicate rooms (IoU threshold).
2. **Enforce minimum room requirements** – expand rooms below regulation
   minimums for area / width / height.
3. **Non-overlap repair** – push-apart overlapping rooms; fallback to full
   repack via ``PolygonPacker`` if overlaps persist.
4. **Corridor-first planning** – carve corridor polygon, allocate room-space.
5. **Corridor accessibility enforcement** – shift rooms that don't touch the
   corridor boundary closer to it.
6. **Doors → connectivity** – place doors on corridor-facing edges, check BFS
   connectivity, retry door insertion if not fully connected.
7. **Travel distance check** – validate max travel distance; try alternate
   corridor strategies if exceeded.
8. **Ontology / KG validation** – run OntologyBridge if available; fallback
   to procedural semantic check.

Returns
-------
(building_fixed, violations, status, repair_trace)
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RepairReport:
    """Repair severity metrics for tracking quality and displacement."""
    severity_score: float  # [0-1], 0.0=no repair, 1.0=major changes
    overlap_fixes: int  # Number of overlap violations fixed
    min_dim_violations_fixed: int  # Number of min-dim violations fixed
    topology_changes: int  # Room merges, splits, drops
    total_displacement_m: float  # Sum of centroid movements (meters)
    stages_applied: List[str]  # List of stage names that made changes
    original_room_count: int
    final_room_count: int

    @property
    def room_count_changed(self) -> bool:
        """Whether room count was modified during repair."""
        return self.original_room_count != self.final_room_count

# ── Phase 3: force-based push-apart feature flags ─────────────────────────────
FORCE_PUSH_ENABLED    = os.getenv("REPAIR_FORCE_PUSH_ENABLED",    "false").lower() == "true"
FORCE_PUSH_MAX_ITERS  = int(os.getenv("REPAIR_FORCE_PUSH_MAX_ITERS",  "120"))
FORCE_PUSH_STEP       = float(os.getenv("REPAIR_FORCE_PUSH_STEP",       "0.5"))
FORCE_PUSH_DAMPING    = float(os.getenv("REPAIR_FORCE_PUSH_DAMPING",    "0.85"))
FORCE_PUSH_MIN_OV     = float(os.getenv("REPAIR_FORCE_PUSH_MIN_OV",     "0.01"))

# ── Phase 4: box optimizer feature flag ───────────────────────────────────────
BOX_OPT_ENABLED = os.getenv("BOX_OPT_ENABLED", "false").lower() == "true"

from shapely.geometry import Polygon, Point, box as shapely_box
from shapely.ops import unary_union

from core.building import Building
from core.room import Room
from core.exit import Exit
from core.corridor import Corridor
from constraints.rule_engine import RuleEngine
from geometry.door_placer import DoorPlacer
from geometry.corridor_first_planner import generate_corridor_first_variants
from geometry.polygon_packer import recursive_pack
from geometry.polygon import (
    snap_building_to_grid,
    enforce_aspect_ratio,
    DEFAULT_ASPECT_LIMITS,
)
from geometry.zoning import assign_room_zones
from geometry.adjacency_intent import adjacency_satisfaction_score
from graph.connectivity import is_fully_connected
from graph.door_graph_path import door_graph_travel_distance
from graph.manhattan_path import max_travel_distance


def _travel_distance(building: Building) -> float:
    """Door/corridor graph travel distance with safe Manhattan fallback."""
    try:
        d = door_graph_travel_distance(building)
        if d is not None and d < 999.0:
            return float(d)
    except Exception:
        pass
    return float(max_travel_distance(building))


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _bbox(room) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in room.polygon]
    ys = [p[1] for p in room.polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _set_rect(room, x1, y1, x2, y2):
    room.polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    room.final_area = (x2 - x1) * (y2 - y1)


def _boundary_bbox(boundary_polygon):
    xs = [p[0] for p in boundary_polygon]
    ys = [p[1] for p in boundary_polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap_area(r1, r2):
    x1a, y1a, x2a, y2a = _bbox(r1)
    x1b, y1b, x2b, y2b = _bbox(r2)
    dx = max(0, min(x2a, x2b) - max(x1a, x1b))
    dy = max(0, min(y2a, y2b) - max(y1a, y1b))
    return dx * dy


def _iou(r1, r2):
    ov = _overlap_area(r1, r2)
    if ov <= 0:
        return 0.0
    a1 = r1.final_area or 1e-6
    a2 = r2.final_area or 1e-6
    return ov / (a1 + a2 - ov)


def _trace(stage: str, action: str, room: str = "") -> Dict:
    return {"stage": stage, "action": action, "room": room}


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 1: Sanitize geometry
# ═══════════════════════════════════════════════════════════════════════════════

def _stage1_sanitize(building: Building, boundary_polygon, trace: list) -> List[str]:
    """Clamp rooms to boundary, drop degenerate, merge IoU-duplicates."""
    violations = []
    bpoly = Polygon(boundary_polygon)
    if not bpoly.is_valid:
        bpoly = bpoly.buffer(0)
    bx0, by0, bx1, by1 = bpoly.bounds

    # 1a. Clamp rooms to boundary polygon (intersection)
    for room in building.rooms:
        if room.polygon is None:
            continue
        rpoly = Polygon(room.polygon)
        if not bpoly.contains(rpoly):
            clipped = rpoly.intersection(bpoly)
            if clipped.is_empty or clipped.area < 0.1:
                violations.append(f"{room.name}: collapsed after boundary clamp")
                trace.append(_trace("sanitize", "collapsed_after_clamp", room.name))
                room.polygon = None
                continue
            # Use bounding box of clipped region (keep rectangles)
            cx0, cy0, cx1, cy1 = clipped.bounds
            _set_rect(room, cx0, cy0, cx1, cy1)
            violations.append(f"{room.name}: clamped to boundary")
            trace.append(_trace("sanitize", "clamped_to_boundary", room.name))

    # 1b. Drop degenerate rooms
    keep = []
    for room in building.rooms:
        if room.polygon is None:
            # KEEP placeholders so we can force a repack later
            if room.provenance and room.provenance.get("source") == "placeholder":
                keep.append(room)
                violations.append(f"{room.name}: placeholder requires repacking")
                trace.append(_trace("sanitize", "kept_placeholder_for_repack", room.name))
            else:
                violations.append(f"{room.name}: no polygon, dropped")
                trace.append(_trace("sanitize", "dropped_no_polygon", room.name))
            continue
        x1, y1, x2, y2 = _bbox(room)
        w, h = x2 - x1, y2 - y1
        if w * h < 0.5 or w < 0.3 or h < 0.3:
            violations.append(f"{room.name}: degenerate ({w:.2f}x{h:.2f}), dropped")
            trace.append(_trace("sanitize", "dropped_degenerate", room.name))
            continue
        keep.append(room)
    building.rooms = keep

    # 1c. Merge near-duplicate rooms (same type, IoU > 0.5)
    merged_out = set()
    for i in range(len(building.rooms)):
        if i in merged_out:
            continue
        for j in range(i + 1, len(building.rooms)):
            if j in merged_out:
                continue
            ri, rj = building.rooms[i], building.rooms[j]
            if ri.room_type != rj.room_type:
                continue
            if ri.polygon is None or rj.polygon is None:
                continue
            if _iou(ri, rj) > 0.5:
                # Keep larger, drop smaller
                if (ri.final_area or 0) >= (rj.final_area or 0):
                    merged_out.add(j)
                    violations.append(f"{rj.name}: merged into {ri.name} (IoU>{0.5})")
                    trace.append(_trace("sanitize", f"merged_into_{ri.name}", rj.name))
                else:
                    merged_out.add(i)
                    violations.append(f"{ri.name}: merged into {rj.name} (IoU>{0.5})")
                    trace.append(_trace("sanitize", f"merged_into_{rj.name}", ri.name))
                    break
    if merged_out:
        building.rooms = [r for idx, r in enumerate(building.rooms) if idx not in merged_out]

    return violations


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 2: Enforce minimum room requirements
# ═══════════════════════════════════════════════════════════════════════════════

def _stage2_enforce_minimums(building: Building, regulation_file: str, trace: list) -> List[str]:
    """Grow rooms that are below regulation minimums."""
    engine = RuleEngine(regulation_file)
    regs = engine.data.get(building.occupancy_type, {}).get("rooms", {})
    violations = []

    for room in building.rooms:
        rule = regs.get(room.room_type)
        if rule is None:
            continue

        min_area = rule["min_area"]
        min_w = rule["min_width"]
        min_h = rule.get("min_height", 0)
        room.set_regulation_constraints(min_area, min_w, min_h)

        if room.polygon is None:
            continue

        x1, y1, x2, y2 = _bbox(room)
        w, h = x2 - x1, y2 - y1
        grown = False

        if w < min_w:
            x2 = x1 + min_w
            grown = True
            violations.append(f"{room.name}: width {w:.2f} < {min_w}, expanded")
            trace.append(_trace("enforce_min", f"width_expanded_{w:.2f}->{min_w}", room.name))

        w = x2 - x1
        h = y2 - y1
        if w * h < min_area:
            needed_h = min_area / max(w, 0.1)
            if needed_h > h:
                y2 = y1 + needed_h
                grown = True
                violations.append(f"{room.name}: area {w*h:.2f} < {min_area}, expanded height")
                trace.append(_trace("enforce_min", f"area_expanded_{w*h:.2f}->{min_area}", room.name))

        if grown:
            _set_rect(room, x1, y1, x2, y2)

    return violations


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 3: Non-overlap repair
# ═══════════════════════════════════════════════════════════════════════════════

def _push_apart(building: Building, boundary_polygon, max_iters: int = 50) -> int:
    """Push overlapping rooms apart. Returns count of remaining overlaps."""
    bx0, by0, bx1, by1 = _boundary_bbox(boundary_polygon)
    rooms = [r for r in building.rooms if r.polygon is not None]

    for _ in range(max_iters):
        moved = False
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                if rooms[i].polygon is None or rooms[j].polygon is None:
                    continue
                ov = _overlap_area(rooms[i], rooms[j])
                if ov < 0.01:
                    continue
                # Push smaller room away from larger
                a, b = (i, j) if (rooms[i].final_area or 0) >= (rooms[j].final_area or 0) else (j, i)
                ax1, ay1, ax2, ay2 = _bbox(rooms[a])
                bx1r, by1r, bx2r, by2r = _bbox(rooms[b])

                cands = []
                pr = ax2 - bx1r
                pl = bx2r - ax1
                pd = ay2 - by1r
                pu = by2r - ay1
                if pr > 0: cands.append(("right", pr))
                if pl > 0: cands.append(("left", pl))
                if pd > 0: cands.append(("down", pd))
                if pu > 0: cands.append(("up", pu))
                if not cands:
                    continue

                direction, dist = min(cands, key=lambda c: c[1])
                nudge = dist + 0.05
                nx1, ny1, nx2, ny2 = bx1r, by1r, bx2r, by2r
                if direction == "right":    nx1 += nudge; nx2 += nudge
                elif direction == "left":   nx1 -= nudge; nx2 -= nudge
                elif direction == "down":   ny1 += nudge; ny2 += nudge
                elif direction == "up":     ny1 -= nudge; ny2 -= nudge

                # Clamp
                if nx1 < bx0: nx2 += bx0 - nx1; nx1 = bx0
                if ny1 < by0: ny2 += by0 - ny1; ny1 = by0
                if nx2 > bx1: nx1 -= nx2 - bx1; nx2 = bx1
                if ny2 > by1: ny1 -= ny2 - by1; ny2 = by1
                nx1, ny1 = max(bx0, nx1), max(by0, ny1)
                nx2, ny2 = min(bx1, nx2), min(by1, ny2)

                if nx2 - nx1 > 0.3 and ny2 - ny1 > 0.3:
                    _set_rect(rooms[b], nx1, ny1, nx2, ny2)
                    moved = True
        if not moved:
            break

    remaining = 0
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            if rooms[i].polygon and rooms[j].polygon and _overlap_area(rooms[i], rooms[j]) > 0.01:
                remaining += 1
    return remaining


def _force_push_apart(
    building: Building,
    boundary_polygon,
    max_iters: int = FORCE_PUSH_MAX_ITERS,
    initial_step: float = FORCE_PUSH_STEP,
    damping: float = FORCE_PUSH_DAMPING,
) -> int:
    """Force-based spring push-apart optimizer (Phase 3).

    Improvements over the greedy ``_push_apart``:

    * **Simultaneous updates** — forces for all pairs are accumulated first,
      then applied at once; prevents the greedy oscillation where moving room A
      to fix pair (A,B) breaks pair (A,C).
    * **Proportional force split** — larger rooms move less (force shared
      inversely proportional to area, like elastic collision).
    * **Damped step size** — step shrinks by ``damping`` each iteration so the
      optimizer settles rather than overshooting.

    Returns the number of remaining overlapping pairs after the run.
    """
    bx0, by0, bx1, by1 = _boundary_bbox(boundary_polygon)
    rooms = [r for r in building.rooms if r.polygon is not None]
    n = len(rooms)
    if n < 2:
        return 0

    step = initial_step

    for _ in range(max_iters):
        forces: List[List[float]] = [[0.0, 0.0] for _ in range(n)]
        has_overlap = False

        for i in range(n):
            for j in range(i + 1, n):
                if rooms[i].polygon is None or rooms[j].polygon is None:
                    continue
                ov = _overlap_area(rooms[i], rooms[j])
                if ov < FORCE_PUSH_MIN_OV:
                    continue
                has_overlap = True

                ax1, ay1, ax2, ay2 = _bbox(rooms[i])
                bx1r, by1r, bx2r, by2r = _bbox(rooms[j])

                # Overlap region dimensions
                ox1 = max(ax1, bx1r);  ox2 = min(ax2, bx2r)
                oy1 = max(ay1, by1r);  oy2 = min(ay2, by2r)
                dx_ov = max(ox2 - ox1, 0.0)
                dy_ov = max(oy2 - oy1, 0.0)

                # Push along the axis with the smaller overlap (cheaper to resolve)
                if dx_ov <= dy_ov:
                    a_cx = (ax1 + ax2) / 2;  b_cx = (bx1r + bx2r) / 2
                    sign = 1.0 if a_cx >= b_cx else -1.0
                    fx_i, fy_i = sign * dx_ov, 0.0
                    fx_j, fy_j = -sign * dx_ov, 0.0
                else:
                    a_cy = (ay1 + ay2) / 2;  b_cy = (by1r + by2r) / 2
                    sign = 1.0 if a_cy >= b_cy else -1.0
                    fx_i, fy_i = 0.0, sign * dy_ov
                    fx_j, fy_j = 0.0, -sign * dy_ov

                # Split force inversely proportional to area
                # (heavier room moves less — simulates elastic collision)
                ai = max(rooms[i].final_area or 1e-6, 1e-6)
                aj = max(rooms[j].final_area or 1e-6, 1e-6)
                total = ai + aj
                wi = aj / total   # room i gets the share driven by room j's mass
                wj = ai / total

                forces[i][0] += fx_i * wi;  forces[i][1] += fy_i * wi
                forces[j][0] += fx_j * wj;  forces[j][1] += fy_j * wj

        if not has_overlap:
            break

        # Apply accumulated forces simultaneously
        for idx in range(n):
            fx, fy = forces[idx]
            if abs(fx) < 1e-9 and abs(fy) < 1e-9:
                continue
            if rooms[idx].polygon is None:
                continue
            x1, y1, x2, y2 = _bbox(rooms[idx])
            w, h = x2 - x1, y2 - y1
            nx1 = x1 + fx * step
            ny1 = y1 + fy * step
            nx2 = nx1 + w
            ny2 = ny1 + h
            # Clamp translate (preserve size)
            if nx1 < bx0: nx2 += bx0 - nx1; nx1 = bx0
            if ny1 < by0: ny2 += by0 - ny1; ny1 = by0
            if nx2 > bx1: nx1 -= nx2 - bx1; nx2 = bx1
            if ny2 > by1: ny1 -= ny2 - by1; ny2 = by1
            nx1, ny1 = max(bx0, nx1), max(by0, ny1)
            nx2, ny2 = min(bx1, nx2), min(by1, ny2)
            if nx2 - nx1 > 0.3 and ny2 - ny1 > 0.3:
                _set_rect(rooms[idx], nx1, ny1, nx2, ny2)

        step *= damping

    # Count remaining overlapping pairs
    remaining = 0
    for i in range(n):
        for j in range(i + 1, n):
            if rooms[i].polygon and rooms[j].polygon and _overlap_area(rooms[i], rooms[j]) > FORCE_PUSH_MIN_OV:
                remaining += 1
    return remaining


def _repack_fallback(building: Building, boundary_polygon, entrance_point, trace: list):
    """Full repack via polygon_packer – throws away learned positions."""
    try:
        bpoly = Polygon(boundary_polygon)
        if not bpoly.is_valid:
            bpoly = bpoly.buffer(0)
        rooms = [r for r in building.rooms if r.final_area and r.final_area > 0]
        rooms.sort(key=lambda r: (r.room_type == "LivingRoom", r.final_area), reverse=True)
        items = [{"room": r, "weight": r.final_area} for r in rooms]
        # Clear existing polygons so packer assigns fresh ones
        for r in rooms:
            r.polygon = None
        recursive_pack(bpoly, items, entrance_pt=entrance_point)
        trace.append(_trace("overlap_repair", "full_repack_applied", ""))
        return True
    except Exception:
        trace.append(_trace("overlap_repair", "repack_failed", ""))
        return False


def _stage3_overlap_repair(building, boundary_polygon, entrance_point, trace) -> List[str]:
    """Push-apart first; optionally refine with box optimizer; fallback to repack."""
    violations = []

    # Step 1: Push-apart (greedy or force-based)
    if FORCE_PUSH_ENABLED:
        remaining = _force_push_apart(building, boundary_polygon)
        push_label = "force_push_apart"
    else:
        remaining = _push_apart(building, boundary_polygon)
        push_label = "push_apart"

    # Step 2: Box optimizer refinement (Phase 4)
    if remaining > 0 and BOX_OPT_ENABLED:
        try:
            from learned.integration.box_optimizer import optimize_box_placement
            opt_result = optimize_box_placement(building, boundary_polygon)
            if opt_result["success"]:
                remaining = opt_result["remaining_overlaps"]
                trace.append(_trace("overlap_repair", f"box_opt_{opt_result['solver']}_applied", ""))
                if remaining == 0:
                    trace.append(_trace("overlap_repair", "box_opt_resolved", ""))
                    return violations
        except Exception as exc:
            trace.append(_trace("overlap_repair", f"box_opt_failed: {exc}", ""))

    # Step 3: Fallback to full repack if overlaps remain or any room lacks a polygon
    needs_repack = remaining > 0
    if not needs_repack:
        for room in building.rooms:
            if room.polygon is None:
                needs_repack = True
                violations.append(f"{room.name} missing geometry → forcing repack")
                trace.append(_trace("overlap_repair", "placeholder_forced_repack", room.name))
                break

    if needs_repack:
        if remaining > 0:
            violations.append(f"{remaining} overlap(s) after {push_label} → repacking")
            trace.append(_trace("overlap_repair", f"{push_label}_incomplete_{remaining}_overlaps", ""))
        ok = _repack_fallback(building, boundary_polygon, entrance_point, trace)
        if not ok:
            violations.append("repack fallback also failed")
    else:
        trace.append(_trace("overlap_repair", f"{push_label}_resolved", ""))
    return violations


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 4: Corridor-first planning
# ═══════════════════════════════════════════════════════════════════════════════

def _stage4_corridor_planning(
    building: Building,
    boundary_polygon,
    entrance_point,
    regulation_file: str,
    trace: list,
) -> Tuple[List[Tuple[Building, str]], List[str]]:
    """Generate corridor variants via existing corridor_first_planner."""
    engine = RuleEngine(regulation_file)
    occ = building.occupancy_type
    # Use Chapter-4 Section 4.8.7 corridor width via proper method
    min_corr_w = engine.get_corridor_min_width(occ)

    violations = []
    variants = generate_corridor_first_variants(
        building,
        boundary_polygon=boundary_polygon,
        entrance_point=entrance_point,
        min_corridor_width=min_corr_w,
    )

    if not variants:
        violations.append("no corridor variants generated")
        trace.append(_trace("corridor", "no_variants", ""))
        variants = [(building, "fallback-none")]
    else:
        trace.append(_trace("corridor", f"generated_{len(variants)}_variants", ""))

    return variants, violations


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 5: Corridor accessibility enforcement
# ═══════════════════════════════════════════════════════════════════════════════

def _stage5_corridor_accessibility(var_building: Building, trace: list) -> List[str]:
    """Shift rooms that don't touch the corridor closer to it."""
    violations = []
    if not var_building.corridors:
        return violations

    corr = var_building.corridors[0]
    if not corr.polygon:
        return violations

    cpoly = Polygon(corr.polygon)
    threshold = 0.10  # metres tolerance

    for room in var_building.rooms:
        if room.polygon is None:
            continue
        rpoly = Polygon(room.polygon)
        dist = rpoly.distance(cpoly)
        if dist <= threshold:
            continue

        # Shift room toward corridor centroid
        rc = rpoly.centroid
        cc = cpoly.centroid
        dx = cc.x - rc.x
        dy = cc.y - rc.y
        length = max(math.hypot(dx, dy), 1e-6)
        shift = dist + 0.05  # shift just enough
        sx = dx / length * shift
        sy = dy / length * shift

        x1, y1, x2, y2 = _bbox(room)
        _set_rect(room, x1 + sx, y1 + sy, x2 + sx, y2 + sy)
        violations.append(f"{room.name}: shifted {shift:.2f}m toward corridor")
        trace.append(_trace("accessibility", f"shifted_{shift:.2f}m", room.name))

    return violations


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 6: Doors + connectivity
# ═══════════════════════════════════════════════════════════════════════════════

def _stage6_doors_connectivity(var_building: Building, min_door_width: float, trace: list) -> List[str]:
    """Place doors, check BFS connectivity, retry if needed."""
    violations = []

    dp = DoorPlacer(var_building, min_door_width)
    dp.place_doors()

    connected = is_fully_connected(var_building)
    if connected:
        trace.append(_trace("doors", "fully_connected", ""))
        return violations

    # Retry: add corridor doors for any unreachable rooms
    violations.append("not fully connected after first door pass")
    trace.append(_trace("doors", "not_connected_retry", ""))

    if var_building.corridors:
        corr = var_building.corridors[0]
        cpoly_pts = corr.polygon if corr.polygon else None
        if cpoly_pts:
            cpoly = Polygon(cpoly_pts)
            from core.door import Door
            for room in var_building.rooms:
                if room.polygon is None:
                    continue
                has_door = any(d.room_a == room or d.room_b == room for d in var_building.doors)
                if has_door:
                    continue
                rpoly = Polygon(room.polygon)
                if rpoly.distance(cpoly) <= 0.25:
                    # Create a synthetic door to corridor
                    rc = rpoly.centroid
                    seg = ((rc.x - min_door_width / 2, rc.y), (rc.x + min_door_width / 2, rc.y))
                    door = Door(room, None, min_door_width, seg, door_type="room_to_circulation")
                    var_building.add_door(door)
                    room.doors.append(door)
                    trace.append(_trace("doors", "synthetic_corridor_door", room.name))

    connected = is_fully_connected(var_building)
    if not connected:
        violations.append("still not fully connected after retry")
        trace.append(_trace("doors", "still_not_connected", ""))

    return violations


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 7: Travel distance check
# ═══════════════════════════════════════════════════════════════════════════════

def _stage7_travel_distance(var_building: Building, max_allowed: float, trace: list) -> List[str]:
    """Check travel distance; log violation if exceeded."""
    violations = []
    travel = _travel_distance(var_building)
    if travel <= max_allowed:
        trace.append(_trace("travel", f"compliant_{travel:.2f}<={max_allowed}", ""))
    else:
        violations.append(f"travel distance {travel:.2f} > allowed {max_allowed}")
        trace.append(_trace("travel", f"exceeded_{travel:.2f}>{max_allowed}", ""))
    return violations


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 8: Ontology / KG validation
# ═══════════════════════════════════════════════════════════════════════════════

def _stage8_ontology(var_building: Building, rule_engine: RuleEngine, trace: list) -> Tuple[List[str], Optional[Dict]]:
    """Run ontology validation if owlready2 is available; fallback to procedural."""
    violations = []
    ont_result = None

    try:
        from ontology.ontology_bridge import OntologyBridge
        bridge = OntologyBridge("ontology/regulatory.owl")
        ont_result = bridge.validate(var_building, rule_engine)
        if ont_result and not ont_result.get("valid", True):
            for v in ont_result.get("violations", []):
                violations.append(f"KG: {v.get('message', v.get('code', 'unknown'))}")
        trace.append(_trace("ontology", f"validated_reasoner={ont_result.get('reasoner', 'n/a')}", ""))
    except Exception as exc:
        trace.append(_trace("ontology", f"fallback_procedural: {exc}", ""))
        # Procedural fallback – connectivity & travel already checked

    return violations, ont_result


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluate a single variant (metrics computation)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_variant(building: Building, regulation_file: str, entrance_point=None) -> Dict:
    """Compute all metrics for a single variant."""
    engine = RuleEngine(regulation_file)
    occ = building.occupancy_type
    max_allowed = engine.get_max_travel_distance(occ)
    total_area_val = sum(r.final_area for r in building.rooms if r.final_area)
    building.total_area = total_area_val
    building.occupant_load = max(1, int(total_area_val / 100 * 8))

    connected = is_fully_connected(building)
    travel_dist = _travel_distance(building)
    zone_map = assign_room_zones(building, entrance_point=entrance_point)
    adj_score, adj_details = adjacency_satisfaction_score(building)

    circ = getattr(building, "corridors", [])
    walkable = round(sum(getattr(c, "walkable_area", 0.0) for c in circ), 2)
    corr_width = round(max((c.width for c in circ), default=0.0), 2)
    conn_exit = all(getattr(c, "connectivity_to_exit", False) for c in circ) if circ else False
    exit_width = building.exit.width if building.exit else 0

    # Corridor-served rooms
    served_names = []
    for c in circ:
        served_names.extend(getattr(c, "connects", []))
    served_names = sorted(set(served_names))

    return {
        "total_area": total_area_val,
        "occupant_load": building.occupant_load,
        "required_exit_width": exit_width,
        "max_travel_distance": travel_dist,
        "max_allowed_travel_distance": max_allowed,
        "travel_distance_compliant": travel_dist <= max_allowed,
        "fully_connected": connected,
        "zone_map": zone_map,
        "adjacency_satisfaction": adj_score,
        "adjacency_details": adj_details,
        "corridor_width": corr_width,
        "circulation_walkable_area": walkable,
        "connectivity_to_exit": conn_exit,
        "corridor_served_ratio": round(len(served_names) / max(len(building.rooms), 1), 4),
        "corridor_connected_room_names": served_names,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main: full 8-stage repair pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def validate_and_repair_generated_layout(
    building: Building,
    boundary_polygon: List[Tuple[float, float]],
    entrance_point: Optional[Tuple[float, float]] = None,
    regulation_file: str = "ontology/regulation_data.json",
    spec: Optional[Dict[str, Any]] = None,
    *,
    run_ontology: bool = True,
) -> Tuple[Building, List[str], str, List[Dict], RepairReport]:
    """
    Full 8-stage deterministic repair pipeline.

    Returns
    -------
    building_best : Building
        Best repaired variant (with corridor + doors).
    all_violations : list[str]
        All violations / repair actions applied.
    status : str
        ``"COMPLIANT"`` or ``"NON_COMPLIANT"``
    repair_trace : list[dict]
        Structured trace of every repair action (stage, action, room).
    repair_report : RepairReport
        Severity metrics and displacement tracking.
    """
    all_violations: List[str] = []
    repair_trace: List[Dict] = []
    engine = RuleEngine(regulation_file)
    occ = building.occupancy_type

    # ── Initialize repair tracking ─────────────────────────────────────────
    original_room_count = len(building.rooms)
    original_centroids = {}  # room_name -> (x, y)
    stages_with_changes = []
    overlap_fixes = 0
    min_dim_fixes = 0
    topology_changes = 0

    # Capture original centroids for displacement tracking
    for room in building.rooms:
        if room.polygon and len(room.polygon) >= 3:
            # Compute centroid
            xs = [p[0] for p in room.polygon]
            ys = [p[1] for p in room.polygon]
            original_centroids[room.name] = (sum(xs) / len(xs), sum(ys) / len(ys))

    initial_trace_length = len(repair_trace)

    # ── Stage 1: Sanitize geometry ────────────────────────────────────────
    stage1_violations_before = len(all_violations)
    all_violations.extend(_stage1_sanitize(building, boundary_polygon, repair_trace))
    if len(all_violations) > stage1_violations_before:
        stages_with_changes.append("sanitize_geometry")
        # Count topology changes (room drops/merges in stage 1)
        topology_changes += len([v for v in all_violations[stage1_violations_before:]
                               if "dropped" in v or "merged" in v])

    if not building.rooms:
        # Create empty repair report for failed case
        repair_report = RepairReport(
            severity_score=1.0,  # Complete failure = maximum severity
            overlap_fixes=0,
            min_dim_violations_fixed=0,
            topology_changes=original_room_count,  # All rooms lost
            total_displacement_m=0.0,
            stages_applied=[],
            original_room_count=original_room_count,
            final_room_count=0,
        )
        return building, all_violations + ["no rooms survived sanitization"], "NON_COMPLIANT", repair_trace, repair_report

    # ── Stage 2: Enforce minimums ─────────────────────────────────────────
    stage2_violations_before = len(all_violations)
    all_violations.extend(_stage2_enforce_minimums(building, regulation_file, repair_trace))
    if len(all_violations) > stage2_violations_before:
        stages_with_changes.append("enforce_minimums")
        min_dim_fixes += len(all_violations) - stage2_violations_before

    # ── Stage 2b: Aspect-ratio enforcement ────────────────────────────────
    aspect_fixes = 0
    for room in building.rooms:
        if enforce_aspect_ratio(room):
            all_violations.append(f"{room.name}: aspect ratio reshaped")
            repair_trace.append(_trace("aspect_ratio", "reshaped", room.name))
            aspect_fixes += 1

    if aspect_fixes > 0:
        stages_with_changes.append("aspect_ratio_enforcement")

    # Re-clamp after growth / reshape (rooms may have expanded outside boundary)
    _stage1_sanitize(building, boundary_polygon, repair_trace)

    # ── Stage 3: Overlap repair ───────────────────────────────────────────
    stage3_violations_before = len(all_violations)
    all_violations.extend(_stage3_overlap_repair(building, boundary_polygon, entrance_point, repair_trace))
    if len(all_violations) > stage3_violations_before:
        stages_with_changes.append("overlap_repair")
        overlap_fixes += len(all_violations) - stage3_violations_before

    # ── Stage 4: Corridor planning ────────────────────────────────────────
    variants, corr_violations = _stage4_corridor_planning(
        building, boundary_polygon, entrance_point, regulation_file, repair_trace,
    )
    all_violations.extend(corr_violations)

    # ── Stages 5-8: per-variant evaluation; pick best ─────────────────────
    min_door_width = engine.get_min_door_width(occ)
    max_allowed_travel = engine.get_max_travel_distance(occ)

    best_variant = None
    best_score = -1.0
    best_metrics: Dict = {}
    best_strategy = ""
    best_variant_violations: List[str] = []
    best_variant_trace: List[Dict] = []

    for var_building, strategy in variants:
        vt: List[Dict] = []
        vv: List[str] = []

        # Stage 5: corridor accessibility
        vv.extend(_stage5_corridor_accessibility(var_building, vt))

        # Stage 6: doors + connectivity
        vv.extend(_stage6_doors_connectivity(var_building, min_door_width, vt))

        # Stage 7: travel distance
        vv.extend(_stage7_travel_distance(var_building, max_allowed_travel, vt))

        # Stage 8: ontology/KG
        ont_violations = []
        ont_result = None
        if run_ontology:
            ont_violations, ont_result = _stage8_ontology(var_building, engine, vt)
            vv.extend(ont_violations)

        # Compute metrics and score
        metrics = evaluate_variant(var_building, regulation_file, entrance_point)
        score = 0.0
        if metrics["fully_connected"] and metrics["travel_distance_compliant"]:
            score += 1.0
        score += 0.3 * metrics.get("adjacency_satisfaction", 0.0)
        if metrics.get("connectivity_to_exit"):
            score += 0.2

        if score > best_score:
            best_variant = var_building
            best_score = score
            best_metrics = metrics
            best_strategy = strategy
            best_variant_violations = vv
            best_variant_trace = vt

    if best_variant is None:
        # Create failure repair report
        repair_report = RepairReport(
            severity_score=1.0,  # Complete failure
            overlap_fixes=overlap_fixes,
            min_dim_violations_fixed=min_dim_fixes,
            topology_changes=topology_changes,
            total_displacement_m=0.0,
            stages_applied=stages_with_changes,
            original_room_count=original_room_count,
            final_room_count=len(building.rooms),
        )
        return building, all_violations + ["no viable variant"], "NON_COMPLIANT", repair_trace, repair_report

    all_violations.extend(best_variant_violations)
    repair_trace.extend(best_variant_trace)

    # ── Final: Grid snap ─────────────────────────────────────────────────
    n_snapped = snap_building_to_grid(best_variant, step=0.15)
    if n_snapped:
        repair_trace.append(_trace("grid_snap", f"snapped_{n_snapped}_polygons", ""))
        if "grid_snap" not in stages_with_changes:
            stages_with_changes.append("grid_snap")

    compliant = best_metrics.get("fully_connected", False) and best_metrics.get("travel_distance_compliant", False)
    status = "COMPLIANT" if compliant else "NON_COMPLIANT"

    # ── Compute repair severity metrics ────────────────────────────────────
    final_room_count = len(best_variant.rooms)

    # Compute total displacement
    total_displacement = 0.0
    displacement_count = 0

    for room in best_variant.rooms:
        if room.name in original_centroids and room.polygon and len(room.polygon) >= 3:
            # Current centroid
            xs = [p[0] for p in room.polygon]
            ys = [p[1] for p in room.polygon]
            current_centroid = (sum(xs) / len(xs), sum(ys) / len(ys))

            # Distance moved
            orig = original_centroids[room.name]
            dist = ((current_centroid[0] - orig[0])**2 + (current_centroid[1] - orig[1])**2)**0.5
            total_displacement += dist
            displacement_count += 1

    # Convert to physical units (assume boundary spans ~15m typical)
    if boundary_polygon:
        xs = [p[0] for p in boundary_polygon]
        ys = [p[1] for p in boundary_polygon]
        boundary_width = max(xs) - min(xs) if xs else 15.0
        boundary_height = max(ys) - min(ys) if ys else 15.0
        boundary_scale = max(boundary_width, boundary_height, 1.0)
        total_displacement_m = total_displacement * boundary_scale
    else:
        total_displacement_m = total_displacement * 15.0  # Fallback estimate

    # Severity score: [0-1] based on extent of changes
    severity_components = []

    # Room count change (major impact)
    if original_room_count != final_room_count:
        severity_components.append(0.4)  # 40% if room count changed

    # Displacement severity (0-30% based on avg movement)
    if displacement_count > 0:
        avg_displacement = total_displacement
        # Normalize to [0, 0.3]: displacement > 0.5 normalized units = 30% severity
        displacement_severity = min(0.3, avg_displacement * 0.6)
        severity_components.append(displacement_severity)

    # Fixes applied (0-30% based on extent)
    total_fixes = overlap_fixes + min_dim_fixes + topology_changes
    if total_fixes > 0:
        # Normalize: 10+ fixes = 30% severity
        fixes_severity = min(0.3, total_fixes / 10.0 * 0.3)
        severity_components.append(fixes_severity)

    # Stages applied (0-20% based on breadth)
    if len(stages_with_changes) > 0:
        # Normalize: 5+ stages = 20% severity
        stage_severity = min(0.2, len(stages_with_changes) / 5.0 * 0.2)
        severity_components.append(stage_severity)

    severity_score = min(1.0, sum(severity_components))

    repair_report = RepairReport(
        severity_score=round(severity_score, 3),
        overlap_fixes=overlap_fixes,
        min_dim_violations_fixed=min_dim_fixes,
        topology_changes=topology_changes,
        total_displacement_m=round(total_displacement_m, 2),
        stages_applied=stages_with_changes,
        original_room_count=original_room_count,
        final_room_count=final_room_count,
    )

    return best_variant, all_violations, status, repair_trace, repair_report
