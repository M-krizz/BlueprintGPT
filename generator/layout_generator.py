import copy
import traceback
from pathlib import Path

from shapely.geometry import Point, Polygon

from core.building import Building
from core.room import Room
from core.exit import Exit
from constraints.rule_engine import RuleEngine
from geometry.allocator import Allocator
from geometry.polygon_packer import PolygonPacker
from geometry.door_placer import DoorPlacer
from geometry.corridor_placer import generate_corridor_variants
from geometry.corridor_first_planner import generate_corridor_first_variants
from geometry.zoning import assign_room_zones


def _resolve_room_names_for_building(building, room_key):
    room_key = str(room_key or "").strip()
    if not room_key:
        return []
    exact = [room.name for room in building.rooms if room.name == room_key]
    if exact:
        return exact
    prefix = f"{room_key}_"
    matches = [room.name for room in building.rooms if room.name.startswith(prefix)]
    if matches:
        return matches
    return [room.name for room in building.rooms if room.room_type == room_key]


def _apply_post_rule_area_guidance(building, spec):
    nl_spec = spec.get("_nl_spec") or {}
    room_size_preferences = nl_spec.get("room_size_preferences") or {}
    if not room_size_preferences:
        return

    scale_map = {
        "larger": 1.18,
        "bigger": 1.18,
        "wider": 1.12,
        "smaller": 0.88,
        "narrower": 0.9,
    }
    room_lookup = {room.name: room for room in building.rooms}
    desired_areas = {
        room.name: float(room.final_area or room.requested_area or 0.0)
        for room in building.rooms
    }
    targeted_names = set()

    for room_key, size_change in room_size_preferences.items():
        scale = scale_map.get(str(size_change).strip().lower())
        if scale is None:
            continue
        for room_name in _resolve_room_names_for_building(building, room_key):
            room = room_lookup.get(room_name)
            if not room:
                continue
            current_area = float(desired_areas.get(room_name, room.final_area or room.requested_area or 0.0))
            min_area = float(room.min_area or 0.0)
            adjusted_area = max(min_area, current_area * scale)
            desired_areas[room_name] = adjusted_area
            targeted_names.add(room_name)

    if not targeted_names:
        return

    base_total = sum(float(room.final_area or room.requested_area or 0.0) for room in building.rooms)
    adjusted_total = sum(desired_areas.values())
    if base_total > 0 and adjusted_total > 0:
        normalization_scale = base_total / adjusted_total
        for room_name in desired_areas:
            desired_areas[room_name] *= normalization_scale

    for room_name, desired_area in desired_areas.items():
        room = room_lookup[room_name]
        min_area = float(room.min_area or 0.0)
        adjusted = max(min_area, desired_area)
        room.requested_area = adjusted
        room.final_area = adjusted
        room.target_area = adjusted
from geometry.adjacency_intent import adjacency_satisfaction_score
from geometry.polygon import snap_building_to_grid, alignment_score, enforce_aspect_ratio
from graph.connectivity import is_fully_connected
from graph.manhattan_path import max_travel_distance
from graph.door_graph_path import door_graph_travel_distance
from generator.ranking import rank_layout_variants
from generator.composition_metrics import composition_quality

# ── Default checkpoint for learned generator ──────────────────────────────────
_DEFAULT_CHECKPOINT = "learned/model/checkpoints/kaggle_test.pt"


def _polygon_area(points):
    if not points or len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _room_area_quality(building):
    room_area_errors = []
    max_error = 0.0
    for room in getattr(building, "rooms", []):
        if not getattr(room, "polygon", None):
            room_area_errors.append(
                {
                    "room": room.name,
                    "target_area": round(float(getattr(room, "target_area", room.final_area or 0.0)), 3),
                    "actual_area": 0.0,
                    "relative_error": 1.0,
                }
            )
            max_error = 1.0
            continue

        target_area = float(getattr(room, "target_area", room.final_area or 0.0))
        actual_area = _polygon_area(room.polygon)
        relative_error = abs(actual_area - target_area) / max(target_area, 1e-6)
        room_area_errors.append(
            {
                "room": room.name,
                "target_area": round(target_area, 3),
                "actual_area": round(actual_area, 3),
                "relative_error": round(relative_error, 4),
            }
        )
        max_error = max(max_error, relative_error)

    return {
        "max_room_area_error": round(max_error, 4),
        "room_area_errors": room_area_errors,
    }


def _composition_quality(building, entrance_point, zone_map, adjacency_details):
    return composition_quality(building, entrance_point, zone_map, adjacency_details)


def _entrance_accessible_without_corridor(building, entrance_point, tolerance=0.25):
    if not entrance_point:
        return False
    point = Point(entrance_point)
    for room in getattr(building, "rooms", []):
        polygon = getattr(room, "polygon", None)
        if not polygon:
            continue
        try:
            room_poly = Polygon(polygon)
        except Exception:
            continue
        if room_poly.boundary.distance(point) <= tolerance:
            return True
    return False


def _build_base(spec, regulation_file):
    """Build and return a base (building, engine, allocation_breakdown, modifications)
    from the spec without running corridor placement or ontology."""
    occupancy = spec.get("occupancy", "Residential")
    building = Building(occupancy_type=occupancy)

    total_area_input = spec.get("total_area")
    area_unit = spec.get("area_unit", "sq.ft")
    allocation_strategy = spec.get("allocation_strategy", "priority_weights")

    for room_data in spec.get("rooms", []):
        requested_area = float(room_data.get("area", 0.0))
        building.add_room(Room(room_data["name"], room_data["type"], requested_area))

    engine = RuleEngine(regulation_file)

    rule_preflight = engine.preflight_validate_spec(spec)
    if not rule_preflight.get("valid", False):
        raise ValueError("Rule preflight failed: " + "; ".join(rule_preflight.get("errors", [])))

    modifications = []
    allocation_breakdown = None

    if total_area_input is not None:
        alloc_mods, allocation_breakdown = engine.allocate_room_areas_from_total(
            building,
            float(total_area_input),
            unit=area_unit,
            strategy=allocation_strategy,
        )
        modifications.extend(alloc_mods)

    modifications.extend(engine.apply_room_rules(building))
    _apply_post_rule_area_guidance(building, spec)
    for room in building.rooms:
        room.target_area = float(room.final_area or room.requested_area or 0.0)
    total_area, occupant_load = engine.compute_building_metrics(building)
    exit_width = engine.compute_exit_width(building)

    # ── Geometry: pack rooms ──────────────────────────────────────────────────
    boundary_polygon = spec.get("boundary_polygon")
    entrance_point = spec.get("entrance_point")
    planner_guidance = spec.get("planner_guidance") or {}
    # Transformer spatial hints (from hybrid pipeline): {room_type: (cx_norm, cy_norm)}
    learned_hints = (
        spec.get("learned_spatial_hints")
        or planner_guidance.get("spatial_hints")
        or {}
    )
    placement_order = planner_guidance.get("room_order") or []
    room_zones = planner_guidance.get("room_zones") or {}
    if boundary_polygon and len(boundary_polygon) >= 3:
        packer = PolygonPacker(
            building, boundary_polygon,
            entrance_point=entrance_point,
            learned_hints=learned_hints if learned_hints else None,
            placement_order=placement_order,
            room_zones=room_zones,
        )
        width, height = packer.allocate()
    else:
        allocator = Allocator(building)
        width, height = allocator.allocate()

    exit_obj = Exit(width=exit_width)
    exit_obj.segment = ((0, 0), (exit_width, 0))
    building.set_exit(exit_obj)

    return (
        building,
        engine,
        allocation_breakdown,
        modifications,
        width,
        height,
        boundary_polygon,
        rule_preflight,
    )


def generate_layout_from_spec(spec, regulation_file, ontology_validator=None):
    """
    Generate 3–5 layout variants (different corridor strategies) and return them
    all. Also returns the first variant as the 'primary' result for backward compat.
    """
    building, engine, allocation_breakdown, modifications, width, height, boundary_polygon, rule_preflight = \
        _build_base(spec, regulation_file)
    planner_guidance = spec.get("planner_guidance") or {}
    learned_hints = (
        spec.get("learned_spatial_hints")
        or planner_guidance.get("spatial_hints")
        or {}
    )
    placement_order = planner_guidance.get("room_order") or []
    room_zones = planner_guidance.get("room_zones") or {}
    adjacency_preferences = planner_guidance.get("adjacency_preferences") or []
    frontage_room = planner_guidance.get("frontage_room")
    layout_pattern = planner_guidance.get("layout_pattern")

    kg_precheck = None
    if ontology_validator is not None and hasattr(ontology_validator, "validate_spec_semantics"):
        kg_precheck = ontology_validator.validate_spec_semantics(spec)
        if not kg_precheck.get("valid", False):
            raise ValueError("KG semantic precheck failed: " + "; ".join(kg_precheck.get("errors", [])))

    # ── Generate corridor variants ────────────────────────────────────────────
    circulation_factor = engine.data[building.occupancy_type]["circulation_factor"]
    min_corridor_width = engine.data[building.occupancy_type].get(
        "corridor", {}).get("min_width", 1.2)
    min_door_width = engine.get_min_door_width(building.occupancy_type)
    max_allowed_travel = engine.get_max_travel_distance(building.occupancy_type)

    if boundary_polygon and len(boundary_polygon) >= 3:
        variants = [(copy.deepcopy(building), "direct-zonal")]
        variants.extend(generate_corridor_first_variants(
            building,
            boundary_polygon=boundary_polygon,
            entrance_point=spec.get("entrance_point"),
            min_corridor_width=min_corridor_width,
            learned_hints=learned_hints if learned_hints else None,
            placement_order=placement_order,
            room_zones=room_zones,
            adjacency_preferences=adjacency_preferences,
            frontage_room=frontage_room,
            layout_pattern=layout_pattern,
        ))
    else:
        variants = generate_corridor_variants(
            building,
            circulation_factor=circulation_factor,
            min_corridor_width=min_corridor_width,
            boundary_polygon=boundary_polygon,
            entrance_point=spec.get("entrance_point"),
        )

    # ── Validate each variant ─────────────────────────────────────────────────
    if not variants:
        variants = [(building, "balanced")]

    validated_variants = []
    for var_building, strategy_name in variants:
        # Aspect ratio enforcement (Disabled to allow arbitrary free-form polygon shapes)
        # for room in var_building.rooms:
        #     enforce_aspect_ratio(room)

        # Grid snapping
        snap_building_to_grid(var_building, step=0.15)

        # doors are generated after circulation carving so room/circulation boundaries are respected
        door_placer = DoorPlacer(var_building, min_door_width)
        door_placer.place_doors()

        connected = is_fully_connected(var_building)
        travel_distance = max_travel_distance(var_building)
        zone_map = assign_room_zones(var_building, entrance_point=spec.get("entrance_point"))
        adjacency_score, adjacency_details = adjacency_satisfaction_score(var_building)
        circulation_spaces = getattr(var_building, "corridors", [])
        skip_corridors = strategy_name == "direct-zonal"
        walkable_area = round(sum(getattr(c, "walkable_area", 0.0) for c in circulation_spaces), 2)
        corridor_width = round(max((c.width for c in circulation_spaces), default=0.0), 2)
        connectivity_to_exit = all(
            getattr(c, "connectivity_to_exit", False) for c in circulation_spaces
        ) if circulation_spaces else False
        if skip_corridors:
            connectivity_to_exit = _entrance_accessible_without_corridor(
                var_building,
                spec.get("entrance_point"),
            )

        # Door-graph travel distance (Dijkstra through doors + corridor)
        door_path_travel = door_graph_travel_distance(var_building)

        # Alignment quality
        align_score = alignment_score(var_building)
        area_quality = _room_area_quality(var_building)
        composition_quality = _composition_quality(
            var_building,
            spec.get("entrance_point"),
            zone_map,
            adjacency_details,
        )

        ont_result = None
        if ontology_validator is not None:
            ont_result = ontology_validator.validate(var_building, engine)

        validated_variants.append({
            "building":      var_building,
            "strategy_name": strategy_name,
            "bounding_box":  {"width": width, "height": height},
            "allocation":    allocation_breakdown,
            "modifications": modifications,
            "metrics": {
                "total_area":               var_building.total_area,
                "occupant_load":            var_building.occupant_load,
                "required_exit_width":      var_building.exit.width if var_building.exit else 0,
                "max_travel_distance":      travel_distance,
                "max_allowed_travel_distance": max_allowed_travel,
                "travel_distance_compliant": travel_distance <= max_allowed_travel,
                "fully_connected":          connected,
                "zone_map":                 zone_map,
                "adjacency_satisfaction":   adjacency_score,
                "adjacency_details":        adjacency_details,
                "corridor_width":           corridor_width,
                "circulation_walkable_area": walkable_area,
                "connectivity_to_exit":     connectivity_to_exit,
                "alignment_score":          align_score,
                "door_path_travel_distance": door_path_travel,
                "max_room_area_error":      area_quality["max_room_area_error"],
                "room_area_errors":         area_quality["room_area_errors"],
                "skip_corridors":           skip_corridors,
                "public_frontage_score":    composition_quality["public_frontage_score"],
                "bedroom_privacy_score":    composition_quality["bedroom_privacy_score"],
                "kitchen_living_score":     composition_quality["kitchen_living_score"],
                "bathroom_access_score":    composition_quality["bathroom_access_score"],
                "architectural_reasonableness": composition_quality["architectural_reasonableness"],
            },
            "ontology":   ont_result,
            "input_spec": spec,
            "spec_validation": spec.get("_spec_validation"),
            "repair": spec.get("_repair"),
            "rule_preflight": rule_preflight,
            "kg_precheck": kg_precheck,
            "source":     "algorithmic",
        })

    # ── Learned-generator variants (Transformer + repair) ─────────────────────
    learned_variants = _generate_learned_variants(
        spec, boundary_polygon, engine, ontology_validator,
        width, height, allocation_breakdown, modifications,
        max_allowed_travel, rule_preflight, kg_precheck,
    )
    validated_variants.extend(learned_variants)

    ranked_variants, recommended_index = rank_layout_variants(validated_variants)

    # Primary result = first variant (backward compatibility)
    primary = ranked_variants[0] if ranked_variants else {}
    primary["layout_variants"] = ranked_variants
    primary["recommended_index"] = recommended_index

    return primary


def _generate_learned_variants(
    spec, boundary_polygon, engine, ontology_validator,
    width, height, allocation_breakdown, modifications,
    max_allowed_travel, rule_preflight, kg_precheck,
    K: int = 5,
) -> list:
    """
    Attempt to generate layout variants from the trained LayoutTransformer.
    Returns a list of validated-variant dicts (same schema as algorithmic ones)
    with additional ``source``, ``repair_trace``, and ``raw_validity`` fields.
    Gracefully returns [] if checkpoint is missing or generation fails.
    """
    checkpoint = spec.get("learned_checkpoint", _DEFAULT_CHECKPOINT)
    if not Path(checkpoint).exists():
        return []

    try:
        from learned.integration.model_generation_loop import generate_best_layout_from_model
        from learned.integration.repair_gate import evaluate_variant
    except ImportError:
        return []

    entrance_point = spec.get("entrance_point")
    occupancy = spec.get("occupancy", "Residential")

    learned_validated = []

    try:
        best_variant, summary = generate_best_layout_from_model(
            spec=spec,
            boundary_poly=boundary_polygon if boundary_polygon else [(0, 0), (width, 0), (width, height), (0, height)],
            entrance=entrance_point,
            checkpoint_path=checkpoint,
            regulation_file="ontology/regulation_data.json",
            K=K,
            max_attempts=max(K * 3, 15),
            temperature=0.85,
        )
    except Exception:
        traceback.print_exc()
        return []

    if best_variant is None:
        return []

    # Collect the best candidate and runners-up from the resample loop
    all_candidates = best_variant.get("all_candidates", [])
    top_candidates = [c for c in all_candidates if c.get("building") is not None][:3]

    for idx, cand in enumerate(top_candidates):
        var_building = cand["building"]
        metrics = cand.get("metrics", {})
        if not metrics:
            metrics = evaluate_variant(var_building, "ontology/regulation_data.json", entrance_point)

        ont_result = None
        if ontology_validator is not None:
            try:
                ont_result = ontology_validator.validate(var_building, engine)
            except Exception:
                pass

        learned_validated.append({
            "building":      var_building,
            "strategy_name": f"learned-gen-{idx+1}",
            "bounding_box":  {"width": width, "height": height},
            "allocation":    allocation_breakdown,
            "modifications": cand.get("violations", []),
            "metrics":       metrics,
            "ontology":      ont_result,
            "input_spec":    spec,
            "spec_validation": spec.get("_spec_validation"),
            "repair":        spec.get("_repair"),
            "rule_preflight": rule_preflight,
            "kg_precheck":   kg_precheck,
            # ── Learned-specific fields ───────────────────────────────────
            "source":        "learned",
            "repair_trace":  cand.get("repair_trace", []),
            "raw_validity":  cand.get("raw_valid", False),
            "generation_summary": {
                "raw_valid_count":     summary.get("raw_valid_count", 0),
                "repaired_valid_count": summary.get("repaired_valid_count", 0),
                "total_attempts":      summary.get("total_attempts", 0),
                "top_failure_reasons": summary.get("top_failure_reasons", {}),
            },
        })

    return learned_validated



