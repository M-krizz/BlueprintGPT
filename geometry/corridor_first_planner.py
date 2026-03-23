import copy
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import unary_union

from core.corridor import Corridor
from geometry.polygon_packer import recursive_pack


STRATEGIES = [
    "balanced",
    "compactness-first",
    "privacy-first",
    "adjacency-first",
    "corridor-minimization",
]


def _get_polygons(geom):
    if geom is None or geom.is_empty:
        return []
    gtype = getattr(geom, "geom_type", "")
    if gtype == "Polygon":
        return [geom]
    if gtype == "MultiPolygon":
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if getattr(g, "geom_type", "") == "Polygon"]


def _dedupe_points(points, tol=1e-6):
    cleaned = []
    for p in points:
        if not cleaned:
            cleaned.append(p)
            continue
        if abs(cleaned[-1][0] - p[0]) > tol or abs(cleaned[-1][1] - p[1]) > tol:
            cleaned.append(p)
    return cleaned


def _farthest_boundary_point(boundary_poly, origin):
    boundary = boundary_poly.boundary
    coords = []
    if hasattr(boundary, "coords"):
        coords.extend(list(boundary.coords))
    else:
        for geom in getattr(boundary, "geoms", []):
            coords.extend(list(getattr(geom, "coords", [])))
    if not coords:
        return origin
    return max(coords, key=lambda p: (p[0] - origin[0]) ** 2 + (p[1] - origin[1]) ** 2)


def _orthogonal_segment_points(p, q):
    if abs(p[0] - q[0]) < 1e-6 or abs(p[1] - q[1]) < 1e-6:
        return [p, q]
    elbow = (q[0], p[1])
    return [p, elbow, q]


def _orthogonal_spine(waypoints):
    if len(waypoints) < 2:
        return waypoints
    points = []
    for i in range(len(waypoints) - 1):
        seg = _orthogonal_segment_points(waypoints[i], waypoints[i + 1])
        if points:
            points.extend(seg[1:])
        else:
            points.extend(seg)
    return _dedupe_points(points)


def _corridor_width(min_corridor_width, strategy_name):
    if strategy_name == "privacy-first":
        return max(min_corridor_width, 1.35)
    if strategy_name == "adjacency-first":
        return max(min_corridor_width, 1.25)
    return max(min_corridor_width, 1.2)


def _strategy_waypoints(boundary_poly, entrance_point, strategy_name):
    c = boundary_poly.centroid
    centroid = (c.x, c.y)
    entrance = entrance_point if entrance_point else centroid
    far = _farthest_boundary_point(boundary_poly, entrance)

    if strategy_name == "compactness-first":
        waypoints = [entrance, centroid]
    elif strategy_name == "corridor-minimization":
        waypoints = [entrance, ((entrance[0] + centroid[0]) / 2.0, entrance[1]), centroid]
    elif strategy_name == "privacy-first":
        waypoints = [entrance, (centroid[0], entrance[1]), centroid, (centroid[0], far[1]), far]
    elif strategy_name == "adjacency-first":
        waypoints = [entrance, (centroid[0], entrance[1]), centroid, (far[0], centroid[1])]
    else:
        waypoints = [entrance, (centroid[0], entrance[1]), centroid]

    return _orthogonal_spine(_dedupe_points(waypoints))


def _orthogonal_strip_buffer(points, width):
    half = width / 2.0
    pieces = []
    for i in range(len(points) - 1):
        p = points[i]
        q = points[i + 1]
        if abs(p[0] - q[0]) < 1e-6:
            x = p[0]
            miny, maxy = sorted([p[1], q[1]])
            pieces.append(box(x - half, miny, x + half, maxy))
        elif abs(p[1] - q[1]) < 1e-6:
            y = p[1]
            minx, maxx = sorted([p[0], q[0]])
            pieces.append(box(minx, y - half, maxx, y + half))
        else:
            pieces.append(LineString([p, q]).buffer(half, cap_style=2, join_style=2))

    if not pieces:
        return Polygon()
    return unary_union(pieces)


def _rooms_sorted_for_packing(building):
    rooms = [r for r in building.rooms if r.final_area and r.final_area > 0]
    rooms.sort(key=lambda r: (r.room_type == "LivingRoom", r.final_area), reverse=True)
    return rooms


def _room_zone(room, room_zones=None):
    room_name = getattr(room, "name", "")
    if room_zones and room_name in room_zones:
        return room_zones[room_name]
    if room.room_type in {"LivingRoom"}:
        return "public"
    if room.room_type in {"Kitchen"}:
        return "service"
    if room.room_type in {"Bedroom", "Bathroom", "WC"}:
        return "private"
    return "service"


def _region_distance_score(region, entrance_point):
    c = region.centroid
    if entrance_point is None:
        return 0.0
    return ((c.x - entrance_point[0]) ** 2 + (c.y - entrance_point[1]) ** 2) ** 0.5


def _split_regions_into_bands(regions, entrance_point):
    if not regions:
        return {"public": [], "service": [], "private": []}

    ranked = sorted(regions, key=lambda r: _region_distance_score(r, entrance_point))
    n = len(ranked)
    one_third = max(1, n // 3)
    two_third = max(one_third + 1, (2 * n) // 3)

    return {
        "public": ranked[:one_third],
        "service": ranked[one_third:two_third],
        "private": ranked[two_third:] or ranked[-1:],
    }


def _zone_priority(zone):
    return {"public": 0, "service": 1, "private": 2}.get(zone, 3)


def _room_adjacency_strength(room_name, adjacency_preferences):
    if not adjacency_preferences:
        return 0.0
    score = 0.0
    for item in adjacency_preferences:
        if room_name in {item.get("a"), item.get("b")}:
            score += float(item.get("score", item.get("weight", 0.0)) or 0.0)
    return score


def _distribute_rooms_across_regions(rooms, regions, entrance_point=None, learned_hints=None):
    if not rooms or not regions:
        return False

    total_room_area = sum(r.final_area for r in rooms)
    total_region_area = sum(r.area for r in regions)
    if total_room_area <= 0 or total_region_area <= 0:
        return False

    room_idx = 0
    for region_idx, region in enumerate(regions):
        if room_idx >= len(rooms):
            break

        next_room_area = float(getattr(rooms[room_idx], "final_area", 0.0) or 0.0)
        remaining_room_area = sum(float(getattr(room, "final_area", 0.0) or 0.0) for room in rooms[room_idx:])
        remaining_region_area = sum(float(regions[idx].area) for idx in range(region_idx, len(regions)))
        if (
            region_idx < len(regions) - 1
            and next_room_area > 0.0
            and region.area < next_room_area * 0.8
            and (remaining_region_area - region.area) >= remaining_room_area * 0.92
        ):
            continue

        if region_idx == len(regions) - 1:
            assigned = rooms[room_idx:]
            room_idx = len(rooms)
        else:
            target = total_room_area * (region.area / total_region_area)
            running = 0.0
            assigned = []
            while room_idx < len(rooms):
                room = rooms[room_idx]
                assigned.append(room)
                running += room.final_area
                room_idx += 1
                if running >= target * 0.9:
                    break

        if not assigned:
            continue

        items = [{"room": room, "weight": room.final_area} for room in assigned]
        recursive_pack(region, items, entrance_pt=entrance_point, learned_hints=learned_hints)

    return room_idx == len(rooms) and all(room.polygon for room in rooms)


def _allocate_rooms_in_regions(
    building,
    regions,
    entrance_point=None,
    learned_hints=None,
    placement_order=None,
    room_zones=None,
    adjacency_preferences=None,
    frontage_room=None,
    layout_pattern=None,
):
    rooms = _rooms_sorted_for_packing(building)
    if not rooms or not regions:
        return False

    order_rank = {room_name: index for index, room_name in enumerate(placement_order or [])}
    rooms.sort(
        key=lambda room: (
            order_rank.get(getattr(room, "name", ""), len(order_rank) + 1),
            _zone_priority(_room_zone(room, room_zones)),
            -_room_adjacency_strength(getattr(room, "name", ""), adjacency_preferences),
            -(room.final_area or 0.0),
        )
    )

    regions_by_distance = sorted(regions, key=lambda region: _region_distance_score(region, entrance_point))
    if len(regions_by_distance) < 3:
        return _distribute_rooms_across_regions(
            rooms,
            regions_by_distance,
            entrance_point=entrance_point,
            learned_hints=learned_hints,
        )

    room_groups = {"public": [], "service": [], "private": []}
    for room in rooms:
        room_groups.setdefault(_room_zone(room, room_zones), []).append(room)

    public_rooms = room_groups.get("public") or []
    if not public_rooms:
        return _distribute_rooms_across_regions(
            rooms,
            regions_by_distance,
            entrance_point=entrance_point,
            learned_hints=learned_hints,
        )

    region_bands = _split_regions_into_bands(regions_by_distance, entrance_point)
    frontage_target = next(
        (room for room in public_rooms if frontage_room and getattr(room, "name", "") == frontage_room),
        None,
    ) or public_rooms[0]
    frontage_candidates = list(region_bands.get("public") or regions_by_distance[:1])
    frontage_region = max(frontage_candidates[: min(2, len(frontage_candidates))], key=lambda region: region.area)

    if not _distribute_rooms_across_regions(
        [frontage_target],
        [frontage_region],
        entrance_point=entrance_point,
        learned_hints=learned_hints,
    ):
        return False

    remaining_public_rooms = [room for room in public_rooms if room != frontage_target]
    room_groups["public"] = remaining_public_rooms

    frontage_region_id = id(frontage_region)
    region_bands["public"] = [region for region in region_bands.get("public", []) if id(region) != frontage_region_id]

    zone_sequence = ["public", "service", "private"]
    if layout_pattern == "compact_frontage":
        zone_sequence = ["public", "service", "private"]
    elif layout_pattern == "zonal_split":
        zone_sequence = ["public", "service", "private"]

    spillover_regions = []
    for zone in zone_sequence:
        zone_rooms = room_groups.get(zone) or []
        zone_regions = list(region_bands.get(zone) or [])
        if not zone_rooms:
            spillover_regions.extend(zone_regions)
            continue

        candidate_regions = zone_regions or spillover_regions
        spillover_regions = []
        if not candidate_regions:
            return False

        zone_rooms.sort(
            key=lambda room: (
                order_rank.get(getattr(room, "name", ""), len(order_rank) + 1),
                -_room_adjacency_strength(getattr(room, "name", ""), adjacency_preferences),
                -(room.final_area or 0.0),
            )
        )
        if not _distribute_rooms_across_regions(
            zone_rooms,
            candidate_regions,
            entrance_point=entrance_point,
            learned_hints=learned_hints,
        ):
            return False

    return all(room.polygon for room in rooms)


def _build_circulation_from_boundary(boundary_poly, entrance_point, strategy_name, min_corridor_width):
    width = _corridor_width(min_corridor_width, strategy_name)
    points = _strategy_waypoints(boundary_poly, entrance_point, strategy_name)
    if len(points) < 2:
        return Polygon(), points, width, False

    strip = _orthogonal_strip_buffer(points, width)

    # 1. Keep corridor internal, but not so deep that it dominates the frontage.
    inward_margin = max(0.35, min(0.55, width * 0.4))
    inward_poly = boundary_poly.buffer(-inward_margin)
    if inward_poly.is_empty:
        inward_poly = boundary_poly.buffer(-0.15)
    if inward_poly.is_empty:
        inward_poly = boundary_poly

    core_circulation = strip.intersection(inward_poly)

    # 2. Add the shortest entrance stem needed to reach the internal spine.
    c = boundary_poly.centroid
    centroid = (c.x, c.y)
    entrance = entrance_point if entrance_point else centroid
    stem_target = points[1] if len(points) > 1 else centroid
    stem = LineString([entrance, stem_target]).buffer(width / 2.0, cap_style=2, join_style=2).intersection(boundary_poly)

    circulation = unary_union([core_circulation, stem])

    connected = True
    if entrance_point is not None and not circulation.is_empty:
        connected = Point(entrance_point).distance(circulation) <= width
    return circulation, points, width, connected


def generate_corridor_first_variants(
    building,
    boundary_polygon,
    entrance_point,
    min_corridor_width,
    learned_hints=None,
    placement_order=None,
    room_zones=None,
    adjacency_preferences=None,
    frontage_room=None,
    layout_pattern=None,
):
    try:
        boundary_poly = Polygon(boundary_polygon)
        if not boundary_poly.is_valid:
            boundary_poly = boundary_poly.buffer(0)
    except Exception:
        return []

    variants = []
    for strategy_name in STRATEGIES:
        b = copy.deepcopy(building)
        circulation_poly, spine_points, width, connected = _build_circulation_from_boundary(
            boundary_poly,
            entrance_point,
            strategy_name,
            min_corridor_width,
        )

        remaining = boundary_poly.difference(circulation_poly)
        regions = sorted(_get_polygons(remaining), key=lambda p: p.area, reverse=True)
        if not regions:
            continue

        ok = _allocate_rooms_in_regions(
            b,
            regions,
            entrance_point=entrance_point,
            learned_hints=learned_hints,
            placement_order=placement_order,
            room_zones=room_zones,
            adjacency_preferences=adjacency_preferences,
            frontage_room=frontage_room,
            layout_pattern=layout_pattern,
        )
        if not ok:
            continue

        corr_polys = _get_polygons(circulation_poly)
        if corr_polys:
            c_poly = max(corr_polys, key=lambda g: g.area)
            connects = []
            for room in b.rooms:
                if not room.polygon:
                    continue
                rp = Polygon(room.polygon)
                if rp.touches(c_poly) or rp.distance(c_poly) <= 0.05:
                    connects.append(room.name)

            corr = Corridor(
                name=f"Circulation_{strategy_name.replace('-', '_')}",
                polygon=[(round(x, 3), round(y, 3)) for x, y in c_poly.exterior.coords],
                width=round(width, 2),
                length=round(max(c_poly.area / max(width, 1e-6), 0.0), 2),
                connects=sorted(set(connects)),
                connectivity_to_exit=connected,
                spine_points=[(round(x, 3), round(y, 3)) for x, y in spine_points],
            )
            b.corridors = [corr]

        variants.append((b, strategy_name))

    return variants




