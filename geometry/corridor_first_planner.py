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


def _room_zone(room):
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


def _pack_rooms_into_region_group(room_group, region_group, entrance_point=None):
    if not room_group or not region_group:
        return

    total_room_area = sum(r.final_area for r in room_group)
    total_region_area = sum(r.area for r in region_group)
    if total_room_area <= 0 or total_region_area <= 0:
        return

    room_idx = 0
    for region_idx, region in enumerate(region_group):
        if room_idx >= len(room_group):
            break

        if region_idx == len(region_group) - 1:
            assigned = room_group[room_idx:]
            room_idx = len(room_group)
        else:
            target = total_room_area * (region.area / total_region_area)
            running = 0.0
            assigned = []
            while room_idx < len(room_group):
                room = room_group[room_idx]
                assigned.append(room)
                running += room.final_area
                room_idx += 1
                if running >= target * 0.9:
                    break

        items = [{"room": room, "weight": room.final_area} for room in assigned]
        recursive_pack(region, items, entrance_pt=entrance_point)


def _allocate_rooms_in_regions(building, regions, entrance_point=None):
    rooms = _rooms_sorted_for_packing(building)
    if not rooms or not regions:
        return False

    bands = _split_regions_into_bands(regions, entrance_point)

    public_rooms = [r for r in rooms if _room_zone(r) == "public"]
    service_rooms = [r for r in rooms if _room_zone(r) == "service"]
    private_rooms = [r for r in rooms if _room_zone(r) == "private"]

    used_region_ids = set()

    def _reserve_regions(region_list):
        selected = []
        for region in region_list:
            rid = id(region)
            if rid not in used_region_ids:
                selected.append(region)
                used_region_ids.add(rid)
        return selected

    _pack_rooms_into_region_group(public_rooms, _reserve_regions(bands["public"]), entrance_point=entrance_point)
    _pack_rooms_into_region_group(service_rooms, _reserve_regions(bands["service"]), entrance_point=entrance_point)

    remaining_regions = [region for region in regions if id(region) not in used_region_ids]
    private_region_pool = _reserve_regions(bands["private"]) + remaining_regions
    _pack_rooms_into_region_group(private_rooms, private_region_pool, entrance_point=entrance_point)

    # fallback: if any rooms still unassigned due to geometry degeneracy, pack them into largest region
    unassigned = [room for room in rooms if not room.polygon]
    if unassigned and regions:
        largest = max(regions, key=lambda r: r.area)
        items = [{"room": room, "weight": room.final_area} for room in unassigned]
        recursive_pack(largest, items, entrance_pt=entrance_point)

    return all(room.polygon for room in rooms)


def _build_circulation_from_boundary(boundary_poly, entrance_point, strategy_name, min_corridor_width):
    width = _corridor_width(min_corridor_width, strategy_name)
    points = _strategy_waypoints(boundary_poly, entrance_point, strategy_name)
    if len(points) < 2:
        return Polygon(), points, width, False

    circulation = _orthogonal_strip_buffer(points, width).intersection(boundary_poly)
    connected = True
    if entrance_point is not None and not circulation.is_empty:
        connected = Point(entrance_point).distance(circulation) <= width
    return circulation, points, width, connected


def generate_corridor_first_variants(building, boundary_polygon, entrance_point, min_corridor_width):
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

        ok = _allocate_rooms_in_regions(b, regions, entrance_point=entrance_point)
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
