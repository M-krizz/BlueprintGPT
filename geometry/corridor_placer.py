import copy
import math
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union
from core.corridor import Corridor


# ── helpers ───────────────────────────────────────────────────────────────────

def _clone(building):
    return copy.deepcopy(building)


def _get_polygons(geom):
    if geom is None or geom.is_empty:
        return []
    if getattr(geom, "geom_type", "") == "Polygon":
        return [geom]
    if getattr(geom, "geom_type", "") == "MultiPolygon":
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if getattr(g, "geom_type", "") == "Polygon"]


def _carve_circulation(
    building,
    circulation_poly,
    name,
    width,
    spine_points,
    connectivity_to_exit=True,
):
    """
    Carves circulation space from room polygons and stores it as walkable space.
    """
    if circulation_poly.is_empty or circulation_poly.area < 0.1:
        return None

    intersected_rooms = []
    actual_corridor_parts = []

    for room in building.rooms:
        if not room.polygon:
            continue

        r_poly = Polygon(room.polygon)
        if not r_poly.is_valid:
            r_poly = r_poly.buffer(0)

        if r_poly.intersects(circulation_poly):
            intersection = r_poly.intersection(circulation_poly)
            if intersection.area > 0.05:
                intersected_rooms.append(room.name)
                actual_corridor_parts.append(intersection)

            new_region = r_poly.difference(circulation_poly)
            polygons = _get_polygons(new_region)
            if polygons:
                largest = max(polygons, key=lambda g: g.area)
                room.polygon = [(round(x, 3), round(y, 3)) for x, y in largest.exterior.coords]

    if not actual_corridor_parts:
        return None

    actual_corridor = unary_union(actual_corridor_parts)
    polygons = _get_polygons(actual_corridor)
    if not polygons:
        return None

    largest = max(polygons, key=lambda g: g.area)
    corridor_length = largest.length / 2.0
    corr = Corridor(
        name=name,
        polygon=[(round(x, 3), round(y, 3)) for x, y in largest.exterior.coords],
        width=round(width, 2),
        length=round(corridor_length, 2),
        connects=list(set(intersected_rooms)),
        connectivity_to_exit=bool(connectivity_to_exit),
        spine_points=[(round(x, 3), round(y, 3)) for x, y in spine_points],
    )
    building.add_corridor(corr)
    return corr


def _building_shape(building):
    """Returns the footprint of the building as a Shapely geometry."""
    polys = [Polygon(r.polygon) for r in building.rooms if r.polygon]
    if not polys:
        return Polygon()
    return unary_union(polys)


def _farthest_boundary_point(shape, origin):
    if shape.is_empty:
        return origin

    boundary = shape.boundary
    candidates = []
    if hasattr(boundary, "coords"):
        candidates.extend(list(boundary.coords))
    else:
        for geom in getattr(boundary, "geoms", []):
            candidates.extend(list(getattr(geom, "coords", [])))

    if not candidates:
        return origin

    farthest = max(candidates, key=lambda p: (p[0] - origin[0]) ** 2 + (p[1] - origin[1]) ** 2)
    return farthest


def _centroid_of_room_types(building, room_types):
    points = []
    for room in building.rooms:
        if room.room_type not in room_types or not room.polygon:
            continue
        poly = Polygon(room.polygon)
        c = poly.centroid
        points.append((c.x, c.y))

    if not points:
        return None

    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def _dedupe_points(points, tol=1e-6):
    cleaned = []
    for p in points:
        if not cleaned:
            cleaned.append(p)
            continue
        if abs(cleaned[-1][0] - p[0]) > tol or abs(cleaned[-1][1] - p[1]) > tol:
            cleaned.append(p)
    return cleaned


def _orthogonal_segment_points(p, q, shape):
    if abs(p[0] - q[0]) < 1e-6 or abs(p[1] - q[1]) < 1e-6:
        return [p, q]

    elbows = [(q[0], p[1]), (p[0], q[1])]
    best_points = [p, elbows[0], q]
    best_score = -1e9

    for elbow in elbows:
        candidate = LineString([p, elbow, q])
        inside_len = candidate.intersection(shape).length
        total_len = candidate.length
        score = inside_len - (total_len - inside_len) * 2.0
        if score > best_score:
            best_score = score
            best_points = [p, elbow, q]

    return best_points


def _orthogonal_spine(waypoints, shape):
    points = []
    for i in range(len(waypoints) - 1):
        segment_points = _orthogonal_segment_points(waypoints[i], waypoints[i + 1], shape)
        if points:
            points.extend(segment_points[1:])
        else:
            points.extend(segment_points)
    return _dedupe_points(points)


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


def _strategy_to_spine_points(building, shape, entrance_point, strategy_name):
    centroid = shape.centroid
    c = (centroid.x, centroid.y)
    entrance = entrance_point if entrance_point else c

    living = _centroid_of_room_types(building, {"LivingRoom"}) or c
    service = _centroid_of_room_types(building, {"Kitchen"})
    private = _centroid_of_room_types(building, {"Bedroom", "Bathroom", "WC"})
    farthest = _farthest_boundary_point(shape, private or c)

    if strategy_name == "compactness-first":
        waypoints = [entrance, living]
    elif strategy_name == "corridor-minimization":
        waypoints = [entrance, ((entrance[0] + living[0]) / 2.0, (entrance[1] + living[1]) / 2.0), living]
    elif strategy_name == "privacy-first":
        waypoints = [entrance, living, private or farthest, farthest]
    elif strategy_name == "adjacency-first":
        waypoints = [entrance, living]
        if service:
            waypoints.append(service)
        if private:
            waypoints.append(private)
    else:
        waypoints = [entrance, living, private or c]

    return _orthogonal_spine(_dedupe_points(waypoints), shape)


def _corridor_width_for_strategy(min_corridor_width, strategy_name):
    if strategy_name == "compactness-first":
        return max(min_corridor_width, 1.0)
    if strategy_name == "privacy-first":
        return max(min_corridor_width, 1.35)
    if strategy_name == "adjacency-first":
        return max(min_corridor_width, 1.25)
    if strategy_name == "corridor-minimization":
        return max(min_corridor_width, 0.95)
    return max(min_corridor_width, 1.2)


def _generate_variant(building, strategy_name, min_corridor_width, entrance_point=None):
    b = _clone(building)
    shape = _building_shape(b)
    if shape.is_empty:
        return b, strategy_name

    points = _strategy_to_spine_points(b, shape, entrance_point, strategy_name)
    if len(points) < 2:
        return b, strategy_name

    width = _corridor_width_for_strategy(min_corridor_width, strategy_name)

    corridor_poly = _orthogonal_strip_buffer(points, width)
    corridor_poly = corridor_poly.intersection(shape)
    connectivity_to_exit = True
    if entrance_point is not None:
        try:
            connectivity_to_exit = Point(entrance_point).distance(corridor_poly) <= (width * 0.75)
        except Exception:
            connectivity_to_exit = True

    _carve_circulation(
        b,
        corridor_poly,
        name=f"Circulation_{strategy_name.replace('-', '_')}",
        width=width,
        spine_points=points,
        connectivity_to_exit=connectivity_to_exit,
    )
    return b, strategy_name


# ── public entry point ────────────────────────────────────────────────────────

def generate_corridor_variants(building, circulation_factor, min_corridor_width,
                                boundary_polygon=None, entrance_point=None):
    """
    Generate circulation-space variants using buffered spine carving.
    """
    variants = []

    strategy_names = [
        "balanced",
        "compactness-first",
        "privacy-first",
        "adjacency-first",
        "corridor-minimization",
    ]

    for strategy_name in strategy_names:
        try:
            b, name = _generate_variant(
                building,
                strategy_name=strategy_name,
                min_corridor_width=min_corridor_width,
                entrance_point=entrance_point,
            )
            variants.append((b, name))
        except Exception as exc:
            print(f"[corridor_placer] {strategy_name} failed: {exc}")

    return variants
