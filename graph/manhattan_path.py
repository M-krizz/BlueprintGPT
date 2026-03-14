from shapely.geometry import Polygon, Point


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def room_centroid(room):
    xs = [p[0] for p in room.polygon]
    ys = [p[1] for p in room.polygon]
    return (sum(xs)/len(xs), sum(ys)/len(ys))


def max_travel_distance(building):
    exit_point = building.exit.segment[0]
    corridors = [c for c in getattr(building, "corridors", []) if getattr(c, "polygon", None)]

    max_distance = 0

    for room in building.rooms:
        if room.polygon is None:
            continue
        centroid = room_centroid(room)
        d = manhattan_distance(centroid, exit_point)

        if corridors:
            try:
                room_pt = Point(centroid)
                exit_pt = Point(exit_point)
                best = d
                for corridor in corridors:
                    c_poly = Polygon(corridor.polygon)
                    room_access = room_pt.distance(c_poly)
                    exit_access = exit_pt.distance(c_poly)
                    candidate = room_access + exit_access + max(corridor.length, 0.0)
                    best = min(best, candidate)
                d = best
            except Exception:
                pass

        max_distance = max(max_distance, d)

    return round(max_distance, 2)