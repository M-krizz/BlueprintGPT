import math


def room_centroid(room):
    polygon = getattr(room, "polygon", None)
    if not polygon:
        return None
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    if not xs or not ys:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def composition_quality(building, entrance_point, zone_map, adjacency_details):
    room_centroids = {}
    for room in getattr(building, "rooms", []):
        centroid = room_centroid(room)
        if centroid is not None:
            room_centroids[room.name] = centroid

    if room_centroids:
        xs = [point[0] for point in room_centroids.values()]
        ys = [point[1] for point in room_centroids.values()]
        diagonal = max(math.hypot(max(xs) - min(xs), max(ys) - min(ys)), 1e-6)
    else:
        diagonal = 1.0

    public_rooms = [room for room in building.rooms if zone_map.get(room.name) == "public"]
    private_rooms = [room for room in building.rooms if zone_map.get(room.name) == "private"]
    living_rooms = [room for room in public_rooms if room.room_type in {"LivingRoom", "DrawingRoom"}]

    public_frontage_score = 0.5
    bedroom_privacy_score = 0.5
    if entrance_point and room_centroids:
        distance_map = {
            room.name: math.hypot(
                room_centroids[room.name][0] - entrance_point[0],
                room_centroids[room.name][1] - entrance_point[1],
            )
            for room in building.rooms
            if room.name in room_centroids
        }
        if living_rooms:
            nearest_living = min(distance_map.get(room.name, diagonal) for room in living_rooms)
            public_frontage_score = max(0.0, min(1.0, 1.0 - nearest_living / diagonal))
        if public_rooms and private_rooms:
            public_mean = sum(distance_map.get(room.name, diagonal) for room in public_rooms) / max(len(public_rooms), 1)
            private_mean = sum(distance_map.get(room.name, diagonal) for room in private_rooms) / max(len(private_rooms), 1)
            bedroom_privacy_score = max(0.0, min(1.0, 0.5 + ((private_mean - public_mean) / diagonal)))

    adjacency_pairs = {item.get("pair"): bool(item.get("satisfied")) for item in adjacency_details or []}
    kitchen_living_score = 1.0 if adjacency_pairs.get("Kitchen<->LivingRoom") else 0.0
    bathroom_access_score = 1.0 if adjacency_pairs.get("Bedroom<->Bathroom") else 0.0

    architectural_reasonableness = round(
        0.35 * public_frontage_score
        + 0.30 * bedroom_privacy_score
        + 0.20 * kitchen_living_score
        + 0.15 * bathroom_access_score,
        4,
    )
    return {
        "public_frontage_score": round(public_frontage_score, 4),
        "bedroom_privacy_score": round(bedroom_privacy_score, 4),
        "kitchen_living_score": round(kitchen_living_score, 4),
        "bathroom_access_score": round(bathroom_access_score, 4),
        "architectural_reasonableness": architectural_reasonableness,
    }
