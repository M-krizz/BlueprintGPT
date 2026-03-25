import math

from shapely.geometry import Polygon


def room_centroid(room):
    polygon = getattr(room, "polygon", None)
    if not polygon:
        return None
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    if not xs or not ys:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _touch_or_near(room_a, room_b, tolerance=0.25):
    polygon_a = getattr(room_a, "polygon", None)
    polygon_b = getattr(room_b, "polygon", None)
    if not polygon_a or not polygon_b:
        return False
    try:
        pa = Polygon(polygon_a)
        pb = Polygon(polygon_b)
    except Exception:
        return False
    if pa.touches(pb):
        return True
    return pa.distance(pb) <= tolerance


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
    kitchens = [room for room in building.rooms if room.room_type == "Kitchen"]
    bathrooms = [room for room in building.rooms if room.room_type in {"Bathroom", "WC"}]

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
    total_room_area = sum(float(getattr(room, "final_area", 0.0) or 0.0) for room in building.rooms)

    living_balance_score = 0.75
    if living_rooms and total_room_area > 0.0:
        living_area = sum(float(getattr(room, "final_area", 0.0) or 0.0) for room in living_rooms)
        living_ratio = living_area / max(total_room_area, 1e-6)
        if 0.16 <= living_ratio <= 0.24:
            living_balance_score = 1.0
        elif living_ratio < 0.16:
            living_balance_score = max(0.0, living_ratio / 0.16)
        else:
            living_balance_score = max(0.0, 1.0 - min(1.0, (living_ratio - 0.24) / 0.16))

    bathroom_public_exposure_score = 1.0
    if bathrooms and public_rooms:
        direct_exposure_hits = 0
        for bathroom in bathrooms:
            for public_room in public_rooms:
                if _touch_or_near(bathroom, public_room):
                    direct_exposure_hits += 1
                    break
        if direct_exposure_hits:
            bathroom_public_exposure_score = max(0.0, 1.0 - (direct_exposure_hits / max(len(bathrooms), 1)) * 0.8)

    master_suite_score = 0.6
    master_bedroom = next((room for room in building.rooms if room.name == "Bedroom_1"), None)
    attached_bathroom = next((room for room in building.rooms if room.name == "Bathroom_1"), None)
    if master_bedroom and attached_bathroom:
        master_suite_score = 1.0 if _touch_or_near(master_bedroom, attached_bathroom) else 0.25

    service_cluster_score = 0.7
    if kitchens and bathrooms:
        service_hits = 0
        for bathroom in bathrooms:
            if any(_touch_or_near(bathroom, kitchen, tolerance=0.35) for kitchen in kitchens):
                service_hits += 1
        service_cluster_score = min(1.0, 0.45 + 0.25 * service_hits)

    architectural_reasonableness = round(
        0.35 * public_frontage_score
        + 0.20 * bedroom_privacy_score
        + 0.12 * kitchen_living_score
        + 0.10 * bathroom_access_score
        + 0.08 * living_balance_score
        + 0.07 * bathroom_public_exposure_score
        + 0.05 * master_suite_score
        + 0.03 * service_cluster_score,
        4,
    )
    return {
        "public_frontage_score": round(public_frontage_score, 4),
        "bedroom_privacy_score": round(bedroom_privacy_score, 4),
        "kitchen_living_score": round(kitchen_living_score, 4),
        "bathroom_access_score": round(bathroom_access_score, 4),
        "living_balance_score": round(living_balance_score, 4),
        "bathroom_public_exposure_score": round(bathroom_public_exposure_score, 4),
        "master_suite_score": round(master_suite_score, 4),
        "service_cluster_score": round(service_cluster_score, 4),
        "architectural_reasonableness": architectural_reasonableness,
    }
