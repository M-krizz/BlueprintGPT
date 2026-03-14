from shapely.geometry import Polygon, Point


# ── Defaults (used when KG is unavailable) ────────────────────────────────────
PUBLIC_TYPES = {"LivingRoom"}
SERVICE_TYPES = {"Kitchen"}
PRIVATE_TYPES = {"Bedroom", "Bathroom", "WC"}


def _zone_maps_from_kg():
    """Try to load richer zone mapping from the OntologyBridge."""
    try:
        from ontology.ontology_bridge import OntologyBridge
        bridge = OntologyBridge("ontology/regulatory.owl")
        full = bridge.get_zone_rules()
        pub = {rt for rt, z in full.items() if z == "public"}
        svc = {rt for rt, z in full.items() if z == "service"}
        prv = {rt for rt, z in full.items() if z == "private"}
        return pub or PUBLIC_TYPES, svc or SERVICE_TYPES, prv or PRIVATE_TYPES
    except Exception:
        return PUBLIC_TYPES, SERVICE_TYPES, PRIVATE_TYPES


def _room_centroid(room):
    if not room.polygon:
        return None
    poly = Polygon(room.polygon)
    c = poly.centroid
    return (c.x, c.y)


def assign_room_zones(building, entrance_point=None):
    """
    Assign each room to Public / Service / Private zone.
    If the type is ambiguous, fall back to distance from entrance.
    Uses KG-derived zone rules when available.
    """
    pub, svc, prv = _zone_maps_from_kg()

    if not building.rooms:
        return {}

    entrance = Point(entrance_point) if entrance_point else None
    centroids = []
    for room in building.rooms:
        c = _room_centroid(room)
        if c is not None:
            centroids.append((room, Point(c)))

    distances = [pt.distance(entrance) for _, pt in centroids] if entrance else []
    if distances:
        near_threshold = sorted(distances)[max(0, int(len(distances) * 0.33) - 1)]
        far_threshold = sorted(distances)[max(0, int(len(distances) * 0.67) - 1)]
    else:
        near_threshold = far_threshold = 0.0

    zone_map = {}
    for room in building.rooms:
        room_type = room.room_type
        if room_type in pub:
            zone_map[room.name] = "public"
            continue
        if room_type in svc:
            zone_map[room.name] = "service"
            continue
        if room_type in prv:
            zone_map[room.name] = "private"
            continue

        if entrance is None:
            zone_map[room.name] = "service"
            continue

        c = _room_centroid(room)
        if c is None:
            zone_map[room.name] = "service"
            continue

        d = Point(c).distance(entrance)
        if d <= near_threshold:
            zone_map[room.name] = "public"
        elif d >= far_threshold:
            zone_map[room.name] = "private"
        else:
            zone_map[room.name] = "service"

    return zone_map
