from shapely.geometry import Polygon


def build_adjacency_intent(room_types=None, use_kg=True):
    """
    Weighted adjacency intent graph (higher weight = stronger preference).

    If *use_kg* is True, queries the OntologyBridge for KG-derived intents;
    falls back to the static graph otherwise.
    """
    if use_kg:
        try:
            from ontology.ontology_bridge import OntologyBridge
            bridge = OntologyBridge("ontology/regulatory.owl")
            intents = bridge.get_adjacency_intents(room_types)
            if intents:
                return intents
        except Exception:
            pass

    # Static fallback
    return [
        ("Kitchen", "LivingRoom", 1.0),
        ("LivingRoom", "Bedroom", 0.5),
        ("Bedroom", "Bathroom", 0.8),
        ("Bedroom", "WC", 0.6),
    ]


def _touch_or_near(room_a, room_b, tolerance=0.25):
    if not room_a.polygon or not room_b.polygon:
        return False
    pa = Polygon(room_a.polygon)
    pb = Polygon(room_b.polygon)
    if pa.touches(pb):
        return True
    return pa.distance(pb) <= tolerance


def adjacency_satisfaction_score(building, intent_edges=None):
    intent_edges = intent_edges or build_adjacency_intent()
    typed_rooms = {}
    for room in building.rooms:
        typed_rooms.setdefault(room.room_type, []).append(room)

    total_weight = sum(weight for _, _, weight in intent_edges) or 1.0
    satisfied_weight = 0.0
    details = []

    for type_a, type_b, weight in intent_edges:
        sat = False
        for room_a in typed_rooms.get(type_a, []):
            for room_b in typed_rooms.get(type_b, []):
                if room_a is room_b:
                    continue
                if _touch_or_near(room_a, room_b):
                    sat = True
                    break
            if sat:
                break

        if sat:
            satisfied_weight += weight
        details.append(
            {
                "pair": f"{type_a}<->{type_b}",
                "weight": weight,
                "satisfied": sat,
            }
        )

    return round(satisfied_weight / total_weight, 4), details
