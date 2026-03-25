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
        ("Kitchen", "DiningRoom", 0.95),
        ("LivingRoom", "DiningRoom", 0.8),
        ("LivingRoom", "Bedroom", 0.15),
        ("Bedroom", "Bathroom", 1.0),
        ("Bedroom", "WC", 0.8),
        ("LivingRoom", "Bathroom", -0.7),
        ("LivingRoom", "WC", -0.75),
        ("Kitchen", "Bathroom", -1.0),
        ("Kitchen", "WC", -1.0),
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
    typed_rooms = {}
    for room in building.rooms:
        typed_rooms.setdefault(room.room_type, []).append(room)

    present_types = sorted(room_type for room_type, rooms in typed_rooms.items() if rooms)
    if intent_edges is None:
        intent_edges = build_adjacency_intent(room_types=present_types or None)

    evaluated_edges = [
        (type_a, type_b, weight)
        for type_a, type_b, weight in intent_edges
        if typed_rooms.get(type_a) and typed_rooms.get(type_b)
    ]
    if not evaluated_edges:
        return 1.0, []

    total_weight = sum(abs(weight) for _, _, weight in evaluated_edges) or 1.0
    satisfied_weight = 0.0
    details = []

    for type_a, type_b, weight in evaluated_edges:
        touching = False
        for room_a in typed_rooms.get(type_a, []):
            for room_b in typed_rooms.get(type_b, []):
                if room_a is room_b:
                    continue
                if _touch_or_near(room_a, room_b):
                    touching = True
                    break
            if touching:
                break

        rule_satisfied = touching if weight >= 0 else not touching
        if rule_satisfied:
            satisfied_weight += abs(weight)
        details.append(
            {
                "pair": f"{type_a}<->{type_b}",
                "weight": abs(weight),
                "raw_weight": weight,
                "relation": "prefer" if weight >= 0 else "avoid",
                "touching": touching,
                "satisfied": rule_satisfied,
            }
        )

    return round(satisfied_weight / total_weight, 4), details
