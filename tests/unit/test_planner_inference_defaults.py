from learned.planner import inference


def test_default_adjacency_preferences_expand_types_to_room_names():
    spec = {
        "rooms": [
            {"name": "LivingRoom_1", "type": "LivingRoom"},
            {"name": "Kitchen_1", "type": "Kitchen"},
            {"name": "Bedroom_1", "type": "Bedroom"},
            {"name": "Bathroom_1", "type": "Bathroom"},
        ]
    }

    prefs = inference._default_adjacency_preferences(spec)

    assert {tuple(sorted((pref["a"], pref["b"]))) for pref in prefs} >= {
        ("Kitchen_1", "LivingRoom_1"),
        ("Bathroom_1", "Bedroom_1"),
    }
    assert all("_" in pref["a"] and "_" in pref["b"] for pref in prefs)


def test_default_adjacency_preferences_ignore_non_contact_relations():
    spec = {
        "rooms": [
            {"name": "LivingRoom_1", "type": "LivingRoom"},
            {"name": "Bedroom_1", "type": "Bedroom"},
            {"name": "Bedroom_2", "type": "Bedroom"},
        ],
        "adjacency": [
            {"source": "LivingRoom", "target": "Bedroom", "relation": "buffer_zone"},
            {"source": "Bedroom", "target": "Bedroom", "relation": "separate"},
            {"source": "LivingRoom", "target": "Bedroom", "relation": "near_to"},
        ],
    }

    prefs = inference._default_adjacency_preferences(spec)

    assert {tuple(sorted((pref["a"], pref["b"]))) for pref in prefs} == {
        ("Bedroom_1", "LivingRoom_1"),
        ("Bedroom_2", "LivingRoom_1"),
    }

def test_default_adjacency_preferences_accept_a_b_type_shape():
    spec = {
        "rooms": [
            {"name": "LivingRoom_1", "type": "LivingRoom"},
            {"name": "Kitchen_1", "type": "Kitchen"},
            {"name": "Bedroom_1", "type": "Bedroom"},
            {"name": "Bathroom_1", "type": "Bathroom"},
        ],
        "adjacency": [
            {"a": "Kitchen", "b": "LivingRoom", "type": "prefer"},
            {"a": "Bathroom", "b": "Bedroom", "type": "prefer"},
        ],
    }

    prefs = inference._default_adjacency_preferences(spec)

    assert {tuple(sorted((pref["a"], pref["b"]))) for pref in prefs} == {
        ("Kitchen_1", "LivingRoom_1"),
        ("Bathroom_1", "Bedroom_1"),
    }


def test_build_room_order_prioritizes_public_service_then_bedrooms_before_bathrooms():
    spatial_hints = {
        "Bathroom_2": [0.2, 0.4],
        "Bedroom_1": [0.7, 0.8],
        "Bedroom_2": [0.3, 0.8],
        "Bathroom_1": [0.75, 0.6],
        "Kitchen_1": [0.2, 0.25],
        "Bedroom_3": [0.5, 0.88],
        "LivingRoom_1": [0.5, 0.22],
    }
    room_zones = {
        "LivingRoom_1": "public",
        "Kitchen_1": "service",
        "Bedroom_1": "private",
        "Bedroom_2": "private",
        "Bedroom_3": "private",
        "Bathroom_1": "private",
        "Bathroom_2": "private",
    }

    order = inference._build_room_order(spatial_hints, "North", room_zones)

    assert order[0] == "LivingRoom_1"
    assert order[1] == "Kitchen_1"
    assert order.index("Bedroom_1") < order.index("Bathroom_1")
    assert order.index("Bedroom_2") < order.index("Bathroom_2")
