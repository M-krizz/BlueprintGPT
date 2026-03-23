from __future__ import annotations

from nl_interface.program_planner import (
    build_room_program,
    build_semantic_spec,
    build_zoning_plan,
    enrich_spec_with_planning,
)


def test_canonical_three_bhk_program_expansion_and_roles():
    spec = {
        "layout_type": "3BHK",
        "rooms": [
            {"type": "Bedroom", "count": 3},
            {"type": "Bathroom", "count": 2},
            {"type": "Kitchen", "count": 1},
            {"type": "LivingRoom", "count": 1},
        ],
        "plot_type": "Custom",
        "entrance_side": "North",
        "preferences": {"adjacency": [], "privacy": {}, "minimize_corridor": False},
    }

    semantic = build_semantic_spec(spec, resolution={"boundary_size": (12.4, 8.9)})
    program = build_room_program(semantic)

    assert program.canonical is True
    assert program.layout_type == "3BHK"
    assert program.required_counts == {
        "Bedroom": 3,
        "LivingRoom": 1,
        "Kitchen": 1,
        "Bathroom": 2,
    }

    roles = {room.name: room.semantic_role for room in program.rooms}
    assert roles["Bedroom_1"] == "master_bedroom"
    assert roles["Bedroom_2"] == "secondary_bedroom"
    assert roles["Bathroom_1"] == "attached_bathroom"
    assert roles["Bathroom_2"] == "common_bathroom"
    assert roles["LivingRoom_1"] == "public_anchor"
    assert roles["Kitchen_1"] == "service_anchor"


def test_zoning_plan_prioritizes_public_frontage_and_named_adjacency():
    spec = {
        "layout_type": "3BHK",
        "rooms": [
            {"type": "Bedroom", "count": 3},
            {"type": "Bathroom", "count": 2},
            {"type": "Kitchen", "count": 1},
            {"type": "LivingRoom", "count": 1},
        ],
        "plot_type": "Custom",
        "entrance_side": "North",
        "preferences": {"adjacency": [], "privacy": {}, "minimize_corridor": False},
    }

    semantic = build_semantic_spec(spec, resolution={"boundary_size": (12.4, 8.9)})
    program = build_room_program(semantic)
    zoning = build_zoning_plan(program, semantic)

    assert zoning.layout_pattern == "zonal_split"
    assert zoning.frontage_room == "LivingRoom_1"
    assert zoning.zone_map["LivingRoom_1"] == "public"
    assert zoning.zone_map["Kitchen_1"] == "service"
    assert zoning.zone_map["Bedroom_1"] == "private"
    assert {"a": "Kitchen_1", "b": "LivingRoom_1", "type": "prefer", "score": 1.0} in zoning.named_adjacency
    assert any(item["a"] == "Bathroom_1" and item["b"] == "Bedroom_1" for item in zoning.named_adjacency)
    assert zoning.spatial_hints["LivingRoom_1"][1] < zoning.spatial_hints["Bedroom_1"][1]
    assert zoning.room_order.index("Bedroom_1") < zoning.room_order.index("Bathroom_1")
    assert zoning.room_order.index("Bedroom_2") < zoning.room_order.index("Bathroom_2")


def test_enrich_spec_with_planning_attaches_structured_contracts():
    spec = {
        "layout_type": "2BHK",
        "rooms": [
            {"type": "Bedroom", "count": 2},
            {"type": "Bathroom", "count": 2},
            {"type": "Kitchen", "count": 1},
            {"type": "LivingRoom", "count": 1},
        ],
        "plot_type": "Custom",
        "entrance_side": "East",
        "preferences": {"adjacency": [], "privacy": {}, "minimize_corridor": False},
    }

    enriched = enrich_spec_with_planning(spec, resolution={"boundary_size": (10.8, 8.2)}, user_prompt="Design a 2BHK")

    assert enriched["semantic_spec"]["layout_type"] == "2BHK"
    assert enriched["room_program"]["layout_type"] == "2BHK"
    assert enriched["zoning_plan"]["layout_pattern"] == "public_front_private_rear"
    assert enriched["zoning_plan"]["room_order"]



def test_custom_residential_program_preserves_explicit_extra_rooms():
    spec = {
        "rooms": [
            {"type": "Bedroom", "count": 3},
            {"type": "Bathroom", "count": 2},
            {"type": "Kitchen", "count": 1},
            {"type": "LivingRoom", "count": 1},
            {"type": "DiningRoom", "count": 1},
            {"type": "Garage", "count": 1},
            {"type": "Store", "count": 1},
        ],
        "plot_type": "Custom",
        "entrance_side": "West",
        "preferences": {"adjacency": [], "privacy": {}, "minimize_corridor": False},
    }

    semantic = build_semantic_spec(
        spec,
        resolution={"boundary_size": (16.0, 11.0)},
        user_prompt="Need a custom house with three bedrooms, dining room, garage, and store",
    )
    program = build_room_program(semantic)

    assert semantic.layout_type == "3BHK"
    assert semantic.shorthand is None
    assert program.canonical is False
    assert program.supported_scope == "custom_residential"
    assert program.required_counts["DiningRoom"] == 1
    assert program.required_counts["Garage"] == 1
    assert program.required_counts["Store"] == 1
    roles = {room.name: room.semantic_role for room in program.rooms}
    assert roles["DiningRoom_1"] == "public_support"
    assert roles["Garage_1"] == "service_support"
    assert roles["Store_1"] == "service_support"



def test_canonical_shorthand_merges_explicit_extra_rooms():
    spec = {
        "layout_type": "3BHK",
        "rooms": [
            {"type": "Bedroom", "count": 3},
            {"type": "Bathroom", "count": 2},
            {"type": "Kitchen", "count": 1},
            {"type": "LivingRoom", "count": 1},
            {"type": "DiningRoom", "count": 1},
            {"type": "Garage", "count": 1},
        ],
        "plot_type": "Custom",
        "entrance_side": "North",
        "preferences": {"adjacency": [], "privacy": {}, "minimize_corridor": False},
    }

    semantic = build_semantic_spec(
        spec,
        resolution={"boundary_size": (14.0, 10.0)},
        user_prompt="Design a 3BHK with dining room and garage",
    )
    program = build_room_program(semantic)

    assert semantic.shorthand == "3BHK"
    assert program.canonical is True
    assert program.supported_scope == "canonical_plus_custom"
    assert program.required_counts["DiningRoom"] == 1
    assert program.required_counts["Garage"] == 1



def test_enrich_spec_with_planning_preserves_edit_preferences():
    base_spec = {
        "layout_type": "3BHK",
        "rooms": [
            {"type": "Bedroom", "count": 3},
            {"type": "Bathroom", "count": 2},
            {"type": "Kitchen", "count": 1},
            {"type": "LivingRoom", "count": 1},
        ],
        "plot_type": "Custom",
        "entrance_side": "North",
        "preferences": {"adjacency": [], "privacy": {}, "minimize_corridor": False},
    }
    baseline = enrich_spec_with_planning(base_spec, resolution={"boundary_size": (12.4, 8.9)}, user_prompt="Design a 3BHK")
    adjusted = enrich_spec_with_planning(
        {
            **base_spec,
            "room_size_preferences": {"Kitchen_1": "larger"},
            "room_position_preferences": {"Bedroom_2": {"direction": "east", "amount": "1m"}},
            "room_swaps": [("Bedroom_1", "Bedroom_3")],
        },
        resolution={"boundary_size": (12.4, 8.9)},
        user_prompt="Move bedroom 2 east and make kitchen larger",
    )

    assert adjusted["room_size_preferences"]["Kitchen_1"] == "larger"
    assert adjusted["room_position_preferences"]["Bedroom_2"]["direction"] == "east"
    assert adjusted["room_swaps"] == [("Bedroom_1", "Bedroom_3")]
    assert (
        adjusted["zoning_plan"]["size_priors"]["Kitchen_1"]["ideal_area_sqm"]
        > baseline["zoning_plan"]["size_priors"]["Kitchen_1"]["ideal_area_sqm"]
    )
    assert (
        adjusted["zoning_plan"]["spatial_hints"]["Bedroom_2"][0]
        > baseline["zoning_plan"]["spatial_hints"]["Bedroom_2"][0]
    )
