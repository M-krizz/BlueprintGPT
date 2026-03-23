from __future__ import annotations

from api import server


def test_derive_resolution_from_spec_uses_total_area(monkeypatch):
    monkeypatch.setattr(
        server,
        "_layout_type_from_spec",
        lambda spec: "3BHK",
    )

    spec = {
        "layout_type": "3BHK",
        "rooms": [
            {"type": "Bedroom", "count": 3},
            {"type": "Bathroom", "count": 2},
            {"type": "Kitchen", "count": 1},
            {"type": "LivingRoom", "count": 1},
        ],
    }
    resolution = {"total_area": 3000.0, "area_unit": "sq.m"}

    derived = server._derive_resolution_from_spec(spec, resolution, [5.0, 0.0])

    width, height = derived["boundary_size"]
    assert round(width * height, 1) >= 2999.0
    assert derived["entrance_point"] == (5.0, 0.0)
    assert derived["boundary_polygon"][2] == [width, height]


def test_derive_resolution_from_spec_uses_entrance_side_when_point_missing(monkeypatch):
    monkeypatch.setattr(server, "_layout_type_from_spec", lambda spec: "3BHK")

    derived = server._derive_resolution_from_spec(
        {
            "layout_type": "3BHK",
            "entrance_side": "North",
            "auto_dimensions": {"width_m": 12.4, "height_m": 8.9},
            "rooms": [{"type": "Bedroom", "count": 3}],
        },
        resolution={},
        entrance_point=None,
    )

    assert derived["boundary_size"] == (12.4, 8.9)
    assert derived["entrance_point"] == (6.2, 0.0)


def test_summarize_inferred_rules_mentions_area_and_adjacency():
    summary = server._summarize_inferred_rules(
        {
            "layout_type": "2BHK",
            "entrance_side": "North",
            "adjacency": [{"source": "Kitchen", "target": "LivingRoom", "relation": "near_to"}],
            "constraint_metadata": {"min_total_area_sqm": 55, "recommended_area_sqm": 75},
        },
        {"total_area": 120.0},
    )

    assert "2BHK" in summary
    assert "120.0 sq.m" in summary
    assert "Kitchen should stay close to LivingRoom" in summary


def test_build_design_conversation_reply_is_conversational():
    reply = server._build_design_conversation_reply(
        design_data={
            "report_status": "COMPLIANT",
            "backend_target": "algorithmic",
            "generated_rooms": {"Bedroom": 3, "Bathroom": 2, "Kitchen": 1, "LivingRoom": 1},
            "metrics": {
                "fully_connected": True,
                "adjacency_satisfaction": 0.31,
                "alignment_score": 0.58,
                "circulation_walkable_area": 20.0,
                "total_area": 110.0,
            },
        },
        spec={
            "layout_type": "3BHK",
            "entrance_side": "North",
            "adjacency": [{"source": "Kitchen", "target": "LivingRoom", "relation": "near_to"}],
        },
        resolution={"boundary_size": (12.4, 8.9)},
        design_count=1,
    )

    assert "I generated 1 layout option" in reply
    assert "corridor is visually dominating" in reply
    assert "Inferred Rules" in reply
    assert "Only one design was generated" not in reply




def test_derive_resolution_from_spec_prefers_explicit_boundary_size_over_auto_dimensions(monkeypatch):
    monkeypatch.setattr(server, "_layout_type_from_spec", lambda spec: "4BHK")

    derived = server._derive_resolution_from_spec(
        {
            "layout_type": "4BHK",
            "entrance_side": "North",
            "auto_dimensions": {"width_m": 14.5, "height_m": 10.4},
            "rooms": [{"type": "Bedroom", "count": 4}],
        },
        resolution={"boundary_size": (100.0, 150.0)},
        entrance_point=None,
    )

    assert derived["boundary_size"] == (100.0, 150.0)
    assert derived["entrance_point"] == (50.0, 0.0)



def test_derive_resolution_from_spec_treats_plot_size_as_site_envelope(monkeypatch):
    monkeypatch.setattr(server, "_layout_type_from_spec", lambda spec: "4BHK")

    derived = server._derive_resolution_from_spec(
        {
            "layout_type": "4BHK",
            "entrance_side": "North",
            "auto_dimensions": {"width_m": 14.5, "height_m": 10.4},
            "rooms": [{"type": "Bedroom", "count": 4}],
        },
        resolution={"boundary_size": (100.0, 150.0), "boundary_role": "site", "boundary_source": "nl_prompt"},
        entrance_point=None,
    )

    assert derived["site_boundary_size"] == (100.0, 150.0)
    assert derived["boundary_size"] == (14.5, 10.4)
    assert derived["entrance_point"] == (7.25, 0.0)
