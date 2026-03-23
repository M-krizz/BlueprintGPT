from nl_interface.runner import _design_gate


def test_design_gate_rejects_large_room_area_drift():
    ok, score, reasons = _design_gate(
        {
            "source": "algorithmic",
            "metrics": {
                "fully_connected": True,
                "travel_distance_compliant": True,
                "max_travel_distance": 10.0,
                "max_allowed_travel_distance": 20.0,
                "adjacency_satisfaction": 0.5,
                "alignment_score": 0.8,
                "corridor_width": 1.2,
                "max_room_area_error": 1.1,
            },
        }
    )

    assert ok is False
    assert "Room area allocation drift too high" in reasons
    assert score < 1.0


def test_design_gate_rejects_excessive_circulation_area():
    ok, score, reasons = _design_gate(
        {
            "source": "algorithmic",
            "metrics": {
                "fully_connected": True,
                "travel_distance_compliant": True,
                "max_travel_distance": 10.0,
                "max_allowed_travel_distance": 20.0,
                "adjacency_satisfaction": 0.55,
                "alignment_score": 0.7,
                "corridor_width": 1.2,
                "circulation_walkable_area": 22.0,
                "total_area": 110.0,
                "max_room_area_error": 0.1,
            },
        }
    )

    assert ok is False
    assert "Circulation area is too high" in reasons
    assert score < 1.0
