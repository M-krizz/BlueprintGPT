from learned.planner.verification import verify_planner_output



def test_verify_planner_output_fails_hard_on_severe_overlap():
    rooms = [
        {
            "name": "LivingRoom_1",
            "type": "LivingRoom",
            "polygon": [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
            "area": 16.0,
            "centroid": (2.0, 2.0),
        },
        {
            "name": "Kitchen_1",
            "type": "Kitchen",
            "polygon": [(2.0, 1.0), (6.0, 1.0), (6.0, 5.0), (2.0, 5.0)],
            "area": 16.0,
            "centroid": (4.0, 3.0),
        },
    ]

    result = verify_planner_output(
        rooms,
        boundary_polygon=[(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)],
        spec={
            "rooms": [
                {"name": "LivingRoom_1", "type": "LivingRoom", "area": 12.0},
                {"name": "Kitchen_1", "type": "Kitchen", "area": 8.0},
            ]
        },
        planner_output={"adjacency_preferences": [{"a": "LivingRoom_1", "b": "Kitchen_1", "score": 1.0}]},
    )

    assert result.passed is False
    assert result.metrics["overlap_free_ratio"] < 0.9
    assert "overlap" in result.metrics["hard_failures"]
    assert any("Overlap detected" in issue for issue in result.issues)
