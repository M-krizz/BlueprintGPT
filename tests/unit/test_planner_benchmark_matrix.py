from learned.planner import benchmark


def test_build_canonical_residential_matrix_generates_cross_product():
    cases = benchmark.build_canonical_residential_matrix(
        layout_types=["1BHK", "3BHK"],
        entrance_sides=["North", "West"],
        size_modes=["minimum", "recommended"],
    )

    assert len(cases) == 8
    first = cases[0]
    assert first["expected_room_counts"]["Kitchen"] == 1
    assert first["expected_room_counts"]["LivingRoom"] == 1
    assert all(case["resolution"]["boundary_size"][0] > 0 for case in cases)


def test_build_canonical_residential_matrix_resolves_entrance_side_points():
    cases = benchmark.build_canonical_residential_matrix(
        layout_types=["2BHK"],
        entrance_sides=["North", "South", "East", "West"],
        size_modes=["recommended"],
    )
    points = {case["entrance_side"]: case["resolution"]["entrance_point"] for case in cases}
    width, height = cases[0]["resolution"]["boundary_size"]

    assert points["North"] == [round(width / 2.0, 3), 0.0]
    assert points["South"] == [round(width / 2.0, 3), round(height, 3)]
    assert points["East"] == [round(width, 3), round(height / 2.0, 3)]
    assert points["West"] == [0.0, round(height / 2.0, 3)]


def test_evaluate_acceptance_accepts_strong_case():
    case = {"expected_room_counts": {"Bedroom": 3, "Bathroom": 2, "Kitchen": 1, "LivingRoom": 1}}
    result = {
        "report_status": "COMPLIANT",
        "generated_rooms": {"Bedroom": 3, "Bathroom": 2, "Kitchen": 1, "LivingRoom": 1},
        "room_coverage": {"missing": [], "extra": []},
        "metrics": {
            "fully_connected": True,
            "connectivity_to_exit": True,
            "adjacency_satisfaction": 0.7,
            "circulation_walkable_area": 12.0,
            "total_area": 110.0,
            "max_room_area_error": 0.15,
            "public_frontage_score": 0.7,
            "bedroom_privacy_score": 0.6,
            "kitchen_living_score": 1.0,
            "bathroom_access_score": 1.0,
            "architectural_reasonableness": 0.72,
        },
    }

    acceptance = benchmark.evaluate_acceptance(result, case)

    assert acceptance["accepted"] is True
    assert acceptance["failed_checks"] == []


def test_evaluate_acceptance_rejects_corridor_heavy_case():
    case = {"expected_room_counts": {"Bedroom": 4, "Bathroom": 2, "Kitchen": 1, "LivingRoom": 1}}
    result = {
        "report_status": "COMPLIANT",
        "generated_rooms": {"Bedroom": 4, "Bathroom": 2, "Kitchen": 1, "LivingRoom": 1},
        "room_coverage": {"missing": [], "extra": []},
        "metrics": {
            "fully_connected": True,
            "connectivity_to_exit": True,
            "adjacency_satisfaction": 0.65,
            "circulation_walkable_area": 28.0,
            "total_area": 120.0,
            "max_room_area_error": 0.2,
            "public_frontage_score": 0.5,
            "bedroom_privacy_score": 0.5,
            "kitchen_living_score": 1.0,
            "bathroom_access_score": 1.0,
            "architectural_reasonableness": 0.62,
        },
    }

    acceptance = benchmark.evaluate_acceptance(result, case)

    assert acceptance["accepted"] is False
    assert "bounded_circulation" in acceptance["failed_checks"]


def test_summarize_benchmark_reports_acceptance_counts():
    payload = benchmark.summarize_benchmark(
        [
            {
                "comparison": {"winner": "planner", "design_score_delta": 0.2, "adjacency_delta": 0.1, "alignment_delta": 0.1},
                "algorithmic": {"acceptance": {"accepted": False}},
                "planner": {"acceptance": {"accepted": True}},
            },
            {
                "comparison": {"winner": "algorithmic", "design_score_delta": -0.1, "adjacency_delta": -0.2, "alignment_delta": 0.0},
                "algorithmic": {"acceptance": {"accepted": True}},
                "planner": {"acceptance": {"accepted": False}},
            },
        ]
    )

    assert payload["acceptance_counts"]["algorithmic"]["accepted"] == 1
    assert payload["acceptance_counts"]["planner"]["accepted"] == 1
    assert payload["winner_counts"]["planner"] == 1
    assert payload["winner_counts"]["algorithmic"] == 1
