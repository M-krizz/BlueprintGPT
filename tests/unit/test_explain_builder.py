from explain.context_builder import build_evidence


def test_build_evidence_minimal():
    variant = {
        "strategy_name": "s1",
        "source": "algorithmic",
        "metrics": {
            "max_travel_distance": 10,
            "max_allowed_travel_distance": 12,
            "travel_distance_compliant": True,
            "adjacency_satisfaction": 0.8,
            "alignment_score": 0.5,
            "door_path_travel_distance": 9,
        },
        "ranking": {"breakdown": {"compactness": 0.7}},
    }
    report = {
        "status": "COMPLIANT",
        "violations": [],
        "circulation_space": {"corridor_width": 1.2, "walkable_area": 5.0},
    }

    evidence = build_evidence(variant, report, variant_id="v1")

    assert evidence["selected_variant_id"] == "v1"
    assert evidence["hard_compliance"]["status"] == "COMPLIANT"
    assert evidence["metrics"]["max_travel_distance"] == 10
    assert evidence["metrics"]["compactness"] == 0.7