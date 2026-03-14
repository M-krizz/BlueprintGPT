import pytest
from generator.ranking import rank_layout_variants, _score_variant

def test_score_variant_perfect():
    variant = {
        "metrics": {
            "fully_connected": True,
            "travel_distance_compliant": True,
            "circulation_walkable_area": 20.0,
            "total_area": 100.0,
            "adjacency_satisfaction": 1.0,
            "max_allowed_travel_distance": 20.0,
            "max_travel_distance": 10.0,
            "alignment_score": 1.0
        }
    }
    score, breakdown = _score_variant(variant)
    assert breakdown["hard_compliance"] == 1.0
    assert score > 0.8  # Perfect metrics should score near highest

def test_rank_layout_variants():
    v1 = {
        "metrics": {"fully_connected": True, "travel_distance_compliant": True}
    }
    v2 = {
        "metrics": {"fully_connected": False, "travel_distance_compliant": False}
    }
    
    ranked, recommended_idx = rank_layout_variants([v2, v1])
    
    assert ranked[0]["metrics"]["fully_connected"] is True
    assert ranked[0]["recommended"] is True
    assert recommended_idx == 0
