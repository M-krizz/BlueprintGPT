from learned.planner.evaluate_rollout import evaluate_benchmark


def test_evaluate_benchmark_holds_when_planner_is_worse():
    result = evaluate_benchmark(
        {
            "summary": {
                "case_count": 4,
                "completed_case_count": 4,
                "winner_counts": {"planner": 1, "algorithmic": 3, "tie": 0},
                "planner_avg_score_delta": -0.018,
                "planner_avg_adjacency_delta": 0.0,
                "planner_avg_alignment_delta": -0.0718,
            },
            "cases": [
                {
                    "planner": {
                        "report_status": "COMPLIANT",
                        "room_coverage": {"missing": [], "extra": []},
                        "metrics": {"fully_connected": True},
                    }
                }
            ],
        }
    )

    assert result["ready_for_rollout"] is False
    assert "hold algorithmic default" in result["recommendation"]


def test_evaluate_benchmark_allows_rollout_when_all_checks_pass():
    result = evaluate_benchmark(
        {
            "summary": {
                "case_count": 2,
                "completed_case_count": 2,
                "winner_counts": {"planner": 2, "algorithmic": 0, "tie": 0},
                "planner_avg_score_delta": 0.02,
                "planner_avg_adjacency_delta": 0.05,
                "planner_avg_alignment_delta": 0.01,
            },
            "cases": [
                {
                    "planner": {
                        "report_status": "COMPLIANT",
                        "room_coverage": {"missing": [], "extra": []},
                        "metrics": {"fully_connected": True},
                    }
                },
                {
                    "planner": {
                        "report_status": "COMPLIANT",
                        "room_coverage": {"missing": [], "extra": []},
                        "metrics": {"fully_connected": True},
                    }
                },
            ],
        }
    )

    assert result["ready_for_rollout"] is True
    assert result["recommendation"] == "enable planner_if_available"
