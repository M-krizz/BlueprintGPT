import json

from explain.context_builder import build_evidence
from explain.llm_explainer import explain
from explain.validator import validate_explanation


def _sample_evidence(status="COMPLIANT"):
    variant = {
        "strategy_name": "s1",
        "source": "algorithmic",
        "metrics": {
            "max_travel_distance": 10.0,
            "max_allowed_travel_distance": 12.0,
            "travel_distance_compliant": True,
            "adjacency_satisfaction": 0.42,
            "circulation_walkable_area": 8.0,
            "total_area": 64.0,
            "corridor_served_ratio": 0.75,
            "fully_connected": True,
        },
        "ranking": {
            "score": 0.88,
            "breakdown": {
                "hard_compliance": 1.0,
                "compactness": 0.8,
                "adjacency": 0.42,
                "travel_margin": 0.16,
                "circulation_ratio": 0.125,
                "alignment": 0.0,
            },
        },
    }
    report = {
        "status": status,
        "source": "algorithmic",
        "violations": [] if status == "COMPLIANT" else ["Travel distance exceeded"],
        "circulation_space": {"corridor_width": 1.2, "walkable_area": 8.0, "connectivity_to_exit": True},
        "checks": {
            "room_minimums": True,
            "connectivity": True,
            "travel_distance": status == "COMPLIANT",
            "exit_width": True,
            "circulation_to_exit": True,
            "llm_spec_valid": True,
            "kg_valid": True,
            "rule_preflight_valid": True,
            "kg_precheck_valid": True,
        },
        "truth_table": [],
        "grounding": {},
    }
    return build_evidence(variant, report, variant_id="v1")


def test_evidence_is_json_serializable():
    evidence = _sample_evidence()
    json.dumps(evidence)


def test_explanation_matches_schema_and_validates():
    evidence = _sample_evidence()
    output = explain(
        evidence,
        ontology_room_types={"Bedroom", "LivingRoom", "Kitchen", "Bathroom"},
        status=evidence["status"],
        llm_fn=None,
    )

    required_keys = {"summary", "why_this_layout", "constraint_justification", "tradeoffs", "suggested_edits", "open_questions"}
    assert required_keys.issubset(output.keys())
    assert isinstance(output["why_this_layout"], list)
    assert isinstance(output["constraint_justification"], list)

    ok, errors = validate_explanation(
        output,
        evidence,
        ontology_room_types={"Bedroom", "LivingRoom", "Kitchen", "Bathroom"},
        status=evidence["status"],
    )
    assert ok, errors


def test_validator_rejects_forbidden_compliance_claim():
    evidence = _sample_evidence(status="NON_COMPLIANT")
    output = {
        "summary": "Fully compliant layout",
        "why_this_layout": ["This is compliant"],
        "constraint_justification": [],
        "tradeoffs": [],
        "suggested_edits": [],
        "open_questions": [],
    }
    ok, errors = validate_explanation(
        output,
        evidence,
        ontology_room_types={"Bedroom"},
        status="NON_COMPLIANT",
    )
    assert not ok
    assert any("Cannot claim compliance" in e or "compliance" in e.lower() for e in errors)


def test_validator_rejects_unknown_room_type():
    evidence = _sample_evidence()
    output = {
        "summary": "ok",
        "why_this_layout": [],
        "constraint_justification": [],
        "tradeoffs": [],
        "suggested_edits": ["MysteryRoom"],
        "open_questions": [],
    }
    ok, errors = validate_explanation(
        output,
        evidence,
        ontology_room_types={"Bedroom"},
        status="COMPLIANT",
    )
    assert not ok
    assert any("Unknown room type" in e for e in errors)
