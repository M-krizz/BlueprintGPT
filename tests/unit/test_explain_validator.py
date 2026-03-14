from explain.validator import validate_explanation


def test_validator_rejects_compliance_claim_on_non_compliant():
    evidence = {}
    output = {"summary": "This is compliant."}
    ok, errors = validate_explanation(output, evidence, ontology_room_types=set(), status="NON_COMPLIANT")
    assert not ok
    assert errors


def test_validator_rejects_unknown_room_type_in_edits():
    evidence = {}
    output = {"summary": "ok", "suggested_edits": [{"action": "resize", "target": "XRoom", "note": ""}]}
    ok, errors = validate_explanation(output, evidence, ontology_room_types={"Bedroom"}, status="COMPLIANT")
    assert not ok
    assert any("Unknown room type" in e for e in errors)


def test_validator_checks_evidence_path():
    evidence = {"metrics": {"adjacency": 0.8}}
    output = {
        "summary": "ok",
        "why_this_layout": [],
        "constraint_justification": [
            {"claim": "c", "evidence_path": "metrics.missing", "value": 0.8},
        ],
        "tradeoffs": [],
        "suggested_edits": [],
        "open_questions": [],
    }
    ok, errors = validate_explanation(output, evidence, ontology_room_types=set(), status="COMPLIANT")
    assert not ok
    assert any("Missing evidence_path" in e for e in errors)