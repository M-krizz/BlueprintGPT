import pytest
from constraints.compliance_report import build_compliance_report

def test_build_compliance_report_compliant():
    result = {
        "metrics": {
            "fully_connected": True,
            "travel_distance_compliant": True,
            "required_exit_width": 1.2,
            "max_travel_distance": 15.0,
            "max_allowed_travel_distance": 20.0,
            "circulation_walkable_area": 20.0,
            "corridor_width": 1.2,
            "connectivity_to_exit": True,
        },
        "modifications": [],
        "spec_validation": {"schema_valid": True, "kg_valid": True},
        "rule_preflight": {"valid": True},
        "kg_precheck": {"valid": True},
        "bounding_box": {"width": 10, "height": 10},
        "allocation": {}
    }
    report = build_compliance_report(result)
    assert report["status"] == "COMPLIANT"
    assert len(report["violations"]) == 0
    assert report["checks"]["connectivity"] is True
    assert report["checks"]["exit_width"] is True

def test_build_compliance_report_non_compliant():
    result = {
        "metrics": {
            "fully_connected": False,
            "travel_distance_compliant": False,
            "required_exit_width": 0.5, # fails width
            "max_travel_distance": 50.0,
            "max_allowed_travel_distance": 20.0,
            "circulation_walkable_area": 20.0,
            "corridor_width": 1.2,
        },
        "modifications": ["Bumped room area"],
        "bounding_box": {"width": 10, "height": 10},
        "ontology": {
            "valid": False,
            "violations": [{"message": "OWL conflict detected."}]
        }
    }
    report = build_compliance_report(result)
    assert report["status"] == "NON_COMPLIANT"
    assert len(report["violations"]) >= 2
