from constraints.repair_loop import validate_and_repair_spec
from constraints.spec_validator import validate_spec
from generator.layout_generator import generate_layout_from_spec
from nl_interface.service import process_user_request


def test_nl_to_backend_spec_runs_legacy_generation_pipeline():
    response = process_user_request(
        "Need 2 bedrooms, 1 bathroom, 1 kitchen and 1 drawing room on a 10 marla plot with a north entrance and minimize corridor.",
        resolution={"boundary_size": (15, 10)},
    )

    assert response["backend_ready"] is True
    spec = response["backend_spec"]

    repaired = validate_and_repair_spec(spec, validate_spec, max_attempts=1)
    assert repaired["validation"]["valid"] is True

    result = generate_layout_from_spec(repaired["spec"], regulation_file="ontology/regulation_data.json")
    variants = result.get("layout_variants", [result])

    assert variants
    assert any(variant.get("source") == "algorithmic" for variant in variants)
    assert result.get("recommended_index") is not None


def test_nl_response_stays_non_ready_without_geometry_resolution():
    response = process_user_request(
        "Need 1 bedroom, 1 bathroom and 1 kitchen on a 5 marla plot with an east entrance."
    )

    assert response["backend_ready"] is False
    assert "boundary_polygon" in response["missing_fields"]
    assert response["backend_spec"] is None
