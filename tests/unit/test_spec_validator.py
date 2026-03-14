import pytest
from constraints.spec_validator import validate_spec

def test_validate_spec_valid(standard_spec):
    res = validate_spec(standard_spec)
    assert res["valid"] is True
    assert res["schema_valid"] is True
    assert res["kg_valid"] is True

def test_validate_spec_invalid_occupancy():
    spec = {"occupancy": "Commercial", "rooms": [{"name": "C1", "type": "Office"}]}
    res = validate_spec(spec)
    assert res["valid"] is False
    assert any("Unsupported occupancy" in e for e in res["errors"])

def test_validate_spec_invalid_room_type():
    spec = {
        "occupancy": "Residential", 
        "total_area": 100,
        "rooms": [{"name": "B1", "type": "Spaceship"}]
    }
    res = validate_spec(spec)
    assert res["valid"] is False
    assert any("unsupported" in e.lower() for e in res["errors"])
