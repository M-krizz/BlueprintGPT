import pytest
from constraints.repair_loop import _repair_once, validate_and_repair_spec

def test_repair_once_defaults():
    spec = {}
    repaired = _repair_once(spec)
    assert repaired["occupancy"] == "Residential"
    assert repaired["area_unit"] == "sq.ft"
    assert repaired["allocation_strategy"] == "priority_weights"

def test_repair_room_aliases():
    spec = {
        "rooms": [
            {"name": "rm1", "type": "hall"}, 
            {"name": "rm2", "type": "wc"},
            {"name": "rm3", "type": "toilet"}
        ]
    }
    repaired = _repair_once(spec)
    types = [r["type"] for r in repaired["rooms"]]
    assert types == ["LivingRoom", "WC", "WC"]

def test_validate_and_repair_loop():
    # A mocked validator that succeeds on the 2nd attempt
    attempts = [0]
    def fake_validator(s):
        attempts[0] += 1
        return {"valid": attempts[0] >= 2, "normalized_spec": s}
    
    res = validate_and_repair_spec({"rooms": [{"type": "hall"}]}, fake_validator, max_attempts=3)
    
    assert res["validation"]["valid"] is True
    assert res["repair_attempts"] == 1
    assert res["spec"]["rooms"][0]["type"] == "LivingRoom"
