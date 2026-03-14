import pytest
from core.building import Building
from core.room import Room
from constraints.rule_engine import RuleEngine

@pytest.fixture
def rule_engine():
    return RuleEngine("ontology/regulation_data.json")

def test_apply_room_rules_minimums(rule_engine):
    b = Building("Residential")
    r1 = Room("Bed1", "Bedroom", 1.0) # Too small
    r1.final_area = 1.0
    b.rooms = [r1]
    
    mods = rule_engine.apply_room_rules(b)
    assert len(mods) > 0
    assert r1.final_area >= 9.0  # Most ontology regulations enforce at least 9 for a master bed

def test_allocate_areas(rule_engine):
    b = Building("Residential")
    b.rooms = [Room("Bed1", "Bedroom", 10.0), Room("Kit1", "Kitchen", 15.0)]

    
    # Priority Strategy
    mods, breakdown = rule_engine.allocate_room_areas_from_total(b, 100, unit="sq.m", strategy="priority_weights")
    assert breakdown["target_usable_area_sqm"] > 0
    assert len(breakdown["rooms"]) == 2
    
    # Equal Surplus Strategy
    mods2, breakdown2 = rule_engine.allocate_room_areas_from_total(b, 100, unit="sq.m", strategy="equal_surplus")
    assert breakdown2["allocation_strategy"] == "equal_surplus"

def test_compute_metrics(rule_engine):
    b = Building("Residential")
    b.rooms = [Room("Bed1", "Bedroom", 12.0)]
    b.rooms[0].final_area = 12.0
    
    total, load = rule_engine.compute_building_metrics(b)
    assert total > 12.0  # Accounts for required circulation factor
    assert load > 0

def test_preflight_validate_spec_fail_small(rule_engine):
    spec = {
        "occupancy": "Residential",
        "total_area": 10,  # Far too small to hold a bedroom + circulation
        "area_unit": "sq.m",
        "rooms": [{"name": "B1", "type": "Bedroom"}]
    }
    result = rule_engine.preflight_validate_spec(spec)
    assert not result["valid"]
    assert len(result["errors"]) > 0
    assert "below minimum required" in result["errors"][0]
