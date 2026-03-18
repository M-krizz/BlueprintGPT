"""Unit tests for Chapter-4 ground truth integration in RuleEngine."""
import pytest
from core.building import Building
from core.room import Room
from constraints.rule_engine import RuleEngine
from constraints.chapter4_helpers import (
    plot_bucket,
    get_min_room_dims,
    get_travel_distance_limit,
    get_corridor_min_width,
    get_exit_capacity,
    get_stair_min_width_by_occupancy,
)


@pytest.fixture
def rule_engine():
    return RuleEngine("ontology/regulation_data.json")


# ─── Plot Bucket Selection Tests ───────────────────────────────────────────

def test_plot_bucket_small():
    """Plot area <= 50 sq.m should use upto_50sqm bucket."""
    assert plot_bucket(30) == "upto_50sqm"
    assert plot_bucket(50) == "upto_50sqm"


def test_plot_bucket_large():
    """Plot area > 50 sq.m should use above_50sqm bucket."""
    assert plot_bucket(50.1) == "above_50sqm"
    assert plot_bucket(100) == "above_50sqm"


def test_rule_engine_plot_bucket(rule_engine):
    """RuleEngine should track plot bucket based on set area."""
    rule_engine.set_plot_area(40)
    assert rule_engine.get_plot_bucket() == "upto_50sqm"

    rule_engine.set_plot_area(80)
    assert rule_engine.get_plot_bucket() == "above_50sqm"


def test_rule_engine_plot_bucket_from_boundary(rule_engine):
    """RuleEngine should compute area from boundary polygon."""
    # 10m x 5m = 50 sq.m
    boundary = [(0, 0), (10, 0), (10, 5), (0, 5)]
    area = rule_engine.set_plot_area_from_boundary(boundary)
    assert abs(area - 50.0) < 0.01
    assert rule_engine.get_plot_bucket() == "upto_50sqm"

    # 12m x 8m = 96 sq.m
    boundary2 = [(0, 0), (12, 0), (12, 8), (0, 8)]
    area2 = rule_engine.set_plot_area_from_boundary(boundary2)
    assert abs(area2 - 96.0) < 0.01
    assert rule_engine.get_plot_bucket() == "above_50sqm"


# ─── Room Minimums by Plot Bucket Tests ────────────────────────────────────

def test_room_dims_small_plot():
    """Room minimums for small plot (<=50 sq.m) from Table 4.2."""
    dims = get_min_room_dims("Bedroom", 40)  # Maps to Habitable
    assert dims["min_area"] == 7.5
    assert dims["min_width"] == 2.1
    assert dims["min_height"] == 2.75

    dims_kit = get_min_room_dims("Kitchen", 40)
    assert dims_kit["min_area"] == 3.3

    dims_bath = get_min_room_dims("Bathroom", 40)
    assert dims_bath["min_area"] == 1.2

    dims_wc = get_min_room_dims("WC", 40)
    assert dims_wc["min_area"] == 1.0


def test_room_dims_large_plot():
    """Room minimums for large plot (>50 sq.m) from Table 4.2."""
    dims = get_min_room_dims("Bedroom", 100)  # Maps to Habitable
    assert dims["min_area"] == 9.5
    assert dims["min_width"] == 2.4
    assert dims["min_height"] == 2.75

    dims_kit = get_min_room_dims("Kitchen", 100)
    assert dims_kit["min_area"] == 4.5

    dims_bath = get_min_room_dims("Bathroom", 100)
    assert dims_bath["min_area"] == 1.8

    dims_wc = get_min_room_dims("WC", 100)
    assert dims_wc["min_area"] == 1.1


def test_apply_room_rules_with_bucket(rule_engine):
    """RuleEngine.apply_room_rules should use plot bucket for Residential."""
    b = Building("Residential")
    r1 = Room("Bed1", "Bedroom", 5.0)  # Too small for any bucket
    r1.final_area = 5.0
    b.rooms = [r1]

    # Small plot - should enforce 7.5 sq.m
    rule_engine.set_plot_area(40)
    mods = rule_engine.apply_room_rules(b)
    assert len(mods) > 0
    assert r1.final_area >= 7.5

    # Large plot - should enforce 9.5 sq.m
    r2 = Room("Bed2", "Bedroom", 5.0)
    r2.final_area = 5.0
    b2 = Building("Residential")
    b2.rooms = [r2]
    rule_engine.set_plot_area(100)
    mods2 = rule_engine.apply_room_rules(b2)
    assert len(mods2) > 0
    assert r2.final_area >= 9.5


# ─── Travel Distance Tests ─────────────────────────────────────────────────

def test_travel_distance_residential():
    """Residential max travel distance = 22.5m."""
    assert get_travel_distance_limit("Residential") == 22.5


def test_travel_distance_assembly():
    """Assembly max travel distance = 30.0m."""
    assert get_travel_distance_limit("Assembly") == 30.0


def test_travel_distance_hazardous():
    """Hazardous max travel distance = 22.5m."""
    assert get_travel_distance_limit("Hazardous") == 22.5


def test_rule_engine_travel_distance(rule_engine):
    """RuleEngine.get_max_travel_distance uses Chapter-4 table."""
    assert rule_engine.get_max_travel_distance("Residential") == 22.5
    assert rule_engine.get_max_travel_distance("Assembly") == 30.0


# ─── Exit Capacity Tests (Table 4.3) ───────────────────────────────────────

def test_exit_capacity_residential():
    """Residential exit capacity: stair=25, ramp=50, door=75."""
    assert get_exit_capacity("Residential", "stair") == 25
    assert get_exit_capacity("Residential", "ramp") == 50
    assert get_exit_capacity("Residential", "door") == 75


def test_exit_capacity_assembly():
    """Assembly exit capacity: stair=40."""
    assert get_exit_capacity("Assembly", "stair") == 40


def test_compute_exit_width_uses_capacity(rule_engine):
    """RuleEngine.compute_exit_width should use Table 4.3 capacity."""
    b = Building("Residential")
    b.occupant_load = 50  # 50 persons

    # For door: 50 / 75 = 0.67 units -> 0.33m (but min is 1.0m)
    width = rule_engine.compute_exit_width(b, "door")
    assert width >= 1.0

    # For stair: 50 / 25 = 2 units -> 1.0m
    stair_width = rule_engine.compute_exit_width(b, "stair")
    assert stair_width >= 1.0


def test_compute_exit_width_high_occupancy(rule_engine):
    """High occupancy should require wider exits."""
    b = Building("Assembly")
    b.occupant_load = 200  # 200 persons

    # For stair: 200 / 40 = 5 units -> 2.5m
    stair_width = rule_engine.compute_exit_width(b, "stair")
    assert stair_width >= 2.5


# ─── Corridor Width Tests (Section 4.8.7) ──────────────────────────────────

def test_corridor_width_residential():
    """Residential dwelling unit corridor = 1.0m."""
    assert get_corridor_min_width("Residential") == 1.0
    assert get_corridor_min_width("Residential", "dwelling_unit") == 1.0


def test_corridor_width_hostel():
    """Residential hostel corridor = 1.25m."""
    assert get_corridor_min_width("Residential", "hostel") == 1.25


def test_corridor_width_assembly():
    """Assembly corridor = 2.0m."""
    assert get_corridor_min_width("Assembly") == 2.0


def test_corridor_width_hospital():
    """Hospital corridor = 2.4m."""
    assert get_corridor_min_width("Institutional", "hospital") == 2.4


def test_rule_engine_corridor_width(rule_engine):
    """RuleEngine.get_corridor_min_width uses Chapter-4 table."""
    assert rule_engine.get_corridor_min_width("Residential") == 1.0
    assert rule_engine.get_corridor_min_width("Assembly") == 2.0


# ─── Stair Width Tests (Section 4.8.6) ─────────────────────────────────────

def test_stair_width_by_occupancy():
    """Stair width varies by occupancy type."""
    assert get_stair_min_width_by_occupancy("Residential", "low_rise") == 0.9
    assert get_stair_min_width_by_occupancy("Residential") == 1.25
    assert get_stair_min_width_by_occupancy("Assembly") == 2.0
    assert get_stair_min_width_by_occupancy("Institutional") == 2.0
    assert get_stair_min_width_by_occupancy("Educational") == 1.5


def test_rule_engine_stair_width_with_bucket(rule_engine):
    """RuleEngine stair width considers both plot bucket and occupancy."""
    # Small plot: stair_width from bucket = 0.75, but occupancy = 0.9
    rule_engine.set_plot_area(40)
    width = rule_engine.get_stair_min_width("Residential", "low_rise")
    assert width == 0.9  # max(0.75, 0.9)

    # Large plot: stair_width from bucket = 0.9, occupancy = 0.9
    rule_engine.set_plot_area(100)
    width2 = rule_engine.get_stair_min_width("Residential", "low_rise")
    assert width2 == 0.9


# ─── Comprehensive Compliance Check Tests ──────────────────────────────────

def test_check_chapter4_compliance_pass(rule_engine):
    """Compliance check passes when all rules satisfied."""
    b = Building("Residential")
    r1 = Room("Bed1", "Bedroom", 12.0)
    r1.final_area = 12.0
    b.rooms = [r1]
    b.occupant_load = 5

    rule_engine.set_plot_area(100)

    result = rule_engine.check_chapter4_compliance(
        b,
        corridor_width=1.2,
        stair_width=1.3,  # Above 1.25m min for non-low-rise Residential
        travel_distance=15.0,
    )

    assert result["compliant"] is True
    assert len(result["violations"]) == 0
    assert result["plot_bucket"] == "above_50sqm"


def test_check_chapter4_compliance_room_fail(rule_engine):
    """Compliance check fails when room area below minimum."""
    b = Building("Residential")
    r1 = Room("Bed1", "Bedroom", 5.0)
    r1.final_area = 5.0  # Below 9.5 for large plot
    b.rooms = [r1]

    rule_engine.set_plot_area(100)

    result = rule_engine.check_chapter4_compliance(b)

    assert result["compliant"] is False
    assert any("room_area" in v["rule"] for v in result["violations"])


def test_check_chapter4_compliance_corridor_fail(rule_engine):
    """Compliance check fails when corridor too narrow."""
    b = Building("Residential")
    b.rooms = []
    b.occupant_load = 5

    rule_engine.set_plot_area(100)

    result = rule_engine.check_chapter4_compliance(
        b,
        corridor_width=0.8,  # Below 1.0m minimum
    )

    assert result["compliant"] is False
    assert any(v["rule"] == "corridor_width" for v in result["violations"])


def test_check_chapter4_compliance_travel_fail(rule_engine):
    """Compliance check fails when travel distance exceeds limit."""
    b = Building("Residential")
    b.rooms = []

    result = rule_engine.check_chapter4_compliance(
        b,
        travel_distance=25.0,  # Above 22.5m limit
    )

    assert result["compliant"] is False
    assert any(v["rule"] == "travel_distance" for v in result["violations"])
