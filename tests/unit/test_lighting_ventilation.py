"""Unit tests for Chapter-4 lighting & ventilation compliance in window_placer."""
import pytest
from geometry.window_placer import (
    compute_window_opening_area,
    check_lighting_ventilation_compliance,
    summarize_lv_compliance,
    ASSUMED_WINDOW_HEIGHT,
)


class MockRoom:
    """Mock room object for testing."""
    def __init__(self, name, room_type, polygon):
        self.name = name
        self.room_type = room_type
        self.polygon = polygon


# ─── Window Opening Area Tests ─────────────────────────────────────────────

def test_compute_window_opening_area():
    """Window opening area = segment length * assumed height."""
    # 2m wide window
    seg = ((0, 0), (2, 0))
    area = compute_window_opening_area(seg)
    expected = 2.0 * ASSUMED_WINDOW_HEIGHT  # 2m * 1.2m = 2.4 sq.m
    assert abs(area - expected) < 0.01


def test_compute_window_opening_area_custom_height():
    """Window opening area with custom height."""
    seg = ((0, 0), (1.5, 0))
    area = compute_window_opening_area(seg, height=1.5)
    expected = 1.5 * 1.5  # 2.25 sq.m
    assert abs(area - expected) < 0.01


# ─── Lighting & Ventilation Compliance Tests ───────────────────────────────

def test_lv_compliance_habitable_room_pass():
    """Habitable room with sufficient window area passes."""
    # 4m x 4m = 16 sq.m floor area -> need 1.6 sq.m opening (10%)
    room = MockRoom("Bed1", "Bedroom", [(0, 0), (4, 0), (4, 4), (0, 4)])
    # 2m wide window at height 1.2m = 2.4 sq.m opening
    windows = [((0, 0), (2, 0))]

    result = check_lighting_ventilation_compliance([room], windows)

    assert result["compliant"] is True
    assert len(result["violations"]) == 0
    assert result["rooms"][0]["floor_area"] == 16.0
    assert result["rooms"][0]["achieved_opening"] >= 1.6


def test_lv_compliance_habitable_room_fail():
    """Habitable room with insufficient window area fails."""
    # 5m x 5m = 25 sq.m floor area -> need 2.5 sq.m opening (10%)
    room = MockRoom("Bed1", "Bedroom", [(0, 0), (5, 0), (5, 5), (0, 5)])
    # 1m wide window at height 1.2m = 1.2 sq.m opening (insufficient)
    windows = [((0, 0), (1, 0))]

    result = check_lighting_ventilation_compliance([room], windows)

    assert result["compliant"] is False
    assert len(result["violations"]) == 1
    assert "Bed1" in result["violations"][0]
    assert result["rooms"][0]["deficit"] > 0


def test_lv_compliance_kitchen_minimum():
    """Kitchen requires at least 1.0 sq.m window opening."""
    # 3m x 2m = 6 sq.m floor area -> 10% = 0.6 sq.m, but kitchen min is 1.0
    room = MockRoom("Kit1", "Kitchen", [(0, 0), (3, 0), (3, 2), (0, 2)])
    # 0.8m wide window at height 1.2m = 0.96 sq.m (below 1.0 min)
    windows = [((0, 0), (0.8, 0))]

    result = check_lighting_ventilation_compliance([room], windows)

    assert result["compliant"] is False
    assert result["rooms"][0]["required_opening"] == 1.0  # Kitchen minimum


def test_lv_compliance_kitchen_pass():
    """Kitchen with 1.0+ sq.m window passes."""
    room = MockRoom("Kit1", "Kitchen", [(0, 0), (3, 0), (3, 2), (0, 2)])
    # 1m wide window at height 1.2m = 1.2 sq.m
    windows = [((0, 0), (1, 0))]

    result = check_lighting_ventilation_compliance([room], windows)

    assert result["compliant"] is True


def test_lv_compliance_bathroom_ventilation():
    """Bathroom requires 0.37 sq.m ventilation opening."""
    # Small bathroom
    room = MockRoom("Bath1", "Bathroom", [(0, 0), (2, 0), (2, 1.5), (0, 1.5)])
    # 0.35m wide window = 0.42 sq.m (passes 0.37)
    windows = [((0, 0), (0.35, 0))]

    result = check_lighting_ventilation_compliance([room], windows)

    assert result["compliant"] is True
    assert result["rooms"][0]["achieved_opening"] >= 0.37


def test_lv_compliance_bathroom_fail_as_warning():
    """Bathroom vent failure is recorded as warning (may use shaft)."""
    room = MockRoom("Bath1", "Bathroom", [(0, 0), (2, 0), (2, 1.5), (0, 1.5)])
    # 0.2m wide window = 0.24 sq.m (below 0.37)
    windows = [((0, 0), (0.2, 0))]

    result = check_lighting_ventilation_compliance([room], windows)

    # Bath/WC failures are warnings, not hard violations
    assert result["compliant"] is True
    assert len(result["warnings"]) == 1
    assert "Bath1" in result["warnings"][0]


def test_lv_compliance_wc():
    """WC requires same ventilation as bathroom."""
    room = MockRoom("WC1", "WC", [(0, 0), (1, 0), (1, 1.2), (0, 1.2)])
    # 0.4m wide window = 0.48 sq.m (passes 0.37)
    windows = [((0, 0), (0.4, 0))]

    result = check_lighting_ventilation_compliance([room], windows)

    assert result["compliant"] is True
    assert result["rooms"][0]["required_opening"] == 0.37


def test_lv_compliance_no_window_room():
    """Room with no windows fails if it requires opening."""
    room = MockRoom("Bed1", "Bedroom", [(0, 0), (4, 0), (4, 4), (0, 4)])
    windows = []  # No windows

    result = check_lighting_ventilation_compliance([room], windows)

    assert result["compliant"] is False
    assert result["rooms"][0]["achieved_opening"] == 0
    assert result["rooms"][0]["deficit"] > 0


def test_lv_compliance_multiple_rooms():
    """Check compliance for multiple rooms."""
    bedroom = MockRoom("Bed1", "Bedroom", [(0, 0), (4, 0), (4, 4), (0, 4)])
    kitchen = MockRoom("Kit1", "Kitchen", [(4, 0), (7, 0), (7, 3), (4, 3)])
    bathroom = MockRoom("Bath1", "Bathroom", [(4, 3), (6, 3), (6, 4.5), (4, 4.5)])

    # Windows on exterior edges
    windows = [
        ((0, 0), (2, 0)),      # Bedroom window
        ((4, 0), (5.5, 0)),    # Kitchen window (1.5m * 1.2 = 1.8 sq.m)
        ((4, 3), (4.5, 3)),    # Bathroom window (0.5m * 1.2 = 0.6 sq.m)
    ]

    result = check_lighting_ventilation_compliance([bedroom, kitchen, bathroom], windows)

    assert result["compliant"] is True
    assert len(result["rooms"]) == 3


def test_lv_compliance_lighting_depth_warning():
    """Large room gets lighting depth warning."""
    # 10m x 4m room - longest dimension > 7.5m
    room = MockRoom("Living", "LivingRoom", [(0, 0), (10, 0), (10, 4), (0, 4)])
    windows = [((0, 0), (3, 0))]  # Adequate window area

    result = check_lighting_ventilation_compliance([room], windows)

    assert len(result["warnings"]) >= 1
    assert any("lighting depth" in w for w in result["warnings"])


# ─── Summary Function Tests ────────────────────────────────────────────────

def test_summarize_lv_compliance():
    """Summary function returns aggregated stats."""
    room1 = MockRoom("Bed1", "Bedroom", [(0, 0), (4, 0), (4, 4), (0, 4)])
    room2 = MockRoom("Kit1", "Kitchen", [(4, 0), (7, 0), (7, 3), (4, 3)])
    windows = [((0, 0), (2, 0)), ((4, 0), (5, 0))]

    summary = summarize_lv_compliance([room1, room2], windows)

    assert "compliant" in summary
    assert "rooms_checked" in summary
    assert "total_floor_area_sqm" in summary
    assert "total_opening_area_sqm" in summary
    assert "opening_ratio_achieved" in summary
    assert summary["rooms_checked"] == 2
