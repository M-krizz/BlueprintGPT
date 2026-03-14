import pytest
from learned.integration.repair_gate import _stage1_sanitize, _iou, _overlap_area
from core.building import Building
from core.room import Room

def test_stage1_sanitize_boundaries():
    b = Building("Residential")
    r1 = Room("Bed", "Bedroom", 12.0)
    # Define an out-of-bounds polygon to trigger boundary clamp logic
    r1.polygon = [(0, 0), (12, 0), (12, 12), (0, 12)]
    b.rooms = [r1]
    
    boundary = [(0, 0), (10, 0), (10, 10), (0, 10)]
    trace = []
    
    violations = _stage1_sanitize(b, boundary, trace)
    assert len(violations) > 0
    assert any("clamped" in v for v in violations)
    
    # Check if polygon was properly clipped to boundary shape
    assert b.rooms[0].polygon == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

def test_stage1_sanitize_degenerate():
    b = Building("Residential")
    r2 = Room("Tiny", "WC", 1.0)
    r2.polygon = [(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)]
    b.rooms = [r2]
    
    boundary = [(0, 0), (10, 0), (10, 10), (0, 10)]
    trace = []
    
    violations = _stage1_sanitize(b, boundary, trace)
    # Should drop the degenerate tiny room
    assert len(b.rooms) == 0
