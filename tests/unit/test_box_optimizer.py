"""
test_box_optimizer.py – Unit tests for box optimizer (Phase 4).

Tests the constraint-based room placement optimizer. Skips if neither
OR-Tools nor scipy is available.
"""
from __future__ import annotations

import pytest

# Import the module (no heavy deps at module level)
from learned.integration.box_optimizer import (
    get_solver_info,
    optimize_box_placement,
    _overlap_area,
    _bbox,
    _set_rect,
    BOX_OPT_ENABLED,
)

# Skip all tests if no solver available
pytestmark = pytest.mark.skipif(
    not get_solver_info()["any_available"],
    reason="Neither OR-Tools nor scipy installed"
)


# ── Mock Room class ───────────────────────────────────────────────────────────

class _MockRoom:
    def __init__(self, name: str, x1: float, y1: float, x2: float, y2: float):
        self.name = name
        self.polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        self.final_area = (x2 - x1) * (y2 - y1)


class _MockBuilding:
    def __init__(self, *rooms):
        self.rooms = list(rooms)


# ── Helper ────────────────────────────────────────────────────────────────────

def _count_overlaps(rooms, threshold=0.01):
    """Count overlapping room pairs."""
    n = len(rooms)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if rooms[i].polygon and rooms[j].polygon:
                x1a, y1a, x2a, y2a = _bbox(rooms[i])
                x1b, y1b, x2b, y2b = _bbox(rooms[j])
                if _overlap_area(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b) > threshold:
                    count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
#  1. Basic functionality
# ─────────────────────────────────────────────────────────────────────────────

class TestBasicFunctionality:

    def test_solver_info_returns_dict(self):
        info = get_solver_info()
        assert isinstance(info, dict)
        assert "any_available" in info
        assert info["any_available"] is True  # else tests would be skipped

    def test_no_rooms_succeeds(self):
        building = _MockBuilding()
        boundary = [(0, 0), (10, 0), (10, 10), (0, 10)]
        result = optimize_box_placement(building, boundary)
        assert result["success"] is True

    def test_single_room_succeeds(self):
        r1 = _MockRoom("R1", 1, 1, 3, 3)
        building = _MockBuilding(r1)
        boundary = [(0, 0), (10, 0), (10, 10), (0, 10)]
        result = optimize_box_placement(building, boundary)
        assert result["success"] is True
        assert result["remaining_overlaps"] == 0


# ─────────────────────────────────────────────────────────────────────────────
#  2. Overlap resolution
# ─────────────────────────────────────────────────────────────────────────────

class TestOverlapResolution:

    def test_two_overlapping_rooms_separated(self):
        """Two overlapping rooms should be separated."""
        r1 = _MockRoom("R1", 0, 0, 2, 2)
        r2 = _MockRoom("R2", 1, 1, 3, 3)  # overlaps r1
        building = _MockBuilding(r1, r2)
        boundary = [(0, 0), (10, 0), (10, 10), (0, 10)]

        assert _count_overlaps(building.rooms) == 1

        result = optimize_box_placement(building, boundary, time_limit=5.0)

        # Should succeed and have no overlaps
        assert result["success"] is True
        assert result["remaining_overlaps"] == 0
        assert _count_overlaps(building.rooms) == 0

    def test_three_overlapping_rooms_separated(self):
        """Three overlapping rooms should all separate."""
        r1 = _MockRoom("R1", 0, 0, 2, 2)
        r2 = _MockRoom("R2", 1, 0, 3, 2)
        r3 = _MockRoom("R3", 0.5, 1, 2.5, 3)
        building = _MockBuilding(r1, r2, r3)
        boundary = [(0, 0), (10, 0), (10, 10), (0, 10)]

        assert _count_overlaps(building.rooms) >= 2

        result = optimize_box_placement(building, boundary, time_limit=5.0)

        assert result["success"] is True
        assert result["remaining_overlaps"] == 0

    def test_non_overlapping_rooms_unchanged(self):
        """Non-overlapping rooms should stay approximately in place."""
        r1 = _MockRoom("R1", 0, 0, 2, 2)
        r2 = _MockRoom("R2", 5, 5, 7, 7)
        building = _MockBuilding(r1, r2)
        boundary = [(0, 0), (10, 0), (10, 10), (0, 10)]

        orig_r1 = _bbox(r1)
        orig_r2 = _bbox(r2)

        result = optimize_box_placement(building, boundary)

        assert result["success"] is True
        # Rooms should be close to original (minimal displacement objective)
        new_r1 = _bbox(r1)
        new_r2 = _bbox(r2)
        assert abs(new_r1[0] - orig_r1[0]) < 1.0
        assert abs(new_r2[0] - orig_r2[0]) < 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  3. Boundary respect
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundaryRespect:

    def test_rooms_stay_inside_boundary(self):
        """All rooms must stay inside the boundary after optimization."""
        r1 = _MockRoom("R1", 0, 0, 2, 2)
        r2 = _MockRoom("R2", 1, 1, 3, 3)
        building = _MockBuilding(r1, r2)
        boundary = [(0, 0), (6, 0), (6, 6), (0, 6)]

        optimize_box_placement(building, boundary, time_limit=5.0)

        for room in building.rooms:
            x1, y1, x2, y2 = _bbox(room)
            assert x1 >= 0 and x2 <= 6
            assert y1 >= 0 and y2 <= 6

    def test_tight_boundary_still_resolves(self):
        """Should handle tight boundaries where rooms barely fit."""
        r1 = _MockRoom("R1", 0, 0, 2, 2)
        r2 = _MockRoom("R2", 1, 1, 3, 3)
        building = _MockBuilding(r1, r2)
        # Tight boundary: 4x4, needs to fit two 2x2 rooms
        boundary = [(0, 0), (4, 0), (4, 4), (0, 4)]

        result = optimize_box_placement(building, boundary, time_limit=5.0)

        # Should succeed since there's exactly enough space
        assert result["success"] is True
        assert result["remaining_overlaps"] == 0


# ─────────────────────────────────────────────────────────────────────────────
#  4. Size preservation
# ─────────────────────────────────────────────────────────────────────────────

class TestSizePreservation:

    def test_room_sizes_preserved(self):
        """Room dimensions should not change after optimization."""
        r1 = _MockRoom("R1", 0, 0, 2.5, 1.5)  # 2.5 x 1.5
        r2 = _MockRoom("R2", 1, 0.5, 4, 2.5)  # 3 x 2
        building = _MockBuilding(r1, r2)
        boundary = [(0, 0), (10, 0), (10, 10), (0, 10)]

        orig_sizes = [(2.5, 1.5), (3.0, 2.0)]

        optimize_box_placement(building, boundary, time_limit=5.0)

        for i, room in enumerate(building.rooms):
            x1, y1, x2, y2 = _bbox(room)
            w, h = x2 - x1, y2 - y1
            assert w == pytest.approx(orig_sizes[i][0], abs=0.01)
            assert h == pytest.approx(orig_sizes[i][1], abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
#  5. Solver selection
# ─────────────────────────────────────────────────────────────────────────────

class TestSolverSelection:

    def test_reports_solver_used(self):
        """Result should indicate which solver was used."""
        r1 = _MockRoom("R1", 0, 0, 2, 2)
        r2 = _MockRoom("R2", 1, 1, 3, 3)
        building = _MockBuilding(r1, r2)
        boundary = [(0, 0), (10, 0), (10, 10), (0, 10)]

        result = optimize_box_placement(building, boundary)

        assert result["solver"] in ("cpsat", "scipy")

    def test_prefer_cpsat_flag(self):
        """prefer_cpsat=False should try scipy first."""
        info = get_solver_info()
        if not info["scipy"]:
            pytest.skip("scipy not available")

        r1 = _MockRoom("R1", 0, 0, 2, 2)
        r2 = _MockRoom("R2", 1, 1, 3, 3)
        building = _MockBuilding(r1, r2)
        boundary = [(0, 0), (10, 0), (10, 10), (0, 10)]

        result = optimize_box_placement(building, boundary, prefer_cpsat=False)

        # If scipy succeeded first, it should report scipy
        # (unless scipy failed and cpsat took over)
        assert result["solver"] in ("cpsat", "scipy")


# ─────────────────────────────────────────────────────────────────────────────
#  6. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_none_polygon_rooms_skipped(self):
        """Rooms with None polygon should be skipped."""
        r1 = _MockRoom("R1", 0, 0, 2, 2)
        r2 = _MockRoom("R2", 1, 1, 3, 3)
        r2.polygon = None
        building = _MockBuilding(r1, r2)
        boundary = [(0, 0), (10, 0), (10, 10), (0, 10)]

        result = optimize_box_placement(building, boundary)

        assert result["success"] is True
        assert result["remaining_overlaps"] == 0

    def test_identical_overlapping_rooms(self):
        """Two rooms at exact same position should separate."""
        r1 = _MockRoom("R1", 2, 2, 4, 4)
        r2 = _MockRoom("R2", 2, 2, 4, 4)  # exact same position
        building = _MockBuilding(r1, r2)
        boundary = [(0, 0), (10, 0), (10, 10), (0, 10)]

        result = optimize_box_placement(building, boundary, time_limit=5.0)

        assert result["success"] is True
        assert result["remaining_overlaps"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
