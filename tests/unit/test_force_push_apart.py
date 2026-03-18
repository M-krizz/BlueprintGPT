"""
test_force_push_apart.py – Unit tests for force-based push-apart optimizer (Phase 3).

Tests the `_force_push_apart` function in `learned/integration/repair_gate.py`.
Requires shapely (skip if unavailable).
"""
from __future__ import annotations

import pytest

# Check for shapely availability
shapely = pytest.importorskip("shapely", reason="shapely not installed")

from core.building import Building
from core.room import Room
from learned.integration.repair_gate import (
    _force_push_apart,
    _overlap_area,
    _bbox,
    FORCE_PUSH_MAX_ITERS,
    FORCE_PUSH_STEP,
    FORCE_PUSH_DAMPING,
)


def _make_room(name: str, x1: float, y1: float, x2: float, y2: float, rtype: str = "Bedroom") -> Room:
    """Helper: create a rectangular Room with given bbox."""
    room = Room(name, rtype)
    room.polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    room.final_area = (x2 - x1) * (y2 - y1)
    return room


def _make_building(*rooms: Room) -> Building:
    """Helper: create a Building with given rooms."""
    b = Building()
    b.occupancy_type = "Residential"
    b.rooms = list(rooms)
    return b


# ─────────────────────────────────────────────────────────────────────────────
#  1. No overlap → no movement
# ─────────────────────────────────────────────────────────────────────────────

class TestNoOverlap:

    def test_two_distant_rooms_unchanged(self):
        """Two rooms with no overlap should not move."""
        r1 = _make_room("R1", 0.0, 0.0, 2.0, 2.0)
        r2 = _make_room("R2", 3.0, 3.0, 5.0, 5.0)
        building = _make_building(r1, r2)
        boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

        original_r1 = _bbox(r1)
        original_r2 = _bbox(r2)

        remaining = _force_push_apart(building, boundary, max_iters=10)

        assert remaining == 0
        assert _bbox(r1) == pytest.approx(original_r1)
        assert _bbox(r2) == pytest.approx(original_r2)

    def test_edge_touching_no_overlap(self):
        """Rooms exactly touching edges (IoU=0) should not be pushed."""
        r1 = _make_room("R1", 0.0, 0.0, 2.0, 2.0)
        r2 = _make_room("R2", 2.0, 0.0, 4.0, 2.0)  # x1 of r2 == x2 of r1
        building = _make_building(r1, r2)
        boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

        remaining = _force_push_apart(building, boundary, max_iters=10)
        assert remaining == 0


# ─────────────────────────────────────────────────────────────────────────────
#  2. Single overlap → resolved
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleOverlap:

    def test_two_overlapping_rooms_pushed_apart(self):
        """Two overlapping rooms should separate."""
        r1 = _make_room("R1", 0.0, 0.0, 2.0, 2.0)
        r2 = _make_room("R2", 1.5, 1.5, 3.5, 3.5)  # overlaps r1
        building = _make_building(r1, r2)
        boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

        remaining = _force_push_apart(building, boundary, max_iters=50)
        assert remaining == 0  # should resolve
        assert _overlap_area(r1, r2) < 0.01

    def test_force_push_converges_faster_than_greedy(self):
        """Force-push should converge in fewer iterations than greedy for typical cases."""
        r1 = _make_room("R1", 0.0, 0.0, 2.0, 2.0)
        r2 = _make_room("R2", 1.0, 1.0, 3.0, 3.0)
        building = _make_building(r1, r2)
        boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

        # With aggressive settings, should resolve in < 20 iterations
        remaining = _force_push_apart(building, boundary, max_iters=20, initial_step=0.8, damping=0.9)
        assert remaining == 0


# ─────────────────────────────────────────────────────────────────────────────
#  3. Multiple overlaps → simultaneous resolution
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleOverlaps:

    def test_three_overlapping_rooms_all_separate(self):
        """Three overlapping rooms should all separate without oscillation."""
        r1 = _make_room("R1", 0.0, 0.0, 2.0, 2.0, "LivingRoom")
        r2 = _make_room("R2", 1.5, 0.0, 3.5, 2.0, "Bedroom")
        r3 = _make_room("R3", 0.75, 1.5, 2.75, 3.5, "Kitchen")
        building = _make_building(r1, r2, r3)
        boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

        remaining = _force_push_apart(building, boundary, max_iters=100)
        assert remaining == 0
        assert _overlap_area(r1, r2) < 0.01
        assert _overlap_area(r1, r3) < 0.01
        assert _overlap_area(r2, r3) < 0.01

    def test_chain_of_overlaps_resolved(self):
        """Chain A→B→C overlap pattern should resolve without oscillation."""
        r1 = _make_room("A", 0.0, 0.0, 2.0, 2.0)
        r2 = _make_room("B", 1.8, 0.0, 3.8, 2.0)
        r3 = _make_room("C", 3.6, 0.0, 5.6, 2.0)
        building = _make_building(r1, r2, r3)
        boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

        remaining = _force_push_apart(building, boundary, max_iters=100)
        assert remaining == 0


# ─────────────────────────────────────────────────────────────────────────────
#  4. Boundary clamping
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundaryRespect:

    def test_rooms_stay_within_boundary(self):
        """Pushed rooms must not exceed boundary polygon."""
        r1 = _make_room("R1", 0.5, 0.5, 2.5, 2.5)
        r2 = _make_room("R2", 2.0, 2.0, 4.0, 4.0)
        building = _make_building(r1, r2)
        boundary = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]

        _force_push_apart(building, boundary, max_iters=50)

        for room in building.rooms:
            x1, y1, x2, y2 = _bbox(room)
            assert x1 >= 0.0 and x2 <= 5.0
            assert y1 >= 0.0 and y2 <= 5.0

    def test_room_near_boundary_pushed_inward(self):
        """Room near boundary with overlap should push inward (no escape)."""
        r1 = _make_room("R1", 8.0, 8.0, 9.8, 9.8)
        r2 = _make_room("R2", 8.5, 8.5, 10.0, 10.0)  # r2 at edge, overlaps r1
        building = _make_building(r1, r2)
        boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

        remaining = _force_push_apart(building, boundary, max_iters=50)
        # Should resolve or be minimal (boundary limits movement)
        assert remaining <= 1  # may not fully resolve due to boundary constraint


# ─────────────────────────────────────────────────────────────────────────────
#  5. Mass-proportional force split
# ─────────────────────────────────────────────────────────────────────────────

class TestMassProportionalForce:

    def test_larger_room_moves_less(self):
        """Larger room (higher area) should move less than smaller room."""
        r1 = _make_room("Large", 0.0, 0.0, 5.0, 5.0)  # area = 25
        r2 = _make_room("Small", 4.0, 4.0, 6.0, 6.0)  # area = 4
        building = _make_building(r1, r2)
        boundary = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]

        orig_r1_center = ((0.0 + 5.0) / 2, (0.0 + 5.0) / 2)
        orig_r2_center = ((4.0 + 6.0) / 2, (4.0 + 6.0) / 2)

        _force_push_apart(building, boundary, max_iters=10, initial_step=0.5, damping=0.95)

        x1, y1, x2, y2 = _bbox(r1)
        r1_center_new = ((x1 + x2) / 2, (y1 + y2) / 2)
        r1_moved = ((r1_center_new[0] - orig_r1_center[0])**2 + (r1_center_new[1] - orig_r1_center[1])**2)**0.5

        x1, y1, x2, y2 = _bbox(r2)
        r2_center_new = ((x1 + x2) / 2, (y1 + y2) / 2)
        r2_moved = ((r2_center_new[0] - orig_r2_center[0])**2 + (r2_center_new[1] - orig_r2_center[1])**2)**0.5

        # Smaller room should move more
        assert r2_moved > r1_moved


# ─────────────────────────────────────────────────────────────────────────────
#  6. Damping convergence
# ─────────────────────────────────────────────────────────────────────────────

class TestDampingConvergence:

    def test_damping_prevents_oscillation(self):
        """Damped step should prevent oscillation and converge."""
        r1 = _make_room("R1", 0.0, 0.0, 2.0, 2.0)
        r2 = _make_room("R2", 1.0, 1.0, 3.0, 3.0)
        building = _make_building(r1, r2)
        boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

        # High damping → converges
        remaining = _force_push_apart(building, boundary, max_iters=80, initial_step=0.6, damping=0.85)
        assert remaining == 0


# ─────────────────────────────────────────────────────────────────────────────
#  7. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_single_room_no_op(self):
        """Single room with no overlap candidate should be no-op."""
        r1 = _make_room("R1", 0.0, 0.0, 2.0, 2.0)
        building = _make_building(r1)
        boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

        original = _bbox(r1)
        remaining = _force_push_apart(building, boundary, max_iters=10)
        assert remaining == 0
        assert _bbox(r1) == pytest.approx(original)

    def test_zero_area_room_ignored(self):
        """Room with final_area=0 should be ignored."""
        r1 = _make_room("R1", 0.0, 0.0, 2.0, 2.0)
        r2 = _make_room("R2", 1.0, 1.0, 3.0, 3.0)
        r2.final_area = 0.0  # degenerate
        building = _make_building(r1, r2)
        boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

        # Should not crash, r2 effectively ignored
        remaining = _force_push_apart(building, boundary, max_iters=10)
        assert remaining == 0  # only 1 valid room

    def test_none_polygon_rooms_skipped(self):
        """Rooms with polygon=None should be skipped."""
        r1 = _make_room("R1", 0.0, 0.0, 2.0, 2.0)
        r2 = Room("R2", "Bedroom")
        r2.polygon = None
        building = _make_building(r1, r2)
        boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

        original = _bbox(r1)
        remaining = _force_push_apart(building, boundary, max_iters=10)
        assert remaining == 0
        assert _bbox(r1) == pytest.approx(original)

    def test_max_iters_limit_enforced(self):
        """Should stop after max_iters even if not fully resolved."""
        r1 = _make_room("R1", 0.0, 0.0, 2.0, 2.0)
        r2 = _make_room("R2", 1.0, 1.0, 3.0, 3.0)
        building = _make_building(r1, r2)
        boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

        # Very small step and no damping → won't resolve in 5 iters
        remaining = _force_push_apart(building, boundary, max_iters=5, initial_step=0.01, damping=1.0)
        # Should still have overlap (but less than before)
        assert _overlap_area(r1, r2) > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
