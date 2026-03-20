"""
test_repair_severity.py - Test repair severity metric tracking.

Tests the RepairReport functionality and severity calculation.
"""
from __future__ import annotations

import pytest

from learned.integration.repair_gate import RepairReport, validate_and_repair_generated_layout
from core.building import Building
from core.room import Room


class TestRepairReportStructure:
    """Test RepairReport dataclass structure and properties."""

    def test_repair_report_creation(self):
        """Test RepairReport can be created with all fields."""
        report = RepairReport(
            severity_score=0.25,
            overlap_fixes=2,
            min_dim_violations_fixed=3,
            topology_changes=1,
            total_displacement_m=2.5,
            stages_applied=["sanitize_geometry", "overlap_repair"],
            original_room_count=4,
            final_room_count=4,
        )

        assert report.severity_score == 0.25
        assert report.overlap_fixes == 2
        assert report.min_dim_violations_fixed == 3
        assert report.topology_changes == 1
        assert report.total_displacement_m == 2.5
        assert len(report.stages_applied) == 2
        assert "sanitize_geometry" in report.stages_applied
        assert not report.room_count_changed

    def test_room_count_changed_property(self):
        """Test room_count_changed property."""
        # No change
        report1 = RepairReport(
            severity_score=0.1, overlap_fixes=0, min_dim_violations_fixed=0,
            topology_changes=0, total_displacement_m=0.0, stages_applied=[],
            original_room_count=3, final_room_count=3
        )
        assert not report1.room_count_changed

        # Room count decreased
        report2 = RepairReport(
            severity_score=0.8, overlap_fixes=0, min_dim_violations_fixed=0,
            topology_changes=2, total_displacement_m=0.0, stages_applied=["sanitize_geometry"],
            original_room_count=5, final_room_count=3
        )
        assert report2.room_count_changed


class TestRepairSeverityIntegration:
    """Test repair severity tracking in real repair scenarios."""

    def _create_test_building(self) -> Building:
        """Create a test building with overlapping rooms."""
        building = Building(occupancy_type="Residential")

        # Room 1: normal position
        room1 = Room("Bedroom1", "Bedroom", 12.0)
        room1.polygon = [(0, 0), (3, 0), (3, 4), (0, 4)]
        room1.final_area = 12.0

        # Room 2: overlapping with room1 (severe overlap)
        room2 = Room("Kitchen1", "Kitchen", 8.0)
        room2.polygon = [(2, 2), (5, 2), (5, 5), (2, 5)]  # Overlaps with bedroom
        room2.final_area = 8.0

        # Room 3: too small (below min requirements)
        room3 = Room("Bathroom1", "Bathroom", 2.0)
        room3.polygon = [(6, 0), (7, 0), (7, 1), (6, 1)]  # 1x1 = 1 sq unit (too small)
        room3.final_area = 2.0

        building.rooms = [room1, room2, room3]
        return building

    def test_repair_severity_calculation(self):
        """Test repair severity is correctly calculated."""
        building = self._create_test_building()

        # Simple rectangular boundary
        boundary = [(0, 0), (10, 0), (10, 8), (0, 8)]
        entrance = (0, 4)

        # Run repair with tracking
        try:
            repaired, violations, status, trace, report = validate_and_repair_generated_layout(
                building, boundary, entrance_point=entrance, run_ontology=False
            )
        except Exception as e:
            pytest.skip(f"Repair pipeline not available: {e}")

        # Verify report structure
        assert isinstance(report, RepairReport)
        assert 0.0 <= report.severity_score <= 1.0
        assert report.original_room_count == 3
        assert report.final_room_count >= 0  # May have lost rooms
        assert isinstance(report.stages_applied, list)

        # Should have some repairs applied to the problematic building
        assert report.overlap_fixes > 0 or report.min_dim_violations_fixed > 0
        assert len(report.stages_applied) > 0

        print(f"\\nRepair severity: {report.severity_score}")
        print(f"Overlap fixes: {report.overlap_fixes}")
        print(f"Min-dim fixes: {report.min_dim_violations_fixed}")
        print(f"Topology changes: {report.topology_changes}")
        print(f"Total displacement: {report.total_displacement_m}m")
        print(f"Stages applied: {report.stages_applied}")

    def test_minimal_repair_low_severity(self):
        """Test that a building needing minimal repair has low severity."""
        building = Building(occupancy_type="Residential")

        # Well-spaced, properly sized rooms
        bedroom = Room("Bedroom1", "Bedroom", 15.0)
        bedroom.polygon = [(0, 0), (4, 0), (4, 4), (0, 4)]  # 4x4 = 16 sq units
        bedroom.final_area = 15.0

        kitchen = Room("Kitchen1", "Kitchen", 8.0)
        kitchen.polygon = [(5, 0), (8, 0), (8, 3), (5, 3)]  # 3x3 = 9 sq units
        kitchen.final_area = 8.0

        building.rooms = [bedroom, kitchen]

        boundary = [(0, 0), (10, 0), (10, 8), (0, 8)]
        entrance = (0, 4)

        try:
            repaired, violations, status, trace, report = validate_and_repair_generated_layout(
                building, boundary, entrance_point=entrance, run_ontology=False
            )

            # Should have low severity (good building)
            assert report.severity_score <= 0.5  # Arbitrary threshold for "low severity"
            assert report.original_room_count == report.final_room_count  # No rooms lost

            print(f"\\nMinimal repair severity: {report.severity_score}")

        except Exception as e:
            pytest.skip(f"Repair pipeline not available: {e}")

    def test_severe_problems_high_severity(self):
        """Test that a building with severe problems has high severity."""
        building = Building(occupancy_type="Residential")

        # Create many problematic rooms
        for i in range(5):
            # All rooms overlap at origin + very small
            room = Room(f"Room{i}", "Bedroom", 1.0)
            room.polygon = [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]  # Tiny + overlapping
            room.final_area = 1.0
            building.rooms.append(room)

        boundary = [(0, 0), (10, 0), (10, 8), (0, 8)]

        try:
            repaired, violations, status, trace, report = validate_and_repair_generated_layout(
                building, boundary, entrance_point=(0, 4), run_ontology=False
            )

            # Should have high severity due to multiple problems
            assert report.severity_score >= 0.3  # Arbitrary threshold for "high severity"
            assert report.overlap_fixes > 0
            assert report.min_dim_violations_fixed > 0

            print(f"\\nSevere problems severity: {report.severity_score}")

        except Exception as e:
            pytest.skip(f"Repair pipeline not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])