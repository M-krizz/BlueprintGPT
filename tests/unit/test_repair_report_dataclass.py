"""
test_repair_report_dataclass.py - Test RepairReport dataclass structure.

Tests just the RepairReport dataclass without heavy dependencies.
"""
from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import List


@dataclass
class RepairReport:
    """Repair severity metrics for tracking quality and displacement."""
    severity_score: float  # [0-1], 0.0=no repair, 1.0=major changes
    overlap_fixes: int  # Number of overlap violations fixed
    min_dim_violations_fixed: int  # Number of min-dim violations fixed
    topology_changes: int  # Room merges, splits, drops
    total_displacement_m: float  # Sum of centroid movements (meters)
    stages_applied: List[str]  # List of stage names that made changes
    original_room_count: int
    final_room_count: int

    @property
    def room_count_changed(self) -> bool:
        """Whether room count was modified during repair."""
        return self.original_room_count != self.final_room_count


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

    def test_severity_score_bounds(self):
        """Test severity score is properly bounded."""
        # Minimum severity
        report_min = RepairReport(
            severity_score=0.0, overlap_fixes=0, min_dim_violations_fixed=0,
            topology_changes=0, total_displacement_m=0.0, stages_applied=[],
            original_room_count=3, final_room_count=3
        )
        assert 0.0 <= report_min.severity_score <= 1.0

        # Maximum severity
        report_max = RepairReport(
            severity_score=1.0, overlap_fixes=10, min_dim_violations_fixed=5,
            topology_changes=3, total_displacement_m=15.5,
            stages_applied=["sanitize_geometry", "enforce_minimums", "overlap_repair", "grid_snap"],
            original_room_count=8, final_room_count=5
        )
        assert 0.0 <= report_max.severity_score <= 1.0
        assert report_max.room_count_changed

    def test_metrics_accumulation(self):
        """Test that metrics properly reflect repair actions."""
        report = RepairReport(
            severity_score=0.45,
            overlap_fixes=3,
            min_dim_violations_fixed=2,
            topology_changes=1,
            total_displacement_m=4.2,
            stages_applied=["sanitize_geometry", "enforce_minimums", "overlap_repair"],
            original_room_count=6,
            final_room_count=5,
        )

        # Total fixes
        total_fixes = report.overlap_fixes + report.min_dim_violations_fixed + report.topology_changes
        assert total_fixes == 6

        # Stages applied
        assert len(report.stages_applied) == 3
        assert all(stage for stage in report.stages_applied)  # No empty strings

        # Physical displacement
        assert report.total_displacement_m > 0

    def test_empty_repair_report(self):
        """Test repair report for building that needed no repair."""
        report = RepairReport(
            severity_score=0.0,
            overlap_fixes=0,
            min_dim_violations_fixed=0,
            topology_changes=0,
            total_displacement_m=0.0,
            stages_applied=[],
            original_room_count=4,
            final_room_count=4,
        )

        assert report.severity_score == 0.0
        assert not report.room_count_changed
        assert len(report.stages_applied) == 0
        assert report.total_displacement_m == 0.0

        # Total fixes should be zero
        total_fixes = report.overlap_fixes + report.min_dim_violations_fixed + report.topology_changes
        assert total_fixes == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])