"""
chapter4_validator.py — Floor-plan-level Chapter-4 compliance validator.

This module provides a unified interface for checking floor plan compliance
against NBC India 2016 Chapter 4 bye-laws. It bridges the Chapter-4 ground truth
(from ontology/regulation_data.json) with the runtime compliance system.

Use this validator for:
- Room size minimums (Table 4.2 with plot bucket selection)
- Egress compliance (travel distance, corridor width, stair width)
- Door dimensions (exit vs internal, habitable vs service)
- Lighting/ventilation requirements
- Exit capacity calculations

This complements the IBC-based ground_truth/validator.py which handles:
- Building-level occupancy classification
- Construction type restrictions
- Height/area limitations
- Fire separation requirements
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from constraints.chapter4_helpers import (
    load_regulation_data,
    plot_bucket,
    get_bucket_rules,
    get_min_room_dims,
    get_door_dims,
    get_stair_width,
    get_travel_distance_limit,
    get_corridor_min_width,
    get_exit_capacity,
    get_stair_min_width_by_occupancy,
    get_exit_door_dims,
    get_opening_ratio,
    get_max_lighting_depth,
    get_kitchen_window_min,
    get_bathroom_vent_min,
    get_shaft_min,
    get_occupant_load_per_100sqm,
    get_chapter4_summary,
    is_habitable,
    is_service,
    chapter4_room_category,
)


@dataclass
class Chapter4Violation:
    """A single Chapter-4 compliance violation."""
    rule_id: str
    rule_section: str
    severity: str  # "CRITICAL", "MAJOR", "MINOR", "INFO"
    message: str
    required_value: Any
    actual_value: Any
    room_name: Optional[str] = None
    remediation: Optional[str] = None


@dataclass
class Chapter4Result:
    """Result of Chapter-4 compliance validation."""
    compliant: bool
    violations: List[Chapter4Violation]
    checks_performed: Dict[str, bool]
    summary: Dict[str, Any]
    plot_bucket: str
    plot_area_sqm: Optional[float]


class Chapter4Validator:
    """
    Floor-plan-level compliance validator using Chapter-4 ground truth.

    Example usage::

        validator = Chapter4Validator(plot_area_sqm=75.0)
        result = validator.validate_building(building, corridor_width=1.2, travel_distance=18.0)
        if not result.compliant:
            for v in result.violations:
                print(f"{v.rule_id}: {v.message}")
    """

    def __init__(
        self,
        plot_area_sqm: Optional[float] = None,
        occupancy: str = "Residential",
        building_height_m: float = 9.0,
        regulation_file: Optional[str] = None,
    ):
        self.plot_area_sqm = plot_area_sqm or 100.0
        self.occupancy = occupancy
        self.building_height_m = building_height_m
        self.reg = load_regulation_data(regulation_file)
        self._bucket = plot_bucket(self.plot_area_sqm)

    def validate_building(
        self,
        building,
        *,
        corridor_width: Optional[float] = None,
        stair_width: Optional[float] = None,
        travel_distance: Optional[float] = None,
        door_widths: Optional[Dict[str, float]] = None,
        opening_areas: Optional[Dict[str, float]] = None,
    ) -> Chapter4Result:
        """
        Validate a Building object against Chapter-4 rules.

        Parameters
        ----------
        building : Building
            Core building object with rooms.
        corridor_width : float, optional
            Measured corridor width (m).
        stair_width : float, optional
            Measured stair width (m).
        travel_distance : float, optional
            Measured max travel distance to exit (m).
        door_widths : dict, optional
            Dict of {room_name: door_width_m}.
        opening_areas : dict, optional
            Dict of {room_name: opening_area_sqm} for lighting/vent check.

        Returns
        -------
        Chapter4Result
            Validation result with violations and summary.
        """
        violations: List[Chapter4Violation] = []
        checks: Dict[str, bool] = {}

        # 1. Room dimensions
        room_violations = self._check_room_dimensions(building)
        violations.extend(room_violations)
        checks["room_dimensions"] = len(room_violations) == 0

        # 2. Corridor width
        if corridor_width is not None:
            corr_violations = self._check_corridor_width(corridor_width)
            violations.extend(corr_violations)
            checks["corridor_width"] = len(corr_violations) == 0

        # 3. Stair width
        if stair_width is not None:
            stair_violations = self._check_stair_width(stair_width)
            violations.extend(stair_violations)
            checks["stair_width"] = len(stair_violations) == 0

        # 4. Travel distance
        if travel_distance is not None:
            travel_violations = self._check_travel_distance(travel_distance)
            violations.extend(travel_violations)
            checks["travel_distance"] = len(travel_violations) == 0

        # 5. Door widths
        if door_widths:
            door_violations = self._check_door_widths(building, door_widths)
            violations.extend(door_violations)
            checks["door_widths"] = len(door_violations) == 0

        # 6. Lighting/ventilation openings
        if opening_areas:
            vent_violations = self._check_openings(building, opening_areas)
            violations.extend(vent_violations)
            checks["openings"] = len(vent_violations) == 0

        # Build summary
        summary = get_chapter4_summary(
            occupancy=self.occupancy,
            plot_area_sqm=self.plot_area_sqm,
            building_height_m=self.building_height_m,
            reg=self.reg,
        )

        compliant = len([v for v in violations if v.severity in ("CRITICAL", "MAJOR")]) == 0

        return Chapter4Result(
            compliant=compliant,
            violations=violations,
            checks_performed=checks,
            summary=summary,
            plot_bucket=self._bucket,
            plot_area_sqm=self.plot_area_sqm,
        )

    def _check_room_dimensions(self, building) -> List[Chapter4Violation]:
        """Check all rooms against Chapter-4 Table 4.2 minimums."""
        violations = []
        for room in getattr(building, "rooms", []):
            room_type = getattr(room, "room_type", None)
            if not room_type:
                continue

            dims = get_min_room_dims(room_type, self.plot_area_sqm, self.reg)
            if dims["min_area"] == 0:
                continue  # Not regulated

            # Get actual dimensions
            actual_area = getattr(room, "final_area", None) or getattr(room, "requested_area", 0)
            actual_width = self._min_dimension(room)

            # Area check
            if actual_area < dims["min_area"] - 0.01:
                violations.append(Chapter4Violation(
                    rule_id=f"room_area_{room.name}",
                    rule_section="Table 4.2",
                    severity="MAJOR",
                    message=f"{room.name} ({room_type}): area {actual_area:.2f} sq.m < min {dims['min_area']} sq.m",
                    required_value=dims["min_area"],
                    actual_value=round(actual_area, 2),
                    room_name=room.name,
                    remediation=f"Increase room area to at least {dims['min_area']} sq.m",
                ))

            # Width check
            if actual_width > 0 and actual_width < dims["min_width"] - 0.01:
                violations.append(Chapter4Violation(
                    rule_id=f"room_width_{room.name}",
                    rule_section="Table 4.2",
                    severity="MAJOR",
                    message=f"{room.name} ({room_type}): width {actual_width:.2f}m < min {dims['min_width']}m",
                    required_value=dims["min_width"],
                    actual_value=round(actual_width, 2),
                    room_name=room.name,
                    remediation=f"Increase room width to at least {dims['min_width']}m",
                ))

        return violations

    def _check_corridor_width(self, corridor_width: float) -> List[Chapter4Violation]:
        """Check corridor width against Section 4.8.7."""
        violations = []
        min_width = get_corridor_min_width(self.occupancy, reg=self.reg)

        if corridor_width < min_width - 0.01:
            violations.append(Chapter4Violation(
                rule_id="corridor_width",
                rule_section="Section 4.8.7",
                severity="MAJOR",
                message=f"Corridor width {corridor_width:.2f}m < min {min_width}m",
                required_value=min_width,
                actual_value=round(corridor_width, 2),
                remediation=f"Widen corridor to at least {min_width}m",
            ))

        return violations

    def _check_stair_width(self, stair_width: float) -> List[Chapter4Violation]:
        """Check stair width against Section 4.8.6 and plot bucket."""
        violations = []

        # By occupancy
        occ_min = get_stair_min_width_by_occupancy(self.occupancy, reg=self.reg)

        # By plot bucket (Residential only)
        plot_min = get_stair_width(self.plot_area_sqm, self.reg) if self.occupancy == "Residential" else 0

        min_width = max(occ_min, plot_min)

        if stair_width < min_width - 0.01:
            violations.append(Chapter4Violation(
                rule_id="stair_width",
                rule_section="Section 4.8.6 / Table 4.2",
                severity="MAJOR",
                message=f"Stair width {stair_width:.2f}m < min {min_width}m",
                required_value=min_width,
                actual_value=round(stair_width, 2),
                remediation=f"Widen stairway to at least {min_width}m",
            ))

        return violations

    def _check_travel_distance(self, travel_distance: float) -> List[Chapter4Violation]:
        """Check travel distance against Section 4.8.4."""
        violations = []
        max_travel = get_travel_distance_limit(self.occupancy, self.reg)

        if travel_distance > max_travel + 0.01:
            violations.append(Chapter4Violation(
                rule_id="travel_distance",
                rule_section="Section 4.8.4",
                severity="CRITICAL",
                message=f"Travel distance {travel_distance:.2f}m > max {max_travel}m",
                required_value=max_travel,
                actual_value=round(travel_distance, 2),
                remediation="Add additional exit or reconfigure layout to reduce travel distance",
            ))

        return violations

    def _check_door_widths(self, building, door_widths: Dict[str, float]) -> List[Chapter4Violation]:
        """Check door widths against Table 4.2 and exit door requirements."""
        violations = []

        for room in getattr(building, "rooms", []):
            room_name = getattr(room, "name", None)
            room_type = getattr(room, "room_type", None)
            if not room_name or room_name not in door_widths:
                continue

            actual_width = door_widths[room_name]
            door_dims = get_door_dims(room_type, self.plot_area_sqm, self.reg)
            min_width = door_dims["min_width"]

            if actual_width < min_width - 0.01:
                violations.append(Chapter4Violation(
                    rule_id=f"door_width_{room_name}",
                    rule_section="Table 4.2",
                    severity="MAJOR",
                    message=f"Door to {room_name}: width {actual_width:.2f}m < min {min_width}m",
                    required_value=min_width,
                    actual_value=round(actual_width, 2),
                    room_name=room_name,
                    remediation=f"Widen door to at least {min_width}m",
                ))

        return violations

    def _check_openings(self, building, opening_areas: Dict[str, float]) -> List[Chapter4Violation]:
        """Check lighting/ventilation openings."""
        violations = []
        min_ratio = get_opening_ratio(self.reg)
        kitchen_min = get_kitchen_window_min(self.reg)
        bath_vent_min = get_bathroom_vent_min(self.reg)

        for room in getattr(building, "rooms", []):
            room_name = getattr(room, "name", None)
            room_type = getattr(room, "room_type", None)
            if not room_name or room_name not in opening_areas:
                continue

            actual_opening = opening_areas[room_name]
            floor_area = getattr(room, "final_area", 0) or getattr(room, "requested_area", 0)

            # Habitable rooms and kitchens need 1/10 ratio
            if is_habitable(room_type) or room_type == "Kitchen":
                required = floor_area * min_ratio
                if actual_opening < required - 0.01:
                    violations.append(Chapter4Violation(
                        rule_id=f"opening_ratio_{room_name}",
                        rule_section="Section 4.9",
                        severity="MINOR",
                        message=f"{room_name}: opening area {actual_opening:.2f} sq.m < required {required:.2f} sq.m (1/10 floor area)",
                        required_value=round(required, 2),
                        actual_value=round(actual_opening, 2),
                        room_name=room_name,
                    ))

            # Kitchen window minimum
            if room_type == "Kitchen" and actual_opening < kitchen_min - 0.01:
                violations.append(Chapter4Violation(
                    rule_id=f"kitchen_window_{room_name}",
                    rule_section="Section 4.9",
                    severity="MINOR",
                    message=f"Kitchen window {actual_opening:.2f} sq.m < min {kitchen_min} sq.m",
                    required_value=kitchen_min,
                    actual_value=round(actual_opening, 2),
                    room_name=room_name,
                ))

            # Bathroom/WC vent minimum
            if room_type in ("Bathroom", "WC", "BathWC") and actual_opening < bath_vent_min - 0.01:
                violations.append(Chapter4Violation(
                    rule_id=f"bathroom_vent_{room_name}",
                    rule_section="Section 4.9",
                    severity="MINOR",
                    message=f"{room_name} vent opening {actual_opening:.2f} sq.m < min {bath_vent_min} sq.m",
                    required_value=bath_vent_min,
                    actual_value=round(actual_opening, 2),
                    room_name=room_name,
                ))

        return violations

    def _min_dimension(self, room) -> float:
        """Get minimum dimension (width or height) of a room from its polygon."""
        polygon = getattr(room, "polygon", None)
        if not polygon or len(polygon) < 3:
            return 0.0

        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        return min(w, h)


# ── Convenience function for compliance reports ──────────────────────────────

def validate_floor_plan_chapter4(
    building,
    plot_area_sqm: float = 100.0,
    occupancy: str = "Residential",
    corridor_width: Optional[float] = None,
    stair_width: Optional[float] = None,
    travel_distance: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convenience function to validate a floor plan against Chapter-4 rules.

    Returns a dict suitable for inclusion in compliance reports.
    """
    validator = Chapter4Validator(plot_area_sqm=plot_area_sqm, occupancy=occupancy)
    result = validator.validate_building(
        building,
        corridor_width=corridor_width,
        stair_width=stair_width,
        travel_distance=travel_distance,
    )

    return {
        "chapter4_compliant": result.compliant,
        "chapter4_violations": [
            {
                "rule_id": v.rule_id,
                "rule_section": v.rule_section,
                "severity": v.severity,
                "message": v.message,
                "required": v.required_value,
                "actual": v.actual_value,
                "room": v.room_name,
            }
            for v in result.violations
        ],
        "chapter4_checks": result.checks_performed,
        "chapter4_plot_bucket": result.plot_bucket,
        "chapter4_summary": result.summary,
    }
