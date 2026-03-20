import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from constraints.chapter4_helpers import (
    polygon_area,
    plot_bucket,
    get_min_room_dims,
    get_door_dims,
    get_stair_width,
    get_travel_distance_limit,
    get_corridor_min_width,
    get_exit_capacity,
    get_stair_min_width_by_occupancy,
    get_exit_door_dims,
    is_habitable,
    is_service,
)


class RuleEngine:
    def __init__(self, regulation_file):
        with open(regulation_file, 'r') as f:
            self.data = json.load(f)
        self._plot_area_sqm: Optional[float] = None

    def set_plot_area(self, plot_area_sqm: float) -> None:
        """Set the plot area for Chapter-4 bucket selection."""
        self._plot_area_sqm = plot_area_sqm

    def set_plot_area_from_boundary(self, boundary_polygon: List[Tuple[float, float]]) -> float:
        """Compute and set plot area from boundary polygon."""
        area = polygon_area(boundary_polygon)
        self._plot_area_sqm = area
        return area

    def get_plot_bucket(self) -> str:
        """Return 'upto_50sqm' or 'above_50sqm' based on set plot area."""
        if self._plot_area_sqm is None:
            return "above_50sqm"  # Default to larger bucket if unknown
        return plot_bucket(self._plot_area_sqm)

    def apply_room_rules(self, building):
        """Apply Chapter-4 room minimum rules using plot bucket selection."""
        occupancy = building.occupancy_type
        plot_area = self._plot_area_sqm or 100.0  # Default >50 if not set

        # Fallback to top-level rooms if occupancy doesn't have detailed room rules
        regulations = self.data.get(occupancy, {}).get("rooms", {})
        modifications = []

        for room in building.rooms:
            # Get Chapter-4 compliant minimums based on plot bucket
            dims = get_min_room_dims(room.room_type, plot_area, self.data)

            if dims["min_area"] == 0:
                # Room type not in regulation - try top-level fallback
                if room.room_type in regulations:
                    rule = regulations[room.room_type]
                    dims = {
                        "min_area": rule.get("min_area", 0),
                        "min_width": rule.get("min_width", 0),
                        "min_height": rule.get("min_height", 0),
                    }
                else:
                    continue

            room.set_regulation_constraints(
                dims["min_area"],
                dims["min_width"],
                dims["min_height"]
            )

            modified = room.enforce_minimums()

            if modified:
                modifications.append(
                    f"{room.name} area increased to {room.final_area} sq.m (minimum requirement)"
                )

        return modifications

    def allocate_room_areas_from_total(self, building, total_area, unit="sq.ft", strategy="priority_weights"):
        data = self.data[building.occupancy_type]
        room_rules = data["rooms"]
        circulation_factor = data["circulation_factor"]

        conversion = 0.09290304 if unit.lower() == "sq.ft" else 1.0
        total_area_sqm = total_area * conversion
        usable_area_sqm = total_area_sqm / (1 + circulation_factor)

        weights = data.get(
            "allocation_weights",
            {
                "Bedroom": 1.3,
                "LivingRoom": 1.5,
                "Kitchen": 1.1,
                "Bathroom": 0.8,
                "WC": 0.7,
            },
        )

        min_area_sum = 0.0
        weighted_rooms = []
        for room in building.rooms:
            if room.room_type not in room_rules:
                continue
            rule = room_rules[room.room_type]
            min_area = float(rule["min_area"])
            min_area_sum += min_area
            weighted_rooms.append((room, min_area, float(weights.get(room.room_type, 1.0))))

        breakdown = {
            "input_total_area": round(total_area, 2),
            "input_unit": unit,
            "input_total_area_sqm": round(total_area_sqm, 2),
            "circulation_factor": circulation_factor,
            "target_usable_area_sqm": round(usable_area_sqm, 2),
            "min_required_area_sqm": round(min_area_sum, 2),
            "allocation_strategy": strategy,
            "rooms": [],
        }

        modifications = []
        if usable_area_sqm < min_area_sum:
            usable_area_sqm = min_area_sum
            total_area_sqm = usable_area_sqm * (1 + circulation_factor)
            modified_total = round(total_area_sqm / conversion, 2)
            modifications.append(
                f"Total area increased to {modified_total} {unit} to satisfy minimum room requirements"
            )
            breakdown["adjusted_total_area"] = modified_total
            breakdown["adjusted_total_area_sqm"] = round(total_area_sqm, 2)

        surplus = max(0.0, usable_area_sqm - min_area_sum)
        breakdown["surplus_area_sqm"] = round(surplus, 2)

        if strategy == "equal_surplus" and weighted_rooms:
            extra_each = surplus / len(weighted_rooms)
            for room, min_area, weight in weighted_rooms:
                allocated = round(min_area + extra_each, 2)
                room.requested_area = allocated
                breakdown["rooms"].append(
                    {
                        "name": room.name,
                        "type": room.room_type,
                        "weight": round(weight, 3),
                        "min_area": round(min_area, 2),
                        "allocated_area": allocated,
                    }
                )
            return modifications, breakdown

        if strategy == "proportional_to_min_area":
            total_min = sum(min_area for _, min_area, _ in weighted_rooms)
            for room, min_area, weight in weighted_rooms:
                extra = (surplus * min_area / total_min) if total_min > 0 else 0
                allocated = round(min_area + extra, 2)
                room.requested_area = allocated
                breakdown["rooms"].append(
                    {
                        "name": room.name,
                        "type": room.room_type,
                        "weight": round(weight, 3),
                        "min_area": round(min_area, 2),
                        "allocated_area": allocated,
                    }
                )
            return modifications, breakdown

        total_weight = sum(weight for _, _, weight in weighted_rooms)
        for room, min_area, weight in weighted_rooms:
            extra = (surplus * weight / total_weight) if total_weight > 0 else 0
            allocated = round(min_area + extra, 2)
            room.requested_area = allocated
            breakdown["rooms"].append(
                {
                    "name": room.name,
                    "type": room.room_type,
                    "weight": round(weight, 3),
                    "min_area": round(min_area, 2),
                    "allocated_area": allocated,
                }
            )

        return modifications, breakdown

    def compute_building_metrics(self, building):
        """Compute total area and occupant load using Chapter-4 tables."""
        occupancy = building.occupancy_type
        data = self.data.get(occupancy, self.data.get("Residential", {}))

        # Compute total usable area
        usable_area = sum(room.final_area for room in building.rooms)

        # Add circulation factor
        circulation_factor = data.get("circulation_factor", 0.12)
        total_area = usable_area * (1 + circulation_factor)

        building.total_area = round(total_area, 2)

        # Compute occupant load using Chapter-4 table
        occupant_table = self.data.get("chapter4_occupant_load", {})
        per_100 = occupant_table.get(occupancy)

        # Handle nested occupancy values (Assembly, Mercantile)
        if isinstance(per_100, dict):
            # Default to the more conservative (higher) value
            per_100 = per_100.get("without_seating_incl_dining",
                      per_100.get("street_and_sales_basement",
                      data.get("occupant_load_per_100sqm", 8.0)))
        elif per_100 is None:
            per_100 = data.get("occupant_load_per_100sqm", 8.0)

        occupant_load = (building.total_area / 100) * per_100
        building.occupant_load = round(occupant_load, 2)

        return building.total_area, building.occupant_load

    def compute_exit_width(self, building, exit_type: str = "door"):
        """Compute required exit width using Table 4.3 exit capacity.

        Args:
            building: Building object with occupant_load set
            exit_type: 'door', 'stair', or 'ramp'

        Returns:
            Required width in meters
        """
        occupancy = building.occupancy_type
        data = self.data.get(occupancy, self.data.get("Residential", {}))

        # Get minimum exit width from Chapter-4 egress or fallback
        exit_door_dims = get_exit_door_dims(occupancy, self.data)
        min_exit_width = exit_door_dims["min_width"]

        # Get capacity per 50cm unit from Table 4.3
        capacity_per_unit = get_exit_capacity(occupancy, exit_type, self.data)

        # Required 50cm units = occupant_load / capacity_per_unit
        if capacity_per_unit > 0:
            required_units = building.occupant_load / capacity_per_unit
            # Convert units to width: each unit = 0.5m
            required_width = required_units * 0.5
        else:
            required_width = min_exit_width

        # Return max of computed and minimum
        return round(max(min_exit_width, required_width), 2)

    def get_min_door_width(self, occupancy_type: str, room_type: str = None) -> float:
        """Get minimum door width based on occupancy and room type.

        For Residential with plot bucket, returns habitable vs service door widths.
        """
        if room_type and occupancy_type == "Residential":
            plot_area = self._plot_area_sqm or 100.0
            dims = get_door_dims(room_type, plot_area, self.data)
            return dims["min_width"]

        # Fallback to occupancy-level door width
        return self.data.get(occupancy_type, {}).get("door", {}).get("min_width", 0.9)

    def get_max_travel_distance(self, occupancy_type: str) -> float:
        """Get maximum travel distance using Chapter-4 egress table."""
        return get_travel_distance_limit(occupancy_type, self.data)

    def get_corridor_min_width(self, occupancy_type: str, sub_type: str = "") -> float:
        """Get minimum corridor width from Chapter-4 Section 4.8.7."""
        return get_corridor_min_width(occupancy_type, sub_type, self.data)

    def get_stair_min_width(self, occupancy_type: str, sub_type: str = "") -> float:
        """Get minimum stair width from Chapter-4 Section 4.8.6."""
        # For Residential, also consider plot bucket
        if occupancy_type == "Residential" and self._plot_area_sqm is not None:
            plot_stair = get_stair_width(self._plot_area_sqm, self.data)
            occupancy_stair = get_stair_min_width_by_occupancy(occupancy_type, sub_type, self.data)
            return max(plot_stair, occupancy_stair)
        return get_stair_min_width_by_occupancy(occupancy_type, sub_type, self.data)

    def check_chapter4_compliance(self, building, corridor_width: float = None,
                                   door_widths: Dict[str, float] = None,
                                   stair_width: float = None,
                                   travel_distance: float = None) -> Dict:
        """Comprehensive Chapter-4 compliance check.

        Returns a dict with:
          - violations: list of {rule, required, actual, message}
          - compliant: bool
          - checks: dict of individual check results
        """
        occupancy = building.occupancy_type
        violations = []
        checks = {}

        # Room area/width/height checks
        plot_area = self._plot_area_sqm or 100.0
        room_violations = []
        for room in building.rooms:
            dims = get_min_room_dims(room.room_type, plot_area, self.data)
            if dims["min_area"] > 0:
                actual_area = getattr(room, "final_area", getattr(room, "requested_area", 0))
                if actual_area < dims["min_area"] - 0.01:
                    room_violations.append({
                        "rule": f"room_area_{room.name}",
                        "required": dims["min_area"],
                        "actual": round(actual_area, 2),
                        "message": f"{room.name} ({room.room_type}): area {actual_area:.2f} < min {dims['min_area']}"
                    })
        checks["room_areas"] = len(room_violations) == 0
        violations.extend(room_violations)

        # Corridor width check
        if corridor_width is not None:
            min_corridor = self.get_corridor_min_width(occupancy)
            checks["corridor_width"] = corridor_width >= min_corridor - 0.01
            if not checks["corridor_width"]:
                violations.append({
                    "rule": "corridor_width",
                    "required": min_corridor,
                    "actual": round(corridor_width, 2),
                    "message": f"Corridor width {corridor_width:.2f}m < min {min_corridor}m"
                })

        # Stair width check
        if stair_width is not None:
            min_stair = self.get_stair_min_width(occupancy)
            checks["stair_width"] = stair_width >= min_stair - 0.01
            if not checks["stair_width"]:
                violations.append({
                    "rule": "stair_width",
                    "required": min_stair,
                    "actual": round(stair_width, 2),
                    "message": f"Stair width {stair_width:.2f}m < min {min_stair}m"
                })

        # Travel distance check
        if travel_distance is not None:
            max_travel = self.get_max_travel_distance(occupancy)
            checks["travel_distance"] = travel_distance <= max_travel + 0.01
            if not checks["travel_distance"]:
                violations.append({
                    "rule": "travel_distance",
                    "required": max_travel,
                    "actual": round(travel_distance, 2),
                    "message": f"Travel distance {travel_distance:.2f}m > max {max_travel}m"
                })

        # Door width checks
        if door_widths:
            door_violations = []
            for room_type, width in door_widths.items():
                min_door = self.get_min_door_width(occupancy, room_type)
                if width < min_door - 0.01:
                    door_violations.append({
                        "rule": f"door_width_{room_type}",
                        "required": min_door,
                        "actual": round(width, 2),
                        "message": f"{room_type} door width {width:.2f}m < min {min_door}m"
                    })
            checks["door_widths"] = len(door_violations) == 0
            violations.extend(door_violations)

        # Exit width check
        if hasattr(building, "occupant_load"):
            required_exit = self.compute_exit_width(building, "door")
            exit_dims = get_exit_door_dims(occupancy, self.data)
            checks["exit_width_required"] = required_exit
            checks["exit_width_min"] = exit_dims["min_width"]

        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "checks": checks,
            "plot_bucket": self.get_plot_bucket(),
            "plot_area_sqm": self._plot_area_sqm,
        }

    def preflight_validate_spec(self, spec):
        occupancy = spec.get("occupancy", "Residential")
        errors = []
        warnings = []

        if occupancy not in self.data:
            return {
                "valid": False,
                "errors": [f"Unsupported occupancy '{occupancy}' in regulation dataset"],
                "warnings": [],
            }

        occupancy_data = self.data[occupancy]
        room_rules = occupancy_data.get("rooms", {})
        circulation_factor = occupancy_data.get("circulation_factor", 0.0)

        rooms = spec.get("rooms", [])
        if not rooms:
            errors.append("Spec has no rooms")
            return {"valid": False, "errors": errors, "warnings": warnings}

        min_required_area = 0.0
        unsupported_types = set()
        for room in rooms:
            room_type = room.get("type")
            if room_type not in room_rules:
                unsupported_types.add(room_type)
                continue
            min_required_area += float(room_rules[room_type].get("min_area", 0.0))

        if unsupported_types:
            errors.append(
                "Unsupported room types for occupancy "
                f"{occupancy}: {', '.join(sorted(str(t) for t in unsupported_types))}"
            )

        total_area_input = spec.get("total_area")
        area_unit = str(spec.get("area_unit", "sq.ft")).lower()
        conversion = 0.09290304 if area_unit == "sq.ft" else 1.0
        if total_area_input is not None:
            try:
                total_sqm = float(total_area_input) * conversion
                required_total_sqm = min_required_area * (1.0 + circulation_factor)
                if total_sqm + 1e-6 < required_total_sqm:
                    errors.append(
                        f"Total area {round(total_sqm, 2)} sq.m is below minimum required "
                        f"{round(required_total_sqm, 2)} sq.m including circulation"
                    )
            except Exception:
                errors.append("total_area must be numeric")

        boundary_polygon = spec.get("boundary_polygon")
        if boundary_polygon and len(boundary_polygon) >= 3:
            boundary_area = self._polygon_area(boundary_polygon)
            if boundary_area <= 0:
                errors.append("Boundary polygon area is zero or invalid")
            elif total_area_input is not None:
                try:
                    total_sqm = float(total_area_input) * conversion
                    mismatch = abs(boundary_area - total_sqm)
                    if total_sqm > 0 and (mismatch / total_sqm) > 0.18:
                        warnings.append(
                            "Boundary area and target total area differ significantly: "
                            f"boundary={round(boundary_area, 2)} sq.m, target={round(total_sqm, 2)} sq.m"
                        )
                except Exception:
                    pass

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "min_required_room_area_sqm": round(min_required_area, 2),
            "circulation_factor": circulation_factor,
        }

    @staticmethod
    def _polygon_area(points):
        if not points or len(points) < 3:
            return 0.0
        area = 0.0
        n = len(points)
        for idx in range(n):
            x1, y1 = points[idx]
            x2, y2 = points[(idx + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0