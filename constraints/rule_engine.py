import json
from pathlib import Path

class RuleEngine:
    def __init__(self, regulation_file):
        with open(regulation_file, 'r') as f:
            self.data = json.load(f)

    def apply_room_rules(self, building):
        regulations = self.data[building.occupancy_type]["rooms"]
        modifications = []

        for room in building.rooms:
            if room.room_type not in regulations:
                continue

            rule = regulations[room.room_type]

            room.set_regulation_constraints(
                rule["min_area"],
                rule["min_width"],
                rule["min_height"]
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
        data = self.data[building.occupancy_type]

        # Compute total usable area
        usable_area = sum(room.final_area for room in building.rooms)

        # Add circulation factor
        circulation_factor = data["circulation_factor"]
        total_area = usable_area * (1 + circulation_factor)

        building.total_area = round(total_area, 2)

        # Compute occupant load
        per_100 = data["occupant_load_per_100sqm"]
        occupant_load = (building.total_area / 100) * per_100
        building.occupant_load = round(occupant_load, 2)

        return building.total_area, building.occupant_load


    def compute_exit_width(self, building):
        data = self.data[building.occupancy_type]
        min_exit_width = data["exit"]["min_width"]

        # For Phase-1, single exit only
        required_width = max(min_exit_width, round(building.occupant_load * 0.01, 2))

        return required_width

    def get_min_door_width(self, occupancy_type):
        return self.data[occupancy_type]["door"]["min_width"]

    def get_max_travel_distance(self, occupancy_type):
        return self.data[occupancy_type]["max_travel_distance"]

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