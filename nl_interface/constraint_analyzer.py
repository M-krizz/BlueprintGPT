"""
constraint_analyzer.py - Intelligent constraint analysis for floor plan generation.

This module analyzes system constraints to produce proper model inputs based on:
1. Layout type (2BHK, 3BHK, 4BHK, studio, etc.)
2. Plot size (determines minimum room dimensions)
3. Building regulations (Indian NBC 2016 Chapter 4)
4. Room adjacency and functional requirements

The ConstraintAnalyzer acts as the intelligent bridge between user input
and model requirements, ensuring generated layouts are compliant and realistic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Configuration ─────────────────────────────────────────────────────────────

# Path to regulation data
REGULATION_DATA_PATH = os.getenv(
    "REGULATION_DATA_PATH",
    str(Path(__file__).parent.parent / "ontology" / "regulation_data.json")
)

# ── Layout Type Definitions ───────────────────────────────────────────────────

# Standard Indian BHK layouts with recommended configurations
BHK_LAYOUTS = {
    "1BHK": {
        "description": "1 Bedroom, 1 Hall, 1 Kitchen",
        "rooms": [
            {"type": "Bedroom", "count": 1, "priority": "high"},
            {"type": "LivingRoom", "count": 1, "priority": "high"},
            {"type": "Kitchen", "count": 1, "priority": "high"},
            {"type": "Bathroom", "count": 1, "priority": "high"},
        ],
        "min_total_area_sqm": 35,
        "recommended_area_sqm": 45,
        "max_total_area_sqm": 60,
        "plot_bucket": "upto_50sqm",
        "adjacency_requirements": [
            {"source": "Bathroom", "target": "Bedroom", "relation": "near_to", "weight": 0.9},
            {"source": "Kitchen", "target": "LivingRoom", "relation": "near_to", "weight": 0.8},
        ],
    },
    "2BHK": {
        "description": "2 Bedrooms, 1 Hall, 1 Kitchen",
        "rooms": [
            {"type": "Bedroom", "count": 2, "priority": "high"},
            {"type": "LivingRoom", "count": 1, "priority": "high"},
            {"type": "Kitchen", "count": 1, "priority": "high"},
            {"type": "Bathroom", "count": 2, "priority": "high"},
        ],
        "min_total_area_sqm": 55,
        "recommended_area_sqm": 75,
        "max_total_area_sqm": 100,
        "plot_bucket": "above_50sqm",
        "adjacency_requirements": [
            {"source": "Bathroom", "target": "Bedroom", "relation": "near_to", "weight": 0.9},
            {"source": "Kitchen", "target": "LivingRoom", "relation": "near_to", "weight": 0.8},
            {"source": "Bedroom", "target": "Bedroom", "relation": "separate", "weight": 0.6},
        ],
    },
    "3BHK": {
        "description": "3 Bedrooms, 1 Hall, 1 Kitchen",
        "rooms": [
            {"type": "Bedroom", "count": 3, "priority": "high"},
            {"type": "LivingRoom", "count": 1, "priority": "high"},
            {"type": "Kitchen", "count": 1, "priority": "high"},
            {"type": "Bathroom", "count": 2, "priority": "high"},
        ],
        "min_total_area_sqm": 85,
        "recommended_area_sqm": 110,
        "max_total_area_sqm": 150,
        "plot_bucket": "above_50sqm",
        "adjacency_requirements": [
            {"source": "Bathroom", "target": "Bedroom", "relation": "near_to", "weight": 0.9},
            {"source": "Kitchen", "target": "LivingRoom", "relation": "near_to", "weight": 0.8},
            {"source": "Bedroom", "target": "Bedroom", "relation": "separate", "weight": 0.6},
            {"source": "LivingRoom", "target": "Bedroom", "relation": "buffer_zone", "weight": 0.5},
        ],
    },
    "4BHK": {
        "description": "4 Bedrooms, 1 Hall, 1 Kitchen",
        "rooms": [
            {"type": "Bedroom", "count": 4, "priority": "high"},
            {"type": "LivingRoom", "count": 1, "priority": "high"},
            {"type": "Kitchen", "count": 1, "priority": "high"},
            {"type": "Bathroom", "count": 2, "priority": "high"},
            {"type": "DiningRoom", "count": 1, "priority": "medium"},
        ],
        "min_total_area_sqm": 120,
        "recommended_area_sqm": 150,
        "max_total_area_sqm": 200,
        "plot_bucket": "above_50sqm",
        "adjacency_requirements": [
            {"source": "Bathroom", "target": "Bedroom", "relation": "near_to", "weight": 0.9},
            {"source": "Kitchen", "target": "DiningRoom", "relation": "adjacent_to", "weight": 0.95},
            {"source": "DiningRoom", "target": "LivingRoom", "relation": "near_to", "weight": 0.8},
            {"source": "Bedroom", "target": "Bedroom", "relation": "separate", "weight": 0.6},
        ],
    },
    "STUDIO": {
        "description": "Studio apartment with combined living/sleeping",
        "rooms": [
            {"type": "LivingRoom", "count": 1, "priority": "high"},  # Acts as combined space
            {"type": "Kitchen", "count": 1, "priority": "high"},
            {"type": "Bathroom", "count": 1, "priority": "high"},
        ],
        "min_total_area_sqm": 25,
        "recommended_area_sqm": 35,
        "max_total_area_sqm": 45,
        "plot_bucket": "upto_50sqm",
        "adjacency_requirements": [
            {"source": "Kitchen", "target": "LivingRoom", "relation": "open_to", "weight": 0.9},
            {"source": "Bathroom", "target": "LivingRoom", "relation": "near_to", "weight": 0.7},
        ],
    },
}

# Room spacing requirements to prevent overlap
ROOM_SPACING = {
    "min_gap_meters": 0.1,  # Minimum gap between rooms (wall thickness)
    "corridor_width": 1.0,  # Standard corridor width
    "door_clearance": 0.9,  # Door swing clearance
}


class ConstraintAnalyzer:
    """
    Analyzes system constraints to produce proper model inputs.

    This class acts as the intelligent coordinator between user input and
    model requirements, ensuring generated layouts are compliant and realistic.
    """

    def __init__(self, regulation_path: str = None):
        """Initialize with regulation data."""
        self.regulation_path = regulation_path or REGULATION_DATA_PATH
        self.regulations = self._load_regulations()
        self._plot_area_sqm: Optional[float] = None

    def _load_regulations(self) -> Dict:
        """Load regulation data from JSON file."""
        try:
            with open(self.regulation_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[WARNING] Failed to load regulations: {e}")
            return {}

    def set_plot_area(self, area_sqm: float) -> None:
        """Set the plot area for constraint calculations."""
        self._plot_area_sqm = area_sqm

    def get_plot_bucket(self, area_sqm: float = None) -> str:
        """Determine plot bucket (upto_50sqm or above_50sqm)."""
        area = area_sqm or self._plot_area_sqm or 100.0
        return "upto_50sqm" if area <= 50.0 else "above_50sqm"

    def get_layout_requirements(self, layout_type: str) -> Dict[str, Any]:
        """
        Get complete requirements for a layout type (2BHK, 3BHK, etc.).

        Returns:
            Dict with rooms, areas, adjacencies, and constraint metadata.
        """
        layout_type = layout_type.upper().replace(" ", "")

        if layout_type not in BHK_LAYOUTS:
            # Try to infer from room count
            if layout_type.endswith("BHK"):
                try:
                    bedroom_count = int(layout_type[:-3])
                    layout_type = f"{bedroom_count}BHK"
                except ValueError:
                    layout_type = "2BHK"  # Default
            else:
                layout_type = "2BHK"  # Default

        if layout_type not in BHK_LAYOUTS:
            layout_type = "2BHK"  # Safe fallback

        layout = BHK_LAYOUTS[layout_type]
        plot_bucket = layout.get("plot_bucket", "above_50sqm")

        # Enhance with regulation-based minimums
        enhanced_rooms = []
        for room_spec in layout["rooms"]:
            room_type = room_spec["type"]
            count = room_spec["count"]

            # Get regulation minimums
            min_dims = self._get_room_minimums(room_type, plot_bucket)

            for i in range(count):
                room_name = f"{room_type}_{i+1}" if count > 1 else room_type
                enhanced_rooms.append({
                    "type": room_type,
                    "name": room_name,
                    "min_area_sqm": min_dims["min_area"],
                    "min_width_m": min_dims["min_width"],
                    "min_height_m": min_dims["min_height"],
                    "priority": room_spec.get("priority", "medium"),
                    "recommended_area_sqm": min_dims["min_area"] * 1.3,  # 30% buffer
                })

        return {
            "layout_type": layout_type,
            "description": layout["description"],
            "rooms": enhanced_rooms,
            "room_count": len(enhanced_rooms),
            "min_total_area_sqm": layout["min_total_area_sqm"],
            "recommended_area_sqm": layout["recommended_area_sqm"],
            "max_total_area_sqm": layout["max_total_area_sqm"],
            "plot_bucket": plot_bucket,
            "adjacency_requirements": layout.get("adjacency_requirements", []),
            "spacing": ROOM_SPACING,
            "constraints_source": "regulation_data.json + BHK_LAYOUTS",
        }

    def _get_room_minimums(self, room_type: str, plot_bucket: str) -> Dict[str, float]:
        """Get minimum dimensions for a room type based on plot bucket."""
        # Try Chapter 4 data first
        chapter4 = self.regulations.get("Residential", {}).get("chapter4_residential", {})
        bucket_data = chapter4.get("plot_buckets", {}).get(plot_bucket, {})
        room_rules = bucket_data.get("rooms", {})

        # Map room types to Chapter 4 categories
        category_map = {
            "Bedroom": "Habitable",
            "LivingRoom": "Habitable",
            "DrawingRoom": "Habitable",
            "DiningRoom": "Habitable",
            "Study": "Habitable",
            "Kitchen": "Kitchen",
            "Bathroom": "Bathroom",
            "WC": "WC",
            "BathWC": "BathWC",
            "Pantry": "Pantry",
            "Garage": "Garage",
        }

        category = category_map.get(room_type, "Habitable")

        if category in room_rules:
            rule = room_rules[category]
            return {
                "min_area": rule.get("min_area", 9.5),
                "min_width": rule.get("min_width", 2.4),
                "min_height": rule.get("min_height", 2.75),
            }

        # Fallback to top-level Residential room rules
        top_level_rooms = self.regulations.get("Residential", {}).get("rooms", {})
        if room_type in top_level_rooms:
            rule = top_level_rooms[room_type]
            return {
                "min_area": rule.get("min_area", 9.5),
                "min_width": rule.get("min_width", 2.4),
                "min_height": rule.get("min_height", 2.75),
            }

        # Ultimate fallback
        return {
            "min_area": 9.5,
            "min_width": 2.4,
            "min_height": 2.75,
        }

    def calculate_optimal_dimensions(self, layout_type: str,
                                     user_total_area: float = None) -> Dict[str, Any]:
        """
        Calculate optimal plot dimensions for a layout type.

        Args:
            layout_type: "2BHK", "3BHK", etc.
            user_total_area: Optional user-specified total area in sqm

        Returns:
            Dict with width, height, and area calculations.
        """
        requirements = self.get_layout_requirements(layout_type)

        # Calculate minimum required area from room minimums
        room_areas_sum = sum(r["min_area_sqm"] for r in requirements["rooms"])

        # Add circulation factor (35% for corridors, walls, etc.)
        circulation_factor = 0.35
        min_area_with_circulation = room_areas_sum * (1 + circulation_factor)

        # Determine target area
        if user_total_area and user_total_area >= min_area_with_circulation:
            target_area = user_total_area
        else:
            target_area = max(
                min_area_with_circulation,
                requirements["recommended_area_sqm"]
            )

        # Calculate dimensions with good aspect ratio (1.0 to 1.5)
        # Target aspect ratio based on room count
        room_count = requirements["room_count"]
        if room_count <= 4:
            aspect_ratio = 1.0  # Square for smaller layouts
        elif room_count <= 6:
            aspect_ratio = 1.2
        else:
            aspect_ratio = 1.4  # More rectangular for larger layouts

        # width * height = target_area
        # width / height = aspect_ratio
        # => width = sqrt(target_area * aspect_ratio)
        import math
        width = math.sqrt(target_area * aspect_ratio)
        height = target_area / width

        # Round to reasonable precision
        width = round(width, 1)
        height = round(height, 1)
        actual_area = width * height

        return {
            "width_m": width,
            "height_m": height,
            "area_sqm": round(actual_area, 2),
            "aspect_ratio": round(width / height, 2),
            "room_areas_sum": round(room_areas_sum, 2),
            "circulation_area": round(actual_area - room_areas_sum, 2),
            "efficiency_percent": round((room_areas_sum / actual_area) * 100, 1),
            "min_required_area": round(min_area_with_circulation, 2),
            "area_sufficient": actual_area >= min_area_with_circulation,
        }

    def validate_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a specification against constraints.

        Returns:
            Dict with validation results, warnings, and suggestions.
        """
        errors = []
        warnings = []
        suggestions = []

        rooms = spec.get("rooms", [])
        if not rooms:
            errors.append("No rooms specified")
            return {"valid": False, "errors": errors, "warnings": warnings, "suggestions": suggestions}

        # Determine layout type from rooms
        bedroom_count = sum(
            r.get("count", 1) for r in rooms
            if r.get("type") == "Bedroom"
        )
        layout_type = f"{bedroom_count}BHK" if bedroom_count > 0 else "STUDIO"

        # Get requirements for this layout type
        requirements = self.get_layout_requirements(layout_type)

        # Check if all required room types are present
        required_types = {r["type"] for r in requirements["rooms"]}
        provided_types = {r.get("type") for r in rooms}

        missing_types = required_types - provided_types
        if missing_types:
            warnings.append(f"Missing recommended room types for {layout_type}: {', '.join(missing_types)}")

        # Check total area
        boundary = spec.get("boundary_polygon")
        if boundary and len(boundary) >= 3:
            boundary_area = self._polygon_area(boundary)
            min_required = requirements["min_total_area_sqm"]

            if boundary_area < min_required:
                errors.append(
                    f"Boundary area ({boundary_area:.1f} sqm) is below minimum "
                    f"required ({min_required:.1f} sqm) for {layout_type}"
                )
                suggestions.append(
                    f"Increase boundary to at least {min_required:.1f} sqm "
                    f"or consider a smaller layout type"
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "inferred_layout_type": layout_type,
            "requirements": requirements,
        }

    def enhance_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance a basic spec with constraint-aware requirements.

        This is the key function that transforms user input into proper model input.
        """
        rooms = spec.get("rooms", [])

        # Determine layout type from rooms
        bedroom_count = sum(
            r.get("count", 1) for r in rooms
            if r.get("type") == "Bedroom"
        )
        layout_type = f"{bedroom_count}BHK" if bedroom_count > 0 else "STUDIO"

        # Get full requirements
        requirements = self.get_layout_requirements(layout_type)

        # Calculate optimal dimensions if not provided
        boundary = spec.get("boundary_polygon")
        if not boundary:
            dimensions = self.calculate_optimal_dimensions(layout_type)
            # Create rectangular boundary
            w, h = dimensions["width_m"], dimensions["height_m"]
            spec["boundary_polygon"] = [[0, 0], [w, 0], [w, h], [0, h]]
            spec["auto_dimensions"] = dimensions

        # Enhance room specifications with minimums
        enhanced_rooms = []
        for room_spec in rooms:
            room_type = room_spec.get("type")
            count = room_spec.get("count", 1)

            # Find matching requirement
            matching_reqs = [r for r in requirements["rooms"] if r["type"] == room_type]
            if matching_reqs:
                req = matching_reqs[0]
                for i in range(count):
                    enhanced_rooms.append({
                        "type": room_type,
                        "count": 1,
                        "min_area_sqm": req["min_area_sqm"],
                        "min_width_m": req["min_width_m"],
                        "recommended_area_sqm": req["recommended_area_sqm"],
                    })
            else:
                # Keep original spec
                enhanced_rooms.append(room_spec)

        # Add adjacency requirements if not specified
        if not spec.get("adjacency"):
            spec["adjacency"] = requirements.get("adjacency_requirements", [])

        spec["rooms"] = enhanced_rooms
        spec["layout_type"] = layout_type
        spec["constraint_metadata"] = {
            "source": "ConstraintAnalyzer",
            "plot_bucket": requirements["plot_bucket"],
            "min_total_area_sqm": requirements["min_total_area_sqm"],
            "room_count": len(enhanced_rooms),
            "spacing": requirements["spacing"],
        }

        return spec

    def get_constraint_summary(self) -> str:
        """
        Generate a human-readable summary of all constraints.
        Used to inform Gemini about the constraint system.
        """
        lines = [
            "=== BlueprintGPT Constraint System ===",
            "",
            "## Layout Types and Room Requirements:",
        ]

        for layout_type, config in BHK_LAYOUTS.items():
            lines.append(f"\n### {layout_type}: {config['description']}")
            lines.append(f"- Area range: {config['min_total_area_sqm']}-{config['max_total_area_sqm']} sqm")
            lines.append(f"- Recommended: {config['recommended_area_sqm']} sqm")
            lines.append(f"- Plot bucket: {config['plot_bucket']}")

            room_types = [f"{r['count']}x {r['type']}" for r in config['rooms']]
            lines.append(f"- Rooms: {', '.join(room_types)}")

        # Add regulation minimums
        lines.append("\n## Room Minimum Dimensions (Indian NBC 2016):")
        lines.append("\n### Plot ≤ 50 sqm:")
        bucket_data = self.regulations.get("Residential", {}).get(
            "chapter4_residential", {}
        ).get("plot_buckets", {}).get("upto_50sqm", {}).get("rooms", {})
        for room_type, dims in bucket_data.items():
            lines.append(f"- {room_type}: {dims.get('min_area')} sqm min, {dims.get('min_width')}m width")

        lines.append("\n### Plot > 50 sqm:")
        bucket_data = self.regulations.get("Residential", {}).get(
            "chapter4_residential", {}
        ).get("plot_buckets", {}).get("above_50sqm", {}).get("rooms", {})
        for room_type, dims in bucket_data.items():
            lines.append(f"- {room_type}: {dims.get('min_area')} sqm min, {dims.get('min_width')}m width")

        lines.append("\n## Important Rules:")
        lines.append("- Rooms must not overlap")
        lines.append("- Minimum wall thickness: 0.1m between rooms")
        lines.append("- Corridor width: 1.0m minimum")
        lines.append("- Each room needs door access")
        lines.append("- Bathrooms should be near bedrooms")
        lines.append("- Kitchen should be near living room or dining")

        return "\n".join(lines)

    @staticmethod
    def _polygon_area(points: List[List[float]]) -> float:
        """Calculate polygon area using shoelace formula."""
        if not points or len(points) < 3:
            return 0.0
        n = len(points)
        area = 0.0
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0


# ── Module-level instance for easy access ─────────────────────────────────────

_analyzer_instance: Optional[ConstraintAnalyzer] = None


def get_analyzer() -> ConstraintAnalyzer:
    """Get the singleton ConstraintAnalyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = ConstraintAnalyzer()
    return _analyzer_instance


def get_layout_requirements(layout_type: str) -> Dict[str, Any]:
    """Convenience function to get layout requirements."""
    return get_analyzer().get_layout_requirements(layout_type)


def calculate_optimal_dimensions(layout_type: str,
                                  user_total_area: float = None) -> Dict[str, Any]:
    """Convenience function to calculate optimal dimensions."""
    return get_analyzer().calculate_optimal_dimensions(layout_type, user_total_area)


def enhance_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to enhance a spec with constraints."""
    return get_analyzer().enhance_spec(spec)


def get_constraint_summary() -> str:
    """Convenience function to get constraint summary."""
    return get_analyzer().get_constraint_summary()
