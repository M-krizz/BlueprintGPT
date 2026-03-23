"""
auto_dimension_selector.py – Automatic dimension selection based on layout type and room count.

This module calculates appropriate plot dimensions based on:
1. Total room count
2. Room types and their space requirements
3. Architectural best practices
4. Building type (residential, commercial, etc.)
"""

from typing import Dict, List, Tuple, Optional
from config.constants import RoomTypes, LayoutStandards, DefaultDimensions
from utils.processing_logger import ProcessingLogger


def calculate_total_area_requirement(rooms: List[Dict]) -> float:
    """Calculate total area requirement for given rooms."""
    total_area = 0

    for room in rooms:
        room_type = room.get("type", "")
        count = room.get("count", 1)

        if room_type in LayoutStandards.ROOM_AREA_STANDARDS:
            # Use ideal area for calculation
            room_area = LayoutStandards.ROOM_AREA_STANDARDS[room_type]["ideal"]
            total_area += room_area * count
        else:
            # Unknown room type - use average
            total_area += 12 * count  # 12 sq.m average

    # Add circulation space
    total_area_with_circulation = total_area * (1 + LayoutStandards.CIRCULATION_FACTOR)

    return total_area_with_circulation


def recommend_dimensions(rooms: List[Dict], building_type: str = "residential") -> Tuple[float, float]:
    """
    Recommend plot dimensions based on room requirements.

    Returns: (width, height) in meters
    """
    total_area = calculate_total_area_requirement(rooms)
    total_rooms = sum(room.get("count", 1) for room in rooms)

    ProcessingLogger.log_auto_dimension_calculation(
        rooms, total_area, (0, 0), 0  # Temporary values, will update below
    )

    # Apply building type modifiers
    if building_type == "commercial":
        total_area *= 1.2  # Commercial needs more space
    elif building_type == "luxury":
        total_area *= 1.5  # Luxury needs much more space

    # Find best aspect ratio that provides enough area
    best_dimensions = None
    best_score = float('inf')

    for aspect_w, aspect_h in LayoutStandards.PREFERRED_ASPECT_RATIOS:
        # Calculate dimensions for this aspect ratio
        # Area = w * h, aspect = w/h, so w = aspect*h, Area = aspect*h²
        h = (total_area / (aspect_w / aspect_h)) ** 0.5
        w = h * (aspect_w / aspect_h)

        # Round to nice numbers (multiples of 0.5m)
        w_rounded = round(w * 2) / 2
        h_rounded = round(h * 2) / 2

        # Ensure minimum dimensions
        w_rounded = max(DefaultDimensions.MIN_WIDTH, w_rounded)
        h_rounded = max(DefaultDimensions.MIN_HEIGHT, h_rounded)

        actual_area = w_rounded * h_rounded

        # Score based on area efficiency and aspect ratio preference
        area_score = abs(actual_area - total_area) / total_area
        aspect_score = abs((w_rounded/h_rounded) - (aspect_w/aspect_h)) * 0.1
        total_score = area_score + aspect_score

        if total_score < best_score:
            best_score = total_score
            best_dimensions = (w_rounded, h_rounded)

    width, height = best_dimensions if best_dimensions else DefaultDimensions.BOUNDARY

    # Calculate efficiency for logging
    efficiency = (total_area / (width * height)) * 100

    # Log the final calculation
    ProcessingLogger.log_auto_dimension_calculation(
        rooms, total_area, (width, height), efficiency
    )

    return width, height


def get_layout_classification(rooms: List[Dict]) -> str:
    """Classify layout type based on room composition."""
    total_rooms = sum(room.get("count", 1) for room in rooms)
    bedrooms = sum(room.get("count", 1) for room in rooms if room.get("type") == RoomTypes.BEDROOM)

    # Standard BHK classification
    if bedrooms >= 1 and any(r.get("type") == RoomTypes.LIVING_ROOM for r in rooms):
        if bedrooms == 1:
            return "1BHK"
        elif bedrooms == 2:
            return "2BHK"
        elif bedrooms == 3:
            return "3BHK"
        elif bedrooms >= 4:
            return f"{bedrooms}BHK"

    # Custom classification
    if total_rooms <= 3:
        return "minimal"
    elif total_rooms <= 6:
        return "medium"
    elif total_rooms <= 9:
        return "large"
    else:
        return "mansion"


def explain_dimension_choice(rooms: List[Dict], width: float, height: float) -> str:
    """Generate explanation for the chosen dimensions."""
    layout_type = get_layout_classification(rooms)
    total_area = calculate_total_area_requirement(rooms)
    actual_area = width * height
    efficiency = (total_area / actual_area) * 100

    room_summary = ", ".join([f"{r.get('count', 1)} {r.get('type')}" for r in rooms])

    explanation = f"""
Layout analysis for {layout_type}:
  Rooms: {room_summary}
  Required: {total_area:.0f} sq.m
  Provided: {actual_area:.0f} sq.m ({efficiency:.1f}% efficiency)

Dimension choice: {width}m x {height}m
  Reasoning: Balanced aspect ratio with adequate circulation space
  Standard: Follows residential design guidelines
""".strip()

    return explanation


def quick_dimension_recommendation(rooms: List[Dict]) -> Tuple[float, float]:
    """Quick dimension recommendation using presets."""
    layout_type = get_layout_classification(rooms)

    if layout_type in LayoutStandards.LAYOUT_DIMENSION_PRESETS:
        width, height = LayoutStandards.LAYOUT_DIMENSION_PRESETS[layout_type]
        ProcessingLogger.logger.debug(f"Quick preset: {layout_type} -> {width}m x {height}m")
        return width, height
    else:
        # Fall back to calculated recommendation
        return recommend_dimensions(rooms)
