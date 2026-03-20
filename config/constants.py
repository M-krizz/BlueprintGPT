"""
Centralized configuration constants for BlueprintGPT.
Eliminates duplicated constants across the codebase.
"""

from typing import Dict, Tuple, List
from enum import Enum


class RoomTypes:
    """Centralized room type constants to replace magic strings."""
    BEDROOM = "Bedroom"
    KITCHEN = "Kitchen"
    BATHROOM = "Bathroom"
    LIVING_ROOM = "LivingRoom"
    DINING_ROOM = "DiningRoom"
    DRAWING_ROOM = "DrawingRoom"
    GARAGE = "Garage"
    STORE = "Store"

    @classmethod
    def all_types(cls) -> List[str]:
        """Get all supported room types."""
        return [
            cls.BEDROOM, cls.KITCHEN, cls.BATHROOM, cls.LIVING_ROOM,
            cls.DINING_ROOM, cls.DRAWING_ROOM, cls.GARAGE, cls.STORE
        ]


class DefaultDimensions:
    """Default dimension constants used across the system."""
    # Default frontend boundary dimensions (width, height) in meters
    BOUNDARY = (12.0, 15.0)

    # Default plot area in square meters
    DEFAULT_AREA = BOUNDARY[0] * BOUNDARY[1]  # 180 sq.m

    # Minimum plot dimensions
    MIN_WIDTH = 6.0
    MIN_HEIGHT = 6.0


class LayoutStandards:
    """Room area standards and layout configuration."""

    # Room area recommendations (in sq.m) based on architectural standards
    ROOM_AREA_STANDARDS: Dict[str, Dict[str, float]] = {
        RoomTypes.BEDROOM: {"min": 9.0, "ideal": 12.0, "max": 20.0},
        RoomTypes.KITCHEN: {"min": 6.0, "ideal": 9.0, "max": 15.0},
        RoomTypes.BATHROOM: {"min": 3.0, "ideal": 4.5, "max": 8.0},
        RoomTypes.LIVING_ROOM: {"min": 12.0, "ideal": 20.0, "max": 40.0},
        RoomTypes.DINING_ROOM: {"min": 8.0, "ideal": 12.0, "max": 20.0},
        RoomTypes.DRAWING_ROOM: {"min": 10.0, "ideal": 15.0, "max": 25.0},
        RoomTypes.GARAGE: {"min": 12.0, "ideal": 18.0, "max": 30.0},
        RoomTypes.STORE: {"min": 2.0, "ideal": 4.0, "max": 8.0},
    }

    # Circulation and wall factor (30-40% additional space)
    CIRCULATION_FACTOR = 0.35

    # Preferred aspect ratios (width : height)
    PREFERRED_ASPECT_RATIOS: List[Tuple[float, float]] = [
        (1.0, 1.0),   # Square
        (1.2, 1.0),   # Slightly rectangular
        (1.5, 1.0),   # Golden ratio-ish
        (1.6, 1.0),   # Common residential
        (2.0, 1.0),   # Wide rectangular
    ]

    # Predefined dimension sets for common layout types
    LAYOUT_DIMENSION_PRESETS: Dict[str, Tuple[float, float]] = {
        "1BHK": (10.0, 12.0),   # 120 sq.m
        "2BHK": (12.0, 15.0),   # 180 sq.m
        "3BHK": (15.0, 18.0),   # 270 sq.m
        "4BHK": (18.0, 20.0),   # 360 sq.m
        "minimal": (8.0, 10.0), # 80 sq.m
        "medium": (12.0, 15.0), # 180 sq.m
        "large": (16.0, 20.0),  # 320 sq.m
        "mansion": (20.0, 25.0) # 500 sq.m
    }


class IntentTypes:
    """Intent classification constants."""
    DESIGN = "design"
    QUESTION = "question"
    CORRECTION = "correction"
    CONVERSATION = "conversation"

    @classmethod
    def all_types(cls) -> List[str]:
        """Get all intent types."""
        return [cls.DESIGN, cls.QUESTION, cls.CORRECTION, cls.CONVERSATION]


class LogCategories:
    """Logging category constants to replace magic strings."""
    USER_INTERACTION = "USER_INTERACTION_ANALYSIS"
    NATURAL_LANGUAGE = "NATURAL_LANGUAGE_PROCESSING"
    SPEC_EXTRACTION = "SPEC_EXTRACTION"
    DIMENSION_PROCESSING = "DIMENSION_PROCESSING"
    AUTO_DIMENSION = "AUTO_DIMENSION_SELECTION"
    GENERATION_PIPELINE = "GENERATION_PIPELINE"
    PROCESSING_SUMMARY = "PROCESSING_SUMMARY"

    # Log message prefixes
    PREFIX_RAW_INPUT = "Raw Input"
    PREFIX_SESSION = "Session"
    PREFIX_BOUNDARY = "Frontend Boundary"
    PREFIX_ENTRANCE = "Frontend Entrance"
    PREFIX_GENERATE_FLAG = "Generate Flag"
    PREFIX_INTENT = "Intent Classification"
    PREFIX_CONFIDENCE = "Intent confidence"
    PREFIX_REASON = "Reason"
    PREFIX_EXTRACTED_ROOMS = "Extracted Rooms"
    PREFIX_SPEC_COMPLETE = "Spec Complete"
    PREFIX_MISSING_FIELDS = "Missing Fields"
    PREFIX_BACKEND_TARGET = "Backend Target"


class APIDefaults:
    """Default values for API parameters."""

    # Default session configuration
    DEFAULT_GENERATE = True
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7

    # Default boundary settings
    DEFAULT_AREA_UNIT = "sq.m"

    # Default entrance positioning
    DEFAULT_ENTRANCE_SIDE = "North"

    # Default building type
    DEFAULT_BUILDING_TYPE = "residential"


class QualityGateSettings:
    """Quality gate thresholds and settings."""

    # Minimum thresholds for design acceptance
    MIN_ADJACENCY_SATISFACTION = 0.2
    MIN_COMPLIANCE_SCORE = 0.8
    MIN_COVERAGE_RATIO = 0.85

    # Maximum allowed values
    MAX_TRAVEL_DISTANCE = 30.0  # meters
    MAX_OVERLAP_RATIO = 0.05    # 5%

    # Quality scoring weights
    QUALITY_WEIGHTS = {
        "compliance": 0.4,
        "realism": 0.25,
        "success": 0.2,
        "repair": 0.15
    }


# Module-level constants for backward compatibility
SUPPORTED_ROOM_TYPES = RoomTypes.all_types()
DEFAULT_BOUNDARY_DIMENSIONS = DefaultDimensions.BOUNDARY
ROOM_AREA_STANDARDS = LayoutStandards.ROOM_AREA_STANDARDS


# Export all configuration classes
__all__ = [
    'RoomTypes', 'DefaultDimensions', 'LayoutStandards', 'IntentTypes',
    'LogCategories', 'APIDefaults', 'QualityGateSettings',
    'SUPPORTED_ROOM_TYPES', 'DEFAULT_BOUNDARY_DIMENSIONS', 'ROOM_AREA_STANDARDS'
]