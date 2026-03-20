"""
Enhanced logging utility for BlueprintGPT with proper levels and formatting.
Replaces excessive print statements with configurable logging.
"""

import logging
import os
import sys
from typing import Dict, Any, Optional, List

# Configure log level from environment
LOG_LEVEL = os.getenv("BLUEPRINT_LOG_LEVEL", "INFO").upper()

# Use direct print for terminal visibility (can be disabled)
USE_DIRECT_PRINT = os.getenv("BLUEPRINT_DIRECT_PRINT", "true").lower() == "true"

# Create logger with proper configuration
logger = logging.getLogger("blueprintgpt")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.propagate = False  # Don't propagate to root logger

# Create console handler with formatting - force immediate flush
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _print_log(level: str, message: str):
    """Print log message directly to ensure terminal visibility."""
    if USE_DIRECT_PRINT:
        print(f"[{level}] {message}", flush=True)


class ProcessingLogger:
    """Centralized logging for BlueprintGPT processing pipeline."""

    # Class-level reference to module logger for convenience
    logger = logger

    @staticmethod
    def log_user_interaction(message: str, session_id: str, boundary: Optional[Dict] = None,
                           entrance_point: Optional[List] = None, generate: bool = True):
        """Log user interaction analysis."""
        boundary_str = f"{boundary.get('width', 'None')}x{boundary.get('height', 'None')}" if boundary else "None"

        msg = f"USER_INTERACTION - Session: {session_id[:8]}..."
        logger.info(msg)
        _print_log("INFO", msg)

        msg2 = f"  Input: '{message[:100]}{'...' if len(message) > 100 else ''}'"
        _print_log("DEBUG", msg2)
        logger.debug(msg2)
        logger.debug(f"  Boundary: {boundary_str}, Entrance: {entrance_point}, Generate: {generate}")

    @staticmethod
    def log_intent_classification(intent: str, confidence: float, reason: str = "",
                                 design_keywords: Optional[List] = None, question_type: Optional[str] = None):
        """Log intent classification results."""
        msg = f"INTENT_CLASSIFICATION - {intent.upper()} (confidence: {confidence:.2f})"
        logger.info(msg)
        _print_log("INFO", msg)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"  Reason: {reason}")
            if design_keywords:
                logger.debug(f"  Design keywords: {', '.join(design_keywords)}")
            if question_type:
                logger.debug(f"  Question type: {question_type}")

    @staticmethod
    def log_spec_extraction(rooms: List[Dict], total_rooms: int, plot_type: Optional[str] = None,
                           entrance_side: Optional[str] = None, adjacency: Optional[List] = None):
        """Log specification extraction results."""
        room_summary = ', '.join([f"{r.get('count', 1)}x{r.get('type')}" for r in rooms])

        msg = f"SPEC_EXTRACTION - {total_rooms} rooms: {room_summary}"
        logger.info(msg)
        _print_log("INFO", msg)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"  Plot type: {plot_type or 'Not specified'}")
            logger.debug(f"  Entrance: {entrance_side or 'Not specified'}")
            logger.debug(f"  Adjacency rules: {len(adjacency or [])}")

    @staticmethod
    def log_dimension_processing(frontend_dims: Optional[tuple], cli_dims: Optional[tuple],
                               final_dims: tuple, auto_calculated: bool = False):
        """Log dimension processing decisions."""
        msg = f"DIMENSION_PROCESSING - Final: {final_dims[0]}m x {final_dims[1]}m"
        logger.info(msg)
        _print_log("INFO", msg)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"  Frontend: {frontend_dims}")
            logger.debug(f"  CLI override: {cli_dims}")
            logger.debug(f"  Auto-calculated: {auto_calculated}")

    @staticmethod
    def log_auto_dimension_calculation(rooms: List[Dict], total_area: float,
                                     recommended_dims: tuple, efficiency: float):
        """Log auto-dimension calculation details."""
        msg = f"AUTO_DIMENSION - {recommended_dims[0]}m x {recommended_dims[1]}m ({efficiency:.1f}% efficient)"
        logger.info(msg)
        _print_log("INFO", msg)

        if logger.isEnabledFor(logging.DEBUG):
            total_rooms = sum(r.get('count', 1) for r in rooms)
            logger.debug(f"  Total rooms: {total_rooms}, Required area: {total_area:.1f} sq.m")
            logger.debug(f"  Provided area: {recommended_dims[0] * recommended_dims[1]:.0f} sq.m")

    @staticmethod
    def log_generation_pipeline(spec_complete: bool, missing_fields: List[str],
                               backend_target: Optional[str], backend_ready: bool):
        """Log generation pipeline status."""
        status = "READY" if spec_complete else f"MISSING_FIELDS({len(missing_fields)})"
        msg = f"GENERATION_PIPELINE - {status}"
        logger.info(msg)
        _print_log("INFO", msg)

        if logger.isEnabledFor(logging.DEBUG):
            if missing_fields:
                logger.debug(f"  Missing: {', '.join(missing_fields)}")
            logger.debug(f"  Backend: {backend_target}, Ready: {backend_ready}")

    @staticmethod
    def log_generation_result(success: bool, design_count: int = 0, error_msg: Optional[str] = None):
        """Log generation result."""
        if success:
            msg = f"GENERATION_RESULT - SUCCESS: {design_count} designs generated"
            logger.info(msg)
            _print_log("INFO", msg)
        else:
            msg = f"GENERATION_RESULT - FAILED: {error_msg[:100] if error_msg else 'Unknown error'}"
            logger.warning(msg)
            _print_log("WARN", msg)

    @staticmethod
    def log_processing_summary(intent: str, should_generate: bool, response_length: int,
                             spec_extracted: bool, correction_parsed: bool):
        """Log processing pipeline summary."""
        msg = f"PROCESSING_SUMMARY - Intent: {intent}, Generate: {should_generate}, Response: {response_length}chars"
        logger.info(msg)
        _print_log("INFO", msg)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"  Spec extracted: {spec_extracted}, Correction parsed: {correction_parsed}")


class DetailedLogger:
    """For cases where detailed debug output is needed (controlled by environment)."""

    @staticmethod
    def enabled() -> bool:
        """Check if detailed logging is enabled."""
        return os.getenv("BLUEPRINT_DETAILED_LOGGING", "false").lower() == "true"

    @staticmethod
    def log_detailed_state(category: str, data: Dict[str, Any]):
        """Log detailed state information only if explicitly enabled."""
        if DetailedLogger.enabled():
            logger.debug(f"DETAILED_{category.upper()}: {data}")


# Export main logger for direct use
__all__ = ['ProcessingLogger', 'DetailedLogger', 'logger']