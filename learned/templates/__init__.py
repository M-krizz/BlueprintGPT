"""
BlueprintGPT Templates Package

Room layout templates for consistent high-quality generation.

Key Components:
- layout_templates: Template engine and predefined layouts
- template_validator: Template quality validation and scoring
- pattern_extractor: Learn patterns from successful layouts

Usage:
    from learned.templates import find_layout_template, apply_layout_template

    # Find best template for specification
    template = find_layout_template(spec)
    if template:
        layout = apply_layout_template(template, boundary_polygon, spec)
"""

from .layout_templates import (
    LayoutTemplate,
    LayoutTemplateEngine,
    LayoutStyle,
    ZoneType,
    RoomTemplate,
    get_global_template_engine,
    find_layout_template,
    apply_layout_template,
)

__all__ = [
    "LayoutTemplate",
    "LayoutTemplateEngine",
    "LayoutStyle",
    "ZoneType",
    "RoomTemplate",
    "get_global_template_engine",
    "find_layout_template",
    "apply_layout_template",
]