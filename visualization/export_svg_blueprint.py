"""
export_svg_blueprint.py â€“ CAD / SmartDraw-style SVG floor-plan renderer.

Produces professional drafting-style output:
    * Merged outer / inner wall strokes (shared walls drawn once)
    * Openings carved for doors with swing arcs
    * Room labels + area dimensions per room
    * Overall boundary dimension strings
    * Title block with scale bar
    * Colour-coded zones (public / service / private)
    * Corridor hatching
    * Optional metric grid overlay

Usage
-----
    python -m visualization.export_svg_blueprint \
        --spec-json outputs/compliance_report.json \
        --output outputs/blueprint.svg

Programmatic
------------
    svg_str = render_svg_blueprint(building, boundary_polygon, title="My Plan")
    Path("output.svg").write_text(svg_str)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree.ElementTree import Element, SubElement, tostring

from core.building import Building
from geometry.window_placer import suggest_window_segments
from geometry.walls import build_wall_segments
from visualization.dimensions import draw_dimension, measure_room_dims
from visualization.render_units import resolve_render_units

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SCALE = 80  # pixels per metre
MARGIN = 70  # px margin around boundary
WALL_WIDTH = 3.0
INNER_WALL_WIDTH = 1.5
DOOR_WIDTH_PX = 2.0
WINDOW_WIDTH_PX = 1.8
LEGEND_W = 160  # reserved px on right for legend

STYLE_PRESET_DOCUMENT = "document"
STYLE_PRESET_PRESENTATION_BLUEPRINT = "presentation_blueprint"

THEME_DOCUMENT = {
    "canvas_bg": "white",
    "sheet_bg": "white",
    "grid_major": "#e0e0e0",
    "grid_minor": "#f1f5f9",
    "outer_wall": "#263238",
    "inner_wall": "#37474f",
    "room_fill": None,
    "room_fill_opacity": "1.0",
    "label_primary": "#1a237e",
    "label_secondary": "#37474f",
    "dimension": "#455a64",
    "corridor_stroke": "#78909c",
    "corridor_label": "#607d8b",
    "door_gap": "white",
    "door_leaf": "#37474f",
    "door_arc": "#455a64",
    "window": "#039be5",
    "window_tick": "#0288d1",
    "entrance": "#d32f2f",
    "legend_text": "#263238",
    "legend_border": "#37474f",
    "footer_bg": "#263238",
    "footer_text": "white",
    "footer_muted": "#b0bec5",
    "title_banner_bg": None,
    "title_banner_text": "#263238",
    "title_banner_rule": "#90a4ae",
    "compass": "#263238",
}

THEME_PRESENTATION_BLUEPRINT = {
    "canvas_bg": "#0b4ea2",
    "sheet_bg": "#0b4ea2",
    "grid_major": "rgba(255,255,255,0.16)",
    "grid_minor": "rgba(255,255,255,0.06)",
    "outer_wall": "#f8fbff",
    "inner_wall": "#edf6ff",
    "room_fill": "#5ca1df",
    "room_fill_opacity": "0.15",
    "label_primary": "#ffffff",
    "label_secondary": "#dbeafe",
    "dimension": "#dbeafe",
    "corridor_stroke": "#dbeafe",
    "corridor_label": "#e2e8f0",
    "door_gap": "#0b4ea2",
    "door_leaf": "#f8fbff",
    "door_arc": "#f8fbff",
    "window": "#f8fbff",
    "window_tick": "#dbeafe",
    "entrance": "#fca5a5",
    "legend_text": "#f8fbff",
    "legend_border": "#dbeafe",
    "footer_bg": "#083f84",
    "footer_text": "#ffffff",
    "footer_muted": "#dbeafe",
    "title_banner_bg": "#0a458f",
    "title_banner_text": "#ffffff",
    "title_banner_rule": "#dbeafe",
    "compass": "#ffffff",
}

_ACTIVE_THEME = dict(THEME_DOCUMENT)

SYMBOL_VIEWBOX_SIZES = {
    "door-swing": (90.0, 90.0),
    "door-double": (180.0, 90.0),
    "window": (100.0, 10.0),
    "bed-double": (120.0, 160.0),
    "bed-single": (72.0, 160.0),
    "kitchen-counter": (240.0, 48.0),
    "dining-table": (96.0, 64.0),
    "chair": (40.0, 40.0),
    "sofa": (160.0, 72.0),
    "toilet": (56.0, 72.0),
    "bathtub": (136.0, 56.0),
    "wardrobe": (96.0, 48.0),
    "north-arrow": (24.0, 40.0),
}


def _set_active_theme(style_preset: str) -> None:
    global _ACTIVE_THEME
    if style_preset == STYLE_PRESET_PRESENTATION_BLUEPRINT:
        _ACTIVE_THEME = dict(THEME_PRESENTATION_BLUEPRINT)
    else:
        _ACTIVE_THEME = dict(THEME_DOCUMENT)


def _theme(key: str):
    return _ACTIVE_THEME.get(key)


def _is_presentation_blueprint() -> bool:
    return _ACTIVE_THEME.get("canvas_bg") == THEME_PRESENTATION_BLUEPRINT["canvas_bg"]

# Zone-based fill colours (muted, architectural)
ZONE_FILL = {
    "public":  "#e8f5e9",
    "service": "#fff3e0",
    "private": "#e3f2fd",
}

ROOM_FILL = {
    "Bedroom":    "#bfdbfe", # blue-200
    "LivingRoom": "#bbf7d0", # green-200
    "Kitchen":    "#fef08a", # yellow-200
    "Bathroom":   "#e9d5ff", # purple-200
    "WC":         "#fbcfe8", # pink-200
    "DiningRoom": "#ffedd5", # orange-200
    "Study":      "#ddd6fe", # violet-200
    "Storage":    "#e5e7eb", # gray-200
    "Balcony":    "#ccfbf1", # teal-200
    "Staircase":  "#d1d5db", # gray-300
    "Garage":     "#fecaca", # red-200
    "Corridor":   "#f3f4f6", # gray-100
}

DEFAULT_FILL = "#f5f5f5"

# Fallback cycling palette for unknown room types
_FILL_PALETTE = [
    "#bfdbfe", "#bbf7d0", "#fef08a", "#e9d5ff",
    "#fbcfe8", "#ffedd5", "#ddd6fe", "#ccfbf1",
    "#fecaca", "#d1fae5", "#fde68a", "#a7f3d0",
]

def _room_fill_color(room_type: str, zone: str = "") -> str:
    """Return a distinct, stable fill color for a room type."""
    if room_type in ROOM_FILL:
        return ROOM_FILL[room_type]
    # Hash-stable assignment for unknown types
    idx = abs(hash(room_type)) % len(_FILL_PALETTE)
    return _FILL_PALETTE[idx]


def _px(metres: float) -> float:
    return metres * SCALE


def _room_bbox(room) -> Tuple[float, float, float, float]:
    if room.polygon is None:
        return (0, 0, 0, 0)
    xs = [p[0] for p in room.polygon]
    ys = [p[1] for p in room.polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _polygon_path(polygon, ox: float, oy: float) -> str:
    """Convert a polygon (list of [x, y] points) to SVG path string with offset."""
    if polygon is None:
        return ""

    # Handle Shapely polygon objects
    if hasattr(polygon, 'exterior'):
        coords = list(polygon.exterior.coords)
    else:
        coords = polygon

    if not coords:
        return ""

    parts = []
    for i, (x, y) in enumerate(coords):
        px = ox + _px(x)
        py = oy + _px(y)
        if i == 0:
            parts.append(f"M{px:.1f},{py:.1f}")
        else:
            parts.append(f"L{px:.1f},{py:.1f}")
    parts.append("Z")  # Close the path
    return " ".join(parts)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SVG building blocks
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _svg_root(width_px: float, height_px: float) -> Element:
    svg = Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": str(int(width_px)),
        "height": str(int(height_px)),
        "viewBox": f"0 0 {int(width_px)} {int(height_px)}",
        "font-family": "'Bahnschrift', 'Segoe UI', Helvetica, Arial, sans-serif",
    })
    # Background
    SubElement(svg, "rect", {
        "width": "100%", "height": "100%", "fill": _theme("canvas_bg") or "white",
    })
    return svg


def _add_defs(svg: Element, boundary_polygon):
    defs = SubElement(svg, "defs")
    corridor_bg = _theme("sheet_bg") if _is_presentation_blueprint() else "#f5f5f5"
    corridor_hatch = _theme("corridor_stroke") if _is_presentation_blueprint() else "#bdbdbd"
    symbol_stroke = _theme("door_arc") if _is_presentation_blueprint() else "#78909c"
    furniture_stroke = _theme("inner_wall") if _is_presentation_blueprint() else "#666"
    furniture_fill = "none" if _is_presentation_blueprint() else "#e0e0e0"
    furniture_fill_light = "none" if _is_presentation_blueprint() else "#f5f5f5"
    accent_color = _theme("window") if _is_presentation_blueprint() else "#2196f3"

    # ── Existing patterns ──────────────────────────────────────────────────
    # Hatch pattern for corridors
    patt = SubElement(defs, "pattern", {
        "id": "corridor-hatch",
        "patternUnits": "userSpaceOnUse",
        "width": "8", "height": "8",
    })
    SubElement(patt, "rect", {
        "width": "8", "height": "8", "fill": corridor_bg,
    })
    SubElement(patt, "path", {
        "d": "M0,8 l8,-8 M-2,2 l4,-4 M6,10 l4,-4",
        "stroke": corridor_hatch, "stroke-width": "0.5",
    })

    # Drop shadow
    filt = SubElement(defs, "filter", {"id": "shadow", "x": "-2%", "y": "-2%",
                                        "width": "104%", "height": "104%"})
    SubElement(filt, "feDropShadow", {
        "dx": "1", "dy": "1", "stdDeviation": "2",
        "flood-opacity": "0.05" if _is_presentation_blueprint() else "0.15",
    })

    # ── Symbol Library ─────────────────────────────────────────────────────

    # Door swing arc symbol (reusable)
    door_symbol = SubElement(defs, "symbol", {
        "id": "door-swing",
        "viewBox": "0 0 90 90",
        "overflow": "visible"
    })
    SubElement(door_symbol, "path", {
        "d": "M 0,0 Q 90,0 90,90",
        "fill": "none",
        "stroke": symbol_stroke,
        "stroke-width": "0.8",
        "stroke-dasharray": "3,2",
    })

    # Double door symbol
    double_door = SubElement(defs, "symbol", {
        "id": "door-double",
        "viewBox": "0 0 180 90",
        "overflow": "visible"
    })
    SubElement(double_door, "path", {
        "d": "M 0,0 Q 90,0 90,90 M 180,0 Q 90,0 90,90",
        "fill": "none",
        "stroke": symbol_stroke,
        "stroke-width": "0.8",
        "stroke-dasharray": "3,2",
    })

    # Window symbol (double line)
    window_symbol = SubElement(defs, "symbol", {
        "id": "window",
        "viewBox": "0 0 100 10",
        "overflow": "visible"
    })
    SubElement(window_symbol, "rect", {
        "x": "0", "y": "3", "width": "100", "height": "4",
        "fill": "none", "stroke": accent_color, "stroke-width": "1.5"
    })
    SubElement(window_symbol, "line", {
        "x1": "50", "y1": "3", "x2": "50", "y2": "7",
        "stroke": accent_color, "stroke-width": "1"
    })

    # ── Furniture Symbols ──────────────────────────────────────────────────

    # Double bed (1.5m x 2.0m)
    bed_double = SubElement(defs, "symbol", {
        "id": "bed-double",
        "viewBox": "0 0 120 160",  # 1.5m x 2.0m at 80px/m
        "overflow": "visible"
    })
    SubElement(bed_double, "rect", {
        "x": "5", "y": "5", "width": "110", "height": "150",
        "fill": furniture_fill, "stroke": furniture_stroke, "stroke-width": "1"
    })
    SubElement(bed_double, "rect", {  # Headboard
        "x": "0", "y": "0", "width": "120", "height": "20",
        "fill": furniture_fill_light, "stroke": furniture_stroke, "stroke-width": "1"
    })

    # Single bed (0.9m x 2.0m)
    bed_single = SubElement(defs, "symbol", {
        "id": "bed-single",
        "viewBox": "0 0 72 160",  # 0.9m x 2.0m at 80px/m
        "overflow": "visible"
    })
    SubElement(bed_single, "rect", {
        "x": "5", "y": "5", "width": "62", "height": "150",
        "fill": furniture_fill, "stroke": furniture_stroke, "stroke-width": "1"
    })
    SubElement(bed_single, "rect", {  # Headboard
        "x": "0", "y": "0", "width": "72", "height": "20",
        "fill": furniture_fill_light, "stroke": furniture_stroke, "stroke-width": "1"
    })

    # Kitchen counter with sink
    kitchen_counter = SubElement(defs, "symbol", {
        "id": "kitchen-counter",
        "viewBox": "0 0 240 48",  # 3.0m x 0.6m counter
        "overflow": "visible"
    })
    SubElement(kitchen_counter, "rect", {
        "x": "0", "y": "0", "width": "240", "height": "48",
        "fill": furniture_fill_light, "stroke": furniture_stroke, "stroke-width": "1"
    })
    # Sink
    SubElement(kitchen_counter, "circle", {
        "cx": "60", "cy": "24", "r": "15",
        "fill": "none", "stroke": accent_color, "stroke-width": "1"
    })
    # Stove burners
    for i, x in enumerate([140, 170, 200, 230]):
        SubElement(kitchen_counter, "circle", {
            "cx": str(x), "cy": "24", "r": "8",
            "fill": "none", "stroke": furniture_stroke, "stroke-width": "1"
        })

    # Dining table (1.2m x 0.8m)
    dining_table = SubElement(defs, "symbol", {
        "id": "dining-table",
        "viewBox": "0 0 96 64",  # 1.2m x 0.8m
        "overflow": "visible"
    })
    SubElement(dining_table, "rect", {
        "x": "0", "y": "0", "width": "96", "height": "64",
        "fill": furniture_fill, "stroke": furniture_stroke, "stroke-width": "1"
    })

    # Chair (0.5m x 0.5m)
    chair = SubElement(defs, "symbol", {
        "id": "chair",
        "viewBox": "0 0 40 40",  # 0.5m x 0.5m
        "overflow": "visible"
    })
    SubElement(chair, "rect", {
        "x": "5", "y": "5", "width": "30", "height": "30",
        "fill": furniture_fill_light, "stroke": furniture_stroke, "stroke-width": "1"
    })
    SubElement(chair, "rect", {  # Backrest
        "x": "5", "y": "0", "width": "30", "height": "10",
        "fill": furniture_fill, "stroke": furniture_stroke, "stroke-width": "1"
    })

    # Sofa (2.0m x 0.9m)
    sofa = SubElement(defs, "symbol", {
        "id": "sofa",
        "viewBox": "0 0 160 72",  # 2.0m x 0.9m
        "overflow": "visible"
    })
    SubElement(sofa, "rect", {
        "x": "0", "y": "10", "width": "160", "height": "52",
        "fill": furniture_fill, "stroke": furniture_stroke, "stroke-width": "1"
    })
    SubElement(sofa, "rect", {  # Backrest
        "x": "0", "y": "0", "width": "160", "height": "20",
        "fill": furniture_fill_light, "stroke": furniture_stroke, "stroke-width": "1"
    })

    # Toilet
    toilet = SubElement(defs, "symbol", {
        "id": "toilet",
        "viewBox": "0 0 56 72",  # 0.7m x 0.9m
        "overflow": "visible"
    })
    SubElement(toilet, "rect", {
        "x": "8", "y": "0", "width": "40", "height": "50",
        "fill": furniture_fill_light, "stroke": accent_color, "stroke-width": "1"
    })
    SubElement(toilet, "rect", {  # Tank
        "x": "12", "y": "52", "width": "32", "height": "20",
        "fill": furniture_fill_light, "stroke": accent_color, "stroke-width": "1"
    })

    # Bathtub (1.7m x 0.7m)
    bathtub = SubElement(defs, "symbol", {
        "id": "bathtub",
        "viewBox": "0 0 136 56",  # 1.7m x 0.7m
        "overflow": "visible"
    })
    SubElement(bathtub, "rect", {
        "x": "0", "y": "0", "width": "136", "height": "56",
        "fill": furniture_fill_light, "stroke": accent_color, "stroke-width": "1.5"
    })

    # Wardrobe (1.2m x 0.6m)
    wardrobe = SubElement(defs, "symbol", {
        "id": "wardrobe",
        "viewBox": "0 0 96 48",  # 1.2m x 0.6m
        "overflow": "visible"
    })
    SubElement(wardrobe, "rect", {
        "x": "0", "y": "0", "width": "96", "height": "48",
        "fill": furniture_fill_light, "stroke": furniture_stroke, "stroke-width": "1"
    })
    # Door handles
    SubElement(wardrobe, "circle", {
        "cx": "24", "cy": "24", "r": "2",
        "fill": furniture_stroke
    })
    SubElement(wardrobe, "circle", {
        "cx": "72", "cy": "24", "r": "2",
        "fill": furniture_stroke
    })

    # ── Directional arrow for layout orientation ──────────────────────────
    north_arrow = SubElement(defs, "symbol", {
        "id": "north-arrow",
        "viewBox": "0 0 24 40",
        "overflow": "visible"
    })
    SubElement(north_arrow, "path", {
        "d": "M 12,4 L 8,16 L 12,12 L 16,16 Z",
        "fill": _theme("compass"), "stroke": _theme("compass"), "stroke-width": "1"
    })
    SubElement(north_arrow, "text", {
        "x": "12", "y": "32", "text-anchor": "middle",
        "font-family": "Arial", "font-size": "8", "fill": _theme("compass")
    })
    north_text = SubElement(north_arrow, "tspan")
    north_text.text = "N"


# ═══════════════════════════════════════════════════════════════════════════════
#  Symbol Usage Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

def _use_symbol(g: Element, symbol_id: str, x: float, y: float,
                rotation: float = 0, scale: float = 1.0,
                rotate_about: str = "origin", **attrs) -> Element:
    """Place a symbol from the library at specified coordinates.

    Parameters
    ----------
    symbol_id : str
        ID of the symbol to use (e.g., "bed-double", "kitchen-counter")
    x, y : float
        Position in SVG coordinates (pixels)
    rotation : float
        Rotation angle in degrees
    scale : float
        Scaling factor
    """
    native_w, native_h = SYMBOL_VIEWBOX_SIZES.get(symbol_id, (100.0, 100.0))
    use_attrs = {
        "href": f"#{symbol_id}",
        "x": f"{x:.1f}",
        "y": f"{y:.1f}",
        "width": f"{native_w * scale:.1f}",
        "height": f"{native_h * scale:.1f}",
    }
    if rotation != 0:
        if rotate_about == "center":
            cx = x + (native_w * scale) / 2.0
            cy = y + (native_h * scale) / 2.0
        else:
            cx = x
            cy = y
        use_attrs["transform"] = f"rotate({rotation},{cx:.1f},{cy:.1f})"
    use_attrs.update(attrs)

    return SubElement(g, "use", use_attrs)


def _fit_symbol_scale(symbol_id: str, max_w: float, max_h: float, *, max_scale: float = 1.0) -> float:
    native_w, native_h = SYMBOL_VIEWBOX_SIZES.get(symbol_id, (100.0, 100.0))
    if native_w <= 0 or native_h <= 0:
        return 1.0
    return max(0.0, min(max_scale, max_w / native_w, max_h / native_h))


def _render_exit_segment(boundary_polygon, entrance_point, door_width_m: float = 1.0):
    """Build a synthetic exit-door segment aligned to the displayed entrance side."""
    if not boundary_polygon or not entrance_point:
        return None

    xs = [p[0] for p in boundary_polygon]
    ys = [p[1] for p in boundary_polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    ex, ey = entrance_point
    half = door_width_m / 2.0
    tol = 1e-6

    if abs(ey - min_y) <= tol:  # North
        cx = max(min_x + half, min(max_x - half, ex))
        return ((cx - half, min_y), (cx + half, min_y))
    if abs(ey - max_y) <= tol:  # South
        cx = max(min_x + half, min(max_x - half, ex))
        return ((cx - half, max_y), (cx + half, max_y))
    if abs(ex - min_x) <= tol:  # West
        cy = max(min_y + half, min(max_y - half, ey))
        return ((min_x, cy - half), (min_x, cy + half))
    if abs(ex - max_x) <= tol:  # East
        cy = max(min_y + half, min(max_y - half, ey))
        return ((max_x, cy - half), (max_x, cy + half))
    return None


def _segments_match(segment_a, segment_b, tol: float = 1e-6) -> bool:
    if not segment_a or not segment_b:
        return False
    (ax1, ay1), (ax2, ay2) = segment_a
    (bx1, by1), (bx2, by2) = segment_b
    same_order = (
        abs(ax1 - bx1) <= tol and abs(ay1 - by1) <= tol and
        abs(ax2 - bx2) <= tol and abs(ay2 - by2) <= tol
    )
    reverse_order = (
        abs(ax1 - bx2) <= tol and abs(ay1 - by2) <= tol and
        abs(ax2 - bx1) <= tol and abs(ay2 - by1) <= tol
    )
    return same_order or reverse_order


def _segment_on_boundary(segment, boundary_polygon, tol: float = 0.02) -> bool:
    if not segment or not boundary_polygon:
        return False
    xs = [p[0] for p in boundary_polygon]
    ys = [p[1] for p in boundary_polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    (x1, y1), (x2, y2) = segment

    if abs(x1 - x2) <= tol:
        return abs(x1 - min_x) <= tol or abs(x1 - max_x) <= tol
    if abs(y1 - y2) <= tol:
        return abs(y1 - min_y) <= tol or abs(y1 - max_y) <= tol
    return False


def _room_door_clearances(room, clearance_px: float = 42.0, tol: float = 0.08) -> Dict[str, float]:
    clearances = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0}
    if not getattr(room, "polygon", None):
        return clearances

    xs = [p[0] for p in room.polygon]
    ys = [p[1] for p in room.polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    for door in getattr(room, "doors", []) or []:
        segment = getattr(door, "segment", None)
        if not segment:
            continue
        (x1, y1), (x2, y2) = segment
        door_clearance = max(clearance_px, _px(getattr(door, "width", 0.9)) + 10.0)
        if abs(x1 - x2) <= tol:
            if abs(x1 - min_x) <= tol:
                clearances["left"] = max(clearances["left"], door_clearance)
            elif abs(x1 - max_x) <= tol:
                clearances["right"] = max(clearances["right"], door_clearance)
        elif abs(y1 - y2) <= tol:
            if abs(y1 - min_y) <= tol:
                clearances["top"] = max(clearances["top"], door_clearance)
            elif abs(y1 - max_y) <= tol:
                clearances["bottom"] = max(clearances["bottom"], door_clearance)
    return clearances


def _prefer_side(primary: str, secondary: str, clearances: Dict[str, float], tolerance_px: float = 6.0) -> str:
    primary_clearance = clearances.get(primary, 0.0)
    secondary_clearance = clearances.get(secondary, 0.0)
    if primary_clearance + tolerance_px < secondary_clearance:
        return primary
    if secondary_clearance + tolerance_px < primary_clearance:
        return secondary
    return primary


def _place_furniture_in_room(g: Element, room, ox: float, oy: float,
                           furniture_enabled: bool = True):
    """Automatically place appropriate furniture in a room based on room type."""
    if not furniture_enabled or not room.polygon or len(room.polygon) < 3:
        return

    # Calculate room bounds and center
    xs = [p[0] for p in room.polygon]
    ys = [p[1] for p in room.polygon]
    room_left, room_right = min(xs), max(xs)
    room_top, room_bottom = min(ys), max(ys)
    room_width = room_right - room_left
    room_height = room_bottom - room_top
    center_x = (room_left + room_right) / 2
    center_y = (room_top + room_bottom) / 2

    # Convert to SVG coordinates
    svg_center_x = ox + _px(center_x)
    svg_center_y = oy + _px(center_y)
    svg_left = ox + _px(room_left)
    svg_top = oy + _px(room_top)
    svg_width = _px(room_width)
    svg_height = _px(room_height)
    pad = 12.0
    door_clearances = _room_door_clearances(room)

    room_type = room.room_type

    if room_type == "Bedroom":
        if svg_width < 150 or svg_height < 110:
            return
        if room_width > room_height:  # Horizontal layout
            bed_scale = _fit_symbol_scale("bed-double", svg_width - 2 * pad, svg_height * 0.62, max_scale=0.9)
            if bed_scale >= 0.55:
                bed_w = SYMBOL_VIEWBOX_SIZES["bed-double"][0] * bed_scale
                bed_h = SYMBOL_VIEWBOX_SIZES["bed-double"][1] * bed_scale
                bed_x = svg_center_x - bed_w / 2
                bed_side = _prefer_side("top", "bottom", door_clearances)
                if bed_side == "top":
                    bed_y = svg_top + pad + door_clearances["top"]
                else:
                    bed_y = svg_top + svg_height - bed_h - pad - door_clearances["bottom"]
                _use_symbol(g, "bed-double", bed_x, bed_y, scale=bed_scale)
                remaining_h = svg_height - bed_h - (2 * pad)
                if remaining_h >= 42 and svg_width >= 180:
                    wardrobe_scale = _fit_symbol_scale("wardrobe", svg_width * 0.55, remaining_h, max_scale=0.9)
                    if wardrobe_scale >= 0.6:
                        wardrobe_w = SYMBOL_VIEWBOX_SIZES["wardrobe"][0] * wardrobe_scale
                        wardrobe_h = SYMBOL_VIEWBOX_SIZES["wardrobe"][1] * wardrobe_scale
                        wardrobe_x = svg_center_x - wardrobe_w / 2
                        if bed_side == "top":
                            wardrobe_y = svg_top + svg_height - wardrobe_h - pad - door_clearances["bottom"]
                        else:
                            wardrobe_y = svg_top + pad + door_clearances["top"]
                        _use_symbol(g, "wardrobe", wardrobe_x, wardrobe_y, scale=wardrobe_scale)
        else:
            bed_scale = _fit_symbol_scale("bed-double", svg_width * 0.78, svg_height - 2 * pad, max_scale=0.82)
            if bed_scale >= 0.5:
                bed_w = SYMBOL_VIEWBOX_SIZES["bed-double"][0] * bed_scale
                bed_h = SYMBOL_VIEWBOX_SIZES["bed-double"][1] * bed_scale
                bed_side = _prefer_side("left", "right", door_clearances)
                if bed_side == "left":
                    bed_x = svg_left + pad + door_clearances["left"]
                else:
                    bed_x = svg_left + svg_width - bed_w - pad - door_clearances["right"]
                bed_y = svg_center_y - bed_h / 2
                _use_symbol(g, "bed-double", bed_x, bed_y, scale=bed_scale)
                remaining_w = svg_width - bed_w - (2 * pad)
                if remaining_w >= 36 and svg_height >= 160:
                    wardrobe_scale = _fit_symbol_scale("wardrobe", remaining_w, svg_height * 0.45, max_scale=0.85)
                    if wardrobe_scale >= 0.55:
                        wardrobe_w = SYMBOL_VIEWBOX_SIZES["wardrobe"][0] * wardrobe_scale
                        wardrobe_h = SYMBOL_VIEWBOX_SIZES["wardrobe"][1] * wardrobe_scale
                        if bed_side == "left":
                            wardrobe_x = svg_left + svg_width - wardrobe_w - pad - door_clearances["right"]
                        else:
                            wardrobe_x = svg_left + pad + door_clearances["left"]
                        wardrobe_y = svg_center_y - wardrobe_h / 2
                        _use_symbol(g, "wardrobe", wardrobe_x, wardrobe_y, scale=wardrobe_scale)

    elif room_type == "Kitchen":
        if svg_width < 110 or svg_height < 90:
            return
        if room_width > room_height and svg_width > 180:
            counter_scale = _fit_symbol_scale("kitchen-counter", svg_width - 2 * pad, svg_height * 0.35, max_scale=0.82)
            if counter_scale >= 0.4:
                counter_w = SYMBOL_VIEWBOX_SIZES["kitchen-counter"][0] * counter_scale
                counter_x = svg_center_x - counter_w / 2
                counter_side = _prefer_side("top", "bottom", door_clearances)
                if counter_side == "top":
                    counter_y = svg_top + pad + door_clearances["top"]
                else:
                    counter_y = svg_top + svg_height - (SYMBOL_VIEWBOX_SIZES["kitchen-counter"][1] * counter_scale) - pad - door_clearances["bottom"]
                _use_symbol(g, "kitchen-counter", counter_x, counter_y, scale=counter_scale)
        elif svg_height > 170 and svg_width > 120:
            counter_scale = _fit_symbol_scale("kitchen-counter", svg_height - 2 * pad, svg_width * 0.42, max_scale=0.72)
            if counter_scale >= 0.38:
                counter_w = SYMBOL_VIEWBOX_SIZES["kitchen-counter"][0] * counter_scale
                counter_h = SYMBOL_VIEWBOX_SIZES["kitchen-counter"][1] * counter_scale
                counter_side = _prefer_side("left", "right", door_clearances)
                if counter_side == "left":
                    counter_x = svg_left + pad + door_clearances["left"]
                else:
                    counter_x = svg_left + svg_width - counter_h - pad - door_clearances["right"]
                counter_y = svg_center_y - counter_w / 2
                _use_symbol(
                    g,
                    "kitchen-counter",
                    counter_x,
                    counter_y,
                    rotation=90,
                    scale=counter_scale,
                    rotate_about="center",
                )

    elif room_type == "LivingRoom" or room_type == "DrawingRoom":
        if svg_width < 170 or svg_height < 100:
            return
        sofa_scale = _fit_symbol_scale("sofa", svg_width * 0.65, svg_height * 0.3, max_scale=0.82)
        if sofa_scale >= 0.45:
            sofa_w = SYMBOL_VIEWBOX_SIZES["sofa"][0] * sofa_scale
            sofa_h = SYMBOL_VIEWBOX_SIZES["sofa"][1] * sofa_scale
            sofa_side = _prefer_side("top", "bottom", door_clearances)
            if sofa_side == "top":
                sofa_y = svg_top + pad + door_clearances["top"]
            else:
                sofa_y = svg_top + svg_height - sofa_h - pad - door_clearances["bottom"]
            sofa_x = svg_center_x - sofa_w / 2
            _use_symbol(g, "sofa", sofa_x, sofa_y, scale=sofa_scale)

    elif room_type == "DiningRoom":
        # Dining table with chairs
        if svg_width > 120 and svg_height > 100:
            table_scale = _fit_symbol_scale("dining-table", svg_width * 0.45, svg_height * 0.35, max_scale=0.9)
            if table_scale < 0.5:
                return
            table_w = SYMBOL_VIEWBOX_SIZES["dining-table"][0] * table_scale
            table_h = SYMBOL_VIEWBOX_SIZES["dining-table"][1] * table_scale
            table_x = svg_center_x - table_w / 2
            table_y = svg_center_y - table_h / 2
            _use_symbol(g, "dining-table", table_x, table_y, scale=table_scale)

            # Chairs around table
            for i, (dx, dy) in enumerate([(-60, -20), (60, -20), (-60, 44), (60, 44)]):
                chair_scale = min(table_scale, 0.8)
                chair_w = SYMBOL_VIEWBOX_SIZES["chair"][0] * chair_scale
                chair_h = SYMBOL_VIEWBOX_SIZES["chair"][1] * chair_scale
                chair_x = table_x + (dx * chair_scale)
                chair_y = table_y + (dy * chair_scale)
                # Check bounds
                if (chair_x > svg_left + 10 and chair_x + chair_w < svg_left + svg_width - 10 and
                    chair_y > svg_top + 10 and chair_y + chair_h < svg_top + svg_height - 10):
                    _use_symbol(g, "chair", chair_x, chair_y, scale=chair_scale)

    elif room_type == "Bathroom" or room_type == "WC":
        if svg_width < 95 or svg_height < 95:
            return
        toilet_scale = _fit_symbol_scale("toilet", svg_width * 0.48, svg_height * 0.5, max_scale=0.8)
        if toilet_scale >= 0.45:
            toilet_w = SYMBOL_VIEWBOX_SIZES["toilet"][0] * toilet_scale
            toilet_h = SYMBOL_VIEWBOX_SIZES["toilet"][1] * toilet_scale
            toilet_side_x = _prefer_side("left", "right", door_clearances)
            toilet_side_y = _prefer_side("top", "bottom", door_clearances)
            toilet_x = svg_left + pad + door_clearances["left"]
            if toilet_side_x == "right":
                toilet_x = svg_left + svg_width - toilet_w - pad - door_clearances["right"]
            toilet_y = svg_top + pad + door_clearances["top"]
            if toilet_side_y == "bottom":
                toilet_y = svg_top + svg_height - toilet_h - pad - door_clearances["bottom"]
            _use_symbol(g, "toilet", toilet_x, toilet_y, scale=toilet_scale)

        if svg_width > 170 and svg_height > 110:
            tub_scale = _fit_symbol_scale("bathtub", svg_width * 0.58, svg_height * 0.28, max_scale=0.8)
            if tub_scale >= 0.4:
                tub_w = SYMBOL_VIEWBOX_SIZES["bathtub"][0] * tub_scale
                tub_x = svg_left + svg_width - tub_w - pad
                tub_y = svg_top + pad
                _use_symbol(g, "bathtub", tub_x, tub_y, scale=tub_scale)


def _door_room_family(room) -> str:
    room_type = getattr(room, "room_type", "") or ""
    if room_type in {"LivingRoom", "DrawingRoom", "DiningRoom", "Lobby", "Foyer"}:
        return "public"
    if room_type in {"Bedroom", "Study", "DressingArea", "PrayerRoom"}:
        return "private"
    if room_type in {"Bathroom", "WC", "Toilet"}:
        return "sanitary"
    if room_type in {"Kitchen", "Store", "Utility", "Pantry", "Laundry", "Garage"}:
        return "service"
    return "other"


def _door_room_center_px(room, ox: float, oy: float) -> Tuple[float, float]:
    min_x, min_y, max_x, max_y = _room_bbox(room)
    return (
        ox + _px((min_x + max_x) / 2.0),
        oy + _px((min_y + max_y) / 2.0),
    )


def _select_door_swing_room(door):
    if getattr(door, "door_type", "") == "exit":
        return None

    room_a = getattr(door, "room_a", None)
    room_b = getattr(door, "room_b", None)
    if room_a is None:
        return room_b
    if room_b is None:
        return room_a

    priority = {
        "sanitary": 4.0,
        "private": 3.0,
        "service": 2.0,
        "other": 1.5,
        "public": 1.0,
    }
    family_a = _door_room_family(room_a)
    family_b = _door_room_family(room_b)
    if priority[family_a] != priority[family_b]:
        return room_a if priority[family_a] > priority[family_b] else room_b

    area_a = float(getattr(room_a, "final_area", 0.0) or getattr(room_a, "requested_area", 0.0) or 0.0)
    area_b = float(getattr(room_b, "final_area", 0.0) or getattr(room_b, "requested_area", 0.0) or 0.0)
    return room_a if area_a <= area_b else room_b


def _door_swing_geometry(door, ox: float, oy: float):
    if getattr(door, "segment", None) is None:
        return None

    (sx1, sy1), (sx2, sy2) = door.segment
    p1 = (ox + _px(sx1), oy + _px(sy1))
    p2 = (ox + _px(sx2), oy + _px(sy2))
    door_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if door_len < 2.0:
        return None

    horizontal = abs(p2[0] - p1[0]) >= abs(p2[1] - p1[1])
    midpoint = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
    swing_room = _select_door_swing_room(door)

    if swing_room is not None:
        center_x, center_y = _door_room_center_px(swing_room, ox, oy)
        if horizontal:
            hinge = p1 if abs(center_x - p1[0]) <= abs(center_x - p2[0]) else p2
            free = p2 if hinge == p1 else p1
            sign = 1.0 if center_y >= midpoint[1] else -1.0
            open_end = (hinge[0], hinge[1] + sign * door_len)
        else:
            hinge = p1 if abs(center_y - p1[1]) <= abs(center_y - p2[1]) else p2
            free = p2 if hinge == p1 else p1
            sign = 1.0 if center_x >= midpoint[0] else -1.0
            open_end = (hinge[0] + sign * door_len, hinge[1])
    else:
        hinge = p1
        free = p2
        if horizontal:
            open_end = (hinge[0], hinge[1] - door_len)
        else:
            open_end = (hinge[0] - door_len, hinge[1])

    return {
        "opening_start": p1,
        "opening_end": p2,
        "hinge": hinge,
        "free": free,
        "open_end": open_end,
    }


def _render_room_aware_door(g: Element, door, ox: float, oy: float):
    geom = _door_swing_geometry(door, ox, oy)
    if geom is None:
        return

    p1 = geom["opening_start"]
    p2 = geom["opening_end"]
    hinge = geom["hinge"]
    free = geom["free"]
    open_end = geom["open_end"]

    SubElement(g, "line", {
        "x1": f"{p1[0]:.1f}", "y1": f"{p1[1]:.1f}",
        "x2": f"{p2[0]:.1f}", "y2": f"{p2[1]:.1f}",
        "stroke": _theme("door_gap"),
        "stroke-width": str(WALL_WIDTH + 2),
    })

    is_exit = getattr(door, "door_type", "") == "exit"
    leaf_stroke = _theme("entrance") if is_exit else _theme("door_leaf")
    arc_stroke = _theme("entrance") if is_exit else _theme("door_arc")

    SubElement(g, "line", {
        "x1": f"{hinge[0]:.1f}", "y1": f"{hinge[1]:.1f}",
        "x2": f"{open_end[0]:.1f}", "y2": f"{open_end[1]:.1f}",
        "stroke": leaf_stroke,
        "stroke-width": "1.8" if is_exit else str(DOOR_WIDTH_PX),
        "stroke-linecap": "round",
    })

    SubElement(g, "path", {
        "d": f"M{free[0]:.1f},{free[1]:.1f} Q{hinge[0]:.1f},{hinge[1]:.1f} {open_end[0]:.1f},{open_end[1]:.1f}",
        "fill": "none",
        "stroke": arc_stroke,
        "stroke-width": "1.8" if is_exit else "1.4",
        "stroke-dasharray": "5,3" if is_exit else "4,3",
        "stroke-linecap": "round",
    })


def _enhanced_draw_door(g: Element, door, ox: float, oy: float, use_symbols: bool = True):
    """Enhanced door rendering with optional symbol library usage."""
    _render_room_aware_door(g, door, ox, oy)




# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Wall rendering via extraction+merge+door-gap carving
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _draw_merged_walls(
    g: Element,
    building: Building,
    boundary_polygon,
    entrance_point,
    ox: float,
    oy: float,
    *,
    wall_snap_step: float = 0.15,
    door_gap_eps: float = 0.1,
    door_attach_tol: float = 0.06,
    show_windows: bool = True,
    window_gap_eps: float = 0.08,
    window_corner_margin: float = 0.25,
    window_min_length: float = 0.55,
    min_overlap_ratio: float = 0.8,
):
    """Draw merged wall segments with door openings carved as real gaps."""
    polygons = []
    for room in building.rooms:
        if room.polygon:
            polygons.append(room.polygon)
    for corr in getattr(building, "corridors", []):
        if getattr(corr, "polygon", None):
            polygons.append(corr.polygon)

    door_segments = [d.segment for d in building.doors if getattr(d, "segment", None)]
    window_segments = []
    if show_windows:
        window_segments = suggest_window_segments(
            building.rooms,
            boundary_polygon,
            door_segments=door_segments,
            entrance_point=entrance_point,
            corner_margin=window_corner_margin,
            min_window_len=window_min_length,
        )

    wall_segments, wall_stats = build_wall_segments(
        polygons,
        door_segments,
        window_segments=window_segments,
        boundary_polygon=boundary_polygon,
        snap_step=wall_snap_step,
        door_eps=door_gap_eps,
        door_attach_tol=door_attach_tol,
        window_eps=window_gap_eps,
        min_overlap_ratio=min_overlap_ratio,
        return_stats=True,
    )

    # Expose renderer wall stats for downstream reporting/debugging.
    building.wall_render_stats = wall_stats
    building.window_segments = window_segments

    for item in wall_segments:
        (x1, y1), (x2, y2) = item["segment"]
        wall_type = item.get("wall_type", "inner")
        if wall_type == "outer":
            stroke = _theme("outer_wall")
            width = WALL_WIDTH
        else:
            stroke = _theme("inner_wall")
            width = INNER_WALL_WIDTH

        SubElement(g, "line", {
            "x1": f"{ox + _px(x1):.1f}", "y1": f"{oy + _px(y1):.1f}",
            "x2": f"{ox + _px(x2):.1f}", "y2": f"{oy + _px(y2):.1f}",
            "stroke": stroke,
            "stroke-width": str(width),
            "stroke-linecap": "round",
        })


def _draw_window(g: Element, segment, ox: float, oy: float):
    (sx1, sy1), (sx2, sy2) = segment
    px1 = ox + _px(sx1)
    py1 = oy + _px(sy1)
    px2 = ox + _px(sx2)
    py2 = oy + _px(sy2)

    # Symbol line for window glazing.
    SubElement(g, "line", {
        "x1": f"{px1:.1f}", "y1": f"{py1:.1f}",
        "x2": f"{px2:.1f}", "y2": f"{py2:.1f}",
        "stroke": _theme("window"),
        "stroke-width": str(WINDOW_WIDTH_PX),
        "stroke-linecap": "round",
    })

    # End ticks mimic a standard drafting window marker.
    tick = 3.0
    if abs(py1 - py2) < 1e-6:
        # Horizontal window segment.
        SubElement(g, "line", {
            "x1": f"{px1:.1f}", "y1": f"{py1 - tick:.1f}",
            "x2": f"{px1:.1f}", "y2": f"{py1 + tick:.1f}",
            "stroke": _theme("window_tick"), "stroke-width": "0.9",
        })
        SubElement(g, "line", {
            "x1": f"{px2:.1f}", "y1": f"{py2 - tick:.1f}",
            "x2": f"{px2:.1f}", "y2": f"{py2 + tick:.1f}",
            "stroke": _theme("window_tick"), "stroke-width": "0.9",
        })
    else:
        # Vertical window segment.
        SubElement(g, "line", {
            "x1": f"{px1 - tick:.1f}", "y1": f"{py1:.1f}",
            "x2": f"{px1 + tick:.1f}", "y2": f"{py1:.1f}",
            "stroke": _theme("window_tick"), "stroke-width": "0.9",
        })
        SubElement(g, "line", {
            "x1": f"{px2 - tick:.1f}", "y1": f"{py2:.1f}",
            "x2": f"{px2 + tick:.1f}", "y2": f"{py2:.1f}",
            "stroke": _theme("window_tick"), "stroke-width": "0.9",
        })


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Overall boundary dimension strings
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _draw_boundary_dims(g: Element, boundary_polygon, ox: float, oy: float, bw=None, bh=None):
    """Draw overall width / height dimension strings along boundary extents."""
    if boundary_polygon and len(boundary_polygon) >= 3:
        xs = [p[0] for p in boundary_polygon]
        ys = [p[1] for p in boundary_polygon]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        bw = x1 - x0
        bh = y1 - y0
    elif bw is not None and bh is not None:
        x0, y0, x1, y1 = 0, 0, bw, bh
    else:
        return

    # Bottom overall width
    dim_y = oy + _px(y1) + 28
    draw_dimension(
        g,
        (ox + _px(x0), dim_y),
        (ox + _px(x1), dim_y),
        text=f"{bw:.2f} m",
        color=_theme("dimension"),
        text_color=_theme("dimension"),
    )

    # Right overall height
    dim_x = ox + _px(x1) + 28
    draw_dimension(
        g,
        (dim_x, oy + _px(y0)),
        (dim_x, oy + _px(y1)),
        text=f"{bh:.2f} m",
        vertical=True,
        color=_theme("dimension"),
        text_color=_theme("dimension"),
    )


# —————————————————————————————————————————————————————————————————————————————
#  Metric grid overlay
# —————————————————————————————————————————————————————————————————————————————

def _draw_grid_overlay(g: Element, boundary_polygon, ox: float, oy: float,
                       step_m: float = 1.0):
    """Light 1 m grid lines inside the boundary extent."""
    if not boundary_polygon:
        return
    xs = [p[0] for p in boundary_polygon]
    ys = [p[1] for p in boundary_polygon]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

    m = x0
    while m <= x1:
        px = ox + _px(m)
        SubElement(g, "line", {
            "x1": f"{px:.1f}", "y1": f"{oy + _px(y0):.1f}",
            "x2": f"{px:.1f}", "y2": f"{oy + _px(y1):.1f}",
            "stroke": _theme("grid_major"), "stroke-width": "0.3",
        })
        m += step_m

    m = y0
    while m <= y1:
        py = oy + _px(m)
        SubElement(g, "line", {
            "x1": f"{ox + _px(x0):.1f}", "y1": f"{py:.1f}",
            "x2": f"{ox + _px(x1):.1f}", "y2": f"{py:.1f}",
            "stroke": _theme("grid_major"), "stroke-width": "0.3",
        })
        m += step_m


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Room rendering
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _draw_room(g: Element, room, ox: float, oy: float, zone: str = "",
               draw_walls: bool = False, show_room_dim_tags: bool = True):
    """Draw room fill, label, and dimension strings.

    When *draw_walls* is False (default), only the filled polygon and text
    are emitted â€” wall strokes are handled separately by ``_draw_merged_walls``
    so shared walls are drawn once with correct thickness.
    """
    if room.polygon is None:
        return
    fill = _theme("room_fill") or _room_fill_color(room.room_type, zone)

    # Room filled polygon
    d = _polygon_path(room.polygon, ox, oy)
    attrs = {
        "d": d,
        "fill": fill,
        "fill-opacity": _theme("room_fill_opacity") or "1.0",
    }
    if draw_walls:
        attrs["stroke"] = _theme("inner_wall")
        attrs["stroke-width"] = str(INNER_WALL_WIDTH)
        attrs["stroke-linejoin"] = "miter"
    else:
        attrs["stroke"] = "none"
    SubElement(g, "path", attrs)



# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _draw_room_label(g, room, ox: float, oy: float):
    """Draw room name + area label drawn AFTER all wall layers so it is always visible."""
    if room.polygon is None:
        return
    xs = [p[0] for p in room.polygon]
    ys = [p[1] for p in room.polygon]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    w_px = (x2 - x1) * 80  # SCALE=80
    h_px = (y2 - y1) * 80
    if w_px < 24 or h_px < 16:
        return
    cx = ox + (x1 + x2) / 2 * 80
    cy = oy + (y1 + y2) / 2 * 80
    from visualization.dimensions import measure_room_dims
    w_m, h_m = measure_room_dims(room.polygon)
    area = w_m * h_m

    from xml.etree.ElementTree import SubElement
    label_text = room.name.replace("_", " ").replace("LivingRoom", "Living Room").replace("DiningRoom", "Dining Room").upper()
    lbl = SubElement(g, "text", {
        "x": f"{cx:.1f}", "y": f"{cy - 4:.1f}",
        "text-anchor": "middle",
        "font-size": "14" if _is_presentation_blueprint() else "13",
        "font-weight": "700",
        "fill": _theme("label_primary"),
    })
    lbl.text = label_text

    dim = SubElement(g, "text", {
        "x": f"{cx:.1f}", "y": f"{cy + 12:.1f}",
        "text-anchor": "middle",
        "font-size": "10",
        "fill": _theme("label_secondary"),
    })
    dim.text = f"{w_m:.2f} m x {h_m:.2f} m"

#  Door rendering (opening + swing arc)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _draw_door(g: Element, door, ox: float, oy: float):
    _render_room_aware_door(g, door, ox, oy)
    return
    if door.segment is None:
        return
    (sx1, sy1), (sx2, sy2) = door.segment
    px1 = ox + _px(sx1)
    py1 = oy + _px(sy1)
    px2 = ox + _px(sx2)
    py2 = oy + _px(sy2)

    # White gap (erase wall)
    SubElement(g, "line", {
        "x1": f"{px1:.1f}", "y1": f"{py1:.1f}",
        "x2": f"{px2:.1f}", "y2": f"{py2:.1f}",
        "stroke": _theme("door_gap"), "stroke-width": str(WALL_WIDTH + 2),
    })

    # Door leaf lines
    SubElement(g, "line", {
        "x1": f"{px1:.1f}", "y1": f"{py1:.1f}",
        "x2": f"{px2:.1f}", "y2": f"{py2:.1f}",
        "stroke": _theme("door_leaf"), "stroke-width": str(DOOR_WIDTH_PX),
    })

    # Swing arc
    door_len = math.hypot(px2 - px1, py2 - py1)
    if door_len < 2:
        return
    # Arc from hinge (px1,py1) sweeping 90Â° with radius = door width
    r = door_len
    # Determine swing direction based on door type (outward for room doors)
    dx = px2 - px1
    dy = py2 - py1
    # Perpendicular direction
    if abs(dy) > abs(dx):
        # Vertical door â†’ arc horizontally
        arc_x = px1 + r
        arc_y = py1
    else:
        # Horizontal door â†’ arc vertically
        arc_x = px1
        arc_y = py1 - r

    is_exit = getattr(door, "door_type", "") == "exit"
    arc_stroke = _theme("entrance") if is_exit else _theme("door_arc")
    arc_width = "1.8" if is_exit else "1.5"
    arc_dash = "5,3" if is_exit else "4,3"

    SubElement(g, "path", {
        "d": f"M{px2:.1f},{py2:.1f} A{r:.1f},{r:.1f} 0 0 1 {arc_x:.1f},{arc_y:.1f}",
        "fill": "none",
        "stroke": arc_stroke,
        "stroke-width": arc_width,
        "stroke-dasharray": arc_dash,
    })


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Corridor rendering
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _draw_corridor(g: Element, corridor, ox: float, oy: float):
    if corridor.polygon is None:
        return
    d = _polygon_path(corridor.polygon, ox, oy)
    SubElement(g, "path", {
        "d": d,
        "fill": "url(#corridor-hatch)",
        "stroke": _theme("corridor_stroke"),
        "stroke-width": "1",
        "stroke-dasharray": "4,2",
    })
    # Corridor label
    if corridor.polygon:
        xs = [p[0] for p in corridor.polygon]
        ys = [p[1] for p in corridor.polygon]
        cx = ox + _px((min(xs) + max(xs)) / 2)
        cy = oy + _px((min(ys) + max(ys)) / 2)
        lbl = SubElement(g, "text", {
            "x": f"{cx:.1f}", "y": f"{cy:.1f}",
            "text-anchor": "middle", "font-size": "7",
            "fill": _theme("corridor_label"), "font-style": "italic",
        })
        lbl.text = "Corridor"


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Boundary + title block
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _draw_boundary(g: Element, boundary_polygon, ox: float, oy: float):
    d = _polygon_path(boundary_polygon, ox, oy)
    SubElement(g, "path", {
        "d": d,
        "fill": "none",
        "stroke": _theme("outer_wall"),
        "stroke-width": str(WALL_WIDTH + 1),
        "stroke-linejoin": "miter",
        "filter": "url(#shadow)",
    })


def _draw_entrance(g: Element, boundary_polygon, entrance_point, ox: float, oy: float, exit_width: float = 1.0):
    if entrance_point is None:
        return

    ex, ey = entrance_point
    px = ox + _px(ex)
    py = oy + _px(ey)
    half = _px(exit_width) / 2.0

    side = "top"
    if boundary_polygon:
        xs = [p[0] for p in boundary_polygon]
        ys = [p[1] for p in boundary_polygon]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        distances = {
            "left": abs(ex - x_min),
            "right": abs(ex - x_max),
            "top": abs(ey - y_min),
            "bottom": abs(ey - y_max),
        }
        side = min(distances, key=distances.get)

    if side in {"top", "bottom"}:
        y = py
        x1 = px - half
        x2 = px + half
        SubElement(g, "line", {
            "x1": f"{x1:.1f}", "y1": f"{y:.1f}",
            "x2": f"{x2:.1f}", "y2": f"{y:.1f}",
            "stroke": _theme("door_gap"), "stroke-width": str(WALL_WIDTH + 3),
        })
        arrow_start_y = y - 15 if side == "top" else y + 15
        arrow_head_y = y - 6 if side == "top" else y + 6
        label_y = y - 22 if side == "top" else y + 24
        SubElement(g, "path", {
            "d": f"M{px:.1f},{arrow_start_y:.1f} L{px:.1f},{y:.1f} "
                 f"L{px - 4:.1f},{arrow_head_y:.1f} M{px:.1f},{y:.1f} "
                 f"L{px + 4:.1f},{arrow_head_y:.1f}",
            "stroke": _theme("entrance"), "stroke-width": "1.5", "fill": "none",
        })
        label_x = px
    else:
        x = px
        y1 = py - half
        y2 = py + half
        SubElement(g, "line", {
            "x1": f"{x:.1f}", "y1": f"{y1:.1f}",
            "x2": f"{x:.1f}", "y2": f"{y2:.1f}",
            "stroke": _theme("door_gap"), "stroke-width": str(WALL_WIDTH + 3),
        })
        arrow_start_x = x - 15 if side == "left" else x + 15
        arrow_head_x = x - 6 if side == "left" else x + 6
        label_x = x - 34 if side == "left" else x + 34
        label_y = py + 3
        SubElement(g, "path", {
            "d": f"M{arrow_start_x:.1f},{py:.1f} L{x:.1f},{py:.1f} "
                 f"L{arrow_head_x:.1f},{py - 4:.1f} M{x:.1f},{py:.1f} "
                 f"L{arrow_head_x:.1f},{py + 4:.1f}",
            "stroke": _theme("entrance"), "stroke-width": "1.5", "fill": "none",
        })

    lbl = SubElement(g, "text", {
        "x": f"{label_x:.1f}", "y": f"{label_y:.1f}",
        "text-anchor": "middle", "font-size": "8", "fill": _theme("entrance"),
        "font-weight": "bold",
    })
    lbl.text = "ENTRANCE"


def _draw_title_block(svg: Element, title: str, width_px: float, height_px: float,
                      total_area: float = 0, occupancy: str = "Residential"):
    tb_h = 60
    tb_y = height_px - tb_h
    g = SubElement(svg, "g")
    SubElement(g, "rect", {
        "x": "0", "y": f"{tb_y:.0f}",
        "width": f"{width_px:.0f}", "height": f"{tb_h}",
        "fill": _theme("footer_bg"),
    })
    t = SubElement(g, "text", {
        "x": f"{MARGIN:.0f}", "y": f"{tb_y + 22:.0f}",
        "font-size": "16", "font-weight": "bold", "fill": _theme("footer_text"),
    })
    t.text = title

    info = SubElement(g, "text", {
        "x": f"{MARGIN:.0f}", "y": f"{tb_y + 38:.0f}",
        "font-size": "11", "fill": _theme("footer_muted"),
    })
    info.text = (f"Occupancy: {occupancy}  |  Total Area: {total_area:.1f} sq.m  |  "
                 f"Scale: 1:{SCALE}  |  GenAI Floor Plan Generator")

    # Scale bar
    bar_x = width_px - LEGEND_W - 20 - _px(3)
    bar_y = tb_y + 28
    bar_w = _px(3)
    SubElement(g, "line", {
        "x1": f"{bar_x:.0f}", "y1": f"{bar_y:.0f}",
        "x2": f"{bar_x + bar_w:.0f}", "y2": f"{bar_y:.0f}",
        "stroke": _theme("footer_text"), "stroke-width": "2",
    })
    for i in range(4):
        tx = bar_x + _px(i)
        SubElement(g, "line", {
            "x1": f"{tx:.0f}", "y1": f"{bar_y - 3:.0f}",
            "x2": f"{tx:.0f}", "y2": f"{bar_y + 3:.0f}",
            "stroke": _theme("footer_text"), "stroke-width": "1",
        })
        st = SubElement(g, "text", {
            "x": f"{tx:.0f}", "y": f"{bar_y + 12:.0f}",
            "text-anchor": "middle", "font-size": "9", "fill": _theme("footer_muted"),
        })
        st.text = f"{i}m"


def _derive_plan_title(building: Building, title: str) -> str:
    cleaned = (title or "").strip()
    generic_titles = {
        "Floor Plan",
        "AI-Generated Floor Plan",
        "NL Interface - Algorithmic Run",
        "NL Interface - Planner Direct Run",
        "NL Interface - Learned Run",
        "NL Interface - Hybrid Run",
    }
    if cleaned and cleaned not in generic_titles:
        return cleaned.upper()

    bedroom_count = sum(1 for room in building.rooms if getattr(room, "room_type", "") == "Bedroom")
    if bedroom_count > 0:
        return f"{bedroom_count} BHK HOME FLOOR PLAN"
    return "RESIDENTIAL FLOOR PLAN"


def _draw_title_banner(svg: Element, title: str, width_px: float) -> None:
    banner_width = min(width_px * 0.56, 760)
    banner_x = (width_px - banner_width) / 2.0
    banner_y = 26
    banner_h = 60

    g = SubElement(svg, "g", {"id": "title-banner"})
    if _theme("title_banner_bg"):
        SubElement(g, "rect", {
            "x": f"{banner_x:.1f}",
            "y": f"{banner_y:.1f}",
            "width": f"{banner_width:.1f}",
            "height": f"{banner_h:.1f}",
            "rx": "10",
            "fill": _theme("title_banner_bg"),
            "fill-opacity": "0.55" if _is_presentation_blueprint() else "1.0",
            "stroke": _theme("title_banner_rule"),
            "stroke-width": "1",
        })
    text = SubElement(g, "text", {
        "x": f"{width_px / 2.0:.1f}",
        "y": f"{banner_y + 30:.1f}",
        "text-anchor": "middle",
        "font-size": "26" if _is_presentation_blueprint() else "20",
        "font-weight": "800",
        "letter-spacing": "1.2",
        "fill": _theme("title_banner_text"),
    })
    text.text = title
    SubElement(g, "line", {
        "x1": f"{banner_x + 40:.1f}",
        "y1": f"{banner_y + banner_h - 12:.1f}",
        "x2": f"{banner_x + banner_width - 40:.1f}",
        "y2": f"{banner_y + banner_h - 12:.1f}",
        "stroke": _theme("title_banner_rule"),
        "stroke-width": "1.4",
        "opacity": "0.9",
    })


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Compass rose
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _draw_compass(svg: Element, x: float, y: float):
    g = SubElement(svg, "g", {"transform": f"translate({x:.0f},{y:.0f})"})
    # N arrow
    SubElement(g, "path", {
        "d": "M0,-20 L5,-5 L0,-10 L-5,-5 Z",
        "fill": _theme("entrance"), "stroke": _theme("compass"), "stroke-width": "0.5",
    })
    n = SubElement(g, "text", {
        "x": "0", "y": "-24", "text-anchor": "middle",
        "font-size": "10", "font-weight": "bold", "fill": _theme("compass"),
    })
    n.text = "N"


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Main renderer
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def render_svg_blueprint(
    building: Building,
    boundary_polygon: List[Tuple[float, float]] = None,
    entrance_point: Tuple[float, float] = None,
    zone_map: Dict[str, str] = None,
    title: str = "Floor Plan",
    *,
    style_preset: str = STYLE_PRESET_PRESENTATION_BLUEPRINT,
    unit: str = "m",
    show_grid: bool = True,
    merge_walls: bool = True,
    wall_snap_step: float = 0.15,
    door_gap_eps: float = 0.1,
    door_attach_tol: float = 0.06,
    show_windows: bool = True,
    window_gap_eps: float = 0.08,
    min_overlap_ratio: float = 0.8,
    show_room_dim_tags: bool = True,
    # ── New symbol library parameters ──────────────────────────────────────
    furniture_enabled: bool = True,
    use_symbol_library: bool = True,
    show_north_arrow: bool = True,
) -> str:
    """Render a Building into a professional SVG blueprint string.

    Parameters
    ----------
    show_grid : bool
        Draw a light 1 m metric grid inside the boundary.
    merge_walls : bool
        Use the wall-merge engine so shared walls are drawn once.
    furniture_enabled : bool
        Automatically place furniture symbols in rooms based on room type.
        Includes beds, sofas, kitchen counters, dining tables, toilets, etc.
    use_symbol_library : bool
        Use reusable SVG symbols for doors and furniture instead of inline SVG.
        Reduces file size and enables consistent styling.
    show_north_arrow : bool
        Display a north arrow symbol for orientation reference.
    """
    _set_active_theme(style_preset)
    units_cfg = resolve_render_units(unit=unit, wall_snap_step=wall_snap_step)
    wall_snap_step = units_cfg.wall_snap_step
    door_gap_eps = units_cfg.door_gap_eps if door_gap_eps == 0.1 else door_gap_eps
    door_attach_tol = units_cfg.door_attach_tol if door_attach_tol == 0.06 else door_attach_tol
    window_gap_eps = units_cfg.window_gap_eps if window_gap_eps == 0.08 else window_gap_eps

    # Determine canvas
    if boundary_polygon:
        bxs = [p[0] for p in boundary_polygon]
        bys = [p[1] for p in boundary_polygon]
        bw, bh = max(bxs) - min(bxs), max(bys) - min(bys)
    else:
        bw = max((r.polygon and max(p[0] for p in r.polygon) or 0) for r in building.rooms)
        bh = max((r.polygon and max(p[1] for p in r.polygon) or 0) for r in building.rooms)

    title_block_h = 50
    width_px = _px(bw) + 2 * MARGIN + 100 + LEGEND_W  # extra for dim + legend
    height_px = _px(bh) + 2 * MARGIN + title_block_h + 60
    ox, oy = MARGIN, MARGIN

    svg = _svg_root(width_px, height_px)
    _add_defs(svg, boundary_polygon)
    display_title = _derive_plan_title(building, title)
    _draw_title_banner(svg, display_title, width_px)

    # Grid overlay (behind everything)
    if show_grid:
        g_grid = SubElement(svg, "g", {"id": "grid"})
        _draw_grid_overlay(g_grid, boundary_polygon, ox, oy)

    # Corridors first (behind rooms)
    g_corr = SubElement(svg, "g", {"id": "corridors"})
    for corridor in getattr(building, "corridors", []):
        _draw_corridor(g_corr, corridor, ox, oy)

    # Rooms (fills only â€“ walls drawn separately when merge_walls=True)
    z = zone_map or {}
    g_rooms = SubElement(svg, "g", {"id": "rooms"})
    for room in building.rooms:
        zone = z.get(room.name, "")
        _draw_room(
            g_rooms,
            room,
            ox,
            oy,
            zone,
            draw_walls=not merge_walls,
            show_room_dim_tags=show_room_dim_tags,
        )

    if furniture_enabled:
        g_furniture = SubElement(svg, "g", {"id": "furniture", "opacity": "0.58" if _is_presentation_blueprint() else "1.0"})
        for room in building.rooms:
            _place_furniture_in_room(g_furniture, room, ox, oy, furniture_enabled=True)

    # Merged wall layer (shared walls drawn once, correct thickness)
    if merge_walls:
        g_walls = SubElement(svg, "g", {"id": "walls"})
        _draw_merged_walls(
            g_walls,
            building,
            boundary_polygon,
            entrance_point,
            ox,
            oy,
            wall_snap_step=wall_snap_step,
            door_gap_eps=door_gap_eps,
            door_attach_tol=door_attach_tol,
            show_windows=show_windows,
            window_gap_eps=window_gap_eps,
            window_corner_margin=units_cfg.window_corner_margin,
            window_min_length=units_cfg.window_min_length,
            min_overlap_ratio=min_overlap_ratio,
        )
    else:
        # Boundary (thick outer wall)
        g_boundary = SubElement(svg, "g", {"id": "boundary"})
        if boundary_polygon:
            _draw_boundary(g_boundary, boundary_polygon, ox, oy)

    # Entrance
    g_entrance = SubElement(svg, "g", {"id": "entrance"})
    exit_w = building.exit.width if building.exit else 1.0
    _draw_entrance(g_entrance, boundary_polygon, entrance_point, ox, oy, exit_w)

    render_exit_segment = _render_exit_segment(boundary_polygon, entrance_point) or (
        getattr(building.exit, "segment", None) if hasattr(building, "exit") and building.exit else None
    )
    # Doors
    g_doors = SubElement(svg, "g", {"id": "doors"})
    for door in building.doors:
        segment = getattr(door, "segment", None)
        if _segment_on_boundary(segment, boundary_polygon) and not _segments_match(segment, render_exit_segment):
            continue
        if use_symbol_library:
            _enhanced_draw_door(g_doors, door, ox, oy, use_symbols=True)
        else:
            _draw_door(g_doors, door, ox, oy)

    if render_exit_segment:
        class ExitDoor:
            segment = render_exit_segment
            door_type = "exit"
        if use_symbol_library:
            _enhanced_draw_door(g_doors, ExitDoor(), ox, oy, use_symbols=True)
        else:
            _draw_door(g_doors, ExitDoor(), ox, oy)

    # Room labels drawn AFTER all walls and doors so they are never obscured
    g_labels = SubElement(svg, "g", {"id": "room-labels"})
    for room in building.rooms:
        _draw_room_label(g_labels, room, ox, oy)

    # Overall boundary dimension strings
    g_dims = SubElement(svg, "g", {"id": "boundary-dims"})
    _draw_boundary_dims(g_dims, boundary_polygon, ox, oy, bw=bw, bh=bh)

    # Title block
    total_area = building.total_area or sum(r.final_area for r in building.rooms if r.final_area)
    _draw_title_block(svg, display_title, width_px, height_px, total_area, building.occupancy_type)

    # Compass
    _draw_compass(svg, width_px - LEGEND_W - 40, 40)

    # Legend -- top-right, outside the plan
    legend_x = width_px - LEGEND_W + 6
    g_legend = SubElement(svg, "g", {"id": "legend",
                                      "transform": f"translate({legend_x:.0f}, {MARGIN:.0f})"})
    hdr = SubElement(g_legend, "text", {
        "x": "0", "y": "14",
        "font-size": "13", "font-weight": "bold", "fill": _theme("legend_text"),
    })
    hdr.text = "Legend"
    ly = 28
    seen_types = set()
    for room in building.rooms:
        if room.room_type in seen_types:
            continue
        seen_types.add(room.room_type)
        fill = ROOM_FILL.get(room.room_type, DEFAULT_FILL)
        SubElement(g_legend, "rect", {
            "x": "0", "y": f"{ly}",
            "width": "20", "height": "20",
            "fill": fill,
            "fill-opacity": _theme("room_fill_opacity") or "1.0",
            "stroke": _theme("legend_border"),
            "stroke-width": "1",
        })
        lt = SubElement(g_legend, "text", {
            "x": "28", "y": f"{ly + 14}",
            "font-size": "12", "fill": _theme("legend_text"),
        })
        lt.text = room.room_type
        ly += 28

    xml_str = tostring(svg, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str


def save_svg_blueprint(
    building: Building,
    output_path: str = "outputs/blueprint.svg",
    boundary_polygon=None,
    entrance_point=None,
    zone_map=None,
    title="Floor Plan",
    style_preset: str = STYLE_PRESET_PRESENTATION_BLUEPRINT,
):
    """Render and save SVG to file."""
    svg_str = render_svg_blueprint(
        building, boundary_polygon, entrance_point, zone_map, title,
        style_preset=style_preset,
    )
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(svg_str, encoding="utf-8")
    print(f"SVG blueprint saved â†’ {p}")
    return p


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  CLI
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def main():
    ap = argparse.ArgumentParser(description="Export SVG blueprint from a building")
    ap.add_argument("--demo", action="store_true", help="Run a demo with the learned generator")
    ap.add_argument("--checkpoint", default="learned/model/checkpoints/kaggle_test.pt")
    ap.add_argument("--boundary", default="15,10")
    ap.add_argument("--output", default="outputs/blueprint.svg")
    ap.add_argument("--title", default="AI-Generated Floor Plan")
    args = ap.parse_args()

    if args.demo:
        from learned.integration.model_generation_loop import generate_best_layout
        from learned.integration.repair_gate import evaluate_variant
        from geometry.zoning import assign_room_zones

        w, h = [float(x) for x in args.boundary.split(",")]
        boundary = [(0, 0), (w, 0), (w, h), (0, h)]
        entrance = (0.2, 0.0)

        print("Generating layout from trained model â€¦")
        result = generate_best_layout(
            checkpoint_path=args.checkpoint,
            boundary_polygon=boundary,
            entrance_point=entrance,
            K=5,
        )

        building = result.get("building")
        if building is None:
            print("Generation failed.")
            return

        zone_map = assign_room_zones(building, entrance_point=entrance)

        save_svg_blueprint(
            building,
            output_path=args.output,
            boundary_polygon=boundary,
            entrance_point=entrance,
            zone_map=zone_map,
            title=args.title,
        )
    else:
        print("Use --demo to generate a layout and export blueprint.")
        print("Or import save_svg_blueprint() directly with your Building object.")


if __name__ == "__main__":
    main()




