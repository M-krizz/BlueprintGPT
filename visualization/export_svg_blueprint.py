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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SVG building blocks
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _svg_root(width_px: float, height_px: float) -> Element:
    svg = Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": str(int(width_px)),
        "height": str(int(height_px)),
        "viewBox": f"0 0 {int(width_px)} {int(height_px)}",
        "font-family": "'Segoe UI', Helvetica, Arial, sans-serif",
    })
    # Background
    SubElement(svg, "rect", {
        "width": "100%", "height": "100%", "fill": "white",
    })
    return svg


def _add_defs(svg: Element, boundary_polygon):
    defs = SubElement(svg, "defs")
    # Hatch pattern for corridors
    patt = SubElement(defs, "pattern", {
        "id": "corridor-hatch",
        "patternUnits": "userSpaceOnUse",
        "width": "8", "height": "8",
    })
    SubElement(patt, "rect", {
        "width": "8", "height": "8", "fill": "#f5f5f5",
    })
    SubElement(patt, "path", {
        "d": "M0,8 l8,-8 M-2,2 l4,-4 M6,10 l4,-4",
        "stroke": "#bdbdbd", "stroke-width": "0.5",
    })
    # Drop shadow
    filt = SubElement(defs, "filter", {"id": "shadow", "x": "-2%", "y": "-2%",
                                        "width": "104%", "height": "104%"})
    SubElement(filt, "feDropShadow", {
        "dx": "1", "dy": "1", "stdDeviation": "2", "flood-opacity": "0.15",
    })


def _polygon_path(polygon, ox: float, oy: float) -> str:
    """Convert polygon [(x,y),...] to SVG path d string."""
    parts = []
    for i, (x, y) in enumerate(polygon):
        cmd = "M" if i == 0 else "L"
        parts.append(f"{cmd}{ox + _px(x):.1f},{oy + _px(y):.1f}")
    parts.append("Z")
    return " ".join(parts)


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
            stroke = "#263238"
            width = WALL_WIDTH
        else:
            stroke = "#37474f"
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
        "stroke": "#039be5",
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
            "stroke": "#0288d1", "stroke-width": "0.9",
        })
        SubElement(g, "line", {
            "x1": f"{px2:.1f}", "y1": f"{py2 - tick:.1f}",
            "x2": f"{px2:.1f}", "y2": f"{py2 + tick:.1f}",
            "stroke": "#0288d1", "stroke-width": "0.9",
        })
    else:
        # Vertical window segment.
        SubElement(g, "line", {
            "x1": f"{px1 - tick:.1f}", "y1": f"{py1:.1f}",
            "x2": f"{px1 + tick:.1f}", "y2": f"{py1:.1f}",
            "stroke": "#0288d1", "stroke-width": "0.9",
        })
        SubElement(g, "line", {
            "x1": f"{px2 - tick:.1f}", "y1": f"{py2:.1f}",
            "x2": f"{px2 + tick:.1f}", "y2": f"{py2:.1f}",
            "stroke": "#0288d1", "stroke-width": "0.9",
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
    )

    # Right overall height
    dim_x = ox + _px(x1) + 28
    draw_dimension(
        g,
        (dim_x, oy + _px(y0)),
        (dim_x, oy + _px(y1)),
        text=f"{bh:.2f} m",
        vertical=True,
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
            "stroke": "#e0e0e0", "stroke-width": "0.3",
        })
        m += step_m

    m = y0
    while m <= y1:
        py = oy + _px(m)
        SubElement(g, "line", {
            "x1": f"{ox + _px(x0):.1f}", "y1": f"{py:.1f}",
            "x2": f"{ox + _px(x1):.1f}", "y2": f"{py:.1f}",
            "stroke": "#e0e0e0", "stroke-width": "0.3",
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
    fill = _room_fill_color(room.room_type, zone)

    # Room filled polygon
    d = _polygon_path(room.polygon, ox, oy)
    attrs = {
        "d": d,
        "fill": fill,
    }
    if draw_walls:
        attrs["stroke"] = "#37474f"
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
    lbl = SubElement(g, "text", {
        "x": f"{cx:.1f}", "y": f"{cy - 4:.1f}",
        "text-anchor": "middle",
        "font-size": "13",
        "font-weight": "700",
        "fill": "#1a237e",
    })
    lbl.text = room.name.replace("_", " ")

    dim = SubElement(g, "text", {
        "x": f"{cx:.1f}", "y": f"{cy + 12:.1f}",
        "text-anchor": "middle",
        "font-size": "10",
        "fill": "#37474f",
    })
    dim.text = f"{area:.1f} m²"

#  Door rendering (opening + swing arc)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _draw_door(g: Element, door, ox: float, oy: float):
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
        "stroke": "white", "stroke-width": str(WALL_WIDTH + 2),
    })

    # Door leaf lines
    SubElement(g, "line", {
        "x1": f"{px1:.1f}", "y1": f"{py1:.1f}",
        "x2": f"{px2:.1f}", "y2": f"{py2:.1f}",
        "stroke": "#37474f", "stroke-width": str(DOOR_WIDTH_PX),
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
    arc_stroke = "#d32f2f" if is_exit else "#455a64"
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
        "stroke": "#78909c",
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
            "fill": "#607d8b", "font-style": "italic",
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
        "stroke": "#263238",
        "stroke-width": str(WALL_WIDTH + 1),
        "stroke-linejoin": "miter",
        "filter": "url(#shadow)",
    })


def _draw_entrance(g: Element, entrance_point, ox: float, oy: float, exit_width: float = 1.0):
    if entrance_point is None:
        return
    ex, ey = entrance_point
    px = ox + _px(ex)
    py = oy + _px(ey)
    ew = _px(exit_width)

    # White gap for entrance
    SubElement(g, "line", {
        "x1": f"{px:.1f}", "y1": f"{py:.1f}",
        "x2": f"{px + ew:.1f}", "y2": f"{py:.1f}",
        "stroke": "white", "stroke-width": str(WALL_WIDTH + 3),
    })
    # Arrow
    SubElement(g, "path", {
        "d": f"M{px + ew/2:.1f},{py + 15:.1f} L{px + ew/2:.1f},{py:.1f} "
             f"L{px + ew/2 - 4:.1f},{py + 6:.1f} M{px + ew/2:.1f},{py:.1f} "
             f"L{px + ew/2 + 4:.1f},{py + 6:.1f}",
        "stroke": "#d32f2f", "stroke-width": "1.5", "fill": "none",
    })
    lbl = SubElement(g, "text", {
        "x": f"{px + ew/2:.1f}", "y": f"{py + 24:.1f}",
        "text-anchor": "middle", "font-size": "8", "fill": "#d32f2f",
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
        "fill": "#263238",
    })
    t = SubElement(g, "text", {
        "x": f"{MARGIN:.0f}", "y": f"{tb_y + 22:.0f}",
        "font-size": "16", "font-weight": "bold", "fill": "white",
    })
    t.text = title

    info = SubElement(g, "text", {
        "x": f"{MARGIN:.0f}", "y": f"{tb_y + 38:.0f}",
        "font-size": "11", "fill": "#b0bec5",
    })
    info.text = (f"Occupancy: {occupancy}  |  Total Area: {total_area:.1f} mÂ²  |  "
                 f"Scale: 1:{SCALE}  |  GenAI Floor Plan Generator")

    # Scale bar
    bar_x = width_px - LEGEND_W - 20 - _px(3)
    bar_y = tb_y + 28
    bar_w = _px(3)
    SubElement(g, "line", {
        "x1": f"{bar_x:.0f}", "y1": f"{bar_y:.0f}",
        "x2": f"{bar_x + bar_w:.0f}", "y2": f"{bar_y:.0f}",
        "stroke": "white", "stroke-width": "2",
    })
    for i in range(4):
        tx = bar_x + _px(i)
        SubElement(g, "line", {
            "x1": f"{tx:.0f}", "y1": f"{bar_y - 3:.0f}",
            "x2": f"{tx:.0f}", "y2": f"{bar_y + 3:.0f}",
            "stroke": "white", "stroke-width": "1",
        })
        st = SubElement(g, "text", {
            "x": f"{tx:.0f}", "y": f"{bar_y + 12:.0f}",
            "text-anchor": "middle", "font-size": "9", "fill": "#b0bec5",
        })
        st.text = f"{i}m"


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Compass rose
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _draw_compass(svg: Element, x: float, y: float):
    g = SubElement(svg, "g", {"transform": f"translate({x:.0f},{y:.0f})"})
    # N arrow
    SubElement(g, "path", {
        "d": "M0,-20 L5,-5 L0,-10 L-5,-5 Z",
        "fill": "#d32f2f", "stroke": "#263238", "stroke-width": "0.5",
    })
    n = SubElement(g, "text", {
        "x": "0", "y": "-24", "text-anchor": "middle",
        "font-size": "10", "font-weight": "bold", "fill": "#263238",
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
) -> str:
    """Render a Building into a professional SVG blueprint string.

    Parameters
    ----------
    show_grid : bool
        Draw a light 1 m metric grid inside the boundary.
    merge_walls : bool
        Use the wall-merge engine so shared walls are drawn once.
    """
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
    _draw_entrance(g_entrance, entrance_point, ox, oy, exit_w)
    g_ent = SubElement(svg, "g", {"id": "entrance"})
    ew = getattr(building.exit, "width", 1.0) if hasattr(building, "exit") and building.exit else 1.0
    _draw_entrance(g_ent, entrance_point, ox, oy, ew)

    # Doors
    g_doors = SubElement(svg, "g", {"id": "doors"})
    for door in building.doors:
        _draw_door(g_doors, door, ox, oy)
        
    if hasattr(building, "exit") and building.exit and getattr(building.exit, "segment", None):
        class ExitDoor:
            segment = building.exit.segment
            door_type = "exit"
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
    _draw_title_block(svg, title, width_px, height_px, total_area, building.occupancy_type)

    # Compass
    _draw_compass(svg, width_px - LEGEND_W - 40, 40)

    # Legend -- top-right, outside the plan
    legend_x = width_px - LEGEND_W + 6
    g_legend = SubElement(svg, "g", {"id": "legend",
                                      "transform": f"translate({legend_x:.0f}, {MARGIN:.0f})"})
    hdr = SubElement(g_legend, "text", {
        "x": "0", "y": "14",
        "font-size": "13", "font-weight": "bold", "fill": "#263238",
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
            "fill": fill, "stroke": "#37474f", "stroke-width": "1",
        })
        lt = SubElement(g_legend, "text", {
            "x": "28", "y": f"{ly + 14}",
            "font-size": "12", "fill": "#263238",
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
):
    """Render and save SVG to file."""
    svg_str = render_svg_blueprint(
        building, boundary_polygon, entrance_point, zone_map, title,
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

