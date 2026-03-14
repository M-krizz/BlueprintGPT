"""
dimensions.py - Reusable helpers for blueprint dimensioning.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple
from xml.etree.ElementTree import SubElement

Point = Tuple[float, float]


def measure_room_dims(room_polygon: Sequence[Point]) -> Tuple[float, float]:
    """Return (width, height) of a polygon's axis-aligned bounding box."""
    xs = [p[0] for p in room_polygon]
    ys = [p[1] for p in room_polygon]
    return max(xs) - min(xs), max(ys) - min(ys)


def draw_dimension(
    g,
    p1: Point,
    p2: Point,
    *,
    text: str,
    vertical: bool = False,
    tick: float = 3.0,
    color: str = "#90a4ae",
    text_color: str = "#78909c",
    font_size: str = "7",
):
    """Draw a compact architectural dimension line with end ticks + label."""
    x1, y1 = p1
    x2, y2 = p2

    SubElement(g, "line", {
        "x1": f"{x1:.1f}", "y1": f"{y1:.1f}",
        "x2": f"{x2:.1f}", "y2": f"{y2:.1f}",
        "stroke": color, "stroke-width": "0.5",
    })

    if vertical:
        SubElement(g, "line", {
            "x1": f"{x1 - tick:.1f}", "y1": f"{y1:.1f}",
            "x2": f"{x1 + tick:.1f}", "y2": f"{y1:.1f}",
            "stroke": color, "stroke-width": "0.5",
        })
        SubElement(g, "line", {
            "x1": f"{x2 - tick:.1f}", "y1": f"{y2:.1f}",
            "x2": f"{x2 + tick:.1f}", "y2": f"{y2:.1f}",
            "stroke": color, "stroke-width": "0.5",
        })
        cx = (x1 + x2) / 2 + 8
        cy = (y1 + y2) / 2
        lbl = SubElement(g, "text", {
            "x": f"{cx:.1f}", "y": f"{cy:.1f}",
            "text-anchor": "middle",
            "font-size": font_size,
            "fill": text_color,
            "transform": f"rotate(-90,{cx:.1f},{cy:.1f})",
        })
    else:
        SubElement(g, "line", {
            "x1": f"{x1:.1f}", "y1": f"{y1 - tick:.1f}",
            "x2": f"{x1:.1f}", "y2": f"{y1 + tick:.1f}",
            "stroke": color, "stroke-width": "0.5",
        })
        SubElement(g, "line", {
            "x1": f"{x2:.1f}", "y1": f"{y2 - tick:.1f}",
            "x2": f"{x2:.1f}", "y2": f"{y2 + tick:.1f}",
            "stroke": color, "stroke-width": "0.5",
        })
        lbl = SubElement(g, "text", {
            "x": f"{(x1 + x2) / 2:.1f}",
            "y": f"{(y1 + y2) / 2 - 3:.1f}",
            "text-anchor": "middle",
            "font-size": font_size,
            "fill": text_color,
        })

    lbl.text = text
