"""
layout_editor.py – Interactive layout editor with drag / resize handles and
instant compliance re-validation.

Usage
-----
    from gui.layout_editor import LayoutEditor
    editor = LayoutEditor(building, boundary_polygon, entrance_point,
                          regulation_file="ontology/regulation_data.json")
    modified_building = editor.run()  # blocks until window is closed
"""
from __future__ import annotations

import copy
import math
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from core.building import Building
from core.room import Room

# ── Colours ───────────────────────────────────────────────────────────────────
ROOM_COLOURS = [
    "#dbeafe", "#dcfce7", "#fef9c3", "#fce7f3",
    "#ede9fe", "#ffedd5", "#f0fdf4", "#fee2e2",
]
CORRIDOR_COLOUR = "#fed7aa"
BOUNDARY_COLOUR = "#1e3a5f"
HANDLE_COLOUR   = "#2563eb"
VIOLATION_COLOUR = "#fca5a5"
OK_COLOUR       = "#bbf7d0"

SNAP_STEP = 0.15  # metres – grid snap resolution


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _snap(val: float, step: float = SNAP_STEP) -> float:
    return round(val / step) * step


def _centroid(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _set_rect(room, x0, y0, x1, y1):
    room.polygon = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    room.final_area = abs((x1 - x0) * (y1 - y0))


# ═══════════════════════════════════════════════════════════════════════════════
#  Quick validation (runs on every edit)
# ═══════════════════════════════════════════════════════════════════════════════

def _quick_validate(building: Building, regulation_file: str) -> Dict[str, List[str]]:
    """Run lightweight checks per room. Returns {room_name: [violation_strings]}."""
    from constraints.rule_engine import RuleEngine

    engine = RuleEngine(regulation_file)
    occ = building.occupancy_type
    regs = engine.data.get(occ, {}).get("rooms", {})

    issues: Dict[str, List[str]] = {}

    for room in building.rooms:
        if room.polygon is None:
            continue
        rule = regs.get(room.room_type, {})
        room_issues: list[str] = []

        x0, y0, x1, y1 = _bbox(room.polygon)
        w, h = x1 - x0, y1 - y0
        area = w * h

        min_area = rule.get("min_area", 0)
        min_w = rule.get("min_width", 0)

        if area < min_area:
            room_issues.append(f"area {area:.1f} < {min_area} m²")
        if min(w, h) < min_w:
            room_issues.append(f"narrow dim {min(w, h):.2f} < {min_w} m")

        # Aspect ratio
        ar = max(w, h) / max(min(w, h), 0.01)
        if ar > 3.0:
            room_issues.append(f"aspect ratio {ar:.1f} too extreme")

        # Overlap check
        from shapely.geometry import Polygon as ShapelyPoly
        rpoly = ShapelyPoly(room.polygon)
        for other in building.rooms:
            if other is room or other.polygon is None:
                continue
            opoly = ShapelyPoly(other.polygon)
            ov = rpoly.intersection(opoly).area
            if ov > 0.05:
                room_issues.append(f"overlaps {other.name} ({ov:.2f} m²)")

        if room_issues:
            issues[room.name] = room_issues

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
#  LayoutEditor
# ═══════════════════════════════════════════════════════════════════════════════

class LayoutEditor:
    """
    Matplotlib-backed interactive layout editor embedded in a Tkinter window.

    Features
    --------
    - Click a room to select it (highlighted border).
    - Drag a room to reposition (snaps to 0.15 m grid).
    - Drag corner handles to resize.
    - Violations overlay updates instantly after each move/resize.
    - Status panel shows per-room issues.
    """

    def __init__(
        self,
        building: Building,
        boundary_polygon: List[Tuple[float, float]],
        entrance_point: Optional[Tuple[float, float]] = None,
        regulation_file: str = "ontology/regulation_data.json",
        parent=None,
    ):
        self.building = copy.deepcopy(building)
        self.boundary = boundary_polygon
        self.entrance = entrance_point
        self.reg_file = regulation_file

        self._selected_room: Optional[Room] = None
        self._drag_mode: Optional[str] = None  # "move" | "resize_<corner>"
        self._drag_start: Optional[Tuple[float, float]] = None
        self._drag_orig_bbox: Optional[Tuple[float, float, float, float]] = None
        self._violations: Dict[str, List[str]] = {}

        # ── Window ────────────────────────────────────────────────────────────
        self.win = tk.Toplevel(parent) if parent else tk.Tk()
        self.win.title("Layout Editor – drag rooms / resize corners")
        self.win.geometry("1100x700")

        self._build_ui()
        self._revalidate()
        self._redraw()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        main = tk.PanedWindow(self.win, orient=tk.HORIZONTAL)
        main.pack(fill="both", expand=True)

        # Left: matplotlib canvas
        left = tk.Frame(main)
        main.add(left, stretch="always")

        self._fig = Figure(figsize=(7, 6), dpi=96)
        self._ax = self._fig.add_subplot(111)
        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=left)
        self._mpl_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._mpl_canvas.mpl_connect("button_press_event", self._on_press)
        self._mpl_canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._mpl_canvas.mpl_connect("button_release_event", self._on_release)

        # Right: info panel
        right = tk.Frame(main, width=280)
        main.add(right, stretch="never")

        tk.Label(right, text="Room Info", font=("Arial", 11, "bold")).pack(pady=6)
        self._info_text = tk.Text(right, width=36, height=14, wrap="word",
                                  state="disabled", font=("Consolas", 9))
        self._info_text.pack(padx=4, fill="both", expand=True)

        tk.Label(right, text="Violations", font=("Arial", 11, "bold"),
                 fg="#991b1b").pack(pady=(10, 2))
        self._viol_text = tk.Text(right, width=36, height=14, wrap="word",
                                  state="disabled", font=("Consolas", 9),
                                  fg="#991b1b")
        self._viol_text.pack(padx=4, fill="both", expand=True)

        # Bottom buttons
        btn = tk.Frame(self.win, pady=6)
        btn.pack(fill="x")
        tk.Button(btn, text="✔ Accept Edits", command=self._accept,
                  bg="#16a34a", fg="white", font=("Arial", 10, "bold"),
                  width=16).pack(side="right", padx=12)
        tk.Button(btn, text="Undo All", command=self._undo_all,
                  width=10).pack(side="right", padx=4)

        self._original = copy.deepcopy(self.building)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _redraw(self):
        ax = self._ax
        ax.clear()
        ax.set_aspect("equal")
        ax.set_title("Click room to select · Drag to move · Drag handle to resize",
                      fontsize=8)

        # Boundary
        if self.boundary and len(self.boundary) >= 3:
            bp = mpatches.Polygon(self.boundary, closed=True,
                                  edgecolor=BOUNDARY_COLOUR, facecolor="#f8fafc",
                                  lw=1.5, linestyle="--", zorder=0)
            ax.add_patch(bp)

        # Rooms
        for idx, room in enumerate(self.building.rooms):
            if room.polygon is None:
                continue
            has_violation = room.name in self._violations
            is_selected = (room is self._selected_room)

            fc = VIOLATION_COLOUR if has_violation else ROOM_COLOURS[idx % len(ROOM_COLOURS)]
            ec = "#dc2626" if has_violation else ("#2563eb" if is_selected else "#6b7280")
            lw = 2.0 if is_selected else (1.2 if has_violation else 0.8)

            rp = mpatches.Polygon(room.polygon, closed=True,
                                  edgecolor=ec, facecolor=fc, lw=lw, zorder=1)
            ax.add_patch(rp)
            cx, cy = _centroid(room.polygon)
            label = room.name if len(room.name) <= 12 else room.name[:11] + "…"
            ax.text(cx, cy, label, fontsize=6, ha="center", va="center",
                    color="#1f2937", zorder=3)

            # Resize handles for selected room
            if is_selected:
                x0, y0, x1, y1 = _bbox(room.polygon)
                for hx, hy in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
                    ax.plot(hx, hy, "s", color=HANDLE_COLOUR, ms=7, zorder=4)

        # Corridors
        for corr in getattr(self.building, "corridors", []):
            if corr.polygon:
                cp = mpatches.Polygon(corr.polygon, closed=True,
                                      edgecolor="#9a3412", facecolor=CORRIDOR_COLOUR,
                                      lw=0.7, alpha=0.6, linestyle="--", zorder=0)
                ax.add_patch(cp)

        ax.autoscale_view()
        ax.set_xticks([])
        ax.set_yticks([])
        self._mpl_canvas.draw_idle()

        # Update info panel
        self._update_info()

    def _update_info(self):
        self._info_text.config(state="normal")
        self._info_text.delete("1.0", "end")
        if self._selected_room and self._selected_room.polygon:
            r = self._selected_room
            x0, y0, x1, y1 = _bbox(r.polygon)
            w, h = x1 - x0, y1 - y0
            lines = [
                f"Name:  {r.name}",
                f"Type:  {r.room_type}",
                f"pos:   ({x0:.2f}, {y0:.2f})",
                f"size:  {w:.2f} × {h:.2f} m",
                f"area:  {w*h:.2f} m²",
            ]
            self._info_text.insert("end", "\n".join(lines))
        else:
            self._info_text.insert("end", "(no room selected)")
        self._info_text.config(state="disabled")

        self._viol_text.config(state="normal")
        self._viol_text.delete("1.0", "end")
        if self._violations:
            for rname, issues in self._violations.items():
                self._viol_text.insert("end", f"─ {rname} ─\n")
                for iss in issues:
                    self._viol_text.insert("end", f"  • {iss}\n")
                self._viol_text.insert("end", "\n")
        else:
            self._viol_text.insert("end", "✓ No violations")
        self._viol_text.config(state="disabled")

    # ── Interaction ───────────────────────────────────────────────────────────

    def _hit_room(self, x, y) -> Optional[Room]:
        """Return room whose bbox contains (x, y), or None."""
        from shapely.geometry import Point, Polygon as SPoly
        pt = Point(x, y)
        for room in reversed(self.building.rooms):
            if room.polygon and SPoly(room.polygon).contains(pt):
                return room
        return None

    def _hit_handle(self, x, y) -> Optional[str]:
        """If near a resize handle of the selected room, return 'bl','br','tr','tl'."""
        if not self._selected_room or not self._selected_room.polygon:
            return None
        x0, y0, x1, y1 = _bbox(self._selected_room.polygon)
        corners = {"bl": (x0, y0), "br": (x1, y0), "tr": (x1, y1), "tl": (x0, y1)}
        # threshold in data units (estimate)
        bx0, by0, bx1, by1 = (0, 0, 20, 15)
        if self.boundary:
            bx0, by0, bx1, by1 = _bbox(self.boundary)
        span = max(bx1 - bx0, by1 - by0, 1)
        thr = span * 0.02
        for label, (cx, cy) in corners.items():
            if math.hypot(x - cx, y - cy) < thr:
                return label
        return None

    def _on_press(self, event):
        if event.inaxes != self._ax or event.xdata is None:
            return
        x, y = event.xdata, event.ydata

        # Check handle first
        handle = self._hit_handle(x, y)
        if handle:
            self._drag_mode = f"resize_{handle}"
            self._drag_start = (x, y)
            self._drag_orig_bbox = _bbox(self._selected_room.polygon)
            return

        # Check room hit
        room = self._hit_room(x, y)
        if room:
            self._selected_room = room
            self._drag_mode = "move"
            self._drag_start = (x, y)
            self._drag_orig_bbox = _bbox(room.polygon)
            self._redraw()
        else:
            self._selected_room = None
            self._drag_mode = None
            self._redraw()

    def _on_motion(self, event):
        if self._drag_mode is None or event.inaxes != self._ax or event.xdata is None:
            return
        room = self._selected_room
        if not room or not room.polygon:
            return

        x, y = event.xdata, event.ydata
        ox, oy = self._drag_start
        ox0, oy0, ox1, oy1 = self._drag_orig_bbox

        if self._drag_mode == "move":
            dx = _snap(x - ox)
            dy = _snap(y - oy)
            _set_rect(room, ox0 + dx, oy0 + dy, ox1 + dx, oy1 + dy)
        elif self._drag_mode.startswith("resize_"):
            corner = self._drag_mode.split("_")[1]
            nx0, ny0, nx1, ny1 = ox0, oy0, ox1, oy1
            if "r" in corner:
                nx1 = _snap(ox1 + (x - ox))
            if "l" in corner:
                nx0 = _snap(ox0 + (x - ox))
            if "t" in corner:
                ny1 = _snap(oy1 + (y - oy))
            if "b" in corner:
                ny0 = _snap(oy0 + (y - oy))
            # Enforce minimum dims
            if nx1 - nx0 < 0.6:
                nx1 = nx0 + 0.6
            if ny1 - ny0 < 0.6:
                ny1 = ny0 + 0.6
            _set_rect(room, nx0, ny0, nx1, ny1)

        self._redraw()

    def _on_release(self, event):
        if self._drag_mode is not None:
            self._drag_mode = None
            self._drag_start = None
            self._drag_orig_bbox = None
            self._revalidate()
            self._redraw()

    # ── Validation ────────────────────────────────────────────────────────────

    def _revalidate(self):
        self._violations = _quick_validate(self.building, self.reg_file)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _undo_all(self):
        self.building = copy.deepcopy(self._original)
        self._selected_room = None
        self._revalidate()
        self._redraw()

    def _accept(self):
        self.win.destroy()

    # ── Public entry ──────────────────────────────────────────────────────────

    def run(self) -> Building:
        """Block until the editor window is closed; return the (possibly edited) building."""
        if isinstance(self.win, tk.Toplevel):
            self.win.grab_set()
            self.win.focus_force()
            self.win.wait_window()
        else:
            self.win.mainloop()
        return self.building


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from core.building import Building
    from core.room import Room

    b = Building("Residential", 100)
    r1 = Room("Bedroom_1", "Bedroom")
    r1.polygon = [(1, 1), (5, 1), (5, 4), (1, 4)]
    r1.final_area = 12.0
    r2 = Room("Kitchen_1", "Kitchen")
    r2.polygon = [(5.5, 1), (9, 1), (9, 3.5), (5.5, 3.5)]
    r2.final_area = 8.75
    b.rooms = [r1, r2]

    boundary = [(0, 0), (15, 0), (15, 10), (0, 10)]
    editor = LayoutEditor(b, boundary)
    result = editor.run()
    print(f"Returned building with {len(result.rooms)} rooms")
