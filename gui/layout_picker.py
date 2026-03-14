"""
LayoutPicker – shows 3-5 layout variants side-by-side and lets the user
choose one before continuing to the compliance report.
"""

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import math


# ── colour palette ─────────────────────────────────────────────────────────────
ROOM_COLOURS = [
    "#dbeafe", "#dcfce7", "#fef9c3", "#fce7f3",
    "#ede9fe", "#ffedd5", "#f0fdf4", "#fee2e2",
]
CORRIDOR_COLOUR = "#fed7aa"   # light orange
BOUNDARY_COLOUR = "#1e3a5f"
DOOR_COLOUR     = "#374151"
EXIT_COLOUR     = "#dc2626"


def _room_centroid(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _draw_variant(ax, variant, boundary_polygon=None):
    """Draw a single layout variant into matplotlib axes `ax`."""
    building = variant["building"]
    ax.set_aspect("equal")
    ax.set_title(variant.get("strategy_name", "Layout"),
                 fontsize=8, fontweight="bold", pad=4)

    # Boundary
    if boundary_polygon and len(boundary_polygon) >= 3:
        bpoly = mpatches.Polygon(boundary_polygon, closed=True,
                                 edgecolor=BOUNDARY_COLOUR,
                                 facecolor="none", lw=1.5, linestyle="--")
        ax.add_patch(bpoly)

    # Rooms
    for idx, room in enumerate(building.rooms):
        if not room.polygon:
            continue
        colour = ROOM_COLOURS[idx % len(ROOM_COLOURS)]
        rpoly = mpatches.Polygon(room.polygon, closed=True,
                                 edgecolor="#6b7280", facecolor=colour, lw=0.8)
        ax.add_patch(rpoly)
        cx, cy = _room_centroid(room.polygon)
        # Abbreviate name if too long
        label = room.name if len(room.name) <= 10 else room.name[:9] + "…"
        ax.text(cx, cy, label, fontsize=5.5, ha="center", va="center", color="#1f2937")

    # Corridors
    for corr in getattr(building, "corridors", []):
        if not corr.polygon:
            continue
        cpoly = mpatches.Polygon(corr.polygon, closed=True,
                                 edgecolor="#9a3412", facecolor=CORRIDOR_COLOUR,
                                 lw=0.7, alpha=0.75,
                                 linestyle="dashed")
        ax.add_patch(cpoly)
        cx, cy = _room_centroid(corr.polygon)
        ax.text(cx, cy, "C", fontsize=4.5, ha="center", va="center", color="#7c2d12")

    # Doors
    for door in building.doors:
        (x1, y1), (x2, y2) = door.segment
        ax.plot([x1, x2], [y1, y2], color=DOOR_COLOUR, lw=1.5)

    # Exit
    if building.exit and building.exit.segment:
        (x1, y1), (x2, y2) = building.exit.segment
        ax.plot([x1, x2], [y1, y2], color=EXIT_COLOUR, lw=2.5)

    # violation count badge
    ont = variant.get("ontology") or {}
    violations = ont.get("violations", [])
    badge_txt = f"✓ 0 violations" if not violations else f"⚠ {len(violations)} violation(s)"
    badge_col = "#166534" if not violations else "#991b1b"
    ax.set_xlabel(badge_txt, fontsize=6, color=badge_col)

    ax.autoscale_view()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


class LayoutPicker:
    """
    Modal Tkinter window that embeds matplotlib thumbnails of each variant.
    Returns the index of the selected variant, or 0 if the user closes.
    """

    def __init__(self, variants, boundary_polygon=None, parent=None, recommended_idx=0):
        self.variants = variants
        self.boundary_polygon = boundary_polygon
        self._selected = recommended_idx if 0 <= recommended_idx < len(variants) else 0

        n = len(variants)
        cols = min(n, 3)
        rows = math.ceil(n / cols)

        self.win = tk.Toplevel(parent) if parent else tk.Tk()
        self.win.title("Choose a Layout Variant")
        self.win.resizable(True, True)

        tk.Label(self.win,
                 text="Select a corridor layout strategy then click 'Use This Layout'",
                 font=("Arial", 11, "bold"), pady=6).pack()

        if variants:
            recommended = variants[self._selected]
            rec_label = (
                f"System recommendation: #{recommended.get('rank', 1)} "
                f"{recommended.get('strategy_name', 'Layout')} "
                f"(score={recommended.get('ranking', {}).get('score', 0.0):.3f})"
            )
            tk.Label(self.win, text=rec_label, fg="#166534", font=("Arial", 9, "bold")).pack()

        # ── matplotlib figure ─────────────────────────────────────────────────
        fig_w = max(5.0 * cols, 8)
        fig_h = max(4.5 * rows, 5)
        self._fig = Figure(figsize=(fig_w, fig_h), dpi=96)
        self._axes = []

        for i, variant in enumerate(variants):
            ax = self._fig.add_subplot(rows, cols, i + 1)
            _draw_variant(ax, variant, boundary_polygon)
            self._axes.append(ax)

        self._fig.tight_layout(pad=1.5)

        canvas = FigureCanvasTkAgg(self._fig, master=self.win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=4)

        # ── selection controls ────────────────────────────────────────────────
        ctrl = tk.Frame(self.win, pady=6)
        ctrl.pack()

        tk.Label(ctrl, text="Choose variant:", font=("Arial", 10)).pack(side="left", padx=4)
        self._var_combo = ttk.Combobox(
            ctrl,
            values=[
                (
                    f"{i+1}. #{v.get('rank', i+1)} {v['strategy_name']} "
                    f"(score={v.get('ranking', {}).get('score', 0.0):.3f})"
                )
                for i, v in enumerate(variants)
            ],
            state="readonly",
            width=52,
        )
        self._var_combo.current(self._selected)
        self._var_combo.pack(side="left", padx=6)

        tk.Button(ctrl, text="✔  Use This Layout",
                  command=self._confirm,
                  bg="#16a34a", fg="white",
                  font=("Arial", 10, "bold"),
                  width=18).pack(side="left", padx=8)

        tk.Button(ctrl, text="Cancel",
                  command=self.win.destroy,
                  width=8).pack(side="left", padx=4)

        self.win.protocol("WM_DELETE_WINDOW", self.win.destroy)

    def _confirm(self):
        self._selected = self._var_combo.current()
        self.win.destroy()

    def run(self):
        self.win.mainloop()
        return self._selected
