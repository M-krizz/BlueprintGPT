"""
BoundaryCanvas – lets the user either draw a freehand polygon or pick a
preset building shape (Rectangle, L, T, U).

Returns (via .run()):
  {
    "boundary_polygon": [(x, y), ...],   # vertices in metres (scaled)
    "target_area_sqft": float,
    "target_area_sqm":  float,
  }
  or None if the user cancels.

Coordinate system:
  Canvas pixels → metres using SCALE px/m.
  The pixel area of the drawn polygon is computed via the shoelace formula,
  then a uniform scale-factor k is applied so that the real-world polygon area
  equals the user-entered sq.ft value (converted to m²).
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox

# ── constants ──────────────────────────────────────────────────────────────────
CANVAS_W = 700
CANVAS_H = 550
SCALE    = 40          # pixels per metre at 1× zoom (1 px = 0.025 m at default)
SQF2SQM  = 0.09290304  # 1 sq.ft → m²

PRESET_SHAPES = {
    "Rectangle": [
        (100, 100), (500, 100), (500, 380), (100, 380)
    ],
    "L-Shape": [
        (100, 100), (500, 100), (500, 250),
        (300, 250), (300, 380), (100, 380)
    ],
    "T-Shape": [
        (100, 100), (500, 100), (500, 220),
        (370, 220), (370, 380), (230, 380),
        (230, 220), (100, 220)
    ],
    "U-Shape": [
        (100, 100), (220, 100), (220, 280),
        (380, 280), (380, 100), (500, 100),
        (500, 400), (100, 400)
    ],
}


def _polygon_area_px(pts):
    """Shoelace formula – returns unsigned area in px²."""
    n = len(pts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _scale_polygon(pts, k, cx, cy):
    """Scale polygon vertices by factor k around centroid (cx, cy)."""
    return [(cx + (x - cx) * k, cy + (y - cy) * k) for x, y in pts]


def _centroid(pts):
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    return cx, cy


def _px_to_m(pts):
    """Convert pixel coords → metre coords (origin = top-left of canvas,
    y-axis flipped so up = positive)."""
    return [(x / SCALE, (CANVAS_H - y) / SCALE) for x, y in pts]


class BoundaryCanvas:
    """Modal Tkinter window for drawing / picking a building boundary."""

    def __init__(self, parent=None):
        self.result = None
        self.vertices = []           # current polygon vertices (px)
        self._draw_ids = []          # canvas item IDs for the current polygon
        self._handle_ids = []        # small squares at each vertex
        self._closed = False
        self._entrance_point = None  # (px_x, px_y)
        self._mode = tk.StringVar(value="draw")

        # ── root window ───────────────────────────────────────────────────────
        self.win = tk.Toplevel(parent) if parent else tk.Tk()
        self.win.title("Draw or Pick Building Boundary")
        self.win.resizable(False, False)

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        top = tk.Frame(self.win, pady=4)
        top.pack(fill="x")

        # mode toggle
        tk.Label(top, text="Mode:", font=("Arial", 10, "bold")).pack(side="left", padx=6)
        tk.Radiobutton(top, text="✏  Draw freehand", variable=self._mode,
                       value="draw", command=self._reset).pack(side="left")
        tk.Radiobutton(top, text="⬛  Pick preset", variable=self._mode,
                       value="preset", command=self._show_presets).pack(side="left")

        # preset buttons row (hidden by default)
        self._preset_frame = tk.Frame(self.win)
        for name in PRESET_SHAPES:
            tk.Button(self._preset_frame, text=name, width=10,
                      command=lambda n=name: self._load_preset(n)).pack(side="left", padx=3)
        # (shown only in preset mode)

        # canvas
        self.canvas = tk.Canvas(self.win, width=CANVAS_W, height=CANVAS_H,
                                bg="#f5f5f0", cursor="crosshair",
                                relief="sunken", bd=2)
        self.canvas.pack(padx=8, pady=4)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<Motion>", self._on_motion)
        self._guide_line = None

        # area entry
        area_row = tk.Frame(self.win)
        area_row.pack(fill="x", padx=10, pady=4)
        tk.Label(area_row, text="Target build area (sq.ft):",
                 font=("Arial", 10)).pack(side="left")
        self._area_var = tk.StringVar()
        tk.Entry(area_row, textvariable=self._area_var, width=12).pack(side="left", padx=6)

        # status label
        self._status = tk.Label(self.win, text="Click on canvas to add vertices. "
                                "Double-click to close polygon.",
                                fg="gray", font=("Arial", 9))
        self._status.pack()

        # entrance toggle hint
        self._entrance_hint = tk.Label(self.win, text="After closing boundary, click on the edge to set Entrance.",
                                        fg="#b91c1c", font=("Arial", 9, "italic"))
        self._entrance_hint.pack_forget() # show only when closed

        # buttons
        btn_row = tk.Frame(self.win, pady=4)
        btn_row.pack()
        tk.Button(btn_row, text="Reset", command=self._reset,
                  width=10).pack(side="left", padx=4)
        tk.Button(btn_row, text="Confirm", command=self._confirm,
                  width=12, bg="#4CAF50", fg="white").pack(side="left", padx=4)
        tk.Button(btn_row, text="Cancel", command=self.win.destroy,
                  width=10).pack(side="left", padx=4)

        # draw grid guide
        self._draw_grid()

    def _draw_grid(self):
        for x in range(0, CANVAS_W, SCALE):
            self.canvas.create_line(x, 0, x, CANVAS_H,
                                    fill="#e0e0e0", tags="grid")
        for y in range(0, CANVAS_H, SCALE):
            self.canvas.create_line(0, y, CANVAS_W, y,
                                    fill="#e0e0e0", tags="grid")
        # axis labels (metres)
        for x in range(0, CANVAS_W, SCALE * 2):
            self.canvas.create_text(x + 2, CANVAS_H - 2,
                                    text=f"{x // SCALE}m",
                                    font=("Arial", 7), fill="#aaa", anchor="sw")

    # ── mode helpers ──────────────────────────────────────────────────────────

    def _show_presets(self):
        self._preset_frame.pack(before=self.canvas, pady=2)
        self._reset()

    def _load_preset(self, name):
        self._reset()
        pts = list(PRESET_SHAPES[name])
        self.vertices = pts
        self._closed = True
        self._mode.set("preset")
        self._redraw()
        self._update_status()

    # ── drawing logic ─────────────────────────────────────────────────────────

    def _on_click(self, event):
        x, y = event.x, event.y
        if self._mode.get() == "preset":
            # In preset mode, we only allow clicking for entrance if closed
            if self._closed:
                self._set_entrance(x, y)
            return
            
        if self._closed:
            # Picking entrance
            self._set_entrance(x, y)
            return

        self.vertices.append((x, y))
        self._redraw()

    def _on_double_click(self, event):
        if self._mode.get() == "preset":
            return
        if len(self.vertices) >= 3:
            # remove last point that was added by the single-click event
            if len(self.vertices) > 3:
                self.vertices.pop()
            self._closed = True
            self._redraw()
            self._update_status()

    def _on_motion(self, event):
        if self._closed or self._mode.get() == "preset":
            return
        if self._guide_line:
            self.canvas.delete(self._guide_line)
        if self.vertices:
            lx, ly = self.vertices[-1]
            self._guide_line = self.canvas.create_line(
                lx, ly, event.x, event.y, fill="#aaa", dash=(4, 4))

    def _set_entrance(self, px, py):
        """Snap (px, py) to the nearest edge of the polygon and set as entrance."""
        if not self.vertices or len(self.vertices) < 3: return
        
        best_pt = None
        min_dist = float('inf')
        
        # Simple point-to-segment distance for each edge
        for i in range(len(self.vertices)):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % len(self.vertices)]
            
            # Snap to segment
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            mag_sq = dx*dx + dy*dy
            if mag_sq < 1e-9: continue
            
            u = ((px - p1[0]) * dx + (py - p1[1]) * dy) / mag_sq
            u = max(0, min(1, u))
            snap_x = p1[0] + u * dx
            snap_y = p1[1] + u * dy
            
            dist = math.sqrt((px - snap_x)**2 + (py - snap_y)**2)
            if dist < min_dist:
                min_dist = dist
                best_pt = (snap_x, snap_y)
        
        if best_pt:
            self._entrance_point = best_pt
            self._redraw()
            self._update_status()

    def _redraw(self):
        self.canvas.delete("poly")
        self.canvas.delete("handle")
        if self._guide_line:
            self.canvas.delete(self._guide_line)
            self._guide_line = None

        if not self.vertices:
            return

        # flat list for tk canvas polygon
        flat = [coord for pt in self.vertices for coord in pt]

        if self._closed and len(self.vertices) >= 3:
            self.canvas.create_polygon(*flat, outline="#2563eb", fill="#dbeafe",
                                       width=2, tags="poly")
            # area label
            area_px = _polygon_area_px(self.vertices)
            area_m2 = area_px / (SCALE ** 2)
            self.canvas.create_text(
                CANVAS_W // 2, 20,
                text=f"Drawn area ≈ {area_m2:.1f} m²  ({area_m2 / SQF2SQM:.0f} sq.ft)  — enters sq.ft target below",
                font=("Arial", 9), fill="#1e40af", tags="poly"
            )
        else:
            # draw polyline
            if len(flat) >= 4:
                self.canvas.create_line(*flat, fill="#2563eb", width=2,
                                        tags="poly", joinstyle="round")

        # vertex handles
        for vx, vy in self.vertices:
            self.canvas.create_rectangle(vx - 4, vy - 4, vx + 4, vy + 4,
                                         fill="#2563eb", outline="white",
                                         tags="handle")

        # entrance point
        if self._entrance_point:
            ex, ey = self._entrance_point
            self.canvas.create_oval(ex - 6, ey - 6, ex + 6, ey + 6,
                                    fill="#ef4444", outline="white", width=2,
                                    tags="handle")
            self.canvas.create_text(ex, ey + 15, text="ENTRANCE", fill="#b91c1c",
                                    font=("Arial", 8, "bold"), tags="handle")

    def _reset(self):
        self.vertices = []
        self._closed = False
        self.canvas.delete("poly")
        self.canvas.delete("handle")
        if self._guide_line:
            self.canvas.delete(self._guide_line)
            self._guide_line = None
        self._status.config(text="Click on canvas to add vertices. "
                            "Double-click to close polygon.")
        self._entrance_point = None
        self._entrance_hint.pack_forget()
        if self._mode.get() == "draw":
            self._preset_frame.pack_forget()

    def _update_status(self):
        if self._closed:
            area_px = _polygon_area_px(self.vertices)
            area_m2 = area_px / (SCALE ** 2)
            self._status.config(
                text=f"Polygon closed — {len(self.vertices)} vertices | "
                     f"Raw area ≈ {area_m2:.1f} m² | Enter target sq.ft below and click Confirm.",
                fg="#166534"
            )
            self._entrance_hint.pack(pady=2)
            if self._entrance_point:
                self._entrance_hint.config(text="Entrance set! Click another edge to change.", fg="#166534")
            else:
                self._entrance_hint.config(text="REQUIRED: Click on the building edge to set Entrance location.", fg="#b91c1c")

    # ── confirmation logic ────────────────────────────────────────────────────

    def _confirm(self):
        if not self._closed or len(self.vertices) < 3:
            messagebox.showwarning("Incomplete", "Please close the polygon first "
                                   "(≥ 3 vertices, double-click to close).")
            return
            
        if not self._entrance_point:
            messagebox.showwarning("Entrance Required", "Please click on any wall/edge to set the building entrance.")
            return

        try:
            target_sqft = float(self._area_var.get())
            if target_sqft <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid Area",
                                   "Please enter a positive number for the target area.")
            return

        target_sqm = target_sqft * SQF2SQM

        # Scale polygon so its area (in m²) matches the target
        area_px = _polygon_area_px(self.vertices)
        area_m2_raw = area_px / (SCALE ** 2)
        if area_m2_raw < 1e-6:
            messagebox.showerror("Error", "Polygon has zero area.")
            return

        k = math.sqrt(target_sqm / area_m2_raw)
        cx, cy = _centroid(self.vertices)
        scaled_px = _scale_polygon(self.vertices, k, cx, cy)

        # Convert to metres (flip Y axis)
        boundary_m = [(x / SCALE, (CANVAS_H - y) / SCALE) for x, y in scaled_px]
        
        # Scale and flip entrance point too
        ex, ey = self._entrance_point
        # Offset from centroid
        ex_scaled = cx + (ex - cx) * k
        ey_scaled = cy + (ey - cy) * k
        entrance_m = (ex_scaled / SCALE, (CANVAS_H - ey_scaled) / SCALE)

        self.result = {
            "boundary_polygon": boundary_m,
            "entrance_point":   entrance_m,
            "target_area_sqft": target_sqft,
            "target_area_sqm":  target_sqm,
        }
        self.win.destroy()

    # ── public entry point ────────────────────────────────────────────────────

    def run(self):
        # If opened as a child Toplevel inside an already-running mainloop,
        # use wait_window() so the call blocks until the window is destroyed.
        # If opened standalone (Tk root), use mainloop() instead.
        if isinstance(self.win, tk.Toplevel):
            self.win.grab_set()          # make it modal (block parent)
            self.win.focus_force()
            self.win.wait_window()       # blocks here until destroy() is called
        else:
            self.win.mainloop()
        return self.result


# ── standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    bc = BoundaryCanvas()
    data = bc.run()
    if data:
        print("Boundary polygon (m):", data["boundary_polygon"])
        print("Target area:", data["target_area_sqft"], "sq.ft /",
              data["target_area_sqm"], "m²")
    else:
        print("Cancelled.")
