import tkinter as tk
from tkinter import ttk, messagebox
import json

from gui.boundary_canvas import BoundaryCanvas


class LayoutForm:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Phase 1 – Layout Input")

        # Load allowed room types
        with open("ontology/regulation_data.json", "r") as f:
            data = json.load(f)
        self.allowed_room_types = list(data["Residential"]["rooms"].keys())

        self._boundary_result = None   # filled by BoundaryCanvas
        self.entries = []
        self.submitted_data = None

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # ── Step 1: Building boundary ─────────────────────────────────────────
        step1 = tk.LabelFrame(self.root, text=" Step 1 – Building Boundary ",
                              font=("Arial", 10, "bold"), padx=8, pady=6)
        step1.grid(row=0, column=0, columnspan=4, sticky="ew", **pad)

        self._boundary_label = tk.Label(
            step1,
            text="No boundary drawn yet.",
            fg="gray", font=("Arial", 9)
        )
        self._boundary_label.pack(side="left", padx=6)

        tk.Button(step1, text="✏  Draw / Pick Building Shape",
                  command=self._open_boundary_canvas,
                  bg="#2563eb", fg="white",
                  font=("Arial", 9, "bold")).pack(side="right", padx=4)

        # ── Step 2: Rooms ─────────────────────────────────────────────────────
        step2 = tk.LabelFrame(self.root, text=" Step 2 – Rooms ",
                              font=("Arial", 10, "bold"), padx=8, pady=6)
        step2.grid(row=1, column=0, columnspan=4, sticky="ew", **pad)

        tk.Label(step2, text="Number of rooms:").grid(row=0, column=0, sticky="w")
        self.room_count_entry = tk.Entry(step2, width=6)
        self.room_count_entry.grid(row=0, column=1, sticky="w", padx=4)

        tk.Button(step2, text="Generate Fields",
                  command=self._create_fields).grid(row=0, column=2, padx=6)

        self.room_frame = tk.Frame(step2)
        self.room_frame.grid(row=1, column=0, columnspan=4, pady=4)

        # ── Submit ────────────────────────────────────────────────────────────
        tk.Button(self.root, text="▶  Generate Layout",
                  command=self._submit,
                  bg="#16a34a", fg="white",
                  font=("Arial", 10, "bold"),
                  width=20).grid(row=2, column=0, columnspan=4, pady=10)

    def _open_boundary_canvas(self):
        bc = BoundaryCanvas(parent=self.root)
        result = bc.run()
        if result:
            self._boundary_result = result
            sqft = result["target_area_sqft"]
            sqm  = result["target_area_sqm"]
            try:
                self._boundary_label.config(
                    text=f"✅  Boundary set — {sqft:.0f} sq.ft  ({sqm:.1f} m²)  |  "
                         f"{len(result['boundary_polygon'])} vertices",
                    fg="#166534"
                )
            except Exception:
                pass   # widget may be briefly invalid right after Toplevel closes

    def _create_fields(self):
        try:
            count = int(self.room_count_entry.get())
            if count < 1:
                raise ValueError
        except ValueError:
            return

        for widget in self.room_frame.winfo_children():
            widget.destroy()
        self.entries = []

        for i in range(count):
            tk.Label(self.room_frame,
                     text=f"Room {i + 1} Name").grid(row=i, column=0, sticky="w")
            name_entry = tk.Entry(self.room_frame, width=14)
            name_entry.grid(row=i, column=1, padx=4)

            tk.Label(self.room_frame, text="Type").grid(row=i, column=2)
            type_combo = ttk.Combobox(
                self.room_frame,
                values=self.allowed_room_types,
                state="readonly",
                width=12
            )
            type_combo.grid(row=i, column=3, padx=4)
            type_combo.current(0)
            self.entries.append((name_entry, type_combo))

    def _submit(self):
        if not self._boundary_result:
            tk.messagebox.showwarning(
                "Missing Boundary",
                "Please draw or pick a building boundary first (Step 1)."
            )
            return

        if not self.entries:
            tk.messagebox.showwarning("No Rooms", "Please generate room fields first.")
            return

        rooms = []
        for name_entry, type_combo in self.entries:
            name = name_entry.get().strip()
            room_type = type_combo.get().strip()
            if name and room_type:
                rooms.append((name, room_type))

        if not rooms:
            return

        self.submitted_data = {
            "total_area":          self._boundary_result["target_area_sqft"],
            "area_unit":           "sq.ft",
            "allocation_strategy": "priority_weights",
            "rooms":               rooms,
            "boundary_polygon":    self._boundary_result["boundary_polygon"],
            "entrance_point":      self._boundary_result.get("entrance_point"),
        }
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.submitted_data