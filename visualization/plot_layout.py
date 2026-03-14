import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from graph.connectivity import is_fully_connected
from graph.manhattan_path import max_travel_distance

ROOM_COLOURS = [
    "#dbeafe", "#dcfce7", "#fef9c3", "#fce7f3",
    "#ede9fe", "#ffedd5", "#f0fdf4", "#fee2e2",
]
CORRIDOR_COLOUR = "#fed7aa"


def room_centroid(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _translate_polygon(polygon, dx, dy):
    return [(x + dx, y + dy) for x, y in polygon]


def _scale_polygon_axis(polygon, sx=1.0, sy=1.0):
    cx, cy = room_centroid(polygon)
    out = []
    for x, y in polygon:
        out.append((cx + (x - cx) * sx, cy + (y - cy) * sy))
    return out


def _poly_bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _issue(rule, entity, actual, required, suggestion):
    return {
        "rule": rule,
        "entity": entity,
        "actual": actual,
        "required": required,
        "suggestion": suggestion,
    }


def _evaluate_live(building, rule_engine=None, ontology_validator=None):
    issues = []

    for room in building.rooms:
        if not getattr(room, "polygon", None):
            continue
        x0, y0, x1, y1 = _poly_bbox(room.polygon)
        width = x1 - x0
        height = y1 - y0
        if getattr(room, "min_width", None) is not None and width < room.min_width:
            issues.append(_issue(
                "Room min width",
                room.name,
                f"{width:.2f}m",
                f"{room.min_width:.2f}m",
                f"Resize room width +{max(0.0, room.min_width - width):.2f}m",
            ))
        if getattr(room, "min_height", None) is not None and height < room.min_height:
            issues.append(_issue(
                "Room min height",
                room.name,
                f"{height:.2f}m",
                f"{room.min_height:.2f}m",
                f"Resize room height +{max(0.0, room.min_height - height):.2f}m",
            ))
        if getattr(room, "final_area", None) is not None and getattr(room, "min_area", None) is not None:
            area = abs((x1 - x0) * (y1 - y0))
            if area < room.min_area:
                issues.append(_issue(
                    "Room min area",
                    room.name,
                    f"{area:.2f}m²",
                    f"{room.min_area:.2f}m²",
                    f"Increase area by {max(0.0, room.min_area - area):.2f}m²",
                ))

    connected = is_fully_connected(building)
    if not connected:
        issues.append(_issue(
            "Connectivity",
            "Building",
            "Disconnected graph",
            "Fully connected",
            "Move rooms or add circulation so every room links into the graph",
        ))

    travel = max_travel_distance(building)
    if rule_engine is not None:
        allowed = rule_engine.get_max_travel_distance(building.occupancy_type)
        if travel > allowed:
            issues.append(_issue(
                "Travel distance",
                "Building",
                f"{travel:.2f}m",
                f"{allowed:.2f}m",
                f"Shorten route by {max(0.0, travel - allowed):.2f}m or move exit/corridor closer",
            ))

    if ontology_validator is not None and rule_engine is not None:
        ont = ontology_validator.validate(building, rule_engine)
        for item in ont.get("violations", []):
            code = item.get("code", "KG")
            entity = item.get("entity", "Building")
            message = item.get("message", "Ontology violation")
            if code == "EXIT_WIDTH":
                issues.append(_issue(
                    "Exit width",
                    entity,
                    "Below minimum",
                    "Minimum code width",
                    "Widen exit or reduce occupant load assumption",
                ))
            elif code == "MIN_AREA":
                issues.append(_issue(
                    "Room min area",
                    entity,
                    "Below minimum",
                    "Code minimum",
                    "Resize room footprint until area meets the minimum",
                ))
            elif code == "CONNECTIVITY":
                issues.append(_issue(
                    "Connectivity",
                    entity,
                    "Disconnected graph",
                    "Fully connected",
                    "Reconnect room cluster to corridor/door network",
                ))
            elif code == "TRAVEL_DISTANCE":
                issues.append(_issue(
                    "Travel distance",
                    entity,
                    "Over limit",
                    "Within code limit",
                    "Shorten path to exit using corridor or room repositioning",
                ))
            else:
                issues.append(_issue(code, entity, message, "Compliant", "Adjust geometry or adjacency for compliance"))

    seen = set()
    unique = []
    for issue in issues:
        key = (issue["rule"], issue["entity"], issue["actual"], issue["required"])
        if key not in seen:
            seen.add(key)
            unique.append(issue)
    return unique


def plot_layout(
    building,
    width,
    height,
    boundary_polygon=None,
    entrance_point=None,
    title="Floor Plan Layout",
    *,
    enable_edit_mode=False,
    rule_engine=None,
    ontology_validator=None,
    edit_step=0.20,
    resize_step=0.06,
):
    """Render floor plan; optional edit mode enables keyboard move/resize with live checks.

    Controls in edit mode:
    - Tab / Shift+Tab: select next/prev room
    - Arrow keys: move selected room
    - Shift+Arrow keys: resize selected room
    """
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#0c4a9a")
    ax.set_facecolor("#0b3f86")

    selected_idx = 0 if building.rooms else -1
    live_issues = _evaluate_live(building, rule_engine=rule_engine, ontology_validator=ontology_validator)

    def _draw():
        ax.clear()
        ax.set_facecolor("#0b3f86")

        violated_entities = {issue["entity"] for issue in live_issues}

        if boundary_polygon and len(boundary_polygon) >= 3:
            bpoly = mpatches.Polygon(
                boundary_polygon,
                closed=True,
                edgecolor="#dbeafe",
                facecolor="none",
                lw=2.2,
                linestyle="-",
                hatch="////",
            )
            ax.add_patch(bpoly)

        if entrance_point:
            ax.plot(
                entrance_point[0],
                entrance_point[1],
                "*",
                markersize=14,
                markeredgecolor="#dbeafe",
                color="#facc15",
            )
            ax.text(
                entrance_point[0],
                entrance_point[1] + 0.5,
                "ENTRANCE",
                color="#dbeafe",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

        for idx, room in enumerate(building.rooms):
            if not room.polygon:
                continue
            colour = ROOM_COLOURS[idx % len(ROOM_COLOURS)]

            room_edge = "#f8fafc"
            room_lw = 1.8
            if room.name in violated_entities:
                room_edge = "#ef4444"
                room_lw = 2.4
            if idx == selected_idx and enable_edit_mode:
                room_edge = "#facc15"
                room_lw = 2.8

            fill = mpatches.Polygon(
                room.polygon,
                closed=True,
                edgecolor="#e2e8f0",
                facecolor=colour,
                lw=1.2,
                alpha=0.42,
            )
            ax.add_patch(fill)

            wall = mpatches.Polygon(
                room.polygon,
                closed=True,
                edgecolor=room_edge,
                facecolor="none",
                lw=room_lw,
                alpha=0.95,
            )
            ax.add_patch(wall)

            cx, cy = room_centroid(room.polygon)
            ax.text(
                cx,
                cy,
                room.name.upper(),
                color="#f8fafc",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

        for corr in getattr(building, "corridors", []):
            if not corr.polygon:
                continue
            cpoly = mpatches.Polygon(
                corr.polygon,
                closed=True,
                edgecolor="#f8fafc",
                facecolor="#1d4ed8",
                lw=1.2,
                alpha=0.55,
                linestyle="-",
                hatch="////",
            )
            ax.add_patch(cpoly)
            cx, cy = room_centroid(corr.polygon)
            ax.text(cx, cy, "CIRC", color="#e2e8f0", ha="center", va="center", fontsize=6.5)

        for door in building.doors:
            (x1, y1), (x2, y2) = door.segment
            ax.plot([x1, x2], [y1, y2], color="#f8fafc", lw=2.6)

        if building.exit and building.exit.segment:
            (x1, y1), (x2, y2) = building.exit.segment
            ax.plot([x1, x2], [y1, y2], color="#fb7185", lw=4)

        margin = max(width, height) * 0.15
        ax.set_xlim(-margin, width + margin)
        ax.set_ylim(-margin, height + margin)
        ax.set_aspect("equal")
        ax.set_xticks(range(int(width) + 2))
        ax.set_yticks(range(int(height) + 2))
        ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.22, color="#93c5fd")
        ax.set_xlabel("Width (m)", color="#dbeafe")
        ax.set_ylabel("Height (m)", color="#dbeafe")
        ax.tick_params(colors="#bfdbfe")

        if boundary_polygon and len(boundary_polygon) >= 3:
            min_x = min(p[0] for p in boundary_polygon)
            max_x = max(p[0] for p in boundary_polygon)
            min_y = min(p[1] for p in boundary_polygon)
            max_y = max(p[1] for p in boundary_polygon)
            dim_offset = margin * 0.45
            ax.plot([min_x, max_x], [max_y + dim_offset, max_y + dim_offset], color="#dbeafe", lw=1.0)
            ax.text(
                (min_x + max_x) / 2,
                max_y + dim_offset + 0.08,
                f"{(max_x - min_x):.2f} m",
                color="#dbeafe",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax.plot([max_x + dim_offset, max_x + dim_offset], [min_y, max_y], color="#dbeafe", lw=1.0)
            ax.text(
                max_x + dim_offset + 0.05,
                (min_y + max_y) / 2,
                f"{(max_y - min_y):.2f} m",
                color="#dbeafe",
                rotation=90,
                ha="left",
                va="center",
                fontsize=8,
            )

        if enable_edit_mode:
            lines = [
                "EDIT MODE: Tab select | Arrows move | Shift+Arrows resize",
                f"Selected: {building.rooms[selected_idx].name if selected_idx >= 0 else 'None'}",
            ]
            if live_issues:
                lines.append("Mentor explanation:")
                for issue in live_issues[:5]:
                    lines.append(
                        f"- {issue['entity']}: {issue['rule']} | required {issue['required']} | got {issue['actual']}"
                    )
                    lines.append(f"  Fix: {issue['suggestion']}")
            else:
                lines.append("No live violations")
            ax.text(
                0.01,
                0.99,
                "\n".join(lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7.5,
                color="#fee2e2" if live_issues else "#dcfce7",
                bbox={"facecolor": "#0f172a", "alpha": 0.72, "edgecolor": "#334155"},
            )

        ax.set_title(title, fontsize=13, fontweight="bold", color="#e2e8f0")
        fig.canvas.draw_idle()

    def _recompute_and_draw():
        nonlocal live_issues
        live_issues = _evaluate_live(building, rule_engine=rule_engine, ontology_validator=ontology_validator)
        _draw()

    def _on_key(event):
        nonlocal selected_idx
        if not enable_edit_mode or selected_idx < 0 or selected_idx >= len(building.rooms):
            return

        room = building.rooms[selected_idx]
        if not room.polygon:
            return

        key = event.key or ""
        lower = key.lower()

        if lower == "tab":
            selected_idx = (selected_idx + 1) % len(building.rooms)
            _draw()
            return
        if lower == "shift+tab":
            selected_idx = (selected_idx - 1) % len(building.rooms)
            _draw()
            return

        moved = False

        if lower in ("left", "right", "up", "down"):
            dx = 0.0
            dy = 0.0
            if lower == "left":
                dx = -edit_step
            elif lower == "right":
                dx = edit_step
            elif lower == "up":
                dy = edit_step
            elif lower == "down":
                dy = -edit_step
            room.polygon = _translate_polygon(room.polygon, dx, dy)
            moved = True

        if lower in ("shift+left", "shift+right", "shift+up", "shift+down"):
            sx = 1.0
            sy = 1.0
            if lower == "shift+left":
                sx = max(0.75, 1.0 - resize_step)
            elif lower == "shift+right":
                sx = 1.0 + resize_step
            elif lower == "shift+up":
                sy = 1.0 + resize_step
            elif lower == "shift+down":
                sy = max(0.75, 1.0 - resize_step)
            room.polygon = _scale_polygon_axis(room.polygon, sx=sx, sy=sy)
            moved = True

        if moved:
            _recompute_and_draw()

    fig.canvas.mpl_connect("key_press_event", _on_key)
    _draw()
    plt.tight_layout()
    plt.show()
