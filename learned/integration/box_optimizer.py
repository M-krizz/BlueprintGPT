"""
box_optimizer.py – Constrained rectangle packing optimizer (Phase 4).

Uses constraint programming (CP-SAT via OR-Tools) or gradient-based optimization
(scipy) to globally minimize room overlaps while respecting boundary and size
constraints.

Pipeline integration
--------------------
Called after ``_force_push_apart`` in Stage 3 of the repair gate when
``BOX_OPT_ENABLED=true``.  This is a *refinement* pass — force-push should have
already reduced most overlaps; the optimizer polishes the result.

Solver hierarchy
----------------
1. **OR-Tools CP-SAT** (``ortools.sat.python.cp_model``) — fast, exact, preferred
2. **scipy.optimize.minimize** (L-BFGS-B) — gradient-free fallback if OR-Tools unavailable
3. **No-op** — if neither is available, returns rooms unchanged

All solvers respect:
- Boundary polygon bbox (rooms must stay inside)
- Minimum room width/height (prevents degenerate shrinkage)
- Original room sizes (translation only, no resize)
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

# ── Feature flags ─────────────────────────────────────────────────────────────
BOX_OPT_ENABLED       = os.getenv("BOX_OPT_ENABLED", "false").lower() == "true"
BOX_OPT_TIME_LIMIT    = float(os.getenv("BOX_OPT_TIME_LIMIT", "2.0"))  # seconds
BOX_OPT_GRID_SCALE    = int(os.getenv("BOX_OPT_GRID_SCALE", "100"))    # coord → int scale
BOX_OPT_MIN_GAP       = float(os.getenv("BOX_OPT_MIN_GAP", "0.05"))    # min gap between rooms

# ── Solver availability ───────────────────────────────────────────────────────
_HAS_ORTOOLS = False
_HAS_SCIPY = False

try:
    from ortools.sat.python import cp_model
    _HAS_ORTOOLS = True
except ImportError:
    pass

try:
    from scipy.optimize import minimize as scipy_minimize
    import numpy as np
    _HAS_SCIPY = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _bbox(room) -> Tuple[float, float, float, float]:
    """Extract (x1, y1, x2, y2) from room.polygon."""
    xs = [p[0] for p in room.polygon]
    ys = [p[1] for p in room.polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _set_rect(room, x1: float, y1: float, x2: float, y2: float):
    """Set room to axis-aligned rectangle."""
    room.polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    room.final_area = (x2 - x1) * (y2 - y1)


def _boundary_bbox(boundary_polygon) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in boundary_polygon]
    ys = [p[1] for p in boundary_polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap_area(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b) -> float:
    """Axis-aligned overlap area between two boxes."""
    dx = max(0.0, min(x2a, x2b) - max(x1a, x1b))
    dy = max(0.0, min(y2a, y2b) - max(y1a, y1b))
    return dx * dy


# ═══════════════════════════════════════════════════════════════════════════════
#  OR-Tools CP-SAT solver
# ═══════════════════════════════════════════════════════════════════════════════

def _solve_cpsat(
    rooms: List,
    boundary: Tuple[float, float, float, float],
    time_limit: float = BOX_OPT_TIME_LIMIT,
    scale: int = BOX_OPT_GRID_SCALE,
    min_gap: float = BOX_OPT_MIN_GAP,
) -> bool:
    """
    Solve room placement using OR-Tools CP-SAT.

    Decision variables: (x1, y1) for each room (integer-scaled).
    Constraints: rooms inside boundary, no pairwise overlap.
    Objective: minimize total displacement from initial positions.

    Returns True if solution found and rooms updated, False otherwise.
    """
    if not _HAS_ORTOOLS:
        return False

    bx0, by0, bx1, by1 = boundary
    n = len(rooms)
    if n < 2:
        return True  # nothing to optimize

    # Extract room sizes (fixed) and initial positions
    sizes = []  # (w, h) for each room
    init_pos = []  # (x1, y1) initial
    for r in rooms:
        x1, y1, x2, y2 = _bbox(r)
        sizes.append((x2 - x1, y2 - y1))
        init_pos.append((x1, y1))

    # Scale to integers
    def to_int(v: float) -> int:
        return int(round(v * scale))

    def to_float(v: int) -> float:
        return v / scale

    bx0_i, by0_i = to_int(bx0), to_int(by0)
    bx1_i, by1_i = to_int(bx1), to_int(by1)
    gap_i = to_int(min_gap)

    model = cp_model.CpModel()

    # Decision variables: x1, y1 for each room
    x_vars = []
    y_vars = []
    for i, (w, h) in enumerate(sizes):
        w_i, h_i = to_int(w), to_int(h)
        # x1 must allow room to fit: x1 + w <= bx1 → x1 <= bx1 - w
        x = model.NewIntVar(bx0_i, bx1_i - w_i, f"x_{i}")
        y = model.NewIntVar(by0_i, by1_i - h_i, f"y_{i}")
        x_vars.append(x)
        y_vars.append(y)

    # No-overlap constraints using AddNoOverlap2D via intervals
    x_intervals = []
    y_intervals = []
    for i, (w, h) in enumerate(sizes):
        w_i, h_i = to_int(w), to_int(h)
        # Interval: [x, x + w) with gap
        x_int = model.NewIntervalVar(x_vars[i], w_i + gap_i, model.NewIntVar(0, bx1_i + w_i + gap_i, f"x_end_{i}"), f"x_int_{i}")
        y_int = model.NewIntervalVar(y_vars[i], h_i + gap_i, model.NewIntVar(0, by1_i + h_i + gap_i, f"y_end_{i}"), f"y_int_{i}")
        x_intervals.append(x_int)
        y_intervals.append(y_int)

    model.AddNoOverlap2D(x_intervals, y_intervals)

    # Objective: minimize total squared displacement from initial positions
    displacements = []
    for i, (ix, iy) in enumerate(init_pos):
        ix_i, iy_i = to_int(ix), to_int(iy)
        # |x - ix| + |y - iy| (Manhattan, since CP-SAT handles linear better)
        dx = model.NewIntVar(0, bx1_i - bx0_i, f"dx_{i}")
        dy = model.NewIntVar(0, by1_i - by0_i, f"dy_{i}")
        model.AddAbsEquality(dx, x_vars[i] - ix_i)
        model.AddAbsEquality(dy, y_vars[i] - iy_i)
        displacements.append(dx)
        displacements.append(dy)

    model.Minimize(sum(displacements))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Update room positions
        for i, r in enumerate(rooms):
            w, h = sizes[i]
            nx1 = to_float(solver.Value(x_vars[i]))
            ny1 = to_float(solver.Value(y_vars[i]))
            _set_rect(r, nx1, ny1, nx1 + w, ny1 + h)
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Scipy fallback solver
# ═══════════════════════════════════════════════════════════════════════════════

def _solve_scipy(
    rooms: List,
    boundary: Tuple[float, float, float, float],
    max_iter: int = 200,
    min_gap: float = BOX_OPT_MIN_GAP,
) -> bool:
    """
    Gradient-free optimization fallback using scipy L-BFGS-B.

    Minimizes overlap + displacement penalty.
    Returns True if solution improved, False otherwise.
    """
    if not _HAS_SCIPY:
        return False

    bx0, by0, bx1, by1 = boundary
    n = len(rooms)
    if n < 2:
        return True

    # Extract sizes and initial positions
    sizes = []
    init_pos = []
    for r in rooms:
        x1, y1, x2, y2 = _bbox(r)
        sizes.append((x2 - x1, y2 - y1))
        init_pos.append((x1, y1))

    # Decision variables: [x1_0, y1_0, x1_1, y1_1, ...]
    x0 = np.array([v for pos in init_pos for v in pos], dtype=np.float64)

    # Bounds: each (x1, y1) must keep room inside boundary
    bounds = []
    for w, h in sizes:
        bounds.append((bx0, bx1 - w))  # x1
        bounds.append((by0, by1 - h))  # y1

    def objective(x):
        """Total overlap + displacement penalty."""
        total_overlap = 0.0
        total_disp = 0.0
        for i in range(n):
            xi, yi = x[2*i], x[2*i + 1]
            wi, hi = sizes[i]
            ix, iy = init_pos[i]
            total_disp += abs(xi - ix) + abs(yi - iy)

            for j in range(i + 1, n):
                xj, yj = x[2*j], x[2*j + 1]
                wj, hj = sizes[j]
                ov = _overlap_area(xi, yi, xi + wi, yi + hi,
                                   xj, yj, xj + wj, yj + hj)
                if ov > 0:
                    total_overlap += ov * 1000  # heavy penalty

        return total_overlap + total_disp * 0.1

    result = scipy_minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-6},
    )

    if result.success or result.fun < objective(x0):
        # Update rooms
        for i, r in enumerate(rooms):
            w, h = sizes[i]
            nx1, ny1 = result.x[2*i], result.x[2*i + 1]
            _set_rect(r, nx1, ny1, nx1 + w, ny1 + h)
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def optimize_box_placement(
    building,
    boundary_polygon: List[Tuple[float, float]],
    *,
    time_limit: float = BOX_OPT_TIME_LIMIT,
    prefer_cpsat: bool = True,
) -> Dict[str, Any]:
    """
    Optimize room placement to eliminate overlaps.

    Parameters
    ----------
    building : Building
        The building with rooms to optimize.
    boundary_polygon : list of (x, y)
        The boundary polygon (uses bbox).
    time_limit : float
        Max solver time in seconds (default 2.0).
    prefer_cpsat : bool
        If True, try OR-Tools first; else try scipy first.

    Returns
    -------
    dict with keys:
        - success: bool
        - solver: str ("cpsat", "scipy", "none")
        - remaining_overlaps: int
    """
    rooms = [r for r in building.rooms if r.polygon is not None]
    boundary = _boundary_bbox(boundary_polygon)

    result = {
        "success": False,
        "solver": "none",
        "remaining_overlaps": 0,
    }

    if len(rooms) < 2:
        result["success"] = True
        return result

    # Try solvers in order
    solvers = [
        ("cpsat", lambda: _solve_cpsat(rooms, boundary, time_limit=time_limit)),
        ("scipy", lambda: _solve_scipy(rooms, boundary)),
    ]
    if not prefer_cpsat:
        solvers = solvers[::-1]

    for name, solve_fn in solvers:
        try:
            if solve_fn():
                result["success"] = True
                result["solver"] = name
                break
        except Exception:
            continue

    # Count remaining overlaps
    n = len(rooms)
    remaining = 0
    for i in range(n):
        for j in range(i + 1, n):
            if rooms[i].polygon and rooms[j].polygon:
                x1a, y1a, x2a, y2a = _bbox(rooms[i])
                x1b, y1b, x2b, y2b = _bbox(rooms[j])
                if _overlap_area(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b) > 0.01:
                    remaining += 1
    result["remaining_overlaps"] = remaining

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Availability check
# ═══════════════════════════════════════════════════════════════════════════════

def get_solver_info() -> Dict[str, bool]:
    """Return which solvers are available."""
    return {
        "ortools_cpsat": _HAS_ORTOOLS,
        "scipy": _HAS_SCIPY,
        "any_available": _HAS_ORTOOLS or _HAS_SCIPY,
    }
