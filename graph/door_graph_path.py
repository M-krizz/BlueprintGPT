"""
door_graph_path.py – Dijkstra-based worst-case travel distance through doors
and corridor space.

Replaces the simplistic centroid→exit Manhattan distance with a graph-based
model where:

Nodes
-----
* Door midpoint of every placed door.
* Exit point (building entrance / exit segment midpoint).
* Corridor spine keypoints (if available).

Edges
-----
* Between every two door midpoints that share a corridor  → Euclidean dist.
* Between a door midpoint and the exit point if both are on the corridor
  or within the same room  → Euclidean dist.
* Between every room-to-room door and the rooms it connects (optional
  intra-room traversal).

The function returns the worst-case shortest path from any room's
door-reachable node to the exit.
"""
from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

from shapely.geometry import Point, Polygon


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mid(seg) -> Tuple[float, float]:
    """Segment midpoint."""
    (x1, y1), (x2, y2) = seg
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point_near_polygon(pt: Tuple[float, float], polygon, tol: float = 0.35) -> bool:
    """Return True if *pt* is within *tol* of *polygon*."""
    if polygon is None:
        return False
    try:
        return Point(pt).distance(Polygon(polygon)) <= tol
    except Exception:
        return False


# ─── Graph construction ──────────────────────────────────────────────────────

def _build_graph(building) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, List[Tuple[str, float]]]]:
    """Build adjacency-list graph from building doors and corridors.

    Returns (nodes, adj) where
        nodes:  {node_id: (x, y)}
        adj:    {node_id: [(neighbour_id, weight), ...]}
    """
    nodes: Dict[str, Tuple[float, float]] = {}
    adj: Dict[str, List[Tuple[str, float]]] = {}

    def _add_edge(a: str, b: str, w: float):
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))

    # Exit node
    exit_pt = None
    if building.exit and building.exit.segment:
        exit_pt = _mid(building.exit.segment)
        nodes["exit"] = exit_pt
        adj.setdefault("exit", [])

    # Corridor spine keypoints
    corridor_polys = []
    for cidx, corr in enumerate(getattr(building, "corridors", [])):
        cpoly = getattr(corr, "polygon", None)
        if cpoly:
            corridor_polys.append(cpoly)
        for kidx, kp in enumerate(getattr(corr, "spine_points", [])):
            nid = f"spine_{cidx}_{kidx}"
            nodes[nid] = (kp[0], kp[1])
            adj.setdefault(nid, [])
            # Connect consecutive spine points
            if kidx > 0:
                prev_nid = f"spine_{cidx}_{kidx - 1}"
                _add_edge(nid, prev_nid, _dist(nodes[nid], nodes[prev_nid]))
            # Connect to exit if close
            if exit_pt and _point_near_polygon(exit_pt, cpoly, 0.5):
                _add_edge(nid, "exit", _dist(nodes[nid], exit_pt))

    # Door nodes
    for didx, door in enumerate(building.doors):
        if door.segment is None:
            continue
        dpt = _mid(door.segment)
        nid = f"door_{didx}"
        nodes[nid] = dpt
        adj.setdefault(nid, [])

        # Tag rooms this door belongs to
        door._graph_rooms = set()
        if door.room_a:
            door._graph_rooms.add(door.room_a.name)
        if door.room_b:
            door._graph_rooms.add(door.room_b.name)

    # Connect doors that share corridor access (both near corridor)
    door_ids = [nid for nid in nodes if nid.startswith("door_")]
    for cpoly in corridor_polys:
        on_corridor = [nid for nid in door_ids if _point_near_polygon(nodes[nid], cpoly, 0.35)]
        # Pairwise edges through corridor
        for i in range(len(on_corridor)):
            for j in range(i + 1, len(on_corridor)):
                w = _dist(nodes[on_corridor[i]], nodes[on_corridor[j]])
                _add_edge(on_corridor[i], on_corridor[j], w)
            # Also connect to exit and spine points on this corridor
            if exit_pt and _point_near_polygon(exit_pt, cpoly, 0.5):
                _add_edge(on_corridor[i], "exit", _dist(nodes[on_corridor[i]], exit_pt))
            for snid in [n for n in nodes if n.startswith("spine_")]:
                if _point_near_polygon(nodes[snid], cpoly, 0.35):
                    _add_edge(on_corridor[i], snid, _dist(nodes[on_corridor[i]], nodes[snid]))

    # Connect room-to-room doors: connect to all other doors of the shared rooms
    door_list = list(building.doors)
    for i, di in enumerate(door_list):
        for j, dj in enumerate(door_list):
            if i >= j:
                continue
            ri = getattr(di, "_graph_rooms", set())
            rj = getattr(dj, "_graph_rooms", set())
            if ri & rj:  # share a room
                ni, nj = f"door_{i}", f"door_{j}"
                if ni in nodes and nj in nodes:
                    _add_edge(ni, nj, _dist(nodes[ni], nodes[nj]))

    # If exit is not already connected, connect it to the nearest door
    if exit_pt and not adj.get("exit"):
        closest = None
        best_d = 1e9
        for nid in door_ids:
            d = _dist(exit_pt, nodes[nid])
            if d < best_d:
                best_d = d
                closest = nid
        if closest:
            _add_edge("exit", closest, best_d)

    return nodes, adj


# ─── Dijkstra ────────────────────────────────────────────────────────────────

def _dijkstra(adj: Dict[str, List[Tuple[str, float]]], source: str) -> Dict[str, float]:
    """Single-source shortest paths from *source*."""
    dist: Dict[str, float] = {source: 0.0}
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


# ─── Public API ───────────────────────────────────────────────────────────────

def door_graph_travel_distance(building) -> float:
    """Compute worst-case travel distance from any room to the exit
    using Dijkstra over the door/corridor graph.

    Returns the distance in metres (rounded to 2 dp), or 999.0 if the
    graph is disconnected or no doors exist.
    """
    if not building.doors or building.exit is None:
        return 999.0

    nodes, adj = _build_graph(building)
    if "exit" not in nodes:
        return 999.0

    dist_from_exit = _dijkstra(adj, "exit")

    # For each room, find the shortest path via any of its doors
    room_door_map: Dict[str, List[str]] = {}
    for didx, door in enumerate(building.doors):
        nid = f"door_{didx}"
        if nid not in nodes:
            continue
        if door.room_a:
            room_door_map.setdefault(door.room_a.name, []).append(nid)
        if door.room_b:
            room_door_map.setdefault(door.room_b.name, []).append(nid)

    worst = 0.0
    for room in building.rooms:
        if room.polygon is None:
            continue
        door_nodes = room_door_map.get(room.name, [])
        if not door_nodes:
            # Room has no door → treat as unreachable
            worst = max(worst, 999.0)
            continue
        best_room = min(dist_from_exit.get(dn, 999.0) for dn in door_nodes)
        # Add intra-room distance: centroid to nearest door
        if room.polygon:
            cx = sum(p[0] for p in room.polygon) / len(room.polygon)
            cy = sum(p[1] for p in room.polygon) / len(room.polygon)
            room_interior = min(
                _dist((cx, cy), nodes[dn]) for dn in door_nodes if dn in nodes
            )
            best_room += room_interior
        worst = max(worst, best_room)

    return round(worst, 2)


def get_room_travel_distances(building) -> Dict[str, float]:
    """Return a dict {room_name: travel_distance_to_exit} for every room."""
    if not building.doors or building.exit is None:
        return {r.name: 999.0 for r in building.rooms}

    nodes, adj = _build_graph(building)
    if "exit" not in nodes:
        return {r.name: 999.0 for r in building.rooms}

    dist_from_exit = _dijkstra(adj, "exit")

    room_door_map: Dict[str, List[str]] = {}
    for didx, door in enumerate(building.doors):
        nid = f"door_{didx}"
        if nid not in nodes:
            continue
        if door.room_a:
            room_door_map.setdefault(door.room_a.name, []).append(nid)
        if door.room_b:
            room_door_map.setdefault(door.room_b.name, []).append(nid)

    result = {}
    for room in building.rooms:
        door_nodes = room_door_map.get(room.name, [])
        if not door_nodes:
            result[room.name] = 999.0
            continue
        best = min(dist_from_exit.get(dn, 999.0) for dn in door_nodes)
        if room.polygon:
            cx = sum(p[0] for p in room.polygon) / len(room.polygon)
            cy = sum(p[1] for p in room.polygon) / len(room.polygon)
            best += min(_dist((cx, cy), nodes[dn]) for dn in door_nodes if dn in nodes)
        result[room.name] = round(best, 2)
    return result
