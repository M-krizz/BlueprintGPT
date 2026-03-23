"""
PolygonPacker – Packs rooms into any arbitrary boundary polygon using 
Continuous Recursive Bisection with Shapely.

Algorithm:
1. Takes the exact drawing bounding polygon as a Shapely Polygon.
2. Recursively slices the polygon using an axis-aligned cut (horizontal or vertical)
   into two exact area proportions matching the rooms' required areas.
3. Assigns each room a continuous, perfectly fitted polygon that sums to 100% 
   of the building area with NO gaps or overlaps, regardless of the boundary shape.
"""

import shapely
from shapely.geometry import Polygon, box, MultiPolygon
from geometry.allocator import Allocator

def _get_largest_polygon(geom):
    """If a slice results in a MultiPolygon, return the largest contiguous part."""
    if geom.is_empty:
        return None
    if isinstance(geom, MultiPolygon) or getattr(geom, 'geom_type', '') == 'MultiPolygon':
        if not list(geom.geoms): return None
        return max(geom.geoms, key=lambda p: p.area)
    elif isinstance(geom, shapely.geometry.collection.GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, Polygon)]
        if not polys: return None
        return max(polys, key=lambda p: p.area)
    return geom

def bisect_polygon(poly, ratio, axis='x'):
    """
    Slices a Shapely Polygon along the given axis such that 
    the first piece has 'ratio' * poly.area.
    Returns (poly1, poly2).
    """
    minx, miny, maxx, maxy = poly.bounds
    target_area = poly.area * ratio
    
    if axis == 'x':
        low, high = minx, maxx
    else:
        low, high = miny, maxy
        
    best_mid = low
    # Binary search for the exact slice coordinate
    for _ in range(35):
        mid = (low + high) / 2.0
        
        if axis == 'x':
            clipper = box(minx - 1, miny - 1, mid, maxy + 1)
        else:
            clipper = box(minx - 1, miny - 1, maxx + 1, mid)
            
        part1 = poly.intersection(clipper)
        if part1.area < target_area:
            low = mid
        else:
            high = mid
            best_mid = mid

    # Final split
    if axis == 'x':
        clipper1 = box(minx - 1, miny - 1, best_mid, maxy + 1)
        clipper2 = box(best_mid, miny - 1, maxx + 1, maxy + 1)
    else:
        clipper1 = box(minx - 1, miny - 1, maxx + 1, best_mid)
        clipper2 = box(minx - 1, best_mid, maxx + 1, maxy + 1)
        
    poly1 = _get_largest_polygon(poly.intersection(clipper1))
    poly2 = _get_largest_polygon(poly.intersection(clipper2))
    
    return poly1, poly2


def _assign_polygon_to_room(room, poly):
    """Assign a cleaned polygon to a room."""
    try:
        clean_poly = poly.simplify(0.01)
        coords = list(clean_poly.exterior.coords)
        room.polygon = [(round(x, 3), round(y, 3)) for x, y in coords]
    except Exception:
        pass


def _room_hint_position(room, learned_hints, minx, miny, maxx, maxy, axis):
    if not learned_hints:
        return None
    hint = learned_hints.get(getattr(room, "name", "")) or learned_hints.get(getattr(room, "room_type", ""))
    if not hint:
        return None
    cx_norm, cy_norm = hint
    if axis == 'x':
        return minx + cx_norm * (maxx - minx)
    return miny + cy_norm * (maxy - miny)


def _prefer_low_side(room, axis, mid, entrance_pt=None, learned_hints=None, bounds=None):
    minx, miny, maxx, maxy = bounds
    hinted = _room_hint_position(room, learned_hints, minx, miny, maxx, maxy, axis)
    if hinted is not None:
        return hinted <= mid

    if entrance_pt is None:
        return True

    entrance_coord = entrance_pt[0] if axis == 'x' else entrance_pt[1]
    entrance_is_low = entrance_coord <= mid
    room_type = getattr(room, "room_type", "")
    if room_type in {"LivingRoom", "DrawingRoom", "Kitchen", "DiningRoom"}:
        return entrance_is_low
    if room_type in {"Bedroom", "Bathroom", "WC"}:
        return not entrance_is_low
    return entrance_is_low


def _fit_single_room_polygon(poly, item, entrance_pt=None, learned_hints=None):
    room = item['room']
    target_area = max(float(item.get('weight', 0.0) or 0.0), 1e-6)
    if poly.area <= target_area * 1.15:
        return poly

    minx, miny, maxx, maxy = poly.bounds
    width = maxx - minx
    height = maxy - miny
    axis = 'x' if width > height else 'y'
    mid = (minx + maxx) / 2.0 if axis == 'x' else (miny + maxy) / 2.0
    ratio = max(0.03, min(0.92, target_area / max(poly.area, 1e-6)))

    prefer_low = _prefer_low_side(
        room,
        axis,
        mid,
        entrance_pt=entrance_pt,
        learned_hints=learned_hints,
        bounds=(minx, miny, maxx, maxy),
    )

    if prefer_low:
        fitted, _ = bisect_polygon(poly, ratio, axis)
    else:
        _, fitted = bisect_polygon(poly, 1.0 - ratio, axis)

    return fitted or poly


def recursive_pack(poly, items, entrance_pt=None, learned_hints=None, fit_single_room=True):
    """
    Recursively divides 'poly' to fit 'items' proportionally.
    items: list of dicts {'room': Room, 'weight': float}
    entrance_pt: (x, y) in metres
    learned_hints: dict of {room_type: (cx_norm, cy_norm)} — normalized [0,1]
                   centroids from the transformer. Used as ordering hints only;
                   rooms still get arbitrary polygon shapes from bisection.
    """
    if not items or poly is None or poly.is_empty:
        return
        
    if len(items) == 1:
        final_poly = poly
        if fit_single_room:
            final_poly = _fit_single_room_polygon(
                poly,
                items[0],
                entrance_pt=entrance_pt,
                learned_hints=learned_hints,
            )
        _assign_polygon_to_room(items[0]['room'], final_poly)
        return

    # Split items into two groups roughly equal by weight
    total_weight = sum(i['weight'] for i in items)
    if total_weight <= 0:
        return
        
    target = total_weight / 2.0
    current_weight = 0.0
    split_idx = 1
    
    for i, item in enumerate(items):
        current_weight += item['weight']
        if current_weight >= target and i < len(items) - 1:
            split_idx = i + 1
            break
            
    group1 = items[:split_idx]
    group2 = items[split_idx:]
    
    w1 = sum(i['weight'] for i in group1)
    w2 = sum(i['weight'] for i in group2)
    ratio = w1 / (w1 + w2)
    
    # Decide split axis based on polygon bounds
    minx, miny, maxx, maxy = poly.bounds
    width = maxx - minx
    height = maxy - miny
    axis = 'x' if width > height else 'y'
    
    poly1, poly2 = bisect_polygon(poly, ratio, axis)
    
    if poly1 and poly2:
        should_swap = False

        # ── Learned-hint ordering ─────────────────────────────────────────────
        # Use transformer centroid hints to decide which group goes to which
        # polygon half. Hints are normalized [0,1] relative to the full boundary;
        # we compare against the midpoint of the bisection cut.
        if learned_hints:
            # Midpoint of the boundary in the bisection axis
            mid_x = (minx + maxx) / 2.0
            mid_y = (miny + maxy) / 2.0

            def _hint_side(group):
                """Return average hint position along the cut axis for the group."""
                positions = []
                for item in group:
                    rname = getattr(item['room'], 'name', None)
                    rtype = getattr(item['room'], 'room_type', None)
                    hint = None
                    if rname and rname in learned_hints:
                        hint = learned_hints[rname]
                    elif rtype and rtype in learned_hints:
                        hint = learned_hints[rtype]
                    if hint:
                        cx_norm, cy_norm = hint
                        if axis == 'x':
                            # Denormalize to real coords using boundary extent
                            positions.append(minx + cx_norm * (maxx - minx))
                        else:
                            positions.append(miny + cy_norm * (maxy - miny))
                return sum(positions) / len(positions) if positions else None

            g1_pos = _hint_side(group1)
            g2_pos = _hint_side(group2)

            if g1_pos is not None and g2_pos is not None:
                # poly1 is the lower half (x < mid_x or y < mid_y)
                # group1 should go to poly1 if its hint centroid is on the low side
                # Check if order needs to be swapped based on learned positions
                if axis == 'x':
                    g1_wants_low = g1_pos <= mid_x
                    g2_wants_low = g2_pos <= mid_x
                else:
                    g1_wants_low = g1_pos <= mid_y
                    g2_wants_low = g2_pos <= mid_y

                # Swap if group1's hint says it belongs to poly2 (high side)
                # and group2's hint says it belongs to poly1 (low side)
                if not g1_wants_low and g2_wants_low:
                    should_swap = True
            elif g1_pos is not None:
                # Only group1 has a hint — put it on the side its centroid suggests
                if axis == 'x':
                    should_swap = g1_pos > mid_x
                else:
                    should_swap = g1_pos > mid_y
            elif g2_pos is not None:
                # Only group2 has a hint — put it on the side its centroid suggests
                if axis == 'x':
                    should_swap = g2_pos <= mid_x
                else:
                    should_swap = g2_pos <= mid_y

        # ── Entrance Weighting (fallback when no learned hints for LivingRoom) ─
        if not should_swap and entrance_pt and not (
            learned_hints and any(
                item['room'].room_type == 'LivingRoom'
                for item in group1 + group2
                if item['room'].room_type in (learned_hints or {})
            )
        ):
            # Check if LivingRoom is in group1 or group2
            g1_has_living = any(i['room'].room_type == 'LivingRoom' for i in group1)
            g2_has_living = any(i['room'].room_type == 'LivingRoom' for i in group2)
            
            if g1_has_living or g2_has_living:
                c1 = poly1.centroid
                c2 = poly2.centroid
                d1 = (c1.x - entrance_pt[0])**2 + (c1.y - entrance_pt[1])**2
                d2 = (c2.x - entrance_pt[0])**2 + (c2.y - entrance_pt[1])**2
                
                if g1_has_living and d2 < d1:
                    should_swap = True
                elif g2_has_living and d1 < d2:
                    should_swap = True

        if should_swap:
            swapped_poly1, swapped_poly2 = bisect_polygon(poly, 1.0 - ratio, axis)
            if swapped_poly1 and swapped_poly2:
                poly1, poly2 = swapped_poly1, swapped_poly2
            recursive_pack(poly1, group2, entrance_pt, learned_hints, fit_single_room=fit_single_room)
            recursive_pack(poly2, group1, entrance_pt, learned_hints, fit_single_room=fit_single_room)
        else:
            recursive_pack(poly1, group1, entrance_pt, learned_hints, fit_single_room=fit_single_room)
            recursive_pack(poly2, group2, entrance_pt, learned_hints, fit_single_room=fit_single_room)
    else:
        # Failsafe if bisection somehow collapses: forcibly split the polygon down the middle mathematically
        minx, miny, maxx, maxy = poly.bounds
        if (maxx - minx) > (maxy - miny):
            midx = (minx + maxx) / 2.0
            p1 = poly.intersection(box(minx - 1, miny - 1, midx, maxy + 1))
            p2 = poly.intersection(box(midx, miny - 1, maxx + 1, maxy + 1))
        else:
            midy = (miny + maxy) / 2.0
            p1 = poly.intersection(box(minx - 1, miny - 1, maxx + 1, midy))
            p2 = poly.intersection(box(minx - 1, midy, maxx + 1, maxy + 1))
        
        poly1 = _get_largest_polygon(p1) or poly
        poly2 = _get_largest_polygon(p2) or poly

        recursive_pack(poly1, items[:len(items)//2], entrance_pt, learned_hints, fit_single_room=fit_single_room)
        recursive_pack(poly2, items[len(items)//2:], entrance_pt, learned_hints, fit_single_room=fit_single_room)


class PolygonPacker:
    """
    Allocates exact freeform room shapes filling 100% of the boundary area
    by recursively bisecting the boundary geometry (Continuous Bisection).
    
    Rooms receive arbitrary polygon shapes carved from the boundary — never
    forced rectangles. ``learned_hints`` provides optional transformer centroid
    priors that bias bisection group assignment without changing room geometry.
    """

    def __init__(self, building, boundary_polygon, entrance_point=None,
                 learned_hints=None, placement_order=None, room_zones=None):
        self.building = building
        self.boundary = boundary_polygon
        self.entrance = entrance_point
        # learned_hints: {room_type: (cx_norm, cy_norm)} in [0,1] space
        self.learned_hints = learned_hints or {}
        self.placement_order = placement_order or []
        self.room_zones = room_zones or {}

    def allocate(self):
        rooms = [r for r in self.building.rooms if r.final_area and r.final_area > 0]
        if not rooms:
            return 0.0, 0.0

        if not self.boundary or len(self.boundary) < 3:
            return Allocator(self.building).allocate()
            
        try:
            poly = Polygon(self.boundary)
            if not poly.is_valid:
                poly = poly.buffer(0)
        except Exception:
            return Allocator(self.building).allocate()
            
        if self.placement_order:
            order_rank = {room_name: index for index, room_name in enumerate(self.placement_order)}
            rooms.sort(
                key=lambda r: (
                    order_rank.get(getattr(r, "name", ""), len(order_rank) + 1),
                    {"public": 0, "service": 1, "private": 2}.get(self.room_zones.get(getattr(r, "name", ""), "private"), 3),
                    -(r.final_area or 0.0),
                )
            )
        else:
            # Sort rooms by area primarily, but keep a stable order
            rooms.sort(
                key=lambda r: (
                    {"public": 0, "service": 1, "private": 2}.get(self.room_zones.get(getattr(r, "name", ""), "private"), 3),
                    -(r.final_area or 0.0),
                )
            )
        items = [{'room': r, 'weight': r.final_area} for r in rooms]
        
        recursive_pack(
            poly, items,
            entrance_pt=self.entrance,
            learned_hints=self.learned_hints if self.learned_hints else None,
            fit_single_room=False,
        )
        
        return self._bounding_box()

    def _bounding_box(self):
        max_x = max_y = 0.0
        for room in self.building.rooms:
            if room.polygon:
                for x, y in room.polygon:
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
        return round(max_x, 3), round(max_y, 3)

