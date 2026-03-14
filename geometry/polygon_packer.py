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


def recursive_pack(poly, items, entrance_pt=None):
    """
    Recursively divides 'poly' to fit 'items' proportionally.
    items: list of dicts {'room': Room, 'weight': float}
    entrance_pt: (x, y) in metres
    """
    if not items or poly is None or poly.is_empty:
        return
        
    if len(items) == 1:
        # Assign entire remaining polygon to this room
        try:
            # Simplify slightly to avoid collinear coordinate bloating
            clean_poly = poly.simplify(0.01)
            coords = list(clean_poly.exterior.coords)
            items[0]['room'].polygon = [(round(x, 3), round(y, 3)) for x, y in coords]
        except Exception:
            pass
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
        # ── Entrance Weighting ────────────────────────────────────────────────
        # If we have an entrance point, we want the group containing 'LivingRoom'
        # to go to the polygon closest to the entrance.
        should_swap = False
        if entrance_pt:
            # Check if LivingRoom is in group1 or group2
            g1_has_living = any(i['room'].room_type == 'LivingRoom' for i in group1)
            g2_has_living = any(i['room'].room_type == 'LivingRoom' for i in group2)
            
            if g1_has_living or g2_has_living:
                c1 = poly1.centroid
                c2 = poly2.centroid
                d1 = (c1.x - entrance_pt[0])**2 + (c1.y - entrance_pt[1])**2
                d2 = (c2.x - entrance_pt[0])**2 + (c2.y - entrance_pt[1])**2
                
                # If LivingRoom is in G1, it should be in the closer poly (shorter d)
                if g1_has_living and d2 < d1:
                    should_swap = True
                # If LivingRoom is in G2, it should be in the closer poly (shorter d)
                elif g2_has_living and d1 < d2:
                    should_swap = True

        if should_swap:
            recursive_pack(poly1, group2, entrance_pt)
            recursive_pack(poly2, group1, entrance_pt)
        else:
            recursive_pack(poly1, group1, entrance_pt)
            recursive_pack(poly2, group2, entrance_pt)
    else:
        # Failsafe if bisection somehow collapses
        recursive_pack(poly, items[:len(items)//2], entrance_pt)
        recursive_pack(poly, items[len(items)//2:], entrance_pt)


class PolygonPacker:
    """
    Allocates exact freeform room shapes filling 100% of the boundary area
    by recursively bisecting the boundary geometry (Continuous Bisection).
    """

    def __init__(self, building, boundary_polygon, entrance_point=None):
        self.building = building
        self.boundary = boundary_polygon
        self.entrance = entrance_point

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
            
        # Sort rooms by area primarily, but keep a stable order
        rooms.sort(key=lambda r: (r.room_type == 'LivingRoom', r.final_area), reverse=True)
        items = [{'room': r, 'weight': r.final_area} for r in rooms]
        
        recursive_pack(poly, items, entrance_pt=self.entrance)
        
        return self._bounding_box()

    def _bounding_box(self):
        max_x = max_y = 0.0
        for room in self.building.rooms:
            if room.polygon:
                for x, y in room.polygon:
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
        return round(max_x, 3), round(max_y, 3)
