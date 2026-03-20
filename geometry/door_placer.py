from core.door import Door
from geometry.adjacency import shared_edge
from shapely.geometry import Polygon, MultiLineString, LineString, Point

# Maximum gap (metres) for which a "bridge door" can be placed between a room
# and the corridor even when they don't physically share a wall.
BRIDGE_GAP_TOLERANCE = 0.30   # 30 cm

class DoorPlacer:

    def __init__(self, building, min_door_width, bridge_gap=BRIDGE_GAP_TOLERANCE):
        self.building = building
        self.min_door_width = min_door_width
        self.bridge_gap = bridge_gap
        # Distance to keep between door centers so their swing arcs don't clash
        self.door_keepout = min_door_width * 2.0

    def _is_too_close_to_existing(self, pt):
        """Check if a candidate door center `pt` is within keepout radius of existing doors."""
        for d in getattr(self.building, "doors", []):
            if d.segment:
                (sx1, sy1), (sx2, sy2) = d.segment
                cx = (sx1 + sx2) / 2.0
                cy = (sy1 + sy2) / 2.0
                dist = ((pt[0] - cx)**2 + (pt[1] - cy)**2)**0.5
                if dist < self.door_keepout:
                    return True
        return False

    def place_doors(self):
        self.building.doors = []
        for room in self.building.rooms:
            room.doors = []

        rooms = self.building.rooms
        rooms_with_circulation_door = set()

        for corridor in getattr(self.building, "corridors", []):
            corridor_poly = getattr(corridor, "polygon", None)
            if not corridor_poly:
                continue

            for room in rooms:
                if not room.polygon or room.name in rooms_with_circulation_door:
                    continue

                # Try exact shared wall first
                segment = self._corridor_facing_segment(room.polygon, corridor_poly)
                if segment is not None:
                    door = Door(room, None, self.min_door_width, segment, door_type="room_to_circulation")
                    self.building.add_door(door)
                    room.doors.append(door)
                    rooms_with_circulation_door.add(room.name)
                    continue

                # Bridge-door: room is close but doesn't share a wall
                bridge_seg = self._bridge_door_segment(room.polygon, corridor_poly)
                if bridge_seg is not None:
                    door = Door(room, None, self.min_door_width, bridge_seg, door_type="room_to_circulation")
                    self.building.add_door(door)
                    room.doors.append(door)
                    rooms_with_circulation_door.add(room.name)

        # ── Room-to-room doors ────────────────────────────────────────────────
        # Only add a room-to-room door if:
        # a) One of the rooms has no door to the circulation (needs access).
        # b) They share a strong adjacency requirement (>0.6 weight).
        from geometry.adjacency_intent import build_adjacency_intent
        intents = build_adjacency_intent()
        intent_map = {}
        for ta, tb, w in intents:
            intent_map.setdefault(ta, {})[tb] = w
            intent_map.setdefault(tb, {})[ta] = w

        for i in range(len(rooms)):
            for j in range(i+1, len(rooms)):
                edge = shared_edge(rooms[i], rooms[j])

                if edge:
                    r1, r2 = rooms[i], rooms[j]
                    has_circ_1 = r1.name in rooms_with_circulation_door
                    has_circ_2 = r2.name in rooms_with_circulation_door
                    
                    # Look up adjacency weight between r1.room_type and r2.room_type
                    weight = intent_map.get(r1.room_type, {}).get(r2.room_type, 0.0)

                    # Condition to add door:
                    # 1. Access needed: one room has NO doors yet
                    # 2. Strong link: weight >= 0.6 AND we don't already have too many doors linking these paths
                    needs_access = (len(r1.doors) == 0) or (len(r2.doors) == 0)
                    strong_link = weight >= 0.6  # e.g., Bedroom <-> Bathroom

                    if needs_access or strong_link:
                        door_segment = self.create_door_segment(edge[0])
                        # If clamping failed (segment too small), skip
                        if door_segment is None:
                            continue
                        
                        door = Door(r1, r2, self.min_door_width, door_segment, door_type="room_to_room")
                        self.building.add_door(door)
                        r1.doors.append(door)
                        r2.doors.append(door)

        # Fallback: ensure every room has at least one door
        for room in rooms:
            if len(room.doors) == 0:
                for other in rooms:
                    if other is room or not other.polygon:
                        continue
                    edge = shared_edge(room, other)
                    if edge:
                        door_segment = self.create_door_segment(edge[0])
                        if door_segment:
                            door = Door(room, other, self.min_door_width, door_segment, door_type="room_to_room")
                            self.building.add_door(door)
                            room.doors.append(door)
                            other.doors.append(door)
                            break

    # ── Bridge door (near-gap) ────────────────────────────────────────────

    def _bridge_door_segment(self, room_polygon, corridor_polygon):
        """Create a door segment for a room that is *close* to but not touching
        the corridor.  Returns None if the gap exceeds ``self.bridge_gap``."""
        try:
            rpoly = Polygon(room_polygon)
            cpoly = Polygon(corridor_polygon)
            gap = rpoly.distance(cpoly)
            if gap > self.bridge_gap or gap < 1e-6:
                return None

            # Nearest points on room boundary & corridor boundary
            from shapely.ops import nearest_points
            rpt, cpt = nearest_points(rpoly.boundary, cpoly.boundary)

            # Build a short perpendicular door segment at the midpoint of the gap
            mx = (rpt.x + cpt.x) / 2.0
            my = (rpt.y + cpt.y) / 2.0
            dx = cpt.x - rpt.x
            dy = cpt.y - rpt.y
            length = max((dx**2 + dy**2) ** 0.5, 1e-9)
            # Perpendicular direction
            px = -dy / length
            py = dx / length
            
            if self._is_too_close_to_existing((mx, my)):
                return None  # Bridge door location overlaps existing door
                
            half = self.min_door_width / 2.0
            return ((mx + px * half, my + py * half),
                    (mx - px * half, my - py * half))
        except Exception:
            return None

    def _corridor_facing_segment(self, room_polygon, corridor_polygon):
        try:
            room_boundary = Polygon(room_polygon).boundary
            corr_boundary = Polygon(corridor_polygon).boundary
            overlap = room_boundary.intersection(corr_boundary)
        except Exception:
            return None

        if overlap.is_empty:
            return None

        line = None
        if isinstance(overlap, LineString):
            line = overlap
        elif isinstance(overlap, MultiLineString):
            line = max(overlap.geoms, key=lambda geom: geom.length, default=None)
        else:
            geoms = [geom for geom in getattr(overlap, "geoms", []) if isinstance(geom, LineString)]
            if geoms:
                line = max(geoms, key=lambda geom: geom.length)

        if line is None or line.length <= 0:
            return None

        (x1, y1) = list(line.coords)[0]
        (x2, y2) = list(line.coords)[-1]
        
        if abs(x1 - x2) >= abs(y1 - y2):
            min_x, max_x = min(x1, x2), max(x1, x2)
            if max_x - min_x < self.min_door_width:
                return None  # Line too short for a door
                
            margin = 0.25 if max_x - min_x >= self.min_door_width + 0.5 else 0.0
            half = self.min_door_width / 2.0
            best_mid = None
            # Scan along the segment for a valid center
            steps = max(int((max_x - min_x) / 0.1), 1)
            for i in range(steps + 1):
                t = i / steps
                cand_mid = (min_x + margin + half) * (1 - t) + (max_x - margin - half) * t
                if not self._is_too_close_to_existing((cand_mid, y1)):
                    best_mid = cand_mid
                    break
                    
            if best_mid is None:
                return None
                
            return ((best_mid - half, y1), (best_mid + half, y1))

        min_y, max_y = min(y1, y2), max(y1, y2)
        if max_y - min_y < self.min_door_width:
            return None
            
        margin = 0.25 if max_y - min_y >= self.min_door_width + 0.5 else 0.0
        half = self.min_door_width / 2.0
        best_mid = None
        steps = max(int((max_y - min_y) / 0.1), 1)
        for i in range(steps + 1):
            t = i / steps
            cand_mid = (min_y + margin + half) * (1 - t) + (max_y - margin - half) * t
            if not self._is_too_close_to_existing((x1, cand_mid)):
                best_mid = cand_mid
                break
                
        if best_mid is None:
            return None
            
        return ((x1, best_mid - half), (x1, best_mid + half))

    def create_door_segment(self, edge):
        (x1, y1), (x2, y2) = edge

        if abs(x1 - x2) >= abs(y1 - y2):
            # Horizontal-ish edge
            min_x, max_x = min(x1, x2), max(x1, x2)
            if max_x - min_x < self.min_door_width:
                return None
                
            margin = 0.25 if max_x - min_x >= self.min_door_width + 0.5 else 0.0
            half = self.min_door_width / 2.0
            best_mid = None
            steps = max(int((max_x - min_x) / 0.1), 1)
            for i in range(steps + 1):
                t = i / steps
                cand_mid = (min_x + margin + half) * (1 - t) + (max_x - margin - half) * t
                if not self._is_too_close_to_existing((cand_mid, y1)):
                    best_mid = cand_mid
                    break
                    
            if best_mid is None:
                return None
                
            return ((best_mid - half, y1), (best_mid + half, y1))
        else:
            # Vertical-ish edge
            min_y, max_y = min(y1, y2), max(y1, y2)
            if max_y - min_y < self.min_door_width:
                return None
                
            margin = 0.25 if max_y - min_y >= self.min_door_width + 0.5 else 0.0
            half = self.min_door_width / 2.0
            best_mid = None
            steps = max(int((max_y - min_y) / 0.1), 1)
            for i in range(steps + 1):
                t = i / steps
                cand_mid = (min_y + margin + half) * (1 - t) + (max_y - margin - half) * t
                if not self._is_too_close_to_existing((x1, cand_mid)):
                    best_mid = cand_mid
                    break
                    
            if best_mid is None:
                return None
                
            return ((x1, best_mid - half), (x1, best_mid + half))