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

        for i in range(len(rooms)):
            for j in range(i+1, len(rooms)):
                edge = shared_edge(rooms[i], rooms[j])

                if edge:
                    door_segment = self.create_door_segment(edge[0])
                    door = Door(rooms[i], rooms[j], self.min_door_width, door_segment, door_type="room_to_room")
                    self.building.add_door(door)
                    rooms[i].doors.append(door)
                    rooms[j].doors.append(door)

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

        (x1, y1), (x2, y2) = list(line.coords)[0], list(line.coords)[-1]
        if abs(x1 - x2) >= abs(y1 - y2):
            mid = (x1 + x2) / 2.0
            half = self.min_door_width / 2.0
            return ((mid - half, y1), (mid + half, y1))

        mid = (y1 + y2) / 2.0
        half = self.min_door_width / 2.0
        return ((x1, mid - half), (x1, mid + half))

    def create_door_segment(self, edge):
        (x1, y1), (x2, y2) = edge

        if x1 == x2:
            mid = (y1 + y2) / 2
            return ((x1, mid - self.min_door_width/2),
                    (x1, mid + self.min_door_width/2))

        if y1 == y2:
            mid = (x1 + x2) / 2
            return ((mid - self.min_door_width/2, y1),
                    (mid + self.min_door_width/2, y1))