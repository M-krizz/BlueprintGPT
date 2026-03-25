import math

from shapely.geometry import LineString, MultiLineString, Polygon

from core.door import Door
from geometry.adjacency import shared_edge

# Maximum gap (metres) for which a "bridge door" can be placed between a room
# and the corridor even when they do not physically share a wall.
BRIDGE_GAP_TOLERANCE = 0.30


class DoorPlacer:
    PUBLIC_ROOM_TYPES = {"LivingRoom", "DrawingRoom", "DiningRoom", "Lobby", "Foyer"}
    PRIVATE_ROOM_TYPES = {"Bedroom", "Study", "DressingArea"}
    SERVICE_ROOM_TYPES = {"Kitchen", "Store", "Utility", "Pantry"}
    SANITARY_ROOM_TYPES = {"Bathroom", "WC", "Toilet"}

    def __init__(self, building, min_door_width, bridge_gap=BRIDGE_GAP_TOLERANCE):
        self.building = building
        self.min_door_width = min_door_width
        self.bridge_gap = bridge_gap
        # Keep door centers separated so swing arcs do not visually collide.
        self.door_keepout = min_door_width * 2.0
        self.corner_margin = max(0.25, min_door_width * 0.4)

    def _distance(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _room_center(self, room):
        polygon = getattr(room, "polygon", None)
        if not polygon:
            return (0.0, 0.0)
        xs = [pt[0] for pt in polygon]
        ys = [pt[1] for pt in polygon]
        return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)

    def _room_family(self, room):
        room_type = getattr(room, "room_type", "") or ""
        if room_type in self.PUBLIC_ROOM_TYPES:
            return "public"
        if room_type in self.PRIVATE_ROOM_TYPES:
            return "private"
        if room_type in self.SANITARY_ROOM_TYPES:
            return "sanitary"
        if room_type in self.SERVICE_ROOM_TYPES:
            return "service"
        return "other"

    def _is_sanitary_public_pair(self, room_a, room_b):
        families = {self._room_family(room_a), self._room_family(room_b)}
        return "sanitary" in families and "public" in families

    def _is_too_close_to_existing(self, pt):
        for door in getattr(self.building, "doors", []):
            if not getattr(door, "segment", None):
                continue
            (sx1, sy1), (sx2, sy2) = door.segment
            center = ((sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0)
            if self._distance(pt, center) < self.door_keepout:
                return True
        return False

    def _overlap_segment(self, edge_a, edge_b):
        (x1, y1), (x2, y2) = edge_a
        (x3, y3), (x4, y4) = edge_b

        if abs(x1 - x2) < 1e-9 and abs(x3 - x4) < 1e-9 and abs(x1 - x3) < 1e-9:
            low = max(min(y1, y2), min(y3, y4))
            high = min(max(y1, y2), max(y3, y4))
            if high - low > 1e-9:
                return ((x1, low), (x1, high))
            return None

        if abs(y1 - y2) < 1e-9 and abs(y3 - y4) < 1e-9 and abs(y1 - y3) < 1e-9:
            low = max(min(x1, x2), min(x3, x4))
            high = min(max(x1, x2), max(x3, x4))
            if high - low > 1e-9:
                return ((low, y1), (high, y1))
            return None

        return None

    def _segment_midpoint(self, segment):
        (x1, y1), (x2, y2) = segment
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _segment_length(self, segment):
        return self._distance(segment[0], segment[1])

    def _candidate_segments_on_edge(self, edge):
        (x1, y1), (x2, y2) = edge
        horizontal = abs(x1 - x2) >= abs(y1 - y2)
        half = self.min_door_width / 2.0
        ratios = (0.5, 0.35, 0.65, 0.2, 0.8)
        candidates = []

        if horizontal:
            min_x, max_x = min(x1, x2), max(x1, x2)
            usable_start = min_x + self.corner_margin + half
            usable_end = max_x - self.corner_margin - half
            if usable_end < usable_start:
                usable_start = min_x + half
                usable_end = max_x - half
            if usable_end < usable_start:
                return []
            if abs(usable_end - usable_start) < 1e-9:
                mids = [usable_start]
            else:
                mids = [usable_start + (usable_end - usable_start) * ratio for ratio in ratios]
            for mid_x in mids:
                segment = ((mid_x - half, y1), (mid_x + half, y1))
                candidates.append(segment)
            return candidates

        min_y, max_y = min(y1, y2), max(y1, y2)
        usable_start = min_y + self.corner_margin + half
        usable_end = max_y - self.corner_margin - half
        if usable_end < usable_start:
            usable_start = min_y + half
            usable_end = max_y - half
        if usable_end < usable_start:
            return []
        if abs(usable_end - usable_start) < 1e-9:
            mids = [usable_start]
        else:
            mids = [usable_start + (usable_end - usable_start) * ratio for ratio in ratios]
        for mid_y in mids:
            segment = ((x1, mid_y - half), (x1, mid_y + half))
            candidates.append(segment)
        return candidates

    def _candidate_score(
        self,
        segment,
        *,
        room_a=None,
        room_b=None,
        is_circulation=False,
        needs_access=False,
        connection_weight=0.0,
    ):
        midpoint = self._segment_midpoint(segment)
        if self._is_too_close_to_existing(midpoint):
            return None

        edge_len = self._segment_length(segment)
        score = 0.0

        # Prefer candidates that stay away from corners and do not hug the edge ends.
        score += min(
            self._distance(midpoint, segment[0]),
            self._distance(midpoint, segment[1]),
        ) / max(self.min_door_width, 0.1)

        if room_a is not None:
            center_a = self._room_center(room_a)
            if room_b is not None:
                center_b = self._room_center(room_b)
                if abs(segment[0][0] - segment[1][0]) < 1e-9:
                    target = (center_a[1] + center_b[1]) / 2.0
                    coord = midpoint[1]
                else:
                    target = (center_a[0] + center_b[0]) / 2.0
                    coord = midpoint[0]
                score -= abs(coord - target) / max(edge_len, 0.1)
            else:
                if abs(segment[0][0] - segment[1][0]) < 1e-9:
                    score -= abs(midpoint[1] - center_a[1]) / max(edge_len, 0.1)
                else:
                    score -= abs(midpoint[0] - center_a[0]) / max(edge_len, 0.1)

        if needs_access:
            score += 0.6

        if is_circulation and room_a is not None:
            family = self._room_family(room_a)
            if family == "public":
                score += 0.8
            elif family == "service":
                score += 0.55
            elif family == "private":
                score += 0.45
            elif family == "sanitary":
                score += 0.25

        if room_a is not None and room_b is not None:
            score += connection_weight * 1.4
            family_a = self._room_family(room_a)
            family_b = self._room_family(room_b)
            families = {family_a, family_b}
            if families == {"private", "sanitary"}:
                score += 1.0
            elif families == {"public", "service"}:
                score += 0.8
            elif families == {"private"}:
                score += 0.25
            elif families == {"public", "sanitary"}:
                score -= 1.25
            elif connection_weight < 0:
                score -= abs(connection_weight) * 3.0

        return score

    def _best_segment_on_edge(
        self,
        edge,
        *,
        room_a=None,
        room_b=None,
        is_circulation=False,
        needs_access=False,
        connection_weight=0.0,
    ):
        best_segment = None
        best_score = None
        for candidate in self._candidate_segments_on_edge(edge):
            score = self._candidate_score(
                candidate,
                room_a=room_a,
                room_b=room_b,
                is_circulation=is_circulation,
                needs_access=needs_access,
                connection_weight=connection_weight,
            )
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_segment = candidate
        return best_segment

    def _has_private_access_alternative(self, sanitary_room, exclude_room, rooms):
        for other in rooms:
            if other is sanitary_room or other is exclude_room or not getattr(other, "polygon", None):
                continue
            shared = shared_edge(sanitary_room, other)
            if not shared:
                continue
            if self._room_family(other) == "private":
                return True
        return False

    def _corridor_facing_segment(self, room_polygon, corridor_polygon, *, room=None):
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

        coords = list(line.coords)
        edge = (coords[0], coords[-1])
        return self._best_segment_on_edge(
            edge,
            room_a=room,
            is_circulation=True,
            needs_access=True,
            connection_weight=0.0,
        )

    def _bridge_door_segment(self, room_polygon, corridor_polygon):
        try:
            rpoly = Polygon(room_polygon)
            cpoly = Polygon(corridor_polygon)
            gap = rpoly.distance(cpoly)
            if gap > self.bridge_gap or gap < 1e-6:
                return None

            from shapely.ops import nearest_points

            rpt, cpt = nearest_points(rpoly.boundary, cpoly.boundary)

            mx = (rpt.x + cpt.x) / 2.0
            my = (rpt.y + cpt.y) / 2.0
            dx = cpt.x - rpt.x
            dy = cpt.y - rpt.y
            length = max((dx ** 2 + dy ** 2) ** 0.5, 1e-9)
            px = -dy / length
            py = dx / length

            if self._is_too_close_to_existing((mx, my)):
                return None

            half = self.min_door_width / 2.0
            return (
                (mx + px * half, my + py * half),
                (mx - px * half, my - py * half),
            )
        except Exception:
            return None

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

                segment = self._corridor_facing_segment(room.polygon, corridor_poly, room=room)
                if segment is not None:
                    door = Door(room, None, self.min_door_width, segment, door_type="room_to_circulation")
                    self.building.add_door(door)
                    room.doors.append(door)
                    rooms_with_circulation_door.add(room.name)
                    continue

                bridge_seg = self._bridge_door_segment(room.polygon, corridor_poly)
                if bridge_seg is not None:
                    door = Door(room, None, self.min_door_width, bridge_seg, door_type="room_to_circulation")
                    self.building.add_door(door)
                    room.doors.append(door)
                    rooms_with_circulation_door.add(room.name)

        from geometry.adjacency_intent import build_adjacency_intent

        intents = build_adjacency_intent()
        intent_map = {}
        for type_a, type_b, weight in intents:
            intent_map.setdefault(type_a, {})[type_b] = weight
            intent_map.setdefault(type_b, {})[type_a] = weight

        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                shared = shared_edge(rooms[i], rooms[j])
                if not shared:
                    continue

                room_a = rooms[i]
                room_b = rooms[j]
                overlap_segment = self._overlap_segment(shared[0], shared[1]) or shared[0]

                weight = intent_map.get(room_a.room_type, {}).get(room_b.room_type, 0.0)
                needs_access = (len(room_a.doors) == 0) or (len(room_b.doors) == 0)
                strong_link = weight >= 0.75
                has_circ_a = room_a.name in rooms_with_circulation_door
                has_circ_b = room_b.name in rooms_with_circulation_door

                if weight < 0 and has_circ_a and has_circ_b:
                    continue

                if self._is_sanitary_public_pair(room_a, room_b):
                    sanitary_room = room_a if self._room_family(room_a) == "sanitary" else room_b
                    public_room = room_b if sanitary_room is room_a else room_a
                    if (
                        sanitary_room.name in rooms_with_circulation_door
                        or self._has_private_access_alternative(sanitary_room, public_room, rooms)
                    ):
                        continue

                if not (needs_access or strong_link):
                    continue

                door_segment = self._best_segment_on_edge(
                    overlap_segment,
                    room_a=room_a,
                    room_b=room_b,
                    needs_access=needs_access,
                    connection_weight=weight,
                )
                if door_segment is None:
                    continue

                door = Door(room_a, room_b, self.min_door_width, door_segment, door_type="room_to_room")
                self.building.add_door(door)
                room_a.doors.append(door)
                room_b.doors.append(door)

        # Fallback: ensure every room still ends up with one usable door.
        for room in rooms:
            if room.doors:
                continue
            fallback_options = []
            for other in rooms:
                if other is room or not getattr(other, "polygon", None):
                    continue
                shared = shared_edge(room, other)
                if not shared:
                    continue
                overlap_segment = self._overlap_segment(shared[0], shared[1]) or shared[0]
                candidate = self._best_segment_on_edge(
                    overlap_segment,
                    room_a=room,
                    room_b=other,
                    needs_access=True,
                    connection_weight=0.0,
                )
                if candidate is None:
                    continue
                score = self._candidate_score(
                    candidate,
                    room_a=room,
                    room_b=other,
                    needs_access=True,
                    connection_weight=0.0,
                )
                if score is not None:
                    fallback_options.append((score, candidate, other))

            if not fallback_options:
                continue

            _, best_segment, other = max(fallback_options, key=lambda item: item[0])
            door = Door(room, other, self.min_door_width, best_segment, door_type="room_to_room")
            self.building.add_door(door)
            room.doors.append(door)
            other.doors.append(door)

    def create_door_segment(self, edge, *, room_a=None, room_b=None, needs_access=False, connection_weight=0.0):
        return self._best_segment_on_edge(
            edge,
            room_a=room_a,
            room_b=room_b,
            needs_access=needs_access,
            connection_weight=connection_weight,
        )
