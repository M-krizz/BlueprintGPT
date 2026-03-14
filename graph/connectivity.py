from collections import deque

def is_fully_connected(building):
    rooms = building.rooms
    if not rooms:
        return True

    visited = set()
    room_neighbors = {room: set() for room in rooms}
    corridor_access_rooms = []

    for door in building.doors:
        if door.room_a in room_neighbors and door.room_b in room_neighbors:
            room_neighbors[door.room_a].add(door.room_b)
            room_neighbors[door.room_b].add(door.room_a)
        elif door.door_type == "room_to_circulation" and door.room_a in room_neighbors:
            corridor_access_rooms.append(door.room_a)

    if len(corridor_access_rooms) > 1:
        for room in corridor_access_rooms:
            room_neighbors[room].update(r for r in corridor_access_rooms if r is not room)

    queue = deque([rooms[0]])

    while queue:
        current = queue.popleft()
        visited.add(current)

        for other in room_neighbors[current]:
            if other not in visited:
                queue.append(other)

    return len(visited) == len(rooms)