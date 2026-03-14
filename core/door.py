class Door:
    def __init__(self, room_a, room_b, width, segment, door_type="room_to_room"):
        """
        segment = ((x1,y1), (x2,y2))
        """
        self.room_a = room_a
        self.room_b = room_b
        self.width = width
        self.segment = segment
        self.door_type = door_type

    def connects(self, room):
        return room == self.room_a or room == self.room_b

    def other_side(self, room):
        if room == self.room_a:
            return self.room_b
        elif room == self.room_b:
            return self.room_a
        return None