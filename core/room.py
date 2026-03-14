class Room:
    def __init__(self, name, room_type, requested_area):
        self.name = name
        self.room_type = room_type
        self.requested_area = requested_area

        self.final_area = None
        self.min_area = None
        self.min_width = None
        self.min_height = None

        self.polygon = None  # Orthogonal polygon (list of vertices)
        self.doors = []

    def set_regulation_constraints(self, min_area, min_width, min_height):
        self.min_area = min_area
        self.min_width = min_width
        self.min_height = min_height

    def enforce_minimums(self):
        if self.requested_area < self.min_area:
            self.final_area = self.min_area
            return True  # modified
        else:
            self.final_area = self.requested_area
            return False

    def __repr__(self):
        return f"{self.name} ({self.room_type}) - Area: {self.final_area}"