class Building:
    def __init__(self, occupancy_type="Residential"):
        self.occupancy_type = occupancy_type
        self.rooms = []
        self.doors = []
        self.corridors = []
        self.exit = None
        self.total_area = 0
        self.occupant_load = 0

    def add_room(self, room):
        self.rooms.append(room)

    def add_door(self, door):
        self.doors.append(door)

    def add_corridor(self, corridor):
        self.corridors.append(corridor)

    def set_exit(self, exit_obj):
        self.exit = exit_obj