import math

class Allocator:

    def __init__(self, building):
        self.building = building

    def compute_dimensions(self, room):
        area = room.final_area
        min_width = room.min_width

        width = max(min_width, math.sqrt(area))
        length = area / width

        return round(width, 2), round(length, 2)

    def allocate(self):
        rooms = sorted(self.building.rooms, key=lambda r: r.final_area, reverse=True)

        total_area = self.building.total_area
        threshold_width = math.sqrt(total_area)

        current_x = 0
        current_y = 0
        row_height = 0

        for room in rooms:
            width, height = self.compute_dimensions(room)

            # Move to next row if threshold exceeded
            if current_x + width > threshold_width:
                current_x = 0
                current_y += row_height
                row_height = 0

            # Assign polygon
            room.polygon = [
                (round(current_x, 2), round(current_y, 2)),
                (round(current_x + width, 2), round(current_y, 2)),
                (round(current_x + width, 2), round(current_y + height, 2)),
                (round(current_x, 2), round(current_y + height, 2))
            ]

            current_x += width
            row_height = max(row_height, height)

        return self.compute_bounding_box()

    def compute_bounding_box(self):
        max_x = 0
        max_y = 0

        for room in self.building.rooms:
            for x, y in room.polygon:
                max_x = max(max_x, x)
                max_y = max(max_y, y)

        return round(max_x, 2), round(max_y, 2)