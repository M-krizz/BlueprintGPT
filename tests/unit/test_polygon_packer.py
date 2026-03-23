from shapely.geometry import Polygon

from core.room import Room
from geometry.polygon_packer import recursive_pack


def _polygon_area(points):
    if not points or len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def test_recursive_pack_single_room_trims_oversized_region_to_target_area():
    room = Room("Bathroom_1", "Bathroom", 6.5)
    room.final_area = 6.5

    recursive_pack(
        Polygon([(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]),
        [{"room": room, "weight": 6.5}],
        entrance_pt=(5.0, 0.0),
    )

    actual_area = _polygon_area(room.polygon)
    assert actual_area < 20.0
    assert abs(actual_area - 6.5) / 6.5 < 0.2


def test_recursive_pack_preserves_group_area_when_hints_flip_split_side():
    living = Room("LivingRoom_1", "LivingRoom", 60.0)
    bedroom = Room("Bedroom_1", "Bedroom", 40.0)
    living.final_area = 60.0
    bedroom.final_area = 40.0

    recursive_pack(
        Polygon([(0.0, 0.0), (0.0, 10.0), (20.0, 10.0), (20.0, 0.0)]),
        [
            {"room": living, "weight": 60.0},
            {"room": bedroom, "weight": 40.0},
        ],
        learned_hints={
            "LivingRoom_1": (0.8, 0.5),
            "Bedroom_1": (0.2, 0.5),
        },
        fit_single_room=False,
    )

    living_area = _polygon_area(living.polygon)
    bedroom_area = _polygon_area(bedroom.polygon)

    assert living_area > bedroom_area
    assert abs((living_area / bedroom_area) - 1.5) < 0.15
