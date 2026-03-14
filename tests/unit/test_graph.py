import pytest
from core.building import Building
from core.room import Room
from core.corridor import Corridor
from core.exit import Exit
from core.door import Door
from graph.door_graph_path import door_graph_travel_distance, get_room_travel_distances

@pytest.fixture
def graph_building():
    b = Building("Residential")
    b.total_area = 100
    r_a = Room("LivingRoom_1", "LivingRoom", 20.0)
    r_a.polygon = [(0, 0), (5, 0), (5, 4), (0, 4)]
    r_a.final_area = 20.0
    r_b = Room("Bedroom_1", "Bedroom", 16.0)
    r_b.polygon = [(5, 0), (9, 0), (9, 4), (5, 4)]
    r_b.final_area = 16.0
    b.rooms = [r_a, r_b]

    corr = Corridor("Corr1", [(0, 4), (9, 4), (9, 5.2), (0, 5.2)], 1.2, 9.0)
    b.corridors = [corr]
    b.exit = Exit(1.0)
    b.exit.segment = ((0, 4.6), (0, 5.6))

    d1 = Door(r_a, None, 0.9, ((2.5, 4.0), (3.4, 4.0)), door_type="room_to_circulation")
    d2 = Door(r_b, None, 0.9, ((7.0, 4.0), (7.9, 4.0)), door_type="room_to_circulation")
    b.doors = [d1, d2]
    return b

def test_door_graph_travel_distance(graph_building):
    travel = door_graph_travel_distance(graph_building)
    assert 0 < travel < 50

def test_get_room_travel_distances(graph_building):
    dists = get_room_travel_distances(graph_building)
    assert "LivingRoom_1" in dists
    assert "Bedroom_1" in dists
