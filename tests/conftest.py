import pytest
from core.building import Building
from core.room import Room

@pytest.fixture
def sample_building():
    b = Building("Residential")
    b.total_area = 100
    r1 = Room("Bedroom_1", "Bedroom", 12.0)
    r1.polygon = [(0, 0), (4, 0), (4, 3), (0, 3)]
    r1.final_area = 12.0
    r2 = Room("Kitchen_1", "Kitchen", 12.0)
    r2.polygon = [(4, 0), (8, 0), (8, 3), (4, 3)]
    r2.final_area = 12.0
    b.rooms = [r1, r2]
    return b

@pytest.fixture
def empty_spec():
    return {
        "occupancy": "Residential",
        "total_area": 100,
        "area_unit": "sq.m",
        "rooms": []
    }

@pytest.fixture
def standard_spec():
    return {
        "occupancy": "Residential",
        "total_area": 100,
        "area_unit": "sq.m",
        "rooms": [
            {"name": "Master", "type": "Bedroom", "area": 15},
            {"name": "Kitchen", "type": "Kitchen", "area": 10}
        ]
    }
