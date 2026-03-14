import pytest
from core.building import Building
from core.room import Room
from core.door import Door
from visualization.export_svg_blueprint import render_svg_blueprint

@pytest.fixture
def svg_building():
    b = Building("Residential")
    b.total_area = 100
    r_a = Room("LivingRoom_1", "LivingRoom", 20.0)
    r_a.polygon = [(0, 0), (5, 0), (5, 4), (0, 4)]
    r_b = Room("Bedroom_1", "Bedroom", 16.0)
    r_b.polygon = [(5, 0), (9, 0), (9, 4), (5, 4)]
    b.rooms = [r_a, r_b]
    
    d1 = Door(r_a, None, 0.9, ((2.5, 4.0), (3.4, 4.0)), door_type="room_to_circulation")
    b.doors = [d1]
    return b

def test_svg_rendering(svg_building):
    boundary = [(0, 0), (9, 0), (9, 5.2), (0, 5.2)]
    svg = render_svg_blueprint(svg_building, boundary, entrance_point=(0, 4.6), title="Test Plan")
    assert len(svg) > 0
    assert 'id="walls"' in svg
    assert 'id="boundary-dims"' in svg
    assert 'id="grid"' in svg
