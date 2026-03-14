import pytest
from core.building import Building
from core.room import Room
from gui.layout_editor import _quick_validate

def test_quick_validate():
    b3 = Building("Residential")
    b3.total_area = 100
    r_small = Room("Bedroom_tiny", "Bedroom", 1.0)
    r_small.polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]  # 1 m2 << 9.5 min
    r_small.final_area = 1.0
    b3.rooms = [r_small]
    
    issues = _quick_validate(b3, "ontology/regulation_data.json")
    assert "Bedroom_tiny" in issues
