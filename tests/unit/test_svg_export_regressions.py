from core.building import Building
from core.exit import Exit
from core.room import Room
from visualization.export_svg_blueprint import render_svg_blueprint


def test_render_svg_blueprint_draws_single_entrance_group():
    building = Building(occupancy_type="Residential")
    room = Room("LivingRoom_1", "LivingRoom", 12.0)
    room.final_area = 12.0
    room.polygon = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    building.add_room(room)

    exit_obj = Exit(width=1.0)
    exit_obj.segment = ((1.5, 0.0), (2.5, 0.0))
    building.set_exit(exit_obj)

    svg = render_svg_blueprint(
        building,
        boundary_polygon=[(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)],
        entrance_point=(1.5, 0.0),
    )

    assert svg.count('id="entrance"') == 1
