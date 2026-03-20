import json
import math

from nl_interface.service import process_user_request


def _room_count(current_spec, room_type):
    for room in current_spec.get("rooms", []):
        if room["type"] == room_type:
            return room["count"]
    return 0


def test_process_user_request_extracts_current_spec_and_missing_geometry():
    response = process_user_request(
        "Need 2 bedrooms, 1 kitchen, 1 bathroom on a 10 marla plot with a north entrance and minimize corridor."
    )

    assert response["current_spec"]["plot_type"] == "10Marla"
    assert response["current_spec"]["entrance_side"] == "North"
    assert _room_count(response["current_spec"], "Bedroom") == 2
    assert _room_count(response["current_spec"], "Kitchen") == 1
    assert response["current_spec"]["preferences"]["minimize_corridor"] is True
    assert response["backend_ready"] is False
    assert response["missing_fields"] == ["boundary_polygon"]


def test_validation_rejects_unsupported_room_mentions():
    response = process_user_request(
        "Need an office and one bedroom on a 5 marla plot with a south entrance."
    )

    assert _room_count(response["current_spec"], "Bedroom") == 1
    assert _room_count(response["current_spec"], "Office") == 0
    assert any("Unsupported room type 'Office'" in error for error in response["validation_errors"])


def test_extracts_typed_adjacency_and_normalizes_weights():
    response = process_user_request(
        "Need 2 bedrooms, 1 dining room, 1 kitchen and 1 garage on a 10 marla plot with an east entrance. "
        "Kitchen adjacent to dining room, bedroom far from garage. Make it feel open and minimize corridor."
    )

    adjacency = response["current_spec"]["preferences"]["adjacency"]
    assert ["Kitchen", "DiningRoom", "adjacent_to"] in adjacency
    assert ["Bedroom", "Garage", "far_from"] in adjacency

    weights = response["current_spec"]["weights"]
    assert math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-4)
    assert weights["privacy"] < (1.0 / 3.0)
    assert response["current_spec"]["preferences"]["minimize_corridor"] is True


def test_feasibility_warning_for_small_plot():
    response = process_user_request(
        "Need 4 bedrooms, 1 kitchen, 1 bathroom and 1 drawing room on a 5 marla plot with a west entrance."
    )

    assert response["feasibility_warnings"]
    assert any("5Marla" in warning for warning in response["feasibility_warnings"])


def test_build_backend_spec_routes_algorithmic_and_maps_drawing_room():
    response = process_user_request(
        "Need 2 bedrooms, 1 bathroom, 1 kitchen and 1 drawing room on a 10 marla plot with a north entrance.",
        resolution={"boundary_size": (15, 10)},
    )

    assert response["backend_ready"] is True
    assert response["backend_target"] == "algorithmic"
    room_types = [room["type"] for room in response["backend_spec"]["rooms"]]
    assert "LivingRoom" in room_types
    assert response["backend_spec"]["entrance_point"] == (7.5, 10.0)


def test_build_backend_spec_routes_learned_for_extended_vocabulary():
    response = process_user_request(
        "Need 2 bedrooms, 1 kitchen, 1 dining room and 1 store on a 10 marla plot with an east entrance.",
        resolution={"boundary_size": (15, 10)},
    )

    assert response["backend_ready"] is True
    assert response["backend_target"] == "learned"
    assert all("name" not in room for room in response["backend_spec"]["rooms"])
    room_types = [room["type"] for room in response["backend_spec"]["rooms"]]
    assert "DiningRoom" in room_types
    assert "Store" in room_types



def test_algorithmic_backend_rooms_are_strict_json_objects():
    response = process_user_request(
        "Need 2 bedrooms, 1 kitchen, 1 bathroom on a 10 marla plot with a north entrance.",
        resolution={"boundary_size": (15, 10)},
    )

    backend_spec = response["backend_spec"]
    assert backend_spec is not None

    serialized = json.dumps(backend_spec)
    assert serialized

    for room in backend_spec["rooms"]:
        assert set(room.keys()) == {"name", "type"}
        assert isinstance(room["name"], str)
        assert isinstance(room["type"], str)

