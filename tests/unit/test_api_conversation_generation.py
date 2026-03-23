from __future__ import annotations

import asyncio

from api import server
from nl_interface import adapter


def test_conversation_generation_fallback_uses_latest_nl_response_spec(monkeypatch):
    server.conversation_manager.sessions.clear()

    captured_room_counts: list[int] = []

    monkeypatch.setattr(
        server,
        "process_nl_message",
        lambda *args, **kwargs: {
            "intent": server.INTENT_DESIGN,
            "intent_confidence": 0.99,
            "should_generate": True,
            "response": "Generating your floor plan designs...",
            "spec": {
                "layout_type": "2BHK",
                "rooms": [
                    {"type": "Bedroom", "count": 2},
                    {"type": "Bathroom", "count": 2},
                    {"type": "Kitchen", "count": 1},
                    {"type": "LivingRoom", "count": 1},
                ],
                "adjacency": [
                    {"source": "Bathroom", "target": "Bedroom", "relation": "near_to"},
                    {"source": "Kitchen", "target": "LivingRoom", "relation": "near_to"},
                ],
                "entrance_side": "South",
                "plot_type": "Custom",
                "style_hints": [],
                "auto_dimensions": {"width_m": 12.4, "height_m": 8.9},
            },
        },
    )

    monkeypatch.setattr(
        server,
        "process_user_request",
        lambda *args, **kwargs: {
            "assistant_text": "ready",
            "current_spec": {
                "layout_type": "2BHK",
                "plot_type": "Custom",
                "entrance_side": "South",
                "rooms": [
                    {"type": "Bedroom", "count": 2},
                    {"type": "Bathroom", "count": 2},
                    {"type": "Kitchen", "count": 1},
                    {"type": "LivingRoom", "count": 1},
                ],
                "preferences": {
                    "adjacency": [
                        ("Bathroom", "Bedroom", "near_to"),
                        ("Kitchen", "LivingRoom", "near_to"),
                    ],
                    "privacy": {},
                    "minimize_corridor": False,
                },
                "weights": {},
            },
            "missing_fields": [],
            "validation_errors": ["force_autogen_path"],
            "backend_ready": False,
            "backend_target": "planner",
            "backend_spec": None,
            "backend_translation_warnings": [],
        },
    )

    def fake_build_backend_spec(current_spec, resolution):
        captured_room_counts.append(sum(int(room.get("count", 1)) for room in current_spec.get("rooms", [])))
        return (
            {
                "rooms": [
                    {"name": "Bedroom_1", "type": "Bedroom"},
                    {"name": "Bedroom_2", "type": "Bedroom"},
                    {"name": "Bathroom_1", "type": "Bathroom"},
                    {"name": "Bathroom_2", "type": "Bathroom"},
                    {"name": "Kitchen_1", "type": "Kitchen"},
                    {"name": "LivingRoom_1", "type": "LivingRoom"},
                ],
                "boundary_polygon": resolution["boundary_polygon"],
                "entrance_point": resolution["entrance_point"],
            },
            [],
        )

    monkeypatch.setattr(adapter, "build_backend_spec", fake_build_backend_spec)
    monkeypatch.setattr(
        server,
        "execute_response",
        lambda nl_response, output_dir, output_prefix: {
            "status": "completed",
            "backend_target": "planner",
            "report_status": "COMPLIANT",
            "artifact_paths": {"svg": "outputs/test.svg", "report": "outputs/test.json"},
            "artifact_urls": {"svg": "/outputs/test.svg"},
            "generated_rooms": {"Bedroom": 2, "Bathroom": 2, "Kitchen": 1, "LivingRoom": 1},
            "metrics": {"fully_connected": True},
            "violations": [],
            "design_score": 1.0,
            "explanation": {"summary": "ok"},
        },
    )
    monkeypatch.setattr(server, "explain_ranked_designs", lambda designs, spec: designs)
    monkeypatch.setattr(server, "generate_comparison_explanation", lambda designs: "comparison")

    body = server.ConversationMessageRequest(
        message="Create a floorplan for 2bhk apartment which has 2200 sqm..",
        boundary=server.Boundary(width=13.0, height=6.5),
        entrance_point=[6.5, 0.0],
        generate=True,
    )

    response = asyncio.run(server.conversation_message(body))

    assert captured_room_counts == [6]
    assert response.spec_complete is True
    assert response.designs is not None
    assert response.designs[0]["generated_rooms"]["Bedroom"] == 2
    assert response.program_summary is not None
    assert "2BHK" in response.program_summary
    assert response.zoning_summary is not None
    assert response.assumptions_used
    assert response.expert_diagnostics is not None
    assert response.expert_diagnostics["room_program"]["layout_type"] == "2BHK"
    assert response.latest_design_summary is not None
    assert response.latest_design_summary["engine"] == "planner"

    server.conversation_manager.sessions.clear()


def test_conversation_generation_ignores_frontend_defaults_without_manual_boundary(monkeypatch):
    server.conversation_manager.sessions.clear()

    captured_boundary_sizes: list[tuple[float, float]] = []

    monkeypatch.setattr(
        server,
        "process_nl_message",
        lambda *args, **kwargs: {
            "intent": server.INTENT_DESIGN,
            "intent_confidence": 0.99,
            "should_generate": True,
            "response": "Generating your floor plan designs...",
            "spec": {
                "layout_type": "3BHK",
                "rooms": [
                    {"type": "Bedroom", "count": 3},
                    {"type": "Bathroom", "count": 2},
                    {"type": "Kitchen", "count": 1},
                    {"type": "LivingRoom", "count": 1},
                ],
                "adjacency": [
                    {"source": "Bathroom", "target": "Bedroom", "relation": "near_to"},
                    {"source": "Kitchen", "target": "LivingRoom", "relation": "near_to"},
                ],
                "entrance_side": "North",
                "plot_type": "Custom",
                "style_hints": [],
                "auto_dimensions": {"width_m": 12.4, "height_m": 8.9},
            },
        },
    )

    monkeypatch.setattr(
        server,
        "process_user_request",
        lambda *args, **kwargs: {
            "assistant_text": "ready",
            "current_spec": {
                "layout_type": "3BHK",
                "plot_type": "Custom",
                "entrance_side": "North",
                "rooms": [
                    {"type": "Bedroom", "count": 3},
                    {"type": "Bathroom", "count": 2},
                    {"type": "Kitchen", "count": 1},
                    {"type": "LivingRoom", "count": 1},
                ],
                "preferences": {
                    "adjacency": [
                        ("Bathroom", "Bedroom", "near_to"),
                        ("Kitchen", "LivingRoom", "near_to"),
                    ],
                    "privacy": {},
                    "minimize_corridor": False,
                },
                "weights": {},
            },
            "missing_fields": [],
            "validation_errors": ["force_autogen_path"],
            "backend_ready": False,
            "backend_target": "algorithmic",
            "backend_spec": None,
            "backend_translation_warnings": [],
        },
    )

    def fake_build_backend_spec(current_spec, resolution):
        captured_boundary_sizes.append(tuple(resolution["boundary_size"]))
        return (
            {
                "rooms": [{"name": "Bedroom_1", "type": "Bedroom"}],
                "boundary_polygon": resolution["boundary_polygon"],
                "entrance_point": resolution["entrance_point"],
            },
            [],
        )

    monkeypatch.setattr(adapter, "build_backend_spec", fake_build_backend_spec)
    monkeypatch.setattr(
        server,
        "execute_response",
        lambda nl_response, output_dir, output_prefix: {
            "status": "completed",
            "backend_target": "algorithmic",
            "report_status": "COMPLIANT",
            "artifact_paths": {"svg": "outputs/test.svg", "report": "outputs/test.json"},
            "artifact_urls": {"svg": "/outputs/test.svg"},
            "generated_rooms": {"Bedroom": 3, "Bathroom": 2, "Kitchen": 1, "LivingRoom": 1},
            "metrics": {"fully_connected": True},
            "violations": [],
            "design_score": 1.0,
            "explanation": {"summary": "ok"},
        },
    )
    monkeypatch.setattr(server, "explain_ranked_designs", lambda designs, spec: designs)
    monkeypatch.setattr(server, "generate_comparison_explanation", lambda designs: "comparison")

    body = server.ConversationMessageRequest(
        message="Design a 3BHK apartment with north entrance",
        boundary=server.Boundary(width=10.0, height=10.0),
        entrance_point=[5.0, 0.0],
        use_manual_boundary=False,
        generate=True,
    )

    response = asyncio.run(server.conversation_message(body))

    assert response.spec_complete is True
    assert captured_boundary_sizes == [(12.4, 8.9)]
    assert response.program_summary is not None
    assert "3BHK" in response.program_summary
    assert response.zoning_summary is not None
    assert response.current_spec["zoning_plan"]["layout_pattern"] == "zonal_split"

    server.conversation_manager.sessions.clear()



def test_conversation_message_correction_regenerates_latest_design(monkeypatch):
    server.conversation_manager.sessions.clear()
    session = server.conversation_manager.create_session()
    session.current_spec = {
        "layout_type": "3BHK",
        "plot_type": "Custom",
        "entrance_side": "North",
        "rooms": [
            {"type": "Bedroom", "count": 3},
            {"type": "Bathroom", "count": 2},
            {"type": "Kitchen", "count": 1},
            {"type": "LivingRoom", "count": 1},
        ],
        "preferences": {"adjacency": [], "privacy": {}, "minimize_corridor": False},
        "weights": {},
    }
    session.resolution = {
        "boundary_size": (12.4, 8.9),
        "boundary_polygon": [(0.0, 0.0), (12.4, 0.0), (12.4, 8.9), (0.0, 8.9)],
        "entrance_point": (6.2, 0.0),
        "area_unit": "sq.m",
    }
    session.add_design(
        {
            "artifact_paths": {"svg": "outputs/original.svg", "report": "outputs/original.json"},
            "artifact_urls": {"svg": "/outputs/original.svg"},
            "design_score": 0.9,
            "metrics": {"fully_connected": True},
            "generated_rooms": {"Bedroom": 3, "Bathroom": 2, "Kitchen": 1, "LivingRoom": 1},
            "violations": [],
            "report_status": "COMPLIANT",
            "winning_source": "algorithmic",
            "explanation": {"summary": "ok"},
        },
        rank=1,
    )

    monkeypatch.setattr(
        server,
        "process_nl_message",
        lambda *args, **kwargs: {
            "intent": server.INTENT_CORRECTION,
            "intent_confidence": 0.95,
            "response": "I can apply that change.",
            "should_generate": False,
            "spec": None,
        },
    )
    monkeypatch.setattr(
        server,
        "handle_correction_request",
        lambda user_request, design_index, session: (
            {
                "changes": [{"type": "move_room", "room": "Kitchen_1", "direction": "east"}],
                "modified_spec": {
                    **session.current_spec,
                    "room_position_preferences": {"Kitchen_1": {"direction": "east", "amount": "1m"}},
                },
                "needs_regeneration": True,
            },
            None,
        ),
    )
    monkeypatch.setattr(
        server,
        "process_user_request",
        lambda *args, **kwargs: {
            "assistant_text": "ready",
            "current_spec": kwargs.get("current_spec") or args[1],
            "missing_fields": [],
            "validation_errors": [],
            "backend_ready": True,
            "backend_target": "algorithmic",
            "backend_spec": {"rooms": [{"name": "Kitchen_1", "type": "Kitchen"}]},
            "backend_translation_warnings": [],
        },
    )
    monkeypatch.setattr(
        server,
        "execute_response",
        lambda nl_response, output_dir, output_prefix: {
            "status": "completed",
            "backend_target": "algorithmic",
            "report_status": "COMPLIANT",
            "artifact_paths": {"svg": "outputs/edited.svg", "report": "outputs/edited.json"},
            "artifact_urls": {"svg": "/outputs/edited.svg"},
            "generated_rooms": {"Bedroom": 3, "Bathroom": 2, "Kitchen": 1, "LivingRoom": 1},
            "metrics": {"fully_connected": True, "adjacency_satisfaction": 0.8, "alignment_score": 0.8},
            "violations": [],
            "design_score": 1.0,
            "winning_source": "algorithmic",
            "explanation": {"summary": "updated"},
        },
    )
    monkeypatch.setattr(server, "explain_ranked_designs", lambda designs, spec: designs)
    monkeypatch.setattr(server, "generate_comparison_explanation", lambda designs: "comparison")

    body = server.ConversationMessageRequest(
        message="Move the kitchen to the east side in the previous layout",
        session_id=session.session_id,
        generate=True,
    )

    response = asyncio.run(server.conversation_message(body))

    assert response.spec_complete is True
    assert response.designs is not None
    assert response.current_spec["room_position_preferences"]["Kitchen_1"]["direction"] == "east"
    assert "updated the current layout request" in response.assistant_text.lower()
    assert response.latest_design_summary is not None
    assert response.latest_design_summary["engine"] == "algorithmic"
    assert server.conversation_manager.get_session(session.session_id).selected_design_index == 1

    server.conversation_manager.sessions.clear()



def test_conversation_generation_drops_prompt_scoped_boundary_for_fresh_design(monkeypatch):
    server.conversation_manager.sessions.clear()
    session = server.conversation_manager.create_session()
    session.resolution = {
        "boundary_size": (10.0, 150.0),
        "boundary_polygon": [(0.0, 0.0), (10.0, 0.0), (10.0, 150.0), (0.0, 150.0)],
        "boundary_source": "nl_prompt",
        "boundary_role": "site",
        "site_boundary_size": (10.0, 150.0),
        "area_unit": "sq.m",
    }

    captured_boundary_sizes = []

    monkeypatch.setattr(
        server,
        "process_nl_message",
        lambda *args, **kwargs: {
            "intent": server.INTENT_DESIGN,
            "intent_confidence": 0.99,
            "should_generate": True,
            "response": "Generating your floor plan designs...",
            "spec": {
                "layout_type": "3BHK",
                "rooms": [
                    {"type": "Bedroom", "count": 3},
                    {"type": "Bathroom", "count": 2},
                    {"type": "Kitchen", "count": 1},
                    {"type": "LivingRoom", "count": 1},
                ],
                "entrance_side": "North",
                "plot_type": "Custom",
                "style_hints": [],
                "auto_dimensions": {"width_m": 12.4, "height_m": 8.9},
            },
        },
    )

    monkeypatch.setattr(
        server,
        "process_user_request",
        lambda *args, **kwargs: {
            "assistant_text": "ready",
            "current_spec": kwargs.get("current_spec") or args[1],
            "missing_fields": [],
            "validation_errors": ["force_autogen_path"],
            "backend_ready": False,
            "backend_target": "algorithmic",
            "backend_spec": None,
            "backend_translation_warnings": [],
        },
    )

    def fake_build_backend_spec(current_spec, resolution):
        captured_boundary_sizes.append(tuple(resolution["boundary_size"]))
        return (
            {
                "rooms": [{"name": "Bedroom_1", "type": "Bedroom"}],
                "boundary_polygon": resolution["boundary_polygon"],
                "entrance_point": resolution["entrance_point"],
            },
            [],
        )

    monkeypatch.setattr(adapter, "build_backend_spec", fake_build_backend_spec)
    monkeypatch.setattr(
        server,
        "execute_response",
        lambda nl_response, output_dir, output_prefix: {
            "status": "completed",
            "backend_target": "algorithmic",
            "report_status": "COMPLIANT",
            "artifact_paths": {"svg": "outputs/test.svg", "report": "outputs/test.json"},
            "artifact_urls": {"svg": "/outputs/test.svg"},
            "generated_rooms": {"Bedroom": 3, "Bathroom": 2, "Kitchen": 1, "LivingRoom": 1},
            "metrics": {"fully_connected": True},
            "violations": [],
            "design_score": 1.0,
            "explanation": {"summary": "ok"},
        },
    )
    monkeypatch.setattr(server, "explain_ranked_designs", lambda designs, spec: designs)
    monkeypatch.setattr(server, "generate_comparison_explanation", lambda designs: "comparison")

    body = server.ConversationMessageRequest(
        message="3bhk plot",
        session_id=session.session_id,
        generate=True,
    )

    response = asyncio.run(server.conversation_message(body))

    assert response.spec_complete is True
    assert captured_boundary_sizes == [(12.4, 8.9)]
    server.conversation_manager.sessions.clear()
