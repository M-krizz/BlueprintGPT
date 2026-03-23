from __future__ import annotations

import asyncio

from api import server


def test_conversation_message_hydrates_client_spec_and_history(monkeypatch):
    server.conversation_manager.sessions.clear()

    captured = {}

    def fake_process_nl_message(message, context, history):
        captured["context"] = context
        captured["history"] = history
        return {
            "intent": server.INTENT_CONVERSATION,
            "intent_confidence": 0.9,
            "response": "Continuing from the same chat.",
            "should_generate": False,
            "spec": None,
        }

    monkeypatch.setattr(server, "process_nl_message", fake_process_nl_message)

    body = server.ConversationMessageRequest(
        message="continue this",
        session_id=None,
        history=[
            {"role": "user", "content": "Need a 2BHK"},
            {"role": "assistant", "content": "I can help with that."},
        ],
        client_spec={
            "rooms": [
                {"type": "Bedroom", "count": 2},
                {"type": "Bathroom", "count": 1},
                {"type": "Kitchen", "count": 1},
                {"type": "LivingRoom", "count": 1},
            ],
            "plot_type": "Custom",
            "entrance_side": "North",
            "preferences": {"adjacency": [], "privacy": {}, "minimize_corridor": False},
            "weights": {},
        },
        generate=False,
    )

    response = asyncio.run(server.conversation_message(body))

    assert response.current_spec["rooms"][0]["type"] == "Bedroom"
    assert captured["context"]["spec"]["rooms"][0]["type"] == "Bedroom"
    assert captured["context"]["semantic_spec"]["layout_type"] == "2BHK"
    assert response.program_summary is not None
    assert response.zoning_summary is not None
    assert any(item.get("content") == "Need a 2BHK" for item in captured["history"])

    server.conversation_manager.sessions.clear()


def test_conversation_message_context_includes_latest_design(monkeypatch):
    server.conversation_manager.sessions.clear()
    session = server.conversation_manager.create_session()
    session.update_spec(
        {
            "rooms": [
                {"type": "Bedroom", "count": 3},
                {"type": "Bathroom", "count": 2},
                {"type": "Kitchen", "count": 1},
                {"type": "LivingRoom", "count": 1},
            ],
            "preferences": {"adjacency": [], "privacy": {}},
            "layout_type": "3BHK",
            "plot_type": "Custom",
            "entrance_side": "North",
        }
    )
    session.add_design(
        {
            "artifact_paths": {"svg": "outputs/test.svg", "report": "outputs/test.json"},
            "artifact_urls": {"svg": "/outputs/test.svg"},
            "design_score": 0.9,
            "metrics": {"fully_connected": True},
            "generated_rooms": {"Bedroom": 3, "Bathroom": 2, "Kitchen": 1, "LivingRoom": 1},
            "violations": [],
            "report_status": "COMPLIANT",
            "winning_source": "planner_direct_rescue",
            "explanation": {"summary": "ok"},
        },
        rank=1,
    )

    captured = {}

    def fake_process_nl_message(message, context, history):
        captured["context"] = context
        return {
            "intent": server.INTENT_CONVERSATION,
            "intent_confidence": 0.9,
            "response": "continuing",
            "should_generate": False,
            "spec": None,
        }

    monkeypatch.setattr(server, "process_nl_message", fake_process_nl_message)

    body = server.ConversationMessageRequest(
        message="can i make some changes in this?",
        session_id=session.session_id,
        generate=False,
    )

    response = asyncio.run(server.conversation_message(body))

    assert "keep iterating on the current design" in response.assistant_text.lower()
    assert captured["context"]["latest_design"]["engine"] == "planner_direct_rescue"
    assert captured["context"]["current_rooms"][0]["type"] == "Bedroom"
    assert response.latest_design_summary is not None
    assert response.latest_design_summary["engine"] == "planner_direct_rescue"
    assert response.program_summary is not None

    server.conversation_manager.sessions.clear()
