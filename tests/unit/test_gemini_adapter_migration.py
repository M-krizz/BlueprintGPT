from __future__ import annotations

from nl_interface import gemini_adapter


def test_resolve_gemini_model_remaps_legacy_models():
    assert gemini_adapter._resolve_gemini_model("gemini-1.5-flash") == "gemini-2.5-flash"
    assert gemini_adapter._resolve_gemini_model("gemini-1.5-pro") == "gemini-2.5-pro"
    assert gemini_adapter._resolve_gemini_model("gemini-2.5-flash") == "gemini-2.5-flash"



def test_extract_spec_from_nl_uses_generation_helper(monkeypatch):
    monkeypatch.setattr(gemini_adapter, "is_available", lambda: True)
    monkeypatch.setattr(
        gemini_adapter,
        "_generate_text",
        lambda prompt, response_mime_type=None: (
            '{"rooms":[{"type":"Bedroom","count":2},{"type":"Bathroom","count":1},'
            '{"type":"Kitchen","count":1},{"type":"LivingRoom","count":1}],'
            '"adjacency":[{"source":"Kitchen","target":"LivingRoom","relation":"near_to"}],'
            '"entrance_side":"South","plot_type":"custom","style_hints":[],"intent":"new_design"}'
        ),
    )
    monkeypatch.setattr(gemini_adapter, "_apply_constraint_enhancement", lambda spec: spec)

    result = gemini_adapter.extract_spec_from_nl("Create a 2BHK house")

    assert len(result["rooms"]) == 4
    assert result["rooms"][0]["type"] == "Bedroom"
    assert result["adjacency"][0]["target"] == "LivingRoom"



def test_chat_response_uses_generation_helper(monkeypatch):
    monkeypatch.setattr(gemini_adapter, "is_available", lambda: True)
    monkeypatch.setattr(gemini_adapter, "_generate_text", lambda prompt, response_mime_type=None: "Hello from Gemini")

    result = gemini_adapter.chat_response("What can you do?", {"state": "initial", "num_designs": 0, "spec": {}}, [])

    assert result == "Hello from Gemini"



def test_generate_text_disables_gemini_after_auth_failure(monkeypatch):
    class FakeModels:
        def generate_content(self, **kwargs):
            raise Exception("403 PERMISSION_DENIED. Your API key was reported as leaked.")

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(gemini_adapter, "_GENAI_SDK", "google.genai")
    monkeypatch.setattr(gemini_adapter, "_gemini_client", FakeClient())
    monkeypatch.setattr(gemini_adapter, "_gemini_ready", True)
    monkeypatch.setattr(gemini_adapter, "_gemini_disabled_reason", None)
    monkeypatch.setattr(gemini_adapter, "GEMINI_ENABLED", True)

    text = gemini_adapter._generate_text("hello")

    assert text is None
    assert gemini_adapter.is_available() is False
    assert "permission_denied" in gemini_adapter._gemini_disabled_reason.lower()


def test_fallback_chat_uses_latest_design_for_change_requests():
    reply = gemini_adapter._fallback_chat(
        "can i make some changes in this?",
        {
            "state": "generated",
            "num_designs": 1,
            "spec": {},
            "latest_design": {
                "rooms": [
                    {"type": "Bedroom", "count": 3},
                    {"type": "Bathroom", "count": 2},
                    {"type": "Kitchen", "count": 1},
                    {"type": "LivingRoom", "count": 1},
                ],
                "engine": "planner_direct_rescue",
                "report_status": "COMPLIANT",
            },
        },
        [],
    )

    assert "Yes. We can keep working on this 3 Bedroom, 2 Bathroom, 1 Kitchen, 1 LivingRoom" in reply
    assert "planner_direct_rescue" in reply



def test_fallback_chat_explains_overlap_without_generic_template():
    reply = gemini_adapter._fallback_chat(
        "the rooms get overlaped in the previous layout you gave what shall i do for it?",
        {
            "state": "generated",
            "num_designs": 1,
            "spec": {},
            "latest_design": {
                "rooms": [{"type": "Bedroom", "count": 3}],
                "engine": "planner_direct_rescue",
                "report_status": "NON_COMPLIANT",
            },
        },
        [],
    )

    assert "geometry was not clean enough" in reply
    assert "repair and regenerate" in reply
    assert "NON_COMPLIANT" in reply



def test_fallback_chat_explains_encoder_decoder_pipeline():
    reply = gemini_adapter._fallback_chat(
        "where is the encoder decoder architecture in this system?",
        {"state": "generated", "num_designs": 1, "spec": {}},
        [],
    )

    assert "natural-language side acts like an encoder" in reply
    assert "geometry stage acts like a decoder" in reply
    assert "verification and repair" in reply



def test_fallback_parse_correction_understands_named_room_adjacency():
    parsed = gemini_adapter._fallback_parse_correction(
        "bedroom1 and bedroom2 should be adjacent",
        0,
        [
            {"name": "Bedroom_1", "type": "Bedroom"},
            {"name": "Bedroom_2", "type": "Bedroom"},
            {"name": "Kitchen_1", "type": "Kitchen"},
        ],
    )

    assert parsed["understood"] is True
    assert parsed["changes"][0] == {
        "type": "change_adjacency",
        "room_a": "Bedroom_1",
        "room_b": "Bedroom_2",
        "relation": "adjacent_to",
    }



def test_classify_intent_treats_room_relationship_edit_as_correction_when_design_exists():
    result = gemini_adapter.classify_intent(
        "i want living room and bedroom to be adjacent",
        {"num_designs": 1, "state": "generated"},
        [],
    )

    assert result["intent"] == gemini_adapter.INTENT_CORRECTION



def test_fallback_parse_correction_understands_to_be_adjacent_phrase():
    parsed = gemini_adapter._fallback_parse_correction(
        "i want living room and bedroom to be adjacent",
        0,
        [
            {"name": "LivingRoom_1", "type": "LivingRoom"},
            {"name": "Bedroom_1", "type": "Bedroom"},
        ],
    )

    assert parsed["understood"] is True
    assert parsed["changes"][0] == {
        "type": "change_adjacency",
        "room_a": "LivingRoom_1",
        "room_b": "Bedroom_1",
        "relation": "adjacent_to",
    }



def test_fallback_intent_classify_treats_relationship_edit_as_correction():
    result = gemini_adapter._fallback_intent_classify(
        "i want kitchen and bathroom to be adjacent",
        has_designs=True,
    )

    assert result["intent"] == gemini_adapter.INTENT_CORRECTION
