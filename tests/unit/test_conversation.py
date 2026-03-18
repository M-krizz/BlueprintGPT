"""
test_conversation.py – Unit tests for conversation management.

Tests the conversation session management and correction handling.
These tests don't require heavy dependencies like shapely or torch.
"""
from __future__ import annotations

import pytest
import time
from unittest.mock import patch, MagicMock

from nl_interface.conversation import (
    ConversationSession,
    ConversationManager,
    Message,
    GeneratedDesign,
)

from nl_interface.correction_handler import (
    apply_corrections_to_spec,
    translate_corrections_to_geometry,
    validate_correction_feasibility,
)


# ─────────────────────────────────────────────────────────────────────────────
#  1. Message Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMessage:

    def test_message_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp > 0
        assert msg.metadata == {}

    def test_message_to_dict(self):
        msg = Message(role="bot", content="Hi there", metadata={"key": "value"})
        d = msg.to_dict()
        assert d["role"] == "bot"
        assert d["content"] == "Hi there"
        assert d["metadata"] == {"key": "value"}
        assert "timestamp" in d


# ─────────────────────────────────────────────────────────────────────────────
#  2. ConversationSession Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConversationSession:

    def test_session_creation(self):
        session = ConversationSession()
        assert session.session_id is not None
        assert len(session.session_id) == 32  # UUID hex
        assert session.state == "initial"
        assert session.messages == []
        assert session.designs == []

    def test_session_with_custom_id(self):
        session = ConversationSession(session_id="custom123")
        assert session.session_id == "custom123"

    def test_add_message(self):
        session = ConversationSession()
        msg = session.add_message("user", "Test message")
        assert len(session.messages) == 1
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "Test message"

    def test_get_history(self):
        session = ConversationSession()
        session.add_message("user", "First")
        session.add_message("bot", "Second")
        session.add_message("user", "Third")

        history = session.get_history()
        assert len(history) == 3
        assert history[0]["content"] == "First"

        limited = session.get_history(limit=2)
        assert len(limited) == 2
        assert limited[0]["content"] == "Second"

    def test_update_spec_rooms(self):
        session = ConversationSession()
        session.update_spec({
            "rooms": [
                {"type": "Bedroom", "count": 2},
                {"type": "Kitchen", "count": 1},
            ]
        })

        assert session.state == "specifying"
        assert len(session.current_spec["rooms"]) == 2

        # Add more rooms – replacement semantics: new values override existing
        session.update_spec({
            "rooms": [
                {"type": "Bedroom", "count": 1},  # Replaces existing 2 with 1
                {"type": "Bathroom", "count": 1},  # New room type
            ]
        })

        room_counts = {r["type"]: r["count"] for r in session.current_spec["rooms"]}
        assert room_counts["Bedroom"] == 1  # Replaced, not added
        assert room_counts["Kitchen"] == 1  # Unchanged
        assert room_counts["Bathroom"] == 1

    def test_update_spec_adjacency(self):
        session = ConversationSession()
        session.update_spec({
            "adjacency": [
                {"source": "Kitchen", "target": "DiningRoom", "relation": "near_to"}
            ]
        })

        adj = session.current_spec["preferences"]["adjacency"]
        assert len(adj) == 1
        assert adj[0] == ("Kitchen", "DiningRoom", "near_to")

        # Add duplicate - should be ignored
        session.update_spec({
            "adjacency": [
                {"source": "Kitchen", "target": "DiningRoom", "relation": "near_to"}
            ]
        })
        assert len(session.current_spec["preferences"]["adjacency"]) == 1

    def test_update_spec_style_hints(self):
        session = ConversationSession()
        session.update_spec({
            "style_hints": ["compact layout", "privacy focused"]
        })

        weights = session.current_spec.get("weights", {})
        # "compact" sets compactness=0.6, "privacy" sets privacy=0.6
        assert weights.get("compactness") == 0.6
        assert weights.get("privacy") == 0.6

    def test_add_design(self):
        session = ConversationSession()
        design_data = {
            "artifact_paths": {"svg": "/path/to/design.svg", "report": "/path/to/report.pdf"},
            "design_score": 0.85,
            "metrics": {"adjacency_satisfaction": 0.9},
            "generated_rooms": {"Bedroom": 2, "Kitchen": 1},
            "violations": [],
        }

        design = session.add_design(design_data, rank=1)

        assert session.state == "generated"
        assert len(session.designs) == 1
        assert design.index == 0
        assert design.rank == 1
        assert design.score == 0.85

    def test_select_design(self):
        session = ConversationSession()
        session.add_design({"artifact_paths": {"svg": "a.svg"}, "design_score": 0.8, "generated_rooms": {}, "metrics": {}, "violations": []}, rank=1)
        session.add_design({"artifact_paths": {"svg": "b.svg"}, "design_score": 0.7, "generated_rooms": {}, "metrics": {}, "violations": []}, rank=2)

        selected = session.select_design(1)
        assert selected is not None
        assert session.selected_design_index == 1
        assert session.state == "correcting"

        # Invalid index
        invalid = session.select_design(99)
        assert invalid is None

    def test_to_dict_and_from_dict(self):
        session = ConversationSession()
        session.add_message("user", "Hello")
        session.update_spec({"rooms": [{"type": "Bedroom", "count": 2}]})

        data = session.to_dict()
        restored = ConversationSession.from_dict(data)

        assert restored.session_id == session.session_id
        assert len(restored.messages) == 1
        assert len(restored.current_spec["rooms"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  3. ConversationManager Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConversationManager:

    def test_create_session(self):
        manager = ConversationManager()
        session = manager.create_session()
        assert session.session_id in manager.sessions

    def test_get_session(self):
        manager = ConversationManager()
        session = manager.create_session()
        retrieved = manager.get_session(session.session_id)
        assert retrieved is session

        # Non-existent
        assert manager.get_session("nonexistent") is None

    def test_get_or_create_session(self):
        manager = ConversationManager()
        session1 = manager.create_session()

        # Get existing
        session2 = manager.get_or_create_session(session1.session_id)
        assert session2 is session1

        # Create new
        session3 = manager.get_or_create_session(None)
        assert session3 is not session1

    def test_delete_session(self):
        manager = ConversationManager()
        session = manager.create_session()
        assert manager.delete_session(session.session_id) is True
        assert session.session_id not in manager.sessions
        assert manager.delete_session("nonexistent") is False

    def test_cleanup_old_sessions(self):
        manager = ConversationManager(max_sessions=2, session_ttl=0.01)  # Very short TTL
        s1 = manager.create_session()
        s2 = manager.create_session()

        time.sleep(0.02)  # Let them expire

        s3 = manager.create_session()  # This should trigger cleanup
        assert s1.session_id not in manager.sessions
        assert s2.session_id not in manager.sessions
        assert s3.session_id in manager.sessions

    def test_max_sessions_cleanup(self):
        manager = ConversationManager(max_sessions=2, session_ttl=3600)
        s1 = manager.create_session()
        s2 = manager.create_session()
        s3 = manager.create_session()  # Should remove oldest

        assert len(manager.sessions) <= 2

    def test_export_import_session(self):
        manager = ConversationManager()
        session = manager.create_session()
        session.add_message("user", "Test")

        exported = manager.export_session(session.session_id)
        assert exported is not None

        # Import into different manager
        manager2 = ConversationManager()
        imported = manager2.import_session(exported)
        assert imported is not None
        assert len(imported.messages) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  4. Correction Handler Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrectionHandler:

    def test_apply_add_room(self):
        spec = {"rooms": [{"type": "Bedroom", "count": 2}]}
        changes = [{"type": "add_room", "room_type": "Bathroom"}]

        modified, applied = apply_corrections_to_spec(spec, changes)

        assert len(applied) == 1
        room_types = {r["type"]: r["count"] for r in modified["rooms"]}
        assert room_types.get("Bathroom") == 1

    def test_apply_add_existing_room(self):
        spec = {"rooms": [{"type": "Bedroom", "count": 2}]}
        changes = [{"type": "add_room", "room_type": "Bedroom"}]

        modified, applied = apply_corrections_to_spec(spec, changes)

        room_types = {r["type"]: r["count"] for r in modified["rooms"]}
        assert room_types["Bedroom"] == 3

    def test_apply_add_room_with_adjacency(self):
        spec = {"rooms": [], "preferences": {"adjacency": []}}
        changes = [{"type": "add_room", "room_type": "Bathroom", "near": "Bedroom"}]

        modified, applied = apply_corrections_to_spec(spec, changes)

        adj = modified["preferences"]["adjacency"]
        assert len(adj) == 1
        assert ("Bathroom", "Bedroom", "near_to") in adj

    def test_apply_remove_room(self):
        spec = {"rooms": [{"type": "Bedroom", "count": 3}, {"type": "Kitchen", "count": 1}]}
        changes = [{"type": "remove_room", "room": "Bedroom_1"}]

        modified, applied = apply_corrections_to_spec(spec, changes)

        assert len(applied) == 1
        room_types = {r["type"]: r["count"] for r in modified["rooms"]}
        assert room_types["Bedroom"] == 2

    def test_apply_remove_last_room(self):
        spec = {"rooms": [{"type": "Store", "count": 1}]}
        changes = [{"type": "remove_room", "room": "Store"}]

        modified, applied = apply_corrections_to_spec(spec, changes)

        # Room with count 0 should be removed
        assert len(modified["rooms"]) == 0

    def test_apply_resize_room(self):
        spec = {}
        changes = [{"type": "resize_room", "room": "Kitchen", "size_change": "larger"}]

        modified, applied = apply_corrections_to_spec(spec, changes)

        assert len(applied) == 1
        assert modified["room_size_preferences"]["Kitchen"] == "larger"

    def test_apply_move_room(self):
        spec = {}
        changes = [{"type": "move_room", "room": "Bedroom", "direction": "left", "amount": "2m"}]

        modified, applied = apply_corrections_to_spec(spec, changes)

        assert len(applied) == 1
        pref = modified["room_position_preferences"]["Bedroom"]
        assert pref["direction"] == "left"
        assert pref["amount"] == "2m"

    def test_apply_swap_rooms(self):
        spec = {}
        changes = [{"type": "swap_rooms", "room_a": "Kitchen", "room_b": "DiningRoom"}]

        modified, applied = apply_corrections_to_spec(spec, changes)

        assert len(applied) == 1
        assert ("Kitchen", "DiningRoom") in modified["room_swaps"]

    def test_apply_change_adjacency(self):
        spec = {"preferences": {"adjacency": [("Kitchen", "LivingRoom", "near_to")]}}
        changes = [{"type": "change_adjacency", "room_a": "Kitchen", "room_b": "LivingRoom", "relation": "adjacent_to"}]

        modified, applied = apply_corrections_to_spec(spec, changes)

        adj = modified["preferences"]["adjacency"]
        # Should replace old adjacency
        assert ("Kitchen", "LivingRoom", "adjacent_to") in adj
        assert ("Kitchen", "LivingRoom", "near_to") not in adj


class TestCorrectionGeometry:

    def test_translate_move_room(self):
        layout = {
            "rooms": [
                {"name": "Kitchen_1", "type": "Kitchen", "polygon": [(0, 0), (2, 0), (2, 2), (0, 2)]}
            ]
        }
        boundary = [(0, 0), (10, 0), (10, 10), (0, 10)]
        changes = [{"type": "move_room", "room": "Kitchen", "direction": "right", "amount": "1m"}]

        modified = translate_corrections_to_geometry(changes, layout, boundary)

        poly = modified["rooms"][0]["polygon"]
        # X coords should be shifted by 1
        assert poly[0][0] == 1  # (0,0) -> (1,0)
        assert poly[1][0] == 3  # (2,0) -> (3,0)

    def test_translate_resize_room(self):
        layout = {
            "rooms": [
                {"name": "Bedroom_1", "type": "Bedroom", "polygon": [(2, 2), (4, 2), (4, 4), (2, 4)]}
            ]
        }
        boundary = [(0, 0), (10, 0), (10, 10), (0, 10)]
        changes = [{"type": "resize_room", "room": "Bedroom", "size_change": "larger"}]

        modified = translate_corrections_to_geometry(changes, layout, boundary)

        # Room should be scaled by 1.2 around centroid
        poly = modified["rooms"][0]["polygon"]
        # Original centroid: (3, 3), size 2x2
        # New size should be ~2.4x2.4
        width = max(p[0] for p in poly) - min(p[0] for p in poly)
        assert width == pytest.approx(2.4, abs=0.1)

    def test_translate_swap_rooms(self):
        layout = {
            "rooms": [
                {"name": "Room_A", "type": "A", "polygon": [(0, 0), (2, 0), (2, 2), (0, 2)]},
                {"name": "Room_B", "type": "B", "polygon": [(5, 5), (7, 5), (7, 7), (5, 7)]},
            ]
        }
        boundary = [(0, 0), (10, 0), (10, 10), (0, 10)]
        changes = [{"type": "swap_rooms", "room_a": "A", "room_b": "B"}]

        modified = translate_corrections_to_geometry(changes, layout, boundary)

        # Polygons should be swapped
        assert modified["rooms"][0]["polygon"] == [(5, 5), (7, 5), (7, 7), (5, 7)]
        assert modified["rooms"][1]["polygon"] == [(0, 0), (2, 0), (2, 2), (0, 2)]


class TestCorrectionValidation:

    def test_add_room_warning(self):
        spec = {"rooms": [{"type": "Bathroom", "count": 5}]}
        changes = [{"type": "add_room", "room_type": "Bathroom"}]

        is_feasible, warnings = validate_correction_feasibility(changes, spec)

        assert is_feasible is True
        assert len(warnings) == 1
        assert "crowd" in warnings[0].lower()

    def test_remove_nonexistent_room(self):
        spec = {"rooms": [{"type": "Bedroom", "count": 1}]}
        changes = [{"type": "remove_room", "room": "Kitchen"}]

        is_feasible, warnings = validate_correction_feasibility(changes, spec)

        assert is_feasible is False
        assert len(warnings) == 1
        assert "kitchen" in warnings[0].lower()  # "No Kitchen found to remove"

    def test_resize_in_tight_space(self):
        # Need many room types (not just count) for < 8 sqm per room
        spec = {"rooms": [
            {"type": "Room1", "count": 1},
            {"type": "Room2", "count": 1},
            {"type": "Room3", "count": 1},
            {"type": "Room4", "count": 1},
            {"type": "Room5", "count": 1},
            {"type": "Room6", "count": 1},
            {"type": "Room7", "count": 1},
        ]}
        resolution = {"total_area": 50}  # ~7 sq m per room < 8 threshold
        changes = [{"type": "resize_room", "room": "Room1", "size_change": "larger"}]

        is_feasible, warnings = validate_correction_feasibility(changes, spec, resolution)

        assert is_feasible is True  # Still feasible, just warned
        assert len(warnings) == 1
        assert "tight" in warnings[0].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
