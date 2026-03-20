"""
conversation.py – Conversation state management for multi-turn interactions.

Manages:
- Session state (spec accumulation across turns)
- Conversation history
- Generated designs and their metadata
- Correction tracking
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.processing_logger import ProcessingLogger
from uuid import uuid4


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class GeneratedDesign:
    """A generated design with its metadata."""
    index: int
    svg_path: str
    report_path: Optional[str]
    score: float
    rank: int
    metrics: Dict[str, Any]
    rooms: List[Dict]
    violations: List[str]
    explanation: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "svg_path": self.svg_path,
            "report_path": self.report_path,
            "score": self.score,
            "rank": self.rank,
            "metrics": self.metrics,
            "rooms": self.rooms,
            "violations": self.violations,
            "explanation": self.explanation,
        }


class ConversationSession:
    """
    Manages a single conversation session with a user.

    Tracks:
    - Accumulated spec from user messages
    - Conversation history
    - Generated designs
    - Current state (initial, specifying, generated, correcting)
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or uuid4().hex
        self.created_at = time.time()
        self.updated_at = time.time()

        # Conversation state
        self.state = "initial"  # initial, specifying, generated, correcting
        self.messages: List[Message] = []

        # Accumulated spec
        self.current_spec: Dict[str, Any] = {
            "rooms": [],
            "preferences": {"adjacency": [], "privacy": {}},
            "weights": {},
            "plot_type": None,
            "entrance_side": None,
        }

        # Resolution (boundary, etc.)
        self.resolution: Optional[Dict[str, Any]] = None

        # Generated designs
        self.designs: List[GeneratedDesign] = []
        self.selected_design_index: Optional[int] = None

        # Correction history
        self.corrections: List[Dict[str, Any]] = []

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> Message:
        """Add a message to the conversation history."""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        self.updated_at = time.time()
        return msg

    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get conversation history as list of dicts."""
        messages = self.messages[-limit:] if limit else self.messages
        return [m.to_dict() for m in messages]

    def update_spec(self, extracted: Dict[str, Any]) -> None:
        """Merge extracted spec data into current spec."""
        ProcessingLogger.logger.debug(f"update_spec: merging {len(extracted.get('rooms', []))} extracted rooms into {len(self.current_spec.get('rooms', []))} existing")

        # Merge rooms (replacement semantics: new values override existing)
        room_counts = {}
        for room in self.current_spec.get("rooms", []):
            rtype = room.get("type")
            room_counts[rtype] = room_counts.get(rtype, 0) + room.get("count", 1)

        for room in extracted.get("rooms", []):
            rtype = room.get("type")
            count = room.get("count", 1)
            # Replace, not add: "2 bedrooms" means 2 total, not 2 more
            room_counts[rtype] = count

        self.current_spec["rooms"] = [
            {"type": rtype, "count": count}
            for rtype, count in room_counts.items()
        ]

        ProcessingLogger.logger.debug(f"update_spec: after merge, {len(self.current_spec['rooms'])} room types")

        # Merge adjacency (additive, deduplicated)
        existing_adj = set()
        for adj in self.current_spec.get("preferences", {}).get("adjacency", []):
            key = (adj[0], adj[1], adj[2]) if isinstance(adj, (list, tuple)) else (adj.get("source"), adj.get("target"), adj.get("relation"))
            existing_adj.add(key)

        for adj in extracted.get("adjacency", []):
            key = (adj.get("source"), adj.get("target"), adj.get("relation"))
            if key not in existing_adj:
                self.current_spec.setdefault("preferences", {}).setdefault("adjacency", []).append(
                    (adj.get("source"), adj.get("target"), adj.get("relation"))
                )
                existing_adj.add(key)

        # Update entrance side
        if extracted.get("entrance_side"):
            self.current_spec["entrance_side"] = extracted["entrance_side"]

        # Update plot type
        if extracted.get("plot_type"):
            self.current_spec["plot_type"] = extracted["plot_type"]

        # Update style hints -> weights
        for hint in extracted.get("style_hints", []):
            if "compact" in hint.lower():
                self.current_spec.setdefault("weights", {})["compactness"] = 0.6
            if "open" in hint.lower():
                # Open plan = less compactness, more circulation
                self.current_spec.setdefault("weights", {})["compactness"] = 0.2
                self.current_spec.setdefault("weights", {})["corridor"] = 0.5
            if "privacy" in hint.lower():
                self.current_spec.setdefault("weights", {})["privacy"] = 0.6

        self.state = "specifying"
        self.updated_at = time.time()

    def set_resolution(self, resolution: Dict[str, Any]) -> None:
        """Set boundary/resolution information."""
        self.resolution = resolution
        self.updated_at = time.time()

    def add_design(self, design_data: Dict[str, Any], rank: int) -> GeneratedDesign:
        """Add a generated design to the session."""
        design = GeneratedDesign(
            index=len(self.designs),
            svg_path=design_data.get("artifact_paths", {}).get("svg", ""),
            report_path=design_data.get("artifact_paths", {}).get("report"),
            score=design_data.get("design_score", 0),
            rank=rank,
            metrics=design_data.get("metrics", {}),
            rooms=[{"type": rtype, "count": count} for rtype, count in design_data.get("generated_rooms", {}).items()],
            violations=design_data.get("violations", []),
            explanation=design_data.get("explanation"),
        )
        self.designs.append(design)
        self.state = "generated"
        self.updated_at = time.time()
        return design

    def clear_designs(self) -> None:
        """Clear all generated designs."""
        self.designs = []
        self.selected_design_index = None
        self.state = "specifying"
        self.updated_at = time.time()

    def select_design(self, index: int) -> Optional[GeneratedDesign]:
        """Select a design for potential correction."""
        if 0 <= index < len(self.designs):
            self.selected_design_index = index
            self.state = "correcting"
            self.updated_at = time.time()
            return self.designs[index]
        return None

    def add_correction(self, correction: Dict[str, Any]) -> None:
        """Record a correction request."""
        self.corrections.append({
            "timestamp": time.time(),
            "design_index": self.selected_design_index,
            "changes": correction.get("changes", []),
        })
        self.updated_at = time.time()

    def get_context(self) -> Dict[str, Any]:
        """Get current context for AI responses."""
        return {
            "session_id": self.session_id,
            "state": self.state,
            "spec": self.current_spec,
            "num_designs": len(self.designs),
            "selected_design": self.selected_design_index,
            "has_resolution": self.resolution is not None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dict for storage/export."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "state": self.state,
            "messages": [m.to_dict() for m in self.messages],
            "current_spec": self.current_spec,
            "resolution": self.resolution,
            "designs": [d.to_dict() for d in self.designs],
            "selected_design_index": self.selected_design_index,
            "corrections": self.corrections,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        """Deserialize session from dict."""
        session = cls(session_id=data.get("session_id"))
        session.created_at = data.get("created_at", time.time())
        session.updated_at = data.get("updated_at", time.time())
        session.state = data.get("state", "initial")
        session.current_spec = data.get("current_spec", {})
        session.resolution = data.get("resolution")
        session.selected_design_index = data.get("selected_design_index")
        session.corrections = data.get("corrections", [])

        for msg_data in data.get("messages", []):
            try:
                session.messages.append(Message(
                    role=msg_data.get("role", "user"),
                    content=msg_data.get("content", ""),
                    timestamp=msg_data.get("timestamp", time.time()),
                    metadata=msg_data.get("metadata", {}),
                ))
            except Exception:
                continue  # Skip malformed messages

        for design_data in data.get("designs", []):
            try:
                session.designs.append(GeneratedDesign(
                    index=design_data.get("index", 0),
                    svg_path=design_data.get("svg_path", ""),
                    report_path=design_data.get("report_path"),
                    score=design_data.get("score", 0),
                    rank=design_data.get("rank", 0),
                    metrics=design_data.get("metrics", {}),
                    rooms=design_data.get("rooms", []),
                    violations=design_data.get("violations", []),
                    explanation=design_data.get("explanation"),
                ))
            except Exception:
                continue  # Skip malformed designs

        return session


class ConversationManager:
    """
    Manages multiple conversation sessions.

    Provides session creation, retrieval, and cleanup.
    """

    def __init__(self, max_sessions: int = 100, session_ttl: float = 3600 * 24):
        self.sessions: Dict[str, ConversationSession] = {}
        self.max_sessions = max_sessions
        self.session_ttl = session_ttl  # 24 hours default

    def create_session(self) -> ConversationSession:
        """Create a new conversation session."""
        self._cleanup_old_sessions()
        session = ConversationSession()
        self.sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get an existing session by ID."""
        session = self.sessions.get(session_id)
        if session:
            session.updated_at = time.time()
        return session

    def get_or_create_session(self, session_id: Optional[str] = None) -> ConversationSession:
        """Get existing session or create new one."""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def _cleanup_old_sessions(self) -> None:
        """Remove sessions older than TTL or when exceeding max."""
        now = time.time()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session.updated_at > self.session_ttl
        ]
        for sid in expired:
            del self.sessions[sid]

        # If still too many, remove oldest
        if len(self.sessions) >= self.max_sessions:
            sorted_sessions = sorted(
                self.sessions.items(),
                key=lambda x: x[1].updated_at,
            )
            to_remove = len(self.sessions) - self.max_sessions + 1
            for sid, _ in sorted_sessions[:to_remove]:
                del self.sessions[sid]

    def export_session(self, session_id: str) -> Optional[str]:
        """Export session as JSON string."""
        session = self.get_session(session_id)
        if session:
            return json.dumps(session.to_dict(), indent=2, default=str)
        return None

    def import_session(self, data: str) -> Optional[ConversationSession]:
        """Import session from JSON string."""
        try:
            session_data = json.loads(data)
            session = ConversationSession.from_dict(session_data)
            self.sessions[session.session_id] = session
            return session
        except Exception:
            return None


# Global conversation manager instance
conversation_manager = ConversationManager()
