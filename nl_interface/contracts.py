from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SemanticSpec:
    building_type: str = "Residential"
    layout_type: Optional[str] = None
    shorthand: Optional[str] = None
    requested_rooms: Dict[str, int] = field(default_factory=dict)
    plot_type: Optional[str] = None
    entrance_side: Optional[str] = None
    total_area_sqm: Optional[float] = None
    boundary_size_m: Optional[List[float]] = None
    style_hints: List[str] = field(default_factory=list)
    adjacency_preferences: List[Dict[str, Any]] = field(default_factory=list)
    privacy_preferences: Dict[str, str] = field(default_factory=dict)
    unsupported_requests: List[str] = field(default_factory=list)
    unresolved_fields: List[str] = field(default_factory=list)
    assumptions_used: List[str] = field(default_factory=list)
    user_prompt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProgramRoom:
    name: str
    type: str
    zone: str
    semantic_role: str
    required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoomProgram:
    layout_type: Optional[str]
    canonical: bool
    rooms: List[ProgramRoom] = field(default_factory=list)
    required_counts: Dict[str, int] = field(default_factory=dict)
    assumptions_used: List[str] = field(default_factory=list)
    deferred_semantics: List[str] = field(default_factory=list)
    supported_scope: str = "canonical_residential"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["rooms"] = [room.to_dict() for room in self.rooms]
        return payload


@dataclass
class ZoningPlan:
    layout_pattern: str
    entrance_frontage_zone: str
    frontage_room: Optional[str]
    named_adjacency: List[Dict[str, Any]] = field(default_factory=list)
    zone_map: Dict[str, str] = field(default_factory=dict)
    spatial_hints: Dict[str, List[float]] = field(default_factory=dict)
    room_order: List[str] = field(default_factory=list)
    size_priors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    heuristics: List[str] = field(default_factory=list)
    assumptions_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationOutcome:
    stage: str
    success: bool
    engine: Optional[str] = None
    report_status: Optional[str] = None
    summary: Optional[str] = None
    assumptions_used: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifact_urls: Dict[str, str] = field(default_factory=dict)
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReplyPayload:
    conversation_state: Dict[str, Any] = field(default_factory=dict)
    clarification_request: Optional[str] = None
    assumptions_used: List[str] = field(default_factory=list)
    program_summary: Optional[str] = None
    zoning_summary: Optional[str] = None
    latest_design_summary: Optional[Dict[str, Any]] = None
    expert_diagnostics: Optional[Dict[str, Any]] = None
    suggested_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
