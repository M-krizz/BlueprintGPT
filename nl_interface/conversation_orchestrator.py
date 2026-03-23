from __future__ import annotations

from typing import Any, Dict, List, Optional

from nl_interface.chat_spec_adapter import ChatSpecAdapter, default_chat_spec_adapter
from nl_interface.contracts import GenerationOutcome, ReplyPayload
from nl_interface.program_planner import enrich_spec_with_planning, summarize_room_program, summarize_zoning_plan
from nl_interface.response_composer import build_generation_summary, compose_followup_reply


class ConversationOrchestrator:
    """Thin orchestration layer over session/spec/planning context."""

    def __init__(self, chat_adapter: Optional[ChatSpecAdapter] = None):
        self.chat_adapter = chat_adapter or default_chat_spec_adapter()

    def process_message(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return self.chat_adapter.process_message(
            user_message,
            context=context or {},
            conversation_history=conversation_history or [],
        )

    def extract_spec(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return self.chat_adapter.extract_spec(user_message, conversation_history=conversation_history or [])

    def chat_reply(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        return self.chat_adapter.chat(
            user_message,
            context=context or {},
            conversation_history=conversation_history or [],
        )

    def enrich_spec(self, spec: Dict[str, Any], resolution: Optional[Dict[str, Any]], user_message: Optional[str]) -> Dict[str, Any]:
        return enrich_spec_with_planning(spec, resolution=resolution, user_prompt=user_message)

    def build_clarification_request(self, semantic_spec: Optional[Dict[str, Any]], room_program: Optional[Dict[str, Any]]) -> Optional[str]:
        semantic_spec = semantic_spec or {}
        unresolved = semantic_spec.get("unresolved_fields") or []
        unsupported = semantic_spec.get("unsupported_requests") or []
        if unsupported:
            return "I can continue, but I need you to simplify or restate unsupported room types or constraints first."
        if "room_program" in unresolved:
            return "I still need the core room program before I can build a layout."
        return None

    def build_reply_payload(
        self,
        *,
        session,
        semantic_spec: Optional[Dict[str, Any]] = None,
        room_program: Optional[Dict[str, Any]] = None,
        zoning_plan: Optional[Dict[str, Any]] = None,
        generation_outcome: Optional[GenerationOutcome] = None,
        clarification_request: Optional[str] = None,
        suggested_actions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        latest_design = session.designs[-1].to_dict() if session.designs else None
        latest_design_summary = build_generation_summary(latest_design) if latest_design else None
        assumptions = []
        assumptions.extend((semantic_spec or {}).get("assumptions_used", []) or [])
        assumptions.extend((room_program or {}).get("assumptions_used", []) or [])
        if generation_outcome:
            assumptions.extend(generation_outcome.assumptions_used)

        payload = ReplyPayload(
            conversation_state={
                "state": session.state,
                "has_design": bool(latest_design),
                "layout_type": (semantic_spec or {}).get("layout_type") or (room_program or {}).get("layout_type"),
            },
            clarification_request=clarification_request,
            assumptions_used=list(dict.fromkeys(assumptions)),
            program_summary=summarize_room_program(room_program),
            zoning_summary=summarize_zoning_plan(zoning_plan),
            latest_design_summary=latest_design_summary,
            expert_diagnostics={
                "semantic_spec": semantic_spec,
                "room_program": room_program,
                "zoning_plan": zoning_plan,
                "generation_outcome": generation_outcome.to_dict() if generation_outcome else None,
            },
            suggested_actions=suggested_actions or [],
        )
        return payload.to_dict()

    def compose_contextual_reply(
        self,
        *,
        user_message: str,
        session,
        fallback_text: str,
        room_program: Optional[Dict[str, Any]] = None,
        zoning_plan: Optional[Dict[str, Any]] = None,
    ) -> str:
        latest_design = session.designs[-1].to_dict() if session.designs else None
        contextual = compose_followup_reply(
            user_message,
            latest_design=latest_design,
            room_program=room_program,
            zoning_plan=zoning_plan,
        )
        return contextual or fallback_text

