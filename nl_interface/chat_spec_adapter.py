from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from nl_interface import gemini_adapter
from nl_interface.gpt_oss_adapter import GPTOSSProviderAdapter


class GeminiProviderAdapter:
    """Compatibility wrapper around the existing Gemini/local adapter."""

    provider_name = "gemini"

    def is_available(self) -> bool:
        return gemini_adapter.is_available()

    def classify_intent(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return gemini_adapter.classify_intent(user_message, context or {}, conversation_history or [])

    def process_message(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return gemini_adapter.process_message(user_message, context or {}, conversation_history or [])

    def extract_spec(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return gemini_adapter.extract_spec_from_nl(user_message, conversation_history or [])

    def parse_correction(
        self,
        user_message: str,
        design_index: int,
        current_rooms: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return gemini_adapter.parse_correction(user_message, design_index, current_rooms or [])

    def chat(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        return gemini_adapter.chat_response(user_message, context or {}, conversation_history or [])

    def rewrite_explanation(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        return self.chat(user_message, context=context, conversation_history=conversation_history)

    def propose_topology(
        self,
        semantic_spec: Dict[str, Any],
        room_program: Dict[str, Any],
        base_zoning_plan: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        del semantic_spec, room_program, base_zoning_plan
        return None


class ChatSpecAdapter:
    """Provider-agnostic adapter boundary for chat and NL-to-spec tasks."""

    def __init__(self, provider: Optional[Any] = None):
        self.provider = provider or GeminiProviderAdapter()

    @property
    def provider_name(self) -> str:
        return getattr(self.provider, "provider_name", "unknown")

    def is_available(self) -> bool:
        return bool(getattr(self.provider, "is_available", lambda: False)())

    def classify_intent(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return self.provider.classify_intent(
            user_message,
            context=context or {},
            conversation_history=conversation_history or [],
        )

    def process_message(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return self.provider.process_message(
            user_message,
            context=context or {},
            conversation_history=conversation_history or [],
        )

    def extract_spec(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return self.provider.extract_spec(user_message, conversation_history=conversation_history or [])

    def parse_correction(
        self,
        user_message: str,
        design_index: int,
        current_rooms: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if hasattr(self.provider, "parse_correction"):
            return self.provider.parse_correction(user_message, design_index, current_rooms or [])
        return {"understood": False, "changes": [], "clarification_needed": None}

    def chat(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        return self.provider.chat(
            user_message,
            context=context or {},
            conversation_history=conversation_history or [],
        )

    def rewrite_explanation(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if hasattr(self.provider, "rewrite_explanation"):
            return self.provider.rewrite_explanation(
                user_message,
                context=context or {},
                conversation_history=conversation_history or [],
            )
        return self.chat(user_message, context=context, conversation_history=conversation_history)

    def propose_topology(
        self,
        semantic_spec: Dict[str, Any],
        room_program: Dict[str, Any],
        base_zoning_plan: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if hasattr(self.provider, "propose_topology"):
            return self.provider.propose_topology(
                semantic_spec,
                room_program,
                base_zoning_plan=base_zoning_plan,
            )
        return None


def _select_provider() -> Any:
    requested = os.getenv("BLUEPRINT_CHAT_PROVIDER", "gemini").strip().lower()
    if requested in {"gpt_oss", "gpt-oss", "gptoss"}:
        provider = GPTOSSProviderAdapter()
        if provider.is_available():
            return provider
    return GeminiProviderAdapter()


def default_chat_spec_adapter() -> ChatSpecAdapter:
    return ChatSpecAdapter(_select_provider())
