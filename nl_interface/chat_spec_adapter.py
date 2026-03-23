from __future__ import annotations

from typing import Any, Dict, List, Optional

from nl_interface import gemini_adapter


class ChatSpecAdapter:
    """Provider-agnostic adapter boundary for chat and NL-to-spec tasks."""

    provider_name = "gemini"

    def is_available(self) -> bool:
        return gemini_adapter.is_available()

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


def default_chat_spec_adapter() -> ChatSpecAdapter:
    return ChatSpecAdapter()
