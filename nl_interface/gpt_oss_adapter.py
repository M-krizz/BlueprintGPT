from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    import httpx  # type: ignore
except ImportError:
    httpx = None

from nl_interface import gemini_adapter
from nl_interface.gpt_oss_schema_prompts import (
    build_chat_prompt,
    build_correction_prompt,
    build_intent_prompt,
    build_spec_prompt,
    build_topology_prompt,
)
from utils.processing_logger import ProcessingLogger


INTENT_DESIGN = gemini_adapter.INTENT_DESIGN
INTENT_QUESTION = gemini_adapter.INTENT_QUESTION
INTENT_CORRECTION = gemini_adapter.INTENT_CORRECTION
INTENT_CONVERSATION = gemini_adapter.INTENT_CONVERSATION


class GPTOSSProviderAdapter:
    """Optional OpenAI-compatible local adapter for gpt-oss style models."""

    provider_name = "gpt_oss"

    def __init__(self) -> None:
        self.enabled = os.getenv("GPT_OSS_ENABLED", "false").strip().lower() == "true"
        self.base_url = os.getenv("GPT_OSS_BASE_URL", "").strip().rstrip("/")
        self.api_key = os.getenv("GPT_OSS_API_KEY", "").strip()
        self.model = os.getenv("GPT_OSS_MODEL", "gpt-oss-20b").strip() or "gpt-oss-20b"
        self.timeout_s = float(os.getenv("GPT_OSS_TIMEOUT_S", "30").strip() or "30")

    def is_available(self) -> bool:
        return self.enabled and bool(self.base_url)

    def classify_intent(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not self.is_available():
            return gemini_adapter.classify_intent(user_message, context or {}, conversation_history or [])
        prompt = build_intent_prompt(
            user_message,
            context=context or {},
            conversation_history=conversation_history or [],
        )
        response = self._request_json(prompt)
        if not response:
            return gemini_adapter.classify_intent(user_message, context or {}, conversation_history or [])
        intent = str(response.get("intent") or "").strip().lower()
        if intent not in {
            INTENT_DESIGN,
            INTENT_QUESTION,
            INTENT_CORRECTION,
            INTENT_CONVERSATION,
        }:
            return gemini_adapter.classify_intent(user_message, context or {}, conversation_history or [])
        return {
            "intent": intent,
            "confidence": float(response.get("confidence", 0.7) or 0.7),
            "reason": str(response.get("reason") or "gpt-oss classification"),
            "design_keywords": list(response.get("design_keywords") or []),
            "question_type": response.get("question_type"),
        }

    def process_message(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        conversation_history = conversation_history or []
        if not self.is_available():
            return gemini_adapter.process_message(user_message, context, conversation_history)

        intent_result = self.classify_intent(user_message, context, conversation_history)
        intent = intent_result.get("intent", INTENT_CONVERSATION)
        result = {
            "intent": intent,
            "intent_confidence": intent_result.get("confidence", 0.0),
            "intent_reason": intent_result.get("reason", ""),
            "response": "",
            "spec": None,
            "correction": None,
            "should_generate": False,
        }

        if intent == INTENT_DESIGN:
            spec = self.extract_spec(user_message, conversation_history=conversation_history)
            result["spec"] = spec
            result["should_generate"] = bool(spec.get("rooms"))
            if spec.get("rooms"):
                room_summary = ", ".join(
                    f"{int(room.get('count', 1) or 1)} {room.get('type')}"
                    for room in spec.get("rooms", [])
                    if room.get("type")
                )
                result["response"] = f"I'll use {room_summary} as the starting program and generate a layout from that."
            else:
                result["response"] = (
                    "I detected a design request, but I still need the room program before I can generate a layout."
                )
        elif intent == INTENT_CORRECTION:
            correction = self.parse_correction(
                user_message,
                design_index=int(context.get("selected_design") or 0),
                current_rooms=context.get("current_rooms") or [],
            )
            result["correction"] = correction
            if correction.get("understood"):
                result["response"] = "I understood the requested design changes and will apply them to the current layout."
            else:
                result["response"] = correction.get(
                    "clarification_needed",
                    "I couldn't fully resolve that modification request against the current design.",
                )
        else:
            result["response"] = self.chat(
                user_message,
                context=context,
                conversation_history=conversation_history,
            )
        return result

    def extract_spec(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not self.is_available():
            return gemini_adapter.extract_spec_from_nl(user_message, conversation_history or [])
        prompt = build_spec_prompt(user_message, conversation_history=conversation_history or [])
        response = self._request_json(prompt)
        if not response:
            return gemini_adapter.extract_spec_from_nl(user_message, conversation_history or [])
        normalized = self._normalize_spec_payload(response)
        return gemini_adapter._apply_constraint_enhancement(normalized)

    def parse_correction(
        self,
        user_message: str,
        design_index: int,
        current_rooms: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not self.is_available():
            return gemini_adapter.parse_correction(user_message, design_index, current_rooms or [])
        prompt = build_correction_prompt(
            user_message,
            design_index=design_index,
            current_rooms=current_rooms or [],
        )
        response = self._request_json(prompt)
        if not response:
            return gemini_adapter.parse_correction(user_message, design_index, current_rooms or [])
        normalized = self._normalize_correction_payload(response, current_rooms or [])
        if not normalized.get("understood"):
            return gemini_adapter.parse_correction(user_message, design_index, current_rooms or [])
        return normalized

    def chat(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if not self.is_available():
            return gemini_adapter.chat_response(user_message, context or {}, conversation_history or [])
        prompt = build_chat_prompt(
            user_message,
            context=context or {},
            conversation_history=conversation_history or [],
        )
        text = self._request_text(prompt)
        if not text:
            return gemini_adapter.chat_response(user_message, context or {}, conversation_history or [])
        return text.strip()

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
        if not self.is_available():
            return None
        prompt = build_topology_prompt(
            semantic_spec,
            room_program,
            base_zoning_plan=base_zoning_plan or {},
        )
        response = self._request_json(prompt)
        if not response:
            return None
        normalized = self._normalize_topology_payload(response, room_program)
        return normalized or None

    def _request_text(self, prompt: str) -> Optional[str]:
        try:
            endpoint = f"{self.base_url}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are BlueprintGPT's gpt-oss integration. "
                            "Respond concisely and stay grounded in the prompt."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
            }
            if httpx is not None:
                response = httpx.post(endpoint, json=payload, headers=headers, timeout=self.timeout_s)
                response.raise_for_status()
                data = response.json()
            else:
                raw_request = urllib_request.Request(
                    endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib_request.urlopen(raw_request, timeout=self.timeout_s) as response:
                    data = json.loads(response.read().decode("utf-8"))
            return str(
                (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
            ).strip() or None
        except (Exception, urllib_error.URLError) as exc:
            ProcessingLogger.logger.warning(f"gpt-oss request failed: {exc}")
            return None

    def _request_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        text = self._request_text(prompt)
        if not text:
            return None
        parsed = _extract_json_object(text)
        if isinstance(parsed, dict):
            return parsed
        return None

    def _normalize_spec_payload(self, response: Dict[str, Any]) -> Dict[str, Any]:
        rooms = []
        for room in response.get("rooms") or []:
            room_type = str(room.get("type") or "").strip()
            count = int(room.get("count", 1) or 1)
            if room_type and count > 0:
                rooms.append({"type": room_type, "count": count})

        preferences = {"adjacency": [], "privacy": {}}
        for item in response.get("preferences", {}).get("adjacency", []) or []:
            if isinstance(item, list) and len(item) >= 3:
                preferences["adjacency"].append([str(item[0]), str(item[1]), str(item[2])])
        for room_type, privacy in (response.get("preferences", {}).get("privacy", {}) or {}).items():
            if room_type:
                preferences["privacy"][str(room_type)] = str(privacy)

        boundary_size = response.get("boundary_size") or {}
        result: Dict[str, Any] = {
            "building_type": str(response.get("building_type") or "Residential"),
            "layout_type": response.get("layout_type"),
            "plot_type": response.get("plot_type"),
            "entrance_side": response.get("entrance_side"),
            "rooms": rooms,
            "preferences": preferences,
            "style_hints": list(response.get("style_hints") or []),
            "assumptions_used": list(response.get("assumptions_used") or []),
            "unresolved_fields": list(response.get("unresolved_fields") or []),
        }
        width = boundary_size.get("width")
        height = boundary_size.get("height")
        unit = str(boundary_size.get("unit") or "m").lower()
        if width is not None and height is not None:
            multiplier = 0.3048 if unit == "ft" else 1.0
            try:
                result["boundary_size"] = [
                    round(float(width) * multiplier, 3),
                    round(float(height) * multiplier, 3),
                ]
            except (TypeError, ValueError):
                pass
        return result

    def _normalize_correction_payload(
        self,
        response: Dict[str, Any],
        current_rooms: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        valid_room_names = {
            str(room.get("name") or room.get("type") or "").strip()
            for room in current_rooms
            if room.get("name") or room.get("type")
        }
        changes: List[Dict[str, Any]] = []
        for change in response.get("changes") or []:
            change_type = str(change.get("type") or "").strip()
            candidate = dict(change)
            if change_type == "change_adjacency":
                room_a = str(candidate.get("room_a") or "").strip()
                room_b = str(candidate.get("room_b") or "").strip()
                if room_a and room_b and (not valid_room_names or (room_a in valid_room_names and room_b in valid_room_names)):
                    candidate["relation"] = str(candidate.get("relation") or "adjacent_to")
                    changes.append(candidate)
            elif change_type in {"move_room", "resize_room", "remove_room"}:
                room_name = str(candidate.get("room") or "").strip()
                if room_name and (not valid_room_names or room_name in valid_room_names):
                    changes.append(candidate)
            elif change_type == "swap_rooms":
                room_a = str(candidate.get("room_a") or "").strip()
                room_b = str(candidate.get("room_b") or "").strip()
                if room_a and room_b and (not valid_room_names or (room_a in valid_room_names and room_b in valid_room_names)):
                    changes.append(candidate)
            elif change_type == "add_room":
                room_type = str(candidate.get("room_type") or "").strip()
                if room_type:
                    changes.append(candidate)
        return {
            "understood": bool(changes),
            "changes": changes,
            "clarification_needed": None if changes else str(response.get("clarification_needed") or ""),
        }

    def _normalize_topology_payload(
        self,
        response: Dict[str, Any],
        room_program: Dict[str, Any],
    ) -> Dict[str, Any]:
        valid_room_names = {
            str(room.get("name") or "").strip()
            for room in room_program.get("rooms", [])
            if room.get("name")
        }
        if not valid_room_names:
            return {}

        result: Dict[str, Any] = {}
        layout_pattern = str(response.get("layout_pattern") or "").strip()
        if layout_pattern:
            result["layout_pattern"] = layout_pattern

        frontage_room = str(response.get("frontage_room") or "").strip()
        if frontage_room in valid_room_names:
            result["frontage_room"] = frontage_room

        for key in ("public_zone", "service_zone", "private_zone"):
            members = [str(name) for name in (response.get(key) or []) if str(name) in valid_room_names]
            if members:
                result[key] = members

        room_order = []
        for room_name in response.get("room_order") or []:
            room_name = str(room_name)
            if room_name in valid_room_names and room_name not in room_order:
                room_order.append(room_name)
        if room_order:
            result["room_order"] = room_order

        size_priors: Dict[str, Any] = {}
        for room_name, prior in (response.get("size_priors") or {}).items():
            room_name = str(room_name)
            if room_name not in valid_room_names:
                continue
            if isinstance(prior, str) and prior.strip():
                size_priors[room_name] = prior.strip().lower()
            elif isinstance(prior, dict):
                size_priors[room_name] = prior
        if size_priors:
            result["size_priors"] = size_priors

        adjacency: List[Dict[str, Any]] = []
        for item in response.get("named_adjacency") or []:
            room_a = str(item.get("a") or "").strip()
            room_b = str(item.get("b") or "").strip()
            relation = str(item.get("type") or "prefer").strip().lower()
            if room_a in valid_room_names and room_b in valid_room_names and room_a != room_b:
                adjacency.append(
                    {
                        "a": room_a,
                        "b": room_b,
                        "type": "avoid" if relation == "avoid" else "prefer",
                        "score": float(item.get("score", 0.8) or 0.8),
                    }
                )
        if adjacency:
            result["named_adjacency"] = adjacency

        heuristics = [str(item).strip() for item in (response.get("heuristics") or []) if str(item).strip()]
        if heuristics:
            result["heuristics"] = heuristics
        return result


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = str(text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                fragment = text[start : index + 1]
                try:
                    parsed = json.loads(fragment)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None
