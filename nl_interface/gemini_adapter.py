"""
gemini_adapter.py â€“ Google Gemini API integration for natural language understanding,
design explanation, and correction parsing.

This module provides:
1. Intent Classification - detect whether user wants design, question, or conversation
2. NL â†’ Spec conversion (parse user intent into system arguments)
3. Design â†’ Explanation (explain generated layouts and rankings)
4. Correction â†’ Delta (parse user corrections into actionable changes)
5. Context-aware Chat - respond to questions and conversations appropriately
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from utils.processing_logger import ProcessingLogger

# Constraint Analyzer for intelligent spec enhancement
try:
    from nl_interface.constraint_analyzer import (
        get_analyzer,
        get_layout_requirements,
        enhance_spec as enhance_spec_with_constraints,
        get_constraint_summary,
    )
    _HAS_CONSTRAINT_ANALYZER = True
except ImportError:
    _HAS_CONSTRAINT_ANALYZER = False
    get_analyzer = None
    get_layout_requirements = None
    enhance_spec_with_constraints = None
    get_constraint_summary = None

# Gemini API
try:
    from google import genai as google_genai
    _GENAI_SDK = "google.genai"
    _HAS_GENAI = True
except ImportError:
    google_genai = None
    try:
        import google.generativeai as legacy_genai
        _GENAI_SDK = "google.generativeai"
        _HAS_GENAI = True
    except ImportError:
        legacy_genai = None
        _GENAI_SDK = None
        _HAS_GENAI = False

# â”€â”€ Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "true").lower() == "true"
DEPRECATED_MODEL_ALIASES = {
    "gemini-1.5-flash": "gemini-2.5-flash",
    "gemini-1.5-pro": "gemini-2.5-pro",
    "gemini-pro": "gemini-2.5-pro",
}


def _resolve_gemini_model(model_name: str) -> str:
    configured = (model_name or "").strip() or "gemini-2.5-flash"
    replacement = DEPRECATED_MODEL_ALIASES.get(configured.lower())
    if replacement:
        ProcessingLogger.logger.warning(
            f"GEMINI_MODEL '{configured}' is deprecated or unavailable; using '{replacement}' instead."
        )
        return replacement
    return configured


GEMINI_MODEL = _resolve_gemini_model(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

# Intent types
INTENT_DESIGN = "design"           # User wants to create/modify a floor plan
INTENT_QUESTION = "question"       # User is asking a question
INTENT_CORRECTION = "correction"   # User wants to modify existing design
INTENT_CONVERSATION = "conversation"  # General chat/greeting


def _is_gemini_auth_error(exc: Exception) -> bool:
    message = str(exc).lower()
    auth_markers = (
        "permission_denied",
        "reported as leaked",
        "invalid api key",
        "api key not valid",
        "api_key_invalid",
        "forbidden",
        "401",
        "403",
    )
    return any(marker in message for marker in auth_markers)


def _disable_gemini(reason: str) -> None:
    global _gemini_ready, _gemini_disabled_reason
    if _gemini_disabled_reason:
        return
    _gemini_ready = False
    _gemini_disabled_reason = reason
    ProcessingLogger.logger.warning(f"Gemini disabled for this process: {reason}")


def _init_gemini() -> bool:
    """Initialize Gemini client. Returns True if successful."""
    global _gemini_client
    if not _HAS_GENAI:
        return False
    if not GEMINI_API_KEY:
        return False
    try:
        if _GENAI_SDK == "google.genai":
            _gemini_client = google_genai.Client(api_key=GEMINI_API_KEY)
        elif _GENAI_SDK == "google.generativeai":
            legacy_genai.configure(api_key=GEMINI_API_KEY)
            _gemini_client = None
        else:
            return False
        return True
    except Exception as exc:
        ProcessingLogger.logger.warning(f"Gemini client initialization failed: {exc}")
        return False


_gemini_client = None
_gemini_disabled_reason = None
_gemini_ready = _init_gemini() if GEMINI_ENABLED else False


def _generate_text(prompt: str, *, response_mime_type: Optional[str] = None) -> Optional[str]:
    """Generate text with the configured Gemini SDK."""
    if not is_available():
        return None

    try:
        if _GENAI_SDK == "google.genai":
            config = {}
            if response_mime_type:
                config["response_mime_type"] = response_mime_type
            response = _gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config or None,
            )
            return getattr(response, "text", None)

        model = legacy_genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return getattr(response, "text", None)
    except Exception as exc:
        if _is_gemini_auth_error(exc):
            _disable_gemini(str(exc))
            return None
        raise


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  System Prompts
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

INTENT_CLASSIFICATION_PROMPT = """You are BlueprintGPT, an AI assistant for architectural floor plan design.
Analyze the user's message and classify their intent into ONE of these categories:

1. **"design"** - User wants to create or generate a new floor plan/layout. Look for:
   - Room specifications (e.g., "3 bedrooms", "2BHK", "kitchen near living room")
   - Plot/area mentions (e.g., "10 marla", "500 sq ft")
   - Building type requests (e.g., "apartment", "house", "residential")
   - Layout preferences (e.g., "open plan", "compact", "privacy-focused")

2. **"correction"** - User wants to modify an existing design. Look for:
   - References to previous designs (e.g., "design #1", "the first one", "this layout")
   - Modification requests (e.g., "move the kitchen", "make bedroom larger", "swap rooms")
   - Specific changes (e.g., "add a bathroom", "remove the garage")

3. **"question"** - User is asking a question about architecture, the system, or designs. Look for:
   - Question words (what, why, how, can, could, would, is, are, does, will)
   - Seeking information or explanations
   - Asking about capabilities or features

4. **"conversation"** - General chat, greetings, or off-topic. Look for:
   - Greetings (hi, hello, hey)
   - Thanks, acknowledgments
   - Non-architectural topics
   - Small talk

Respond with ONLY a JSON object:
{
  "intent": "design|correction|question|conversation",
  "confidence": 0.0-1.0,
  "reason": "brief explanation",
  "design_keywords": ["list", "of", "design", "keywords", "found"],
  "question_type": "if question, what type: capability|explanation|clarification|general"
}

User message: "{user_message}"
Conversation state: {state}
Previous designs exist: {has_designs}
"""

SPEC_EXTRACTION_PROMPT_BASE = """You are an intelligent AI assistant for an architectural floor plan generator.
Your task is to extract structured specifications from natural language descriptions and apply building constraints.

## IMPORTANT: Layout Type Requirements

You must understand the minimum requirements for each layout type:

### 1BHK (35-60 sqm recommended):
- 1 Bedroom (min 7.5 sqm)
- 1 LivingRoom (min 7.5 sqm)
- 1 Kitchen (min 3.3 sqm)
- 1 Bathroom (min 1.2 sqm)

### 2BHK (55-100 sqm recommended):
- 2 Bedrooms (each min 9.5 sqm)
- 1 LivingRoom (min 9.5 sqm)
- 1 Kitchen (min 4.5 sqm)
- 1 Bathroom (min 1.8 sqm)

### 3BHK (85-150 sqm recommended):
- 3 Bedrooms (each min 9.5 sqm)
- 1 LivingRoom (min 9.5 sqm)
- 1 Kitchen (min 4.5 sqm)
- 2 Bathrooms (each min 1.8 sqm)

### 4BHK (120-200 sqm recommended):
- 4 Bedrooms (each min 9.5 sqm)
- 1 LivingRoom (min 9.5 sqm)
- 1 Kitchen (min 4.5 sqm)
- 2 Bathrooms (each min 1.8 sqm)
- 1 DiningRoom (optional, min 9.5 sqm)

## Extraction Rules:

1. **rooms**: List of room types and counts
   - For "3BHK": Extract as 3 Bedrooms + 1 LivingRoom + 1 Kitchen + 2 Bathrooms
   - For "2BHK": Extract as 2 Bedrooms + 1 LivingRoom + 1 Kitchen + 1 Bathroom
   - Always include all mandatory rooms for the layout type

2. **adjacency**: Room relationships based on architectural best practices:
   - Bathroom should be near Bedroom
   - Kitchen should be near LivingRoom or DiningRoom
   - Bedrooms should have privacy (separation from public spaces)

3. **entrance_side**: North, South, East, or West (default: South)

4. **plot_type**: "5marla", "10marla", "20marla", or "custom"

5. **layout_type**: Inferred layout type ("1BHK", "2BHK", "3BHK", "4BHK", "STUDIO")

6. **style_hints**: Any style preferences

Valid room types: Bedroom, Kitchen, Bathroom, LivingRoom, DiningRoom, DrawingRoom, Garage, Store, Pantry, WC

## Response Format:

Respond ONLY with a valid JSON object:
{
  "layout_type": "3BHK",
  "rooms": [
    {"type": "Bedroom", "count": 3},
    {"type": "LivingRoom", "count": 1},
    {"type": "Kitchen", "count": 1},
    {"type": "Bathroom", "count": 2}
  ],
  "adjacency": [
    {"source": "Bathroom", "target": "Bedroom", "relation": "near_to"},
    {"source": "Kitchen", "target": "LivingRoom", "relation": "near_to"}
  ],
  "entrance_side": "South",
  "plot_type": "10marla",
  "style_hints": [],
  "intent": "new_design"
}

If user is requesting a correction, set intent to "correction" and populate corrections field.
"""

# Dynamic prompt with constraint summary (built at runtime)
def _build_spec_extraction_prompt() -> str:
    """Build the spec extraction prompt with dynamic constraint information."""
    base = SPEC_EXTRACTION_PROMPT_BASE

    # Add constraint summary if available
    if _HAS_CONSTRAINT_ANALYZER and get_constraint_summary:
        try:
            constraint_info = get_constraint_summary()
            return f"{base}\n\n## System Constraints:\n{constraint_info}"
        except Exception:
            pass

    return base

# For backward compatibility
SPEC_EXTRACTION_PROMPT = SPEC_EXTRACTION_PROMPT_BASE

EXPLANATION_PROMPT = """You are an AI assistant explaining architectural floor plan designs to users.

Given the design data below, provide a clear, friendly explanation that covers:
1. **Overview**: Brief summary of what was generated
2. **Room Layout**: How rooms are arranged and why
3. **Strengths**: What works well in this design
4. **Weaknesses**: Any limitations or trade-offs
5. **Ranking Reason**: Why this design is ranked at position {rank} out of {total}
6. **Suggestions**: What could be improved (if the user wants to request changes)

Be conversational and helpful. Use simple language. Mention specific rooms and their positions.
If there are compliance violations, explain them simply.

Design Data:
{design_data}

Metrics:
- Design Score: {score}
- Compliance Status: {compliance_status}
- Travel Distance: {travel_distance}m (max allowed: {max_travel}m)
- Room Coverage: {room_coverage}
- Violations: {violations}
"""

CORRECTION_PROMPT = """You are an AI assistant parsing user corrections for floor plans.

The user wants to modify design #{design_index}. Parse their request into actionable changes.

Current design has these rooms: {current_rooms}

User's correction request: "{user_request}"

Extract the changes as a JSON object:
{
  "understood": true,
  "changes": [
    {"type": "move_room", "room": "Kitchen", "direction": "right", "amount": "2m"},
    {"type": "resize_room", "room": "Bedroom_1", "dimension": "width", "change": "+1m"},
    {"type": "swap_rooms", "room_a": "Kitchen", "room_b": "DiningRoom"},
    {"type": "add_room", "room_type": "Bathroom", "near": "Bedroom_1"},
    {"type": "remove_room", "room": "Store"},
    {"type": "change_adjacency", "room_a": "Kitchen", "room_b": "LivingRoom", "relation": "adjacent_to"}
  ],
  "clarification_needed": null
}

If the request is ambiguous, set "understood" to false and explain in "clarification_needed".
"""

CONTEXT_AWARE_CHAT_PROMPT = """You are BlueprintGPT, an AI assistant specialized in architectural floor plan design.
You help users create and refine floor plans through natural conversation.

**Your Capabilities:**
- Generate floor plan layouts based on specifications (room counts, adjacency preferences)
- Support Indian BHK formats (2BHK, 3BHK) and standard room specifications
- Handle various plot sizes (5 marla, 10 marla, custom dimensions)
- Apply architectural best practices for room placement and flow
- Explain design decisions and trade-offs
- Accept corrections and modifications to generated designs

**Supported Room Types:**
- Bedroom, Kitchen, Bathroom, LivingRoom (Hall), DiningRoom, DrawingRoom, Garage, Store

**Response Guidelines:**
1. Be helpful and conversational - not just technical
2. If the user asks a question, ANSWER it thoroughly
3. If the user asks about room types, list the supported types clearly
4. If the user seems confused, offer guidance on how to use the system
5. If no designs exist yet, guide them on how to request one
6. Reference specific architectural concepts when relevant
7. Be specific about what you can and cannot do

**Current Context:**
- Session state: {state}
- Designs generated: {num_designs}
- Current specification: {spec_summary}
- Selected design: {selected_design}

**Conversation history:**
{history}

**User message:** {user_message}

Respond naturally and helpfully. If the user is asking about architectural design concepts, explain them clearly.
If they want to create a design, guide them on what information you need.
If they have questions about a generated design, explain it in detail."""


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Core Functions
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract a JSON object from text, handling markdown fences and nested braces."""
    # Strip markdown code fences if present
    stripped = re.sub(r'```(?:json)?\s*', '', text).strip().rstrip('`')

    # Try parsing the whole thing first (common case: response is pure JSON)
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass

    # Find all { positions and try parsing from each one
    for i, ch in enumerate(stripped):
        if ch == '{':
            # Walk forward tracking brace depth
            depth = 0
            for j in range(i, len(stripped)):
                if stripped[j] == '{':
                    depth += 1
                elif stripped[j] == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = stripped[i:j+1]
                        try:
                            return json.loads(candidate)
                        except (json.JSONDecodeError, ValueError):
                            break  # This opening brace didn't work, try next
    return None

def is_available() -> bool:
    """Check if Gemini is available and configured."""
    return _gemini_ready and GEMINI_ENABLED and not _gemini_disabled_reason


def classify_intent(
    user_message: str,
    context: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Classify user intent to route message appropriately.

    Returns:
        Dict with keys: intent, confidence, reason, design_keywords, question_type
    """
    context = context or {}
    state = context.get("state", "initial")
    has_designs = context.get("num_designs", 0) > 0

    # First try quick local classification for common patterns
    local_result = _quick_intent_classify(user_message, has_designs)
    if local_result and local_result.get("confidence", 0) >= 0.9:
        ProcessingLogger.logger.debug(f"Quick intent classification: {local_result['intent']} (confidence: {local_result['confidence']})")
        return local_result

    # Fall back to Gemini for complex cases
    if not is_available():
        result = _fallback_intent_classify(user_message, has_designs)
        ProcessingLogger.logger.debug(f"Fallback intent classification: {result['intent']} (confidence: {result['confidence']})")
        return result

    try:
        prompt = INTENT_CLASSIFICATION_PROMPT.format(
            user_message=user_message,
            state=state,
            has_designs=has_designs,
        )

        text = _generate_text(prompt, response_mime_type="application/json")
        if not text:
            return _fallback_intent_classify(user_message, has_designs)

        result = _extract_json_from_text(text)
        if result and "intent" in result:
            ProcessingLogger.logger.debug(f"Gemini intent classification: {result['intent']} (confidence: {result.get('confidence', 0)})")
            return result

        return _fallback_intent_classify(user_message, has_designs)

    except Exception as e:
        ProcessingLogger.logger.warning(f"Gemini intent classification failed: {e}")
        return _fallback_intent_classify(user_message, has_designs)


def _quick_intent_classify(user_message: str, has_designs: bool) -> Optional[Dict[str, Any]]:
    """Quick local intent classification for obvious patterns."""
    msg_lower = user_message.lower().strip()

    # Obvious greetings
    greeting_patterns = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "thanks", "thank you", "bye", "goodbye"]
    if msg_lower in greeting_patterns or any(msg_lower.startswith(g + " ") or msg_lower.startswith(g + ",") or msg_lower.startswith(g + "!") for g in greeting_patterns):
        return {
            "intent": INTENT_CONVERSATION,
            "confidence": 0.95,
            "reason": "Greeting or social message detected",
            "design_keywords": [],
            "question_type": None
        }

    # Obvious questions - check FIRST before design indicators
    question_starters = ["what", "why", "how", "can you", "could you", "will you", "is it", "are there", "does", "do you", "which", "where", "when", "who"]
    is_question = msg_lower.endswith("?") or any(msg_lower.startswith(q) for q in question_starters)

    # Strong design indicators - words that strongly suggest creating something new
    design_action_words = [
        r'\bcreate\b', r'\bgenerate\b', r'\bbuild\b', r'\bdesign\s+a\b', r'\bmake\s+a\b',
        r'\bi\s+need\b', r'\bi\s+want\b', r'\bplan\s+for\b'
    ]
    design_spec_patterns = [
        r'\d+\s*bhk', r'\d+\s*bedroom', r'\d+\s*bathroom',
        r'marla', r'floor\s*plan', r'layout',
        r'entrance\s+(on|from)', r'north\s+entrance', r'south\s+entrance'
    ]

    has_design_action = any(re.search(pattern, msg_lower) for pattern in design_action_words)
    has_design_spec = any(re.search(pattern, msg_lower) for pattern in design_spec_patterns)

    # Correction indicators (only if designs exist)
    correction_patterns = [
        r'\bmove\s+(the\s+)?(\w+)', r'\bshift\s+(the\s+)?(\w+)',
        r'\bmake\s+(the\s+)?(\w+)\s+(larger|bigger|smaller)',
        r'\bswap\b', r'\bchange\s+(the\s+)?position',
        r'\bmodify\b', r'\badjust\b', r'\bresize\b',
        r'\bshould\s+be\s+(adjacent|next\s+to|near|close)\b',
        r'\bto\s+be\s+(adjacent|next\s+to|near|close)\b',
        r'\bkeep\s+.+\s+(adjacent|next\s+to|near|close)\b',
        r'\bplace\s+.+\s+(near|next\s+to|beside|adjacent\s+to)\b',
        r'\bcloser\s+to\b',
        r'design\s*#?\d+', r'the\s+(first|second|third)\s+(design|one)',
        r'\badd\s+a\s+', r'\bremove\s+(the\s+)?'
    ]

    correction_found = has_designs and any(re.search(pattern, msg_lower) for pattern in correction_patterns)

    # If it's a question AND doesn't have strong design action words, it's a question
    if is_question and not has_design_action:
        # Determine question type
        question_type = "general"
        if any(word in msg_lower for word in ["can you", "able to", "possible", "support"]):
            question_type = "capability"
        elif any(word in msg_lower for word in ["why", "explain", "what is", "how does"]):
            question_type = "explanation"
        elif any(word in msg_lower for word in ["mean", "understand", "clarify"]):
            question_type = "clarification"

        return {
            "intent": INTENT_QUESTION,
            "confidence": 0.88,
            "reason": "Question detected",
            "design_keywords": [],
            "question_type": question_type
        }

    # Check for corrections before design (corrections take precedence if designs exist)
    if correction_found:
        return {
            "intent": INTENT_CORRECTION,
            "confidence": 0.9,
            "reason": "Correction request for existing design",
            "design_keywords": [],
            "question_type": None
        }

    # Now check for design intent
    if has_design_action or has_design_spec:
        keywords = []
        for pattern in design_action_words + design_spec_patterns:
            match = re.search(pattern, msg_lower)
            if match:
                keywords.append(match.group(0))
        return {
            "intent": INTENT_DESIGN,
            "confidence": 0.92,
            "reason": f"Design keywords detected: {', '.join(keywords[:3])}",
            "design_keywords": keywords,
            "question_type": None
        }

    # Not confident enough for quick classification
    return None


def _fallback_intent_classify(user_message: str, has_designs: bool) -> Dict[str, Any]:
    """Fallback intent classification when Gemini unavailable."""
    msg_lower = user_message.lower().strip()

    # Check if it's a question FIRST (questions take priority over keyword matching)
    is_question = "?" in user_message or any(
        msg_lower.startswith(q) for q in ["what", "why", "how", "can", "could", "would", "is", "are", "does", "do"]
    )

    # Check for design-related keywords
    design_keywords = []
    design_patterns = {
        "bhk": r'\d+\s*bhk',
        "bedroom": r'\d*\s*bedroom',
        "bathroom": r'\d*\s*bathroom',
        "kitchen": r'kitchen',
        "living": r'living\s*room',
        "marla": r'\d+\s*marla',
        "floor plan": r'floor\s*plan',
        "layout": r'layout',
    }

    for key, pattern in design_patterns.items():
        if re.search(pattern, msg_lower):
            design_keywords.append(key)

    correction_found = has_designs and any(
        re.search(pattern, msg_lower)
        for pattern in [
            r'\bmove\s+(the\s+)?(\w+)', r'\bshift\s+(the\s+)?(\w+)',
            r'\bmake\s+(the\s+)?(\w+)\s+(larger|bigger|smaller)',
            r'\bswap\b', r'\bchange\s+(the\s+)?position',
            r'\bmodify\b', r'\badjust\b', r'\bresize\b',
            r'\bshould\s+be\s+(adjacent|next\s+to|near|close)\b',
            r'\bto\s+be\s+(adjacent|next\s+to|near|close)\b',
            r'\bkeep\s+.+\s+(adjacent|next\s+to|near|close)\b',
            r'\bplace\s+.+\s+(near|next\s+to|beside|adjacent\s+to)\b',
            r'\bcloser\s+to\b',
            r'design\s*#?\d+', r'the\s+(first|second|third)\s+(design|one)',
            r'\badd\s+a\s+', r'\bremove\s+(the\s+)?'
        ]
    )

    # Determine intent - prioritize questions over design keywords
    if is_question:
        # It's a question - classify as QUESTION even if it contains design keywords
        intent = INTENT_QUESTION
        confidence = 0.8
    elif correction_found:
        intent = INTENT_CORRECTION
        confidence = 0.8
    elif design_keywords:
        intent = INTENT_DESIGN
        confidence = min(0.7 + len(design_keywords) * 0.05, 0.9)
    else:
        intent = INTENT_CONVERSATION
        confidence = 0.6

    return {
        "intent": intent,
        "confidence": confidence,
        "reason": f"Fallback classification - {'question' if is_question else 'keyword analysis'}",
        "design_keywords": design_keywords,
        "question_type": "general" if intent == INTENT_QUESTION else None
    }


def process_message(
    user_message: str,
    context: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for processing user messages with intent-aware routing.

    Returns:
        Dict containing:
        - intent: classified intent type
        - response: appropriate response based on intent
        - spec: extracted spec if intent is design (None otherwise)
        - correction: parsed correction if intent is correction (None otherwise)
    """
    context = context or {}

    ProcessingLogger.logger.debug(f"Starting NL processing for: '{user_message[:100]}{'...' if len(user_message) > 100 else ''}'")
    ProcessingLogger.logger.debug(f"Context: state={context.get('state')}, designs={context.get('num_designs', 0)}")

    # Step 1: Classify intent
    intent_result = classify_intent(user_message, context, conversation_history)
    intent = intent_result.get("intent", INTENT_CONVERSATION)

    # Log intent classification using ProcessingLogger
    ProcessingLogger.log_intent_classification(
        intent=intent,
        confidence=intent_result.get('confidence', 0),
        reason=intent_result.get('reason', ''),
        design_keywords=intent_result.get('design_keywords'),
        question_type=intent_result.get('question_type')
    )

    result = {
        "intent": intent,
        "intent_confidence": intent_result.get("confidence", 0),
        "intent_reason": intent_result.get("reason", ""),
        "response": "",
        "spec": None,
        "correction": None,
        "should_generate": False,
    }

    # Step 2: Route based on intent
    if intent == INTENT_DESIGN:
        ProcessingLogger.logger.info("Extracting design specifications from input")

        # Extract specification for design generation
        spec = extract_spec_from_nl(user_message, conversation_history)
        result["spec"] = spec
        result["should_generate"] = True

        # Log extracted specification details
        rooms = spec.get("rooms", [])
        total_rooms = sum(r.get("count", 1) for r in rooms)
        ProcessingLogger.log_spec_extraction(
            rooms=rooms,
            total_rooms=total_rooms,
            plot_type=spec.get('plot_type'),
            entrance_side=spec.get('entrance_side'),
            adjacency=spec.get('adjacency')
        )

        # Generate a helpful response about what we understood
        if rooms:
            room_summary = ", ".join([f"{r.get('count', 1)} {r.get('type')}" for r in rooms])
            result["response"] = f"I'll create a floor plan with {room_summary}. "
            if spec.get("plot_type"):
                result["response"] += f"Plot type: {spec['plot_type']}. "
            if spec.get("entrance_side"):
                result["response"] += f"Entrance: {spec['entrance_side']}. "
            result["response"] += "Let me generate some design options for you."
        else:
            result["response"] = "I detected you want to create a design, but I need more details. Please specify the rooms you need (e.g., '3BHK' or '2 bedrooms, 1 kitchen, 2 bathrooms')."
            result["should_generate"] = False
            ProcessingLogger.logger.warning("No rooms extracted - design generation cancelled")

    elif intent == INTENT_CORRECTION:
        # Parse correction for existing design
        selected_design = context.get("selected_design")
        current_rooms = context.get("current_rooms", [])
        correction = parse_correction(user_message, selected_design or 0, current_rooms)
        result["correction"] = correction

        if correction.get("understood"):
            changes = correction.get("changes", [])
            change_summary = ", ".join([f"{c.get('type')}: {c.get('room', c.get('room_type', ''))}" for c in changes[:3]])
            result["response"] = f"I understand you want to: {change_summary}. I'll apply these changes to your design."
        else:
            result["response"] = correction.get("clarification_needed", "I couldn't understand the correction. Could you be more specific about what you'd like to change?")

    elif intent == INTENT_QUESTION:
        # Generate helpful answer
        result["response"] = chat_response(user_message, context, conversation_history)

    else:  # INTENT_CONVERSATION
        # General conversational response
        result["response"] = chat_response(user_message, context, conversation_history)

    # Log processing summary
    ProcessingLogger.log_processing_summary(
        intent=intent,
        should_generate=result['should_generate'],
        response_length=len(result.get('response', '')),
        spec_extracted=bool(result.get('spec')),
        correction_parsed=bool(result.get('correction'))
    )

    return result


def extract_spec_from_nl(user_text: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Extract structured specification from natural language using Gemini.

    Falls back to regex-based extraction if Gemini is unavailable.
    Applies constraint-aware enhancement to ensure proper model inputs.
    """
    ProcessingLogger.logger.debug(f"extract_spec_from_nl called with text: '{user_text[:80]}...'")
    ProcessingLogger.logger.debug(f"Gemini available: {is_available()}")

    if not is_available():
        ProcessingLogger.logger.debug("Using fallback extraction (Gemini not available)")
        result = _fallback_extract(user_text)
        result = _apply_constraint_enhancement(result)
        ProcessingLogger.logger.debug(f"Fallback result: {result}")
        return result

    try:
        ProcessingLogger.logger.debug(f"Calling Gemini API with model: {GEMINI_MODEL}")

        # Build context with conversation history
        messages = []
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                messages.append(f"{msg['role']}: {msg['content']}")

        context = "\n".join(messages) if messages else ""

        # Use constraint-aware prompt
        spec_prompt = _build_spec_extraction_prompt()
        full_prompt = f"{spec_prompt}\n\nConversation context:\n{context}\n\nUser message: {user_text}"

        text = _generate_text(full_prompt, response_mime_type="application/json")
        if not text:
            ProcessingLogger.logger.debug("Gemini returned empty response, using fallback")
            result = _fallback_extract(user_text)
            result = _apply_constraint_enhancement(result)
            return result
        text = text.strip()

        ProcessingLogger.logger.debug(f"Gemini response (first 200 chars): {text[:200]}")

        # Extract JSON from response
        result = _extract_json_from_text(text)
        if result:
            ProcessingLogger.logger.debug(f"Gemini extracted JSON: {result}")
            # Apply constraint enhancement
            result = _apply_constraint_enhancement(result)
            ProcessingLogger.logger.debug(f"Constraint-enhanced result: {result}")
            return result

        ProcessingLogger.logger.debug("No JSON found in Gemini response, using fallback")
        result = _fallback_extract(user_text)
        result = _apply_constraint_enhancement(result)
        ProcessingLogger.logger.debug(f"Fallback result: {result}")
        return result

    except Exception as e:
        ProcessingLogger.logger.warning(f"Gemini extraction failed: {e}")
        result = _fallback_extract(user_text)
        result = _apply_constraint_enhancement(result)
        ProcessingLogger.logger.debug(f"Fallback result: {result}")
        return result


def _apply_constraint_enhancement(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply constraint-aware enhancement to a specification.

    This ensures that:
    1. All room types have proper minimum dimensions
    2. Layout type requirements are satisfied
    3. Adjacency requirements are included
    4. Boundary dimensions are calculated if not provided
    """
    if not _HAS_CONSTRAINT_ANALYZER or not enhance_spec_with_constraints:
        ProcessingLogger.logger.debug("Constraint analyzer not available, skipping enhancement")
        return spec

    try:
        enhanced = enhance_spec_with_constraints(spec)

        # Log comprehensive constraint analysis
        layout_type = enhanced.get('layout_type', 'unknown')
        rooms = enhanced.get('rooms', [])
        metadata = enhanced.get('constraint_metadata', {})
        auto_dims = enhanced.get('auto_dimensions', {})

        ProcessingLogger.log_constraint_analysis(
            layout_type=layout_type,
            room_count=len(rooms),
            min_area=metadata.get('min_total_area_sqm', 0),
            recommended_area=metadata.get('recommended_area_sqm', 0) if 'recommended_area_sqm' in metadata else metadata.get('min_total_area_sqm', 0) * 1.3,
            plot_bucket=metadata.get('plot_bucket', 'unknown'),
            auto_dimensions=(auto_dims.get('width_m'), auto_dims.get('height_m')) if auto_dims else None
        )

        # Log room requirements if debug enabled
        ProcessingLogger.log_room_requirements(rooms)

        # Log adjacency requirements
        adjacencies = enhanced.get('adjacency', [])
        ProcessingLogger.log_adjacency_requirements(adjacencies)

        return enhanced
    except Exception as e:
        ProcessingLogger.logger.warning(f"Constraint enhancement failed: {e}")
        return spec


def explain_design(
    design_data: Dict,
    rank: int,
    total_designs: int,
    metrics: Optional[Dict] = None,
) -> str:
    """
    Generate a natural language explanation for a design.

    Falls back to template-based explanation if Gemini is unavailable.
    """
    if not is_available():
        return _fallback_explain(design_data, rank, total_designs, metrics)

    try:
        score = metrics.get("design_score", 0) if metrics else 0
        compliance = metrics.get("compliance_status", "Unknown") if metrics else "Unknown"
        travel = metrics.get("travel_distance", "N/A") if metrics else "N/A"
        max_travel = metrics.get("max_travel_distance", "N/A") if metrics else "N/A"
        coverage = metrics.get("room_coverage", {}) if metrics else {}
        violations = metrics.get("violations", []) if metrics else []

        prompt = EXPLANATION_PROMPT.format(
            rank=rank,
            total=total_designs,
            design_data=json.dumps(design_data, indent=2, default=str),
            score=score,
            compliance_status=compliance,
            travel_distance=travel,
            max_travel=max_travel,
            room_coverage=json.dumps(coverage),
            violations=json.dumps(violations),
        )

        text = _generate_text(prompt)
        if not text:
            return _fallback_explain(design_data, rank, total_designs, metrics)
        return text.strip()

    except Exception as e:
        ProcessingLogger.logger.warning(f"Gemini explanation failed: {e}")
        return _fallback_explain(design_data, rank, total_designs, metrics)


def parse_correction(
    user_request: str,
    design_index: int,
    current_rooms: List[Dict],
) -> Dict[str, Any]:
    """
    Parse user's correction request into actionable changes.

    Falls back to simple keyword matching if Gemini is unavailable.
    """
    if not is_available():
        return _fallback_parse_correction(user_request, design_index, current_rooms)

    try:
        rooms_str = ", ".join([r.get("name", r.get("type", "Unknown")) for r in current_rooms])

        prompt = CORRECTION_PROMPT.format(
            design_index=design_index,
            current_rooms=rooms_str,
            user_request=user_request,
        )

        text = _generate_text(prompt, response_mime_type="application/json")
        if not text:
            return _fallback_parse_correction(user_request, design_index, current_rooms)
        text = text.strip()

        # Extract JSON from response
        parsed = _extract_json_from_text(text)
        if parsed:
            return parsed

        return _fallback_parse_correction(user_request, design_index, current_rooms)

    except Exception as e:
        ProcessingLogger.logger.warning(f"Gemini correction parsing failed: {e}")
        return _fallback_parse_correction(user_request, design_index, current_rooms)


def chat_response(
    user_message: str,
    context: Dict[str, Any],
    conversation_history: Optional[List[Dict]] = None,
) -> str:
    """
    Generate a context-aware conversational response about the design process.
    Enhanced to properly answer questions and provide helpful guidance.
    """
    if not is_available():
        return _fallback_chat(user_message, context, conversation_history)

    try:
        # Build conversation history
        history = ""
        if conversation_history:
            for msg in conversation_history[-10:]:
                role = msg.get('role', 'user').upper()
                content = msg.get('content', '')[:200]  # Truncate long messages
                history += f"\n{role}: {content}"

        # Build spec summary
        spec = context.get('spec', {})
        rooms = spec.get('rooms', [])
        spec_summary = "No specification yet"
        if rooms:
            room_list = ", ".join([f"{r.get('count', 1)} {r.get('type')}" for r in rooms])
            spec_summary = f"Rooms: {room_list}"
            if spec.get('plot_type'):
                spec_summary += f", Plot: {spec['plot_type']}"
            if spec.get('entrance_side'):
                spec_summary += f", Entrance: {spec['entrance_side']}"

        # Build selected design info
        selected = context.get('selected_design')
        selected_info = f"Design #{selected + 1}" if selected is not None else "None selected"

        prompt = CONTEXT_AWARE_CHAT_PROMPT.format(
            state=context.get('state', 'initial'),
            num_designs=context.get('num_designs', 0),
            spec_summary=spec_summary,
            selected_design=selected_info,
            history=history if history else "No previous messages",
            user_message=user_message,
        )

        text = _generate_text(prompt)
        if not text:
            return _fallback_chat(user_message, context, conversation_history)
        return text.strip()

    except Exception as e:
        ProcessingLogger.logger.warning(f"Gemini chat failed: {e}")
        return _fallback_chat(user_message, context, conversation_history)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Fallback Functions (when Gemini unavailable)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _canonical_room_from_label(label: str, current_rooms: Optional[List[Dict]] = None) -> str:
    """Convert a room label like 'living room' or 'bedroom1' to a canonical room type or room name."""
    from nl_interface.constants import ROOM_LABELS
    label_lower = label.lower().strip()
    compact = re.sub(r"[\s_]+", "", label_lower)

    for room in current_rooms or []:
        room_name = str(room.get("name") or "").strip()
        room_type = str(room.get("type") or "").strip()
        if not room_name and not room_type:
            continue
        aliases = {
            room_name.lower(),
            re.sub(r"[\s_]+", "", room_name.lower()),
            room_type.lower(),
            re.sub(r"[\s_]+", "", room_type.lower()),
        }
        if label_lower in aliases or compact in aliases:
            return room_name or room_type

    for canonical, labels in ROOM_LABELS.items():
        normalized_labels = {re.sub(r"[\s_]+", "", value.lower()) for value in labels}
        if (
            label_lower in labels
            or label_lower == canonical.lower()
            or compact in normalized_labels
            or compact == re.sub(r"[\s_]+", "", canonical.lower())
        ):
            return canonical
    return label.capitalize()


def _fallback_extract(user_text: str) -> Dict[str, Any]:
    """Simple regex-based extraction when Gemini is unavailable."""
    ProcessingLogger.logger.debug(f"Starting fallback extraction for: '{user_text[:80]}...'")
    from nl_interface.service import _extract_from_text

    extracted = _extract_from_text(user_text)
    ProcessingLogger.logger.debug(f"Raw extraction - rooms: {extracted.get('rooms', [])}")
    ProcessingLogger.logger.debug(f"Raw extraction - plot_type: {extracted.get('plot_type')}, entrance_side: {extracted.get('entrance_side')}")

    # rooms is already a list of dicts [{"type": ..., "count": ...}]
    rooms = extracted.get("rooms", [])

    adjacency = []
    for src, tgt, rel in extracted.get("preferences", {}).get("adjacency", []):
        adjacency.append({"source": src, "target": tgt, "relation": rel})

    # Extract style hints from weights and interpretation notes
    style_hints = []
    weights = extracted.get("weights", {})
    if weights:
        if weights.get("compactness", 0) > 0.4:
            style_hints.append("compact")
        if weights.get("privacy", 0) > 0.4:
            style_hints.append("private")
    notes = extracted.get("interpretation_notes", [])
    for note in notes:
        note_lower = note.lower()
        if "compact" in note_lower:
            style_hints.append("compact")
        if "open" in note_lower:
            style_hints.append("open")
        if "privacy" in note_lower or "private" in note_lower:
            style_hints.append("private")

    result = {
        "rooms": rooms,
        "adjacency": adjacency,
        "entrance_side": extracted.get("entrance_side"),
        "plot_type": extracted.get("plot_type"),
        "style_hints": list(set(style_hints)),  # deduplicate
        "corrections": None,
        "intent": "new_design",
    }
    ProcessingLogger.logger.debug(f"Fallback result: rooms={result['rooms']}, plot_type={result['plot_type']}, entrance_side={result['entrance_side']}")
    return result


def _fallback_explain(
    design_data: Dict,
    rank: int,
    total_designs: int,
    metrics: Optional[Dict] = None,
) -> str:
    """Template-based explanation when Gemini is unavailable."""
    metrics = metrics or {}

    rooms = design_data.get("generated_rooms", {})
    room_list = ", ".join([f"{count} {rtype}" for rtype, count in rooms.items()])

    score = metrics.get("design_score", 0)
    violations = metrics.get("violations", [])
    compliance = "COMPLIANT" if not violations else "NON_COMPLIANT"

    explanation = f"""## Design #{rank} of {total_designs}

**Rooms Generated:** {room_list or 'None'}

**Design Score:** {score:.2f}/1.0

**Compliance Status:** {compliance}

"""

    if rank == 1:
        explanation += "**Why #1:** This design scored highest based on room arrangement, compliance, and adjacency satisfaction.\n\n"
    else:
        explanation += f"**Ranking:** This design ranked #{rank} due to "
        if violations:
            explanation += f"compliance issues: {', '.join(violations[:3])}\n\n"
        else:
            explanation += "lower overall optimization score.\n\n"

    if violations:
        explanation += f"**Issues to Address:**\n"
        for v in violations[:5]:
            explanation += f"- {v}\n"
        explanation += "\n"

    explanation += "**To request changes:** Tell me what you'd like to modify (e.g., 'make the kitchen larger', 'move bedroom away from entrance')."

    return explanation


def _fallback_parse_correction(
    user_request: str,
    design_index: int,
    current_rooms: List[Dict],
) -> Dict[str, Any]:
    """Simple keyword-based correction parsing when Gemini is unavailable."""
    changes = []
    user_lower = user_request.lower()

    # Build room name pattern from known room labels
    from nl_interface.constants import ROOM_LABELS
    all_labels = sorted(
        {label for labels in ROOM_LABELS.values() for label in labels},
        key=len, reverse=True,
    )
    room_pat = "|".join(re.escape(l) for l in all_labels)
    # Also match single-word capitalized names as fallback
    room_or_word = rf"(?:{room_pat}(?:\s*\d+)?|\w+)"
    article = r"(?:the|a|an)\s+"

    # Detect move requests: "move (the) kitchen left"
    move_patterns = [
        (rf"move\s+{article}?({room_or_word})\s+(left|right|up|down)", "move_room"),
        (rf"shift\s+{article}?({room_or_word})\s+(left|right|up|down)", "move_room"),
    ]
    for pattern, change_type in move_patterns:
        match = re.search(pattern, user_lower)
        if match:
            room_name = _canonical_room_from_label(match.group(1), current_rooms)
            changes.append({
                "type": change_type,
                "room": room_name,
                "direction": match.group(2),
            })

    # Detect resize requests: "make (the) kitchen larger"
    resize_patterns = [
        (rf"make\s+{article}?({room_or_word})\s+(larger|bigger|smaller|bigger|wider|narrower)", "resize_room"),
        (rf"(increase|decrease)\s+{article}?({room_or_word})\s+size", "resize_room"),
    ]
    for pattern, change_type in resize_patterns:
        match = re.search(pattern, user_lower)
        if match:
            if "make" in pattern:
                room_name = _canonical_room_from_label(match.group(1), current_rooms)
            else:
                room_name = _canonical_room_from_label(match.group(2), current_rooms)
            size_dir = "larger" if any(w in user_lower for w in ("larger", "bigger", "increase", "wider")) else "smaller"
            changes.append({
                "type": change_type,
                "room": room_name,
                "size_change": size_dir,
            })

    adjacency_patterns = [
        rf"({room_or_word})\s+(?:and|with)\s+({room_or_word})\s+should\s+be\s+(adjacent|next to|near|close)",
        rf"({room_or_word})\s+(?:and|with)\s+({room_or_word})\s+to\s+be\s+(adjacent|next to|near|close)",
        rf"keep\s+({room_or_word})\s+(?:and|with)\s+({room_or_word})\s+(adjacent|near|close)",
        rf"place\s+({room_or_word})\s+(?:near|next to|beside|adjacent to)\s+({room_or_word})",
    ]
    for pattern in adjacency_patterns:
        match = re.search(pattern, user_lower)
        if not match:
            continue
        room_a = _canonical_room_from_label(match.group(1), current_rooms)
        room_b = _canonical_room_from_label(match.group(2), current_rooms)
        changes.append({
            "type": "change_adjacency",
            "room_a": room_a,
            "room_b": room_b,
            "relation": "adjacent_to",
        })
        break

    # Detect swap requests: "swap kitchen and bedroom"
    swap_match = re.search(rf"swap\s+{article}?({room_or_word})\s+(?:and|with)\s+{article}?({room_or_word})", user_lower)
    if swap_match:
        changes.append({
            "type": "swap_rooms",
            "room_a": _canonical_room_from_label(swap_match.group(1), current_rooms),
            "room_b": _canonical_room_from_label(swap_match.group(2), current_rooms),
        })

    # Detect add/remove
    add_match = re.search(rf"add\s+{article}?({room_or_word})", user_lower)
    if add_match:
        changes.append({"type": "add_room", "room_type": _canonical_room_from_label(add_match.group(1), current_rooms)})

    remove_match = re.search(rf"remove\s+{article}?({room_or_word})", user_lower)
    if remove_match:
        changes.append({"type": "remove_room", "room": _canonical_room_from_label(remove_match.group(1), current_rooms)})

    return {
        "understood": len(changes) > 0,
        "changes": changes,
        "clarification_needed": None if changes else "I couldn't understand your correction request. Please be more specific about what you'd like to change.",
    }


def _summarize_room_program(rooms: List[Dict]) -> str:
    parts = []
    for room in rooms or []:
        room_type = room.get("type") or room.get("name") or "Room"
        count = int(room.get("count", 1) or 1)
        parts.append(f"{count} {room_type}")
    return ", ".join(parts)



def _latest_design_summary(context: Dict[str, Any]) -> Tuple[Dict[str, Any], str, Optional[str], Optional[str]]:
    latest_design = context.get("latest_design") or {}
    latest_rooms = latest_design.get("rooms") or []
    program = _summarize_room_program(latest_rooms)
    if not program:
        spec = context.get("spec", {}) or {}
        program = _summarize_room_program(spec.get("rooms", []))
    return latest_design, program, latest_design.get("engine"), latest_design.get("report_status")



def _fallback_chat(
    user_message: str,
    context: Dict[str, Any],
    conversation_history: Optional[List[Dict]] = None,
) -> str:
    """Natural fallback chat when Gemini is unavailable."""
    state = context.get("state", "initial")
    num_designs = context.get("num_designs", 0)
    msg_lower = user_message.lower().strip()
    spec = context.get("spec", {}) or {}
    latest_design, design_program, design_engine, design_status = _latest_design_summary(context)
    previous_user_messages = [
        msg.get("content", "")
        for msg in (conversation_history or [])
        if msg.get("role") == "user" and msg.get("content")
    ]
    previous_context = previous_user_messages[-2] if len(previous_user_messages) >= 2 else None
    msg_words = set(re.findall(r"[a-z]+", msg_lower))
    has_greeting = bool({"hi", "hello", "hey"} & msg_words) or any(
        phrase in msg_lower for phrase in ["good morning", "good afternoon"]
    )

    if has_greeting:
        if design_program:
            return f"I still have your latest plan in context: {design_program}. You can change it, ask why something happened, or tell me to generate a revised version."
        return "I’m ready. Tell me the home you want, including the room program, area or plot size, entrance side, and any adjacency preferences."

    if any(t in msg_lower for t in ["thank", "thanks", "appreciate"]):
        return "We can keep going from the same plan. If you want, tell me the next change and I’ll carry the current context forward."

    if any(term in msg_lower for term in ["encoder", "decoder", "architecture", "planner", "how does this work", "how do you generate", "model pipeline"]):
        return (
            "The system is split into stages. First, the natural-language side acts like an encoder: it reads your prompt, resolves the room program, area, entrance, and adjacency intent into a structured spec. "
            "Second, a planner predicts room relationships and rough placement intent. Third, the geometry stage acts like a decoder: it turns that structured plan into room shapes and doors. "
            "After that, verification and repair check overlap, connectivity, area compliance, and layout quality before anything is returned. "
            "So the encoder/decoder split is there conceptually and in runtime flow, even though the current repo still mixes learned and deterministic components in the final generation path."
        )

    if num_designs > 0 and any(term in msg_lower for term in ["can i make", "can i change", "make some changes", "modify this", "change this", "refine this", "edit this"]):
        base = f"Yes. We can keep working on this {design_program or 'plan'}"
        if design_engine:
            base += f" from the `{design_engine}` path"
        base += ". Tell me exactly what to change."
        return base + " For example: `make the living room larger`, `move the kitchen closer to the living room`, `keep bedrooms away from the entrance`, or `regenerate with less corridor area`."

    if any(term in msg_lower for term in ["overlap", "overlapped", "overlapping", "intersect", "clash", "collide"]):
        detail = "That means the previous layout geometry was not clean enough to trust as a final residential plan."
        if design_engine:
            detail += f" In your case, the last design came from `{design_engine}`."
        if design_status:
            detail += f" Its current status is `{design_status}`."
        return (
            detail
            + " The right response is not to ignore it. The system should either repair and regenerate the layout or reject it and explain why. "
            + "If you want to continue from the same request, I can do one of three things next: regenerate with stricter no-overlap checks, keep the same program but relax adjacency pressure, or shift to a more conservative layout strategy."
        )

    if any(q in msg_lower for q in ["room type", "room types", "which room", "what room", "support"]):
        return (
            "For residential plans, I support Bedroom, Kitchen, Bathroom, LivingRoom, DiningRoom, DrawingRoom, Garage, Store, Study, WC, and Balcony. "
            "You can specify them directly, or use formats like `2BHK` and `3BHK`, and I’ll expand them into the full room program."
        )

    if any(q in msg_lower for q in ["what can you", "can you", "able to", "capabilities", "what do you do"]):
        return (
            "I can interpret a floor-plan request, keep context across turns, generate or revise layouts, explain why a layout passed or failed, and translate high-level requests like privacy or adjacency into planning rules. "
            "If you already have a design on screen, you can now ask for targeted changes without restating the full brief."
        )

    if "how" in msg_lower and ("work" in msg_lower or "use" in msg_lower):
        return (
            "Use it as a conversation. Start with the room program, then add area or plot size, entrance side, and any adjacency or privacy preferences. "
            "After a layout is generated, you can refine it with follow-ups like `make the kitchen larger`, `reduce corridor area`, or `keep bedrooms farther from the entrance`."
        )

    if state == "initial":
        return "I’m ready. Describe the home you want, for example: `3BHK with 180 sq.m and north entrance`, or `2-bedroom home with kitchen near living room`."

    if state == "specifying":
        if design_program:
            reply = f"I’m tracking the current brief as {design_program}."
            if previous_context:
                reply += f" Earlier, you said: `{previous_context}`."
            reply += " If you want, I can generate now, or you can still refine the area, entrance, privacy, or room relationships."
            return reply
        return "Tell me the room program first, such as `2BHK`, `3 bedrooms with 2 bathrooms`, or `studio with open kitchen`."

    if state == "generated" or num_designs > 0:
        summary = f"I still have your latest design in context"
        if design_program:
            summary += f": {design_program}"
        if design_engine:
            summary += f", generated through `{design_engine}`"
        if design_status:
            summary += f", with status `{design_status}`"
        summary += "."
        return summary + " Ask for a change, ask why a rule was applied, or tell me to regenerate with a different priority."

    if state == "correcting":
        return "I’m ready to modify the current layout. Tell me the change in plain language, and I’ll keep the rest of the design context intact."

    return "I’m still on the same floor-plan conversation. Tell me the next change, the question you want answered, or the new constraint you want me to apply."

