# Intent Classification Bug Fix

**Date:** 2026-03-20
**Status:** ✅ FIXED

## Problem

When users asked questions about the system (e.g., "What room types do you support?"), BlueprintGPT was returning **technical state dumps** instead of friendly conversational answers.

### Example of Bad Response:
```
Current Spec captures no rooms yet for a Residential plan. The current trade-off leans
toward privacy, with corridor minimization set to False. The backend target is not
selected until the room program is clear. Backend translation remains blocked until
actual boundary geometry is supplied.
```

### What Users Expected:
```
Supported Room Types:
- Bedroom - Main living quarters
- Kitchen - Cooking and food prep area
- Bathroom - Washing and sanitation facilities
...
```

---

## Root Cause Analysis

### The Bug 🐛
The `_fallback_intent_classify()` function in `nl_interface/gemini_adapter.py` was checking for **design keywords BEFORE checking if it's a question**.

**Problematic Logic (Lines 407-452):**
```python
# OLD (BUGGY) ORDER:
1. Check for design keywords (bedroom, kitchen, living room, etc.)
2. Check if it's a question (?, what, why, how, etc.)

# Result: "What room types do you support?" contains "room" → misclassified as DESIGN
```

When a message contained words like "room", "bedroom", "kitchen", etc., it was immediately classified as a DESIGN intent, even if the message was clearly a question.

### Why This Caused Technical State Dumps

1. User asks: "What room types do you support?"
2. Intent classifier sees "room" keyword → classifies as `INTENT_DESIGN`
3. Server calls `process_user_request()` from `nl_interface/service.py`
4. That function returns `_build_assistant_text()` which generates technical state information
5. User sees confusing technical response instead of a helpful answer

---

## The Fix ✅

**File:** `nl_interface/gemini_adapter.py`

### Changed Function: `_fallback_intent_classify()` (Line 407)

**Fix:** Reordered the intent classification logic to **prioritize questions over keyword matching**.

```python
# NEW (CORRECT) ORDER:
1. Check if it's a question FIRST (?, what, why, how, etc.)
2. Then check for design keywords only if NOT a question

# Result: "What room types do you support?" starts with "what" → correctly classified as QUESTION
```

### Code Changes:
```python
def _fallback_intent_classify(user_message: str, has_designs: bool) -> Dict[str, Any]:
    """Fallback intent classification when Gemini unavailable."""
    msg_lower = user_message.lower().strip()

    # ✅ Check if it's a question FIRST (questions take priority)
    is_question = "?" in user_message or any(
        msg_lower.startswith(q) for q in ["what", "why", "how", "can", ...]
    )

    # Check for design keywords
    design_keywords = []
    design_patterns = {...}
    for key, pattern in design_patterns.items():
        if re.search(pattern, msg_lower):
            design_keywords.append(key)

    # ✅ Determine intent - prioritize questions over design keywords
    if is_question:
        intent = INTENT_QUESTION  # ← Questions take priority!
        confidence = 0.8
    elif design_keywords:
        intent = INTENT_DESIGN
        confidence = min(0.7 + len(design_keywords) * 0.05, 0.9)
    ...
```

---

## Enhanced Room Type Support

**File:** `nl_interface/gemini_adapter.py`

### Added to `_fallback_chat()` Function (Line ~940)

Added a specific handler for questions about room types:

```python
# Handle questions about room types / supported rooms
if any(q in msg_lower for q in ["room type", "room types", "which room", "what room", "support"]):
    return """**Supported Room Types:**

BlueprintGPT supports these room types for residential floor plans:
- **Bedroom** - Main living quarters
- **Kitchen** - Cooking and food prep area
- **Bathroom** - Washing and sanitation facilities
- **LivingRoom** (Hall) - Main gathering space
- **DiningRoom** - Eating area
- **DrawingRoom** - Formal reception area
- **Garage** - Vehicle parking
- **Store** - Storage space

You can specify rooms in several ways:
- **BHK format**: "3BHK" = 3 Bedrooms + 1 LivingRoom + 1 Kitchen + 2 Bathrooms
- **Direct specification**: "2 bedrooms, 1 kitchen, 1 living room, 2 bathrooms"
- **With adjacencies**: "3BHK with kitchen near dining room"

Try it! For example: "Design a 2BHK apartment" or "Create a 3-bedroom house"""
```

---

## Enhanced Gemini Prompt

**File:** `nl_interface/gemini_adapter.py`

### Updated `CONTEXT_AWARE_CHAT_PROMPT` (Line 189)

Added explicit room type information to the Gemini prompt:

```python
**Supported Room Types:**
- Bedroom, Kitchen, Bathroom, LivingRoom (Hall), DiningRoom, DrawingRoom, Garage, Store

**Response Guidelines:**
...
3. If the user asks about room types, list the supported types clearly
...
```

This ensures that when Gemini API is available, it also provides detailed room type information.

---

## Impact

### Before Fix:
- ❌ Questions about system capabilities → Technical state dumps
- ❌ Poor user experience - confusing responses
- ❌ Users couldn't understand what the system supports

### After Fix:
- ✅ Questions → Helpful, conversational answers
- ✅ Clear explanation of supported room types
- ✅ Examples showing how to use the system
- ✅ User-friendly guidance for new users

---

## Testing

### Test Case 1: Room Types Question
**Input:** "What room types do you support?"
**Expected:** Detailed list of 8 supported room types with descriptions
**Status:** ✅ PASS

### Test Case 2: Capabilities Question
**Input:** "What can you do?"
**Expected:** List of 4 main capabilities with examples
**Status:** ✅ PASS

### Test Case 3: Design Request (should still work)
**Input:** "Create a 3BHK apartment"
**Expected:** Design generation with spec extraction
**Status:** ✅ PASS (not affected by fix)

### Test Case 4: Mixed Keywords
**Input:** "How does room placement work?"
**Expected:** Explanation of room placement logic (QUESTION intent)
**Status:** ✅ PASS

---

## Files Modified

1. **`nl_interface/gemini_adapter.py`**
   - Fixed `_fallback_intent_classify()` function (lines 407-452)
   - Enhanced `_fallback_chat()` with room type handler (lines ~940)
   - Updated `CONTEXT_AWARE_CHAT_PROMPT` with room type list (lines 189-223)

---

## How to Test

1. **Open browser:** `http://localhost:8000/`
2. **Try these questions:**
   - "What room types do you support?"
   - "Which rooms can I add?"
   - "What can you do?"
   - "How do I use this system?"
3. **Expected:** Friendly, helpful responses with clear information

---

## Lessons Learned

1. **Intent classification order matters:** Always check for high-confidence patterns (questions, greetings) BEFORE keyword matching
2. **Question detection is strong:** Messages with "?", "what", "why", "how" are almost always questions
3. **Fallback handling is critical:** Even when Gemini API is available, fallback logic needs to be robust
4. **User experience > Technical accuracy:** Users want helpful answers, not internal state dumps

---

## Related Files

- `nl_interface/gemini_adapter.py` - Intent classification and chat response
- `nl_interface/service.py` - `process_user_request()` generates technical state (used for design intent only)
- `api/server.py` - Routes messages based on intent
- `BUGFIXES_SUMMARY.md` - Previous bug fixes

---

**Status:** ✅ System now correctly identifies questions and provides user-friendly conversational responses!
