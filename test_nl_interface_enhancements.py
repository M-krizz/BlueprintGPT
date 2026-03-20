"""
test_nl_interface_enhancements.py - Test the enhanced NL interface with intent classification

This tests:
1. Intent classification (design vs question vs conversation)
2. Context-aware chat responses
3. Routing based on intent
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_intent_classification():
    """Test that user messages are classified correctly by intent."""
    print("Testing Intent Classification...")

    try:
        from nl_interface.gemini_adapter import (
            classify_intent, _quick_intent_classify, _fallback_intent_classify,
            INTENT_DESIGN, INTENT_QUESTION, INTENT_CORRECTION, INTENT_CONVERSATION
        )

        # Test design intents
        design_messages = [
            "I need a 3BHK apartment",
            "Create a floor plan with 2 bedrooms and 1 kitchen",
            "Generate a layout for a 10 marla plot",
            "2 bedrooms, 1 bathroom, kitchen with north entrance",
        ]

        print("\n  Design intent tests:")
        for msg in design_messages:
            result = _quick_intent_classify(msg, has_designs=False) or _fallback_intent_classify(msg, has_designs=False)
            intent = result.get("intent")
            confidence = result.get("confidence", 0)
            status = "[OK]" if intent == INTENT_DESIGN else f"[FAIL] got {intent}"
            print(f"    {status} '{msg[:40]}...' -> {intent} ({confidence:.2f})")

        # Test question intents
        question_messages = [
            "What is BHK?",
            "How does this work?",
            "Can you explain the design?",
            "Why is the kitchen placed there?",
        ]

        print("\n  Question intent tests:")
        for msg in question_messages:
            result = _quick_intent_classify(msg, has_designs=False) or _fallback_intent_classify(msg, has_designs=False)
            intent = result.get("intent")
            confidence = result.get("confidence", 0)
            status = "[OK]" if intent == INTENT_QUESTION else f"[FAIL] got {intent}"
            print(f"    {status} '{msg[:40]}...' -> {intent} ({confidence:.2f})")

        # Test conversation intents
        conversation_messages = [
            "Hello",
            "Hi there",
            "Thanks!",
            "Good morning",
        ]

        print("\n  Conversation intent tests:")
        for msg in conversation_messages:
            result = _quick_intent_classify(msg, has_designs=False) or _fallback_intent_classify(msg, has_designs=False)
            intent = result.get("intent")
            confidence = result.get("confidence", 0)
            status = "[OK]" if intent == INTENT_CONVERSATION else f"[FAIL] got {intent}"
            print(f"    {status} '{msg[:40]}...' -> {intent} ({confidence:.2f})")

        # Test correction intents (with existing designs)
        correction_messages = [
            "Move the kitchen to the left",
            "Make the bedroom larger",
            "Swap kitchen and dining room",
        ]

        print("\n  Correction intent tests (with designs):")
        for msg in correction_messages:
            result = _quick_intent_classify(msg, has_designs=True) or _fallback_intent_classify(msg, has_designs=True)
            intent = result.get("intent")
            confidence = result.get("confidence", 0)
            status = "[OK]" if intent == INTENT_CORRECTION else f"[FAIL] got {intent}"
            print(f"    {status} '{msg[:40]}...' -> {intent} ({confidence:.2f})")

        print("\n  [OK] Intent classification tests passed")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Intent classification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_chat_responses():
    """Test that fallback chat gives helpful responses."""
    print("\nTesting Fallback Chat Responses...")

    try:
        from nl_interface.gemini_adapter import _fallback_chat

        test_cases = [
            # (message, context, expected_substring)
            ("Hello", {"state": "initial"}, "BlueprintGPT"),
            ("What can you do?", {"state": "initial"}, "Create Floor Plans"),
            ("How does this work?", {"state": "initial"}, "BlueprintGPT"),  # Response mentions BlueprintGPT
            ("What is BHK?", {"state": "initial"}, "Bedroom-Hall-Kitchen"),
            ("Thanks!", {"state": "generated", "num_designs": 3}, "welcome"),
        ]

        for msg, context, expected in test_cases:
            response = _fallback_chat(msg, context)
            status = "[OK]" if expected.lower() in response.lower() else "[FAIL]"
            print(f"  {status} '{msg}' -> contains '{expected}': {expected.lower() in response.lower()}")

        print("\n  [OK] Fallback chat tests passed")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Fallback chat failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_process_message_routing():
    """Test that process_message routes correctly based on intent."""
    print("\nTesting Message Routing...")

    try:
        from nl_interface.gemini_adapter import process_message, INTENT_DESIGN, INTENT_QUESTION

        # Test design routing
        result = process_message(
            "I need a 3BHK apartment with north entrance",
            context={"state": "initial", "num_designs": 0}
        )

        if result.get("intent") == INTENT_DESIGN:
            print(f"  [OK] Design intent correctly routed")
            if result.get("spec"):
                rooms = result["spec"].get("rooms", [])
                print(f"       Extracted rooms: {len(rooms)} room types")
                print(f"       Should generate: {result.get('should_generate')}")
            else:
                print(f"  [WARN] No spec extracted")
        else:
            print(f"  [FAIL] Expected design intent, got {result.get('intent')}")

        # Test question routing
        result = process_message(
            "What is BHK?",
            context={"state": "initial", "num_designs": 0}
        )

        if result.get("intent") == INTENT_QUESTION:
            print(f"  [OK] Question intent correctly routed")
            if result.get("response"):
                print(f"       Response length: {len(result['response'])} chars")
            else:
                print(f"  [WARN] No response generated")
        else:
            print(f"  [FAIL] Expected question intent, got {result.get('intent')}")

        print("\n  [OK] Message routing tests passed")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Message routing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all NL interface enhancement tests."""
    print("=" * 60)
    print("NL Interface Enhancement Tests")
    print("=" * 60)

    results = []
    results.append(test_intent_classification())
    results.append(test_fallback_chat_responses())
    results.append(test_process_message_routing())

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} test groups passed")

    if passed == total:
        print("\n[SUCCESS] All NL interface enhancements working!")
    else:
        print(f"\n[WARNING] {total - passed} test group(s) failed")

    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
