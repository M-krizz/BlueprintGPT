import asyncio

from api import server


def test_server_nl_wrappers_delegate_to_chat_adapter(monkeypatch):
    captured = {}

    def fake_process_message(user_message, context=None, conversation_history=None):
        captured["process"] = {
            "user_message": user_message,
            "context": context,
            "conversation_history": conversation_history,
        }
        return {"intent": server.INTENT_CONVERSATION, "response": "ok", "should_generate": False}

    def fake_chat(user_message, context=None, conversation_history=None):
        captured["chat"] = {
            "user_message": user_message,
            "context": context,
            "conversation_history": conversation_history,
        }
        return "adapter reply"

    monkeypatch.setattr(server.conversation_orchestrator.chat_adapter, "process_message", fake_process_message)
    monkeypatch.setattr(server.conversation_orchestrator.chat_adapter, "chat", fake_chat)

    nl_result = server.process_nl_message("hello", {"state": "test"}, [{"role": "user", "content": "hi"}])
    chat_result = server.gemini_chat("hello", {"state": "test"}, [{"role": "user", "content": "hi"}])

    assert nl_result["response"] == "ok"
    assert chat_result == "adapter reply"
    assert captured["process"]["user_message"] == "hello"
    assert captured["chat"]["context"]["state"] == "test"


def test_conversation_message_uses_adapter_backed_process_message(monkeypatch):
    server.conversation_manager.sessions.clear()
    captured = {}

    def fake_process_message(user_message, context=None, conversation_history=None):
        captured["user_message"] = user_message
        captured["context"] = context
        captured["history"] = conversation_history
        return {
            "intent": server.INTENT_CONVERSATION,
            "intent_confidence": 0.95,
            "response": "Continuing through the adapter.",
            "should_generate": False,
            "spec": None,
        }

    monkeypatch.setattr(server.conversation_orchestrator.chat_adapter, "process_message", fake_process_message)

    body = server.ConversationMessageRequest(message="continue", generate=False)
    response = asyncio.run(server.conversation_message(body))

    assert response.assistant_text == "Continuing through the adapter."
    assert captured["user_message"] == "continue"
    assert captured["history"]

    server.conversation_manager.sessions.clear()
