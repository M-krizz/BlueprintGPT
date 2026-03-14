import json

from nl_interface.runner import _resolve_llm_fn
from explain.validator import validate_explanation


def test_llm_fallback_without_key():
    llm_fn, warning, provider, model = _resolve_llm_fn("gemini", None, None)
    assert llm_fn is None
    assert warning is not None
    assert provider == "gemini"
    assert model == "gemini-pro"


def test_validator_rejects_unsupported_number():
    evidence = {"metrics": {"adjacency": 0.5}}
    output = {
        "summary": "Unsupported 99",
        "why_this_layout": ["still mentions 42"],
        "constraint_justification": [],
        "tradeoffs": [],
        "suggested_edits": [],
        "open_questions": [],
    }
    ok, errors = validate_explanation(output, evidence, ontology_room_types={"Bedroom"}, status="COMPLIANT")
    assert not ok
    assert any("Unsupported numeric claim" in e for e in errors)


def test_llm_metadata_has_no_secret_leak():
    secret = "SECRET_KEY_SHOULD_NOT_LEAK"
    _, warning, provider, model = _resolve_llm_fn("gemini", secret, None)
    meta = {
        "used": False,
        "provider": provider,
        "model": model,
        "latency_ms": None,
        "warning": warning,
        "evidence_hash": "abc",
    }
    serialized = json.dumps(meta)
    assert secret not in serialized
    assert "api_key" not in serialized
