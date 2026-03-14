import os
import pytest

httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from api.server import app


client = TestClient(app)


def _chat(payload):
    return client.post("/chat/generate", json=payload)


def test_chat_generate_algorithmic_smoke():
    resp = _chat(
        {
            "prompt": "Need 2 bedrooms, 1 bathroom, 1 kitchen and 1 drawing room on a 10 marla plot with a north entrance and minimize corridor.",
            "boundary": {"width": 15, "height": 10},
            "area_unit": "sq.m",
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["backend_ready"] is True
    assert data["backend_target"] == "algorithmic"
    assert data["execution"]["status"] == "completed"
    assert "artifact_urls" in data["execution"]
    assert data["execution"]["artifact_urls"].get("svg", "").startswith("/outputs/")


def test_chat_generate_learned_smoke():
    os.environ["BLUEPRINTGPT_ALLOW_DESIGN_FAIL"] = "1"
    resp = _chat(
        {
            "prompt": "Need 2 bedrooms, 1 bathroom, 1 kitchen and 1 garage on a 10 marla plot with a north entrance.",
            "boundary": {"width": 15, "height": 10},
            "area_unit": "sq.m",
        }
    )
    os.environ.pop("BLUEPRINTGPT_ALLOW_DESIGN_FAIL", None)

    assert resp.status_code == 200
    data = resp.json()
    assert data["backend_ready"] is True
    assert data["backend_target"] == "learned"
    assert data["execution"]["status"] == "completed"
    assert data["execution"]["artifact_urls"].get("svg", "").startswith("/outputs/")
