from pathlib import Path
from uuid import uuid4

import pytest

from nl_interface.runner import execute_response


def test_execute_response_blocks_when_not_ready():
    with pytest.raises(ValueError):
        execute_response({"backend_ready": False, "backend_target": "algorithmic", "backend_spec": None})


def test_execute_response_dispatches_learned(monkeypatch):
    called = {}

    def fake_run(spec, **kwargs):
        called["spec"] = spec
        called["kwargs"] = kwargs
        return {"status": "completed", "backend_target": "learned"}

    monkeypatch.setattr("nl_interface.runner.run_learned_backend", fake_run)

    response = {
        "backend_ready": True,
        "backend_target": "learned",
        "backend_spec": {"rooms": [{"type": "DiningRoom"}]},
    }
    work_dir = Path("outputs") / f"test_nl_runner_unit_{uuid4().hex}"

    result = execute_response(response, output_dir=str(work_dir), output_prefix="runner_test")

    assert result["status"] == "completed"
    assert result["backend_target"] == "learned"
    assert called["spec"] == response["backend_spec"]
    assert called["kwargs"]["output_prefix"] == "runner_test"


def test_execute_response_missing_checkpoint_raises(monkeypatch):
    # Force env override to a missing checkpoint to ensure early, clear error.
    work_dir = Path("outputs") / f"test_nl_runner_missing_ckpt_{uuid4().hex}"
    missing = work_dir / "missing_checkpoint.pt"
    monkeypatch.setenv("BLUEPRINTGPT_CHECKPOINT", str(missing))

    response = {
        "backend_ready": True,
        "backend_target": "learned",
        "backend_spec": {"rooms": [{"type": "LivingRoom"}]},
    }

    with pytest.raises(ValueError) as excinfo:
        execute_response(response, output_dir=str(work_dir))

    assert "checkpoint" in str(excinfo.value).lower()
