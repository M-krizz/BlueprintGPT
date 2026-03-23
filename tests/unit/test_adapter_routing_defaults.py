from __future__ import annotations

from nl_interface import adapter


def test_resolve_core_backend_defaults_to_algorithmic(monkeypatch):
    monkeypatch.delenv(adapter.AUTO_CORE_BACKEND_ENV, raising=False)
    monkeypatch.setattr(adapter, "_planner_checkpoint_exists", lambda: True)

    backend, policy, checkpoint_ready = adapter._resolve_core_backend()

    assert backend == "algorithmic"
    assert policy == "algorithmic"
    assert checkpoint_ready is True


def test_resolve_core_backend_planner_if_available_promotes_with_checkpoint(monkeypatch):
    monkeypatch.setenv(adapter.AUTO_CORE_BACKEND_ENV, "planner_if_available")
    monkeypatch.setattr(adapter, "_planner_checkpoint_exists", lambda: True)

    backend, policy, checkpoint_ready = adapter._resolve_core_backend()

    assert backend == "planner"
    assert policy == "planner_if_available"
    assert checkpoint_ready is True
