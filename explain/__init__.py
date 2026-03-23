"""Explanation module - provides layout explanations."""

from explain.context_builder import build_evidence
from explain.llm_explainer import explain

__all__ = ["build_evidence", "explain"]
