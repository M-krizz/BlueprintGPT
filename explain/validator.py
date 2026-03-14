"""Validate explainer output against evidence to prevent hallucinations."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

SCHEMA = {
    "summary": str,
    "why_this_layout": list,
    "constraint_justification": list,
    "tradeoffs": list,
    "suggested_edits": list,
    "open_questions": list,
}


def validate_explanation(
    output: Dict,
    evidence: Dict,
    *,
    ontology_room_types: Set[str],
    status: str,
) -> (bool, List[str]):
    errors: List[str] = []

    errors.extend(_validate_schema(output))

    # Compliance consistency
    if status == "NON_COMPLIANT" and _claims_compliance(output):
        errors.append("Cannot claim compliance when status is NON_COMPLIANT")

    # Room types check in suggested edits and text fields
    for edit in output.get("suggested_edits", []) or []:
        target = _maybe_extract_room(edit)
        if target and target not in ontology_room_types:
            errors.append(f"Unknown room type in suggested_edits: {target}")
    errors.extend(_detect_unknown_rooms_in_text(output, ontology_room_types))

    # Evidence path existence and numeric coherence
    for item in output.get("constraint_justification", []) or []:
        path = item.get("evidence_path")
        if path and not _path_exists(evidence, path):
            errors.append(f"Missing evidence_path: {path}")
        if path and _path_exists(evidence, path) and "value" in item:
            expected = _get_path(evidence, path)
            if _is_number(expected) and _is_number(item.get("value")):
                if float(expected) != float(item.get("value")):
                    errors.append(f"Value mismatch for {path}: expected {expected}, got {item.get('value')}")

    errors.extend(_detect_unsupported_numbers(output, evidence))

    return len(errors) == 0, errors


def _claims_compliance(output: Dict) -> bool:
    text_fields = [output.get("summary", ""), output.get("why_this_layout", "")]
    for field in text_fields:
        if isinstance(field, str) and "compliant" in field.lower():
            return True
        if isinstance(field, list):
            for item in field:
                if isinstance(item, str) and "compliant" in item.lower():
                    return True
    return False


def _path_exists(obj: Dict, path: str) -> bool:
    parts = path.split(".")
    cur = obj
    for part in parts:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
            continue
        return False
    return True


def _get_path(obj: Dict, path: str) -> Any:
    parts = path.split(".")
    cur: Any = obj
    for part in parts:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float))


def _maybe_extract_room(edit: Any) -> str:
    if isinstance(edit, dict):
        return str(edit.get("target", ""))
    if isinstance(edit, str):
        return edit
    return ""


def _validate_schema(output: Dict) -> List[str]:
    errors: List[str] = []
    for key, typ in SCHEMA.items():
        if key not in output:
            errors.append(f"Missing required key: {key}")
            continue
        if not isinstance(output[key], typ):
            errors.append(f"Key '{key}' has wrong type; expected {typ.__name__}")
    return errors


def _detect_unknown_rooms_in_text(output: Dict, ontology_room_types: Set[str]) -> List[str]:
    errors: List[str] = []
    room_pattern = re.compile(r"\b([A-Z][A-Za-z]*Room)\b")
    text_fields = []
    for key in ("summary", "why_this_layout", "tradeoffs", "suggested_edits", "open_questions"):
        val = output.get(key)
        if isinstance(val, str):
            text_fields.append(val)
        elif isinstance(val, list):
            text_fields.extend([x for x in val if isinstance(x, str)])
    for text in text_fields:
        for match in room_pattern.findall(text):
            if match not in ontology_room_types:
                errors.append(f"Unknown room type referenced: {match}")
    return errors


def _detect_unsupported_numbers(output: Dict, evidence: Dict) -> List[str]:
    errors: List[str] = []
    evidence_numbers = _collect_numbers(evidence)
    evidence_numbers_rounded = {round(num, 6) for num in evidence_numbers}

    text_fields: List[str] = []
    for key in ("summary", "why_this_layout", "tradeoffs", "suggested_edits", "open_questions"):
        val = output.get(key)
        if isinstance(val, str):
            text_fields.append(val)
        elif isinstance(val, list):
            text_fields.extend([x for x in val if isinstance(x, str)])
    for text in text_fields:
        for num in _numbers_in_text(text):
            if round(num, 6) not in evidence_numbers_rounded:
                errors.append(f"Unsupported numeric claim: {num}")
    return errors


def _numbers_in_text(text: str) -> List[float]:
    nums: List[float] = []
    for match in re.findall(r"-?\d+(?:\.\d+)?", text):
        try:
            nums.append(float(match))
        except ValueError:
            continue
    return nums


def _collect_numbers(obj: Any) -> List[float]:
    found: List[float] = []
    if isinstance(obj, (int, float)):
        found.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            found.extend(_collect_numbers(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_collect_numbers(v))
    return found
