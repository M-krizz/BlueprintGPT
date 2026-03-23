from __future__ import annotations

from typing import Any, Dict, List, Optional

from nl_interface.program_planner import summarize_room_program, summarize_zoning_plan


def compose_followup_reply(
    message: str,
    *,
    latest_design: Optional[Dict[str, Any]] = None,
    room_program: Optional[Dict[str, Any]] = None,
    zoning_plan: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    lowered = (message or "").strip().lower()
    if not lowered:
        return None

    program_summary = summarize_room_program(room_program)
    zoning_summary = summarize_zoning_plan(zoning_plan)
    design_engine = (latest_design or {}).get("engine")
    report_status = (latest_design or {}).get("report_status")

    if any(phrase in lowered for phrase in ("can i make", "make some changes", "modify", "change this", "refine this", "edit this")):
        details = program_summary or "your current layout"
        return (
            f"Yes. We can keep iterating on the current design. I still have {details.lower()} in context. "
            f"Tell me what you want to change, for example bedroom privacy, kitchen placement, bathroom access, or plot size."
        )

    if "overlap" in lowered or "overlapped" in lowered:
        reply = "If rooms overlap, I should treat that as a generation-quality problem rather than a cosmetic issue. "
        if latest_design:
            reply += (
                f"The latest design came from the `{design_engine or 'current'}` engine and is marked `{report_status or 'UNKNOWN'}`. "
                "I can regenerate it with stronger zoning and spacing guidance, or explain which part of the plan likely caused the conflict."
            )
        else:
            reply += "I can regenerate with stricter non-overlap and privacy zoning once we have a concrete program."
        return reply

    if "why failed" in lowered or "why did it fail" in lowered or "what happened" in lowered:
        if zoning_summary:
            return (
                "The plan fails when the geometry stage cannot satisfy the room program, zoning intent, and access constraints at the same time. "
                f"The current planning intent is: {zoning_summary}"
            )

    if "encoder" in lowered or "decoder" in lowered or "architecture" in lowered:
        return (
            "Right now the system behaves as a structured pipeline: the NL layer interprets the request, the room-program and zoning layer decides what spaces should exist and how they relate, "
            "the geometry engine turns that into room shapes, and validation plus repair check the result before rendering. "
            "The goal of the current refactor is to make those boundaries explicit instead of leaving them mixed together."
        )

    return None


def build_generation_summary(design_data: Dict[str, Any]) -> Dict[str, Any]:
    metrics = design_data.get("metrics", {}) or {}
    svg_url = None
    artifact_urls = design_data.get("artifact_urls", {}) or {}
    if artifact_urls.get("svg"):
        svg_url = artifact_urls.get("svg")
    elif design_data.get("svg_path"):
        svg_path = str(design_data.get("svg_path", "")).replace("\\", "/")
        if svg_path.startswith("outputs/"):
            svg_url = "/" + svg_path
    engine = (
        design_data.get("winning_source")
        or design_data.get("backend_target")
        or design_data.get("engine")
    )
    return {
        "engine": engine,
        "report_status": design_data.get("report_status"),
        "adjacency_satisfaction": metrics.get("adjacency_satisfaction"),
        "alignment_score": metrics.get("alignment_score"),
        "fully_connected": metrics.get("fully_connected"),
        "max_travel_distance": metrics.get("max_travel_distance"),
        "artifact_urls": artifact_urls,
        "svg_url": svg_url,
    }


def compose_conversational_design_reply(
    *,
    layout_type: Optional[str],
    report_status: Optional[str],
    engine: Optional[str],
    requested_program: Optional[str],
    zoning_summary: Optional[str],
    assumptions: List[str],
    suggestions: List[str],
) -> str:
    intro = (
        f"I generated a layout for your {layout_type or 'residential'} request. "
        f"It is currently marked `{report_status or 'UNKNOWN'}` and the selected engine was `{engine or 'unknown'}`."
    )
    body = []
    if requested_program:
        body.append(f"The interpreted room program is: {requested_program}")
    if zoning_summary:
        body.append(f"Planning intent: {zoning_summary}")
    if assumptions:
        body.append("Assumptions used: " + "; ".join(assumptions[:4]) + ".")
    if suggestions:
        body.append("Next steps: " + " ".join(suggestions[:3]))
    return "\n\n".join([intro] + body)
