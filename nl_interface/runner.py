"""Execute backend-ready NL responses through the existing smoke pipelines."""

from __future__ import annotations

import os
import json
import math
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from constraints.compliance_report import build_compliance_report, save_compliance_report
from explain.context_builder import build_evidence
from explain.llm_explainer import explain
from constraints.repair_loop import validate_and_repair_spec
from constraints.spec_validator import validate_spec
from generator.layout_generator import generate_layout_from_spec
from generator.ranking import rank_layout_variants
from learned.integration.model_generation_loop import generate_best_layout_from_model
from visualization.export_svg_blueprint import save_svg_blueprint


__all__ = [
    "execute_response",
    "run_algorithmic_backend",
    "run_planner_backend",
    "run_planner_direct_backend",
    "run_learned_backend",
    "run_hybrid_backend",
]

DEFAULT_REGULATION_FILE = "ontology/regulation_data.json"
DEFAULT_CHECKPOINT = "learned/model/checkpoints/improved_v1.pt"
CHECKPOINT_ENV_VAR = "BLUEPRINTGPT_CHECKPOINT"
DEFAULT_PLANNER_CHECKPOINT = "learned/planner/checkpoints/room_planner.pt"
PLANNER_CHECKPOINT_ENV_VAR = "BLUEPRINTGPT_PLANNER_CHECKPOINT"
STRUCTURED_PLANNER_GUIDANCE_SOURCES = {
    "zoning-plan",
    "compact_algorithmic_baseline",
}


def execute_response(
    response: Dict,
    *,
    output_dir: str = "outputs",
    output_prefix: Optional[str] = None,
    regulation_file: str = DEFAULT_REGULATION_FILE,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    device: str = "cpu",
    k: int = 10,
    top_m: int = 3,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 0,
    llm_provider: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> Dict:
    if not response.get("backend_ready"):
        raise ValueError("NL response is not backend-ready. Resolve the missing fields before execution.")

    backend_target = response.get("backend_target")
    backend_spec = deepcopy(response.get("backend_spec") or {})

    llm_fn, llm_warning, llm_used_provider, llm_used_model = _resolve_llm_fn(
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )
    if backend_target == "algorithmic":
        return run_algorithmic_backend(
            backend_spec,
            output_dir=output_dir,
            output_prefix=output_prefix,
            regulation_file=regulation_file,
            llm_fn=llm_fn,
            llm_provider=llm_used_provider,
            llm_model=llm_used_model,
            llm_warning=llm_warning,
        )
    if backend_target == "planner":
        return run_planner_backend(
            backend_spec,
            output_dir=output_dir,
            output_prefix=output_prefix,
            regulation_file=regulation_file,
            device=device,
            llm_fn=llm_fn,
            llm_provider=llm_used_provider,
            llm_model=llm_used_model,
            llm_warning=llm_warning,
        )
    if backend_target == "planner_direct":
        return run_planner_direct_backend(
            backend_spec,
            output_dir=output_dir,
            output_prefix=output_prefix,
            regulation_file=regulation_file,
            device=device,
            skip_corridors=True,
            llm_fn=llm_fn,
            llm_provider=llm_used_provider,
            llm_model=llm_used_model,
            llm_warning=llm_warning,
        )
    if backend_target == "learned":
        return run_learned_backend(
            backend_spec,
            output_dir=output_dir,
            output_prefix=output_prefix,
            regulation_file=regulation_file,
            checkpoint_path=checkpoint_path,
            device=device,
            k=k,
            top_m=top_m,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            llm_fn=llm_fn,
            llm_provider=llm_used_provider,
            llm_model=llm_used_model,
            llm_warning=llm_warning,
        )
    if backend_target == "hybrid":
        return run_hybrid_backend(
            backend_spec,
            output_dir=output_dir,
            output_prefix=output_prefix,
            regulation_file=regulation_file,
            checkpoint_path=checkpoint_path,
            device=device,
            k=k,
            top_m=top_m,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            llm_fn=llm_fn,
            llm_provider=llm_used_provider,
            llm_model=llm_used_model,
            llm_warning=llm_warning,
        )
    raise ValueError(f"Unsupported backend target '{backend_target}'.")


def run_algorithmic_backend(
    spec: Dict,
    *,
    output_dir: str = "outputs",
    output_prefix: Optional[str] = None,
    regulation_file: str = DEFAULT_REGULATION_FILE,
    llm_fn=None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_warning: Optional[str] = None,
    allow_auto_boundary_recovery: bool = True,
) -> Dict:
    output_paths = _output_paths(output_dir, output_prefix, "algorithmic")

    repaired = validate_and_repair_spec(spec, validate_spec, max_attempts=3)
    working_spec = repaired["spec"]
    working_spec["_spec_validation"] = repaired.get("validation", {})
    working_spec["_repair"] = {"repair_attempts": repaired.get("repair_attempts", 0)}
    working_spec["learned_checkpoint"] = "__disabled_checkpoint__.pt"

    result = _quiet_call(generate_layout_from_spec, working_spec, regulation_file=regulation_file)
    algo_variants = [v for v in result.get("layout_variants", [result]) if v.get("source") == "algorithmic"]
    ranked_algo, _ = rank_layout_variants(algo_variants)

    passed, rejected, design_stats = _design_filter(ranked_algo)
    debug_gate_bypassed = False
    if not passed:
        if not rejected:
            raise ValueError("No layout variants were generated.")
        top_rejection = rejected[0]
        critical_reasons = {
            "Not fully connected",
            "Travel distance exceeds limit",
            "Adjacency satisfaction below 0.2",
            "Alignment score below 0.45",
            "Corridor width below 1.0 m",
            "Room area allocation drift too high",
            "Circulation area is too high",
        }
        rejection_reasons = set(top_rejection.get("_design_reasons") or [])
        if rejection_reasons & critical_reasons:
            compact_program = (
                len(working_spec.get("rooms", []) or []) <= 6
                and 0.0 < float(working_spec.get("total_area", 0.0) or 0.0) <= 80.0
            )
            recovery_reasons = {
                "Room area allocation drift too high",
                "Circulation area is too high",
                "Corridor width below 1.0 m",
            }
            if compact_program:
                recovery_reasons.update(
                    {
                        "Not fully connected",
                        "Travel distance exceeds limit",
                    }
                )
            if allow_auto_boundary_recovery and rejection_reasons & recovery_reasons:
                recovery_spec, recovery_note = _inflate_rectangular_boundary_spec(
                    working_spec,
                    rejection_reasons=top_rejection.get("_design_reasons") or [],
                )
                if recovery_spec is not None:
                    try:
                        recovery_output_prefix = f"{output_prefix}_expanded" if output_prefix else None
                        recovered_result = run_algorithmic_backend(
                            recovery_spec,
                            output_dir=output_dir,
                            output_prefix=recovery_output_prefix,
                            regulation_file=regulation_file,
                            llm_fn=llm_fn,
                            llm_provider=llm_provider,
                            llm_model=llm_model,
                            llm_warning=llm_warning,
                            allow_auto_boundary_recovery=False,
                        )
                        recovered_result["auto_boundary_adjustment"] = recovery_note
                        recovered_result.setdefault("algorithmic_attempt", {
                            "reasons": top_rejection.get("_design_reasons") or ["unknown failure"],
                            "strategy_name": top_rejection.get("strategy_name"),
                        })
                        if recovered_result.get("backend_target") == "algorithmic" and not recovered_result.get("winning_source"):
                            recovered_result["winning_source"] = "algorithmic_boundary_recovery"
                        return recovered_result
                    except Exception:
                        pass
            try:
                fallback_output_prefix = f"{output_prefix}_planner_direct_rescue" if output_prefix else None
                fallback_result = run_planner_direct_backend(
                    working_spec,
                    output_dir=output_dir,
                    output_prefix=fallback_output_prefix,
                    regulation_file=regulation_file,
                    llm_fn=llm_fn,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    llm_warning=llm_warning,
                )
                if fallback_result.get("report_status") == "COMPLIANT":
                    fallback_result["winning_source"] = "planner_direct_rescue"
                    fallback_result["algorithmic_attempt"] = {
                        "reasons": top_rejection.get("_design_reasons") or ["unknown failure"],
                        "strategy_name": top_rejection.get("strategy_name"),
                    }
                    return fallback_result
            except Exception:
                pass
            raise ValueError(
                "Design-quality gate rejected all algorithmic variants: "
                + ", ".join(top_rejection.get("_design_reasons") or ["unknown failure"])
            )
        # Soft fallback only for non-critical ranking misses.
        print("[DEBUG] Design-quality gate rejected all variants — showing best soft-rejected variant.")
        passed = [top_rejection]
        debug_gate_bypassed = True

    chosen = passed[0]
    building = chosen["building"]

    svg_path = _quiet_call(
        save_svg_blueprint,
        building,
        output_path=str(output_paths["svg"]),
        boundary_polygon=working_spec.get("boundary_polygon"),
        entrance_point=working_spec.get("entrance_point"),
        title="NL Interface - Algorithmic Run",
    )

    report = build_compliance_report(chosen)

    ontology_rooms = _load_ontology_room_types(regulation_file)

    alternatives_evidence = []
    for alt in passed[1:4]:
        alternatives_evidence.append(
            {
                "strategy_name": alt.get("strategy_name"),
                "design_score": alt.get("_design_score"),
                "design_reasons": alt.get("_design_reasons", []),
                "metrics": _safe_metric_snapshot(alt.get("metrics", {})),
                "diversity_from_selected": _diversity_score(chosen, alt),
            }
        )

    evidence = build_evidence(
        chosen,
        report,
        variant_id=chosen.get("strategy_name"),
        design_score=chosen.get("_design_score"),
        design_reasons=chosen.get("_design_reasons", []),
        alternatives=alternatives_evidence,
        design_filter_stats=design_stats,
    )

    import hashlib

    evidence_hash = hashlib.sha256(json.dumps(evidence, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    llm_latency_ms = None
    if llm_fn is not None:
        import time

        _start = time.time()
        explanation = explain(
            evidence,
            ontology_room_types=ontology_rooms,
            status=report.get("status", "UNKNOWN"),
            llm_fn=llm_fn,
        )
        llm_latency_ms = round((time.time() - _start) * 1000, 2)
    else:
        explanation = explain(
            evidence,
            ontology_room_types=ontology_rooms,
            status=report.get("status", "UNKNOWN"),
            llm_fn=None,
        )
    report["explanation"] = explanation
    save_compliance_report(report, str(output_paths["report"]))

    requested_rooms = _spec_room_counts(working_spec)
    generated_rooms = _building_room_counts(building)

    llm_warning_effective = llm_warning
    if llm_fn is not None and llm_warning_effective is None and isinstance(explanation, dict):
        if explanation.get("_llm_validation_errors"):
            llm_warning_effective = "validation_failed"

    return {
        "status": "completed",
        "backend_target": "algorithmic",
        "strategy_name": chosen.get("strategy_name"),
        "design_score": chosen.get("_design_score"),
        "design_reasons": chosen.get("_design_reasons", []),
        "report_status": report.get("status"),
        "explanation": explanation,
        "debug_gate_bypassed": debug_gate_bypassed,
        "llm": {
            "used": llm_fn is not None,
            "provider": llm_provider or "deterministic",
            "model": llm_model,
            "latency_ms": llm_latency_ms,
            "warning": llm_warning_effective,
            "evidence_hash": evidence_hash,
        },
        "requested_rooms": requested_rooms,
        "generated_rooms": generated_rooms,
        "room_coverage": _room_coverage(requested_rooms, generated_rooms),
        "violations": report.get("violations", []),
        "alternatives": [
            {
                "strategy_name": alt.get("strategy_name"),
                "design_score": alt.get("_design_score"),
                "design_reasons": alt.get("_design_reasons", []),
                "diversity_from_selected": _diversity_score(chosen, alt),
                "metrics": _safe_metric_snapshot(alt.get("metrics", {})),
            }
            for alt in passed[:3]
        ],
        "design_filter_stats": design_stats,
        "artifact_paths": {
            "svg": str(svg_path),
            "report": str(output_paths["report"]),
        },
        "metrics": {
            "total_area": chosen.get("metrics", {}).get("total_area"),
            "fully_connected": chosen.get("metrics", {}).get("fully_connected"),
            "max_travel_distance": chosen.get("metrics", {}).get("max_travel_distance"),
            "max_allowed_travel_distance": chosen.get("metrics", {}).get("max_allowed_travel_distance"),
            "adjacency_satisfaction": chosen.get("metrics", {}).get("adjacency_satisfaction"),
            "alignment_score": chosen.get("metrics", {}).get("alignment_score"),
            "corridor_width": chosen.get("metrics", {}).get("corridor_width"),
            "circulation_walkable_area": chosen.get("metrics", {}).get("circulation_walkable_area"),
            "connectivity_to_exit": chosen.get("metrics", {}).get("connectivity_to_exit"),
            "door_path_travel_distance": chosen.get("metrics", {}).get("door_path_travel_distance"),
            "public_frontage_score": chosen.get("metrics", {}).get("public_frontage_score"),
            "bedroom_privacy_score": chosen.get("metrics", {}).get("bedroom_privacy_score"),
            "kitchen_living_score": chosen.get("metrics", {}).get("kitchen_living_score"),
            "bathroom_access_score": chosen.get("metrics", {}).get("bathroom_access_score"),
            "architectural_reasonableness": chosen.get("metrics", {}).get("architectural_reasonableness"),
            "max_room_area_error": chosen.get("metrics", {}).get("max_room_area_error"),
        },
    }


def run_planner_backend(
    spec: Dict,
    *,
    output_dir: str = "outputs",
    output_prefix: Optional[str] = None,
    regulation_file: str = DEFAULT_REGULATION_FILE,
    device: str = "cpu",
    llm_fn=None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_warning: Optional[str] = None,
) -> Dict:
    from learned.planner.inference import (
        DEFAULT_PLANNER_CHECKPOINT,
        predict_planner_guidance,
    )

    baseline_spec = _spec_without_learned_planner_guidance(spec)
    inference_spec = _spec_without_all_planner_guidance(spec)
    existing_guidance = deepcopy((baseline_spec.get("planner_guidance") or {}))
    room_count = len(baseline_spec.get("rooms", []) or [])
    total_area = float(baseline_spec.get("total_area") or 0.0)
    compact_program_baseline_only = room_count <= 6 and (0.0 < total_area <= 80.0)

    if compact_program_baseline_only:
        baseline_result = run_algorithmic_backend(
            deepcopy(baseline_spec),
            output_dir=output_dir,
            output_prefix=output_prefix,
            regulation_file=regulation_file,
            llm_fn=llm_fn,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_warning=llm_warning,
        )
        baseline_result["winning_source"] = "algorithmic_compact_baseline"
        baseline_result["planner_fallback_reason"] = "compact_program_prefers_stable_algorithmic_baseline"
        baseline_result["backend_target"] = "planner"
        baseline_result["planner_guidance"] = existing_guidance or {
            "source": "compact_algorithmic_baseline",
            "spatial_hints": {},
            "room_order": [],
            "adjacency_preferences": [],
        }
        baseline_result["planner_summary"] = {
            "source": baseline_result["planner_guidance"].get("source", "compact_algorithmic_baseline"),
            "checkpoint_path": None,
            "room_order_count": len(baseline_result["planner_guidance"].get("room_order", [])),
            "spatial_hint_count": len(baseline_result["planner_guidance"].get("spatial_hints", {})),
            "adjacency_preferences_count": len(baseline_result["planner_guidance"].get("adjacency_preferences", [])),
            "compact_baseline_only": True,
        }
        return baseline_result

    planner_checkpoint = os.getenv(PLANNER_CHECKPOINT_ENV_VAR, DEFAULT_PLANNER_CHECKPOINT)
    planner_guidance = predict_planner_guidance(
        inference_spec,
        checkpoint_path=planner_checkpoint,
        device=device,
    )

    guided_spec = deepcopy(baseline_spec)
    guided_spec["planner_guidance"] = planner_guidance
    guided_spec["learned_spatial_hints"] = planner_guidance.get("spatial_hints", {})
    baseline_result = None
    if room_count <= 6 or (0.0 < total_area <= 80.0):
        try:
            baseline_result = run_algorithmic_backend(
                deepcopy(baseline_spec),
                output_dir=output_dir,
                output_prefix=output_prefix,
                regulation_file=regulation_file,
                llm_fn=llm_fn,
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_warning=llm_warning,
            )
        except ValueError:
            baseline_result = None

    try:
        result = run_algorithmic_backend(
            guided_spec,
            output_dir=output_dir,
            output_prefix=output_prefix,
            regulation_file=regulation_file,
            llm_fn=llm_fn,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_warning=llm_warning,
        )
        result["winning_source"] = "planner_packer"
    except ValueError as exc:
        base_rescue_result = baseline_result
        if base_rescue_result is None:
            rescue_spec = deepcopy(baseline_spec)
            try:
                base_rescue_result = run_algorithmic_backend(
                    rescue_spec,
                    output_dir=output_dir,
                    output_prefix=output_prefix,
                    regulation_file=regulation_file,
                    llm_fn=llm_fn,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    llm_warning=llm_warning,
                )
            except ValueError:
                base_rescue_result = None

        direct_output_prefix = f"{output_prefix}_planner_direct" if output_prefix else None
        direct_result = run_planner_direct_backend(
            guided_spec,
            output_dir=output_dir,
            output_prefix=direct_output_prefix,
            regulation_file=regulation_file,
            device=device,
            skip_corridors=True,
            planner_guidance=planner_guidance,
            llm_fn=llm_fn,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_warning=llm_warning,
        )
        if direct_result.get("report_status") == "COMPLIANT":
            result = direct_result
            result["winning_source"] = "planner_direct_fallback"
        elif base_rescue_result is not None:
            result = base_rescue_result
            result["planner_direct_attempt"] = {
                "report_status": direct_result.get("report_status"),
                "violations": direct_result.get("violations", []),
                "artifact_paths": direct_result.get("artifact_paths", {}),
            }
            result["winning_source"] = "algorithmic_rescue"
        else:
            raise
        result["planner_fallback_reason"] = str(exc)

    result["backend_target"] = "planner"
    result["planner_guidance"] = planner_guidance
    result["planner_summary"] = {
        "source": planner_guidance.get("source"),
        "checkpoint_path": planner_guidance.get("checkpoint_path"),
        "room_order_count": len(planner_guidance.get("room_order", [])),
        "spatial_hint_count": len(planner_guidance.get("spatial_hints", {})),
        "adjacency_preferences_count": len(planner_guidance.get("adjacency_preferences", [])),
    }
    return result
def run_planner_direct_backend(
    spec: Dict,
    *,
    output_dir: str = "outputs",
    output_prefix: Optional[str] = None,
    regulation_file: str = DEFAULT_REGULATION_FILE,
    device: str = "cpu",
    skip_corridors: bool = True,
    planner_guidance: Optional[Dict] = None,
    llm_fn=None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_warning: Optional[str] = None,
) -> Dict:
    """
    Generate layout directly from planner model without algorithmic packing.

    This backend:
    1. Calls predict_planner_guidance() to get model outputs
    2. Uses geometry_synthesis to convert centroids+area_ratios to room polygons
    3. Builds a graph-valid room/door model for reporting and travel checks
    4. Optionally skips corridor carving while keeping connectivity evaluation honest
    """
    from core.building import Building
    from core.exit import Exit
    from core.room import Room
    from learned.planner.inference import predict_planner_guidance
    from learned.planner.geometry_synthesis import synthesize_room_geometry
    from learned.planner.verification import verify_planner_output, generate_verification_summary

    output_paths = _output_paths(output_dir, output_prefix, "planner_direct")

    repaired = validate_and_repair_spec(spec, validate_spec, max_attempts=3)
    working_spec = repaired["spec"]
    working_spec["_spec_validation"] = repaired.get("validation", {})
    working_spec["_repair"] = {"repair_attempts": repaired.get("repair_attempts", 0)}

    if planner_guidance is None:
        planner_checkpoint = os.getenv(PLANNER_CHECKPOINT_ENV_VAR, DEFAULT_PLANNER_CHECKPOINT)
        planner_guidance = predict_planner_guidance(
            working_spec,
            checkpoint_path=planner_checkpoint,
            device=device,
        )

    working_spec["planner_guidance"] = planner_guidance
    working_spec["learned_spatial_hints"] = planner_guidance.get("spatial_hints", {})

    boundary_polygon = working_spec.get("boundary_polygon")
    entrance_point = working_spec.get("entrance_point")

    if not boundary_polygon:
        raise ValueError("boundary_polygon required for planner_direct backend")

    boundary_tuples = [tuple(p) for p in boundary_polygon]
    entrance_tuple = tuple(entrance_point) if entrance_point else None

    synthesized_rooms = synthesize_room_geometry(
        planner_guidance,
        boundary_tuples,
        working_spec,
        entrance_point=entrance_tuple,
    )

    verification = verify_planner_output(
        synthesized_rooms,
        boundary_tuples,
        working_spec,
        planner_output=planner_guidance,
    )
    verification_dict = verification.to_dict()

    occupancy = working_spec.get("occupancy", "Residential")
    building = Building(occupancy_type=occupancy)

    for room_data in synthesized_rooms:
        room = Room(room_data["name"], room_data["type"], room_data["area"])
        room.final_area = room_data["area"]
        room.target_area = room_data["area"]
        room.polygon = room_data["polygon"]
        building.add_room(room)

    exit_width = 1.0
    ex = Exit(width=exit_width)
    if entrance_tuple:
        ex.segment = _planner_direct_exit_segment(boundary_tuples, entrance_tuple, exit_width)
    building.set_exit(ex)

    _attach_planner_direct_doors(
        building,
        synthesized_rooms,
        boundary_polygon=boundary_tuples,
        entrance_point=entrance_tuple,
    )

    metrics = _planner_direct_metrics(
        building,
        verification=verification_dict,
        regulation_file=regulation_file,
        boundary_polygon=boundary_tuples,
        entrance_point=entrance_tuple,
        spec=working_spec,
        skip_corridors=skip_corridors,
    )

    variant = {
        "source": "planner_direct",
        "building": building,
        "allocation": None,
        "modifications": [],
        "metrics": metrics,
        "verification": verification_dict,
        "input_spec": working_spec,
        "spec_validation": working_spec.get("_spec_validation"),
        "repair": working_spec.get("_repair"),
        "rule_preflight": {"valid": True, "errors": [], "warnings": []},
        "kg_precheck": {},
        "bounding_box": _bbox_from_boundary(boundary_polygon),
        "strategy_name": "planner-direct-synthesis",
    }

    design_score = (
        0.30 * verification.metrics.get("overlap_free_ratio", 0)
        + 0.25 * verification.metrics.get("boundary_containment", 0)
        + 0.25 * verification.metrics.get("area_compliance", 0)
        + 0.20 * verification.metrics.get("adjacency_satisfaction", 0)
    )
    variant["_design_score"] = round(design_score, 4)
    variant["_design_reasons"] = verification.issues

    svg_path = _quiet_call(
        save_svg_blueprint,
        building,
        output_path=str(output_paths["svg"]),
        boundary_polygon=boundary_polygon,
        entrance_point=entrance_point,
        title="NL Interface - Planner Direct Run",
    )

    report = build_compliance_report(variant)
    report["verification"] = verification_dict
    report["verification_summary"] = generate_verification_summary(verification)

    ontology_rooms = _load_ontology_room_types(regulation_file)

    evidence = build_evidence(
        variant,
        report,
        variant_id="planner_direct",
        design_score=design_score,
        design_reasons=verification.issues,
        alternatives=[],
        design_filter_stats={"accepted_count": 1, "rejected_count": 0},
    )

    import hashlib
    evidence_hash = hashlib.sha256(json.dumps(evidence, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    llm_latency_ms = None
    if llm_fn is not None:
        import time
        _start = time.time()
        explanation = explain(
            evidence,
            ontology_room_types=ontology_rooms,
            status=report.get("status", "UNKNOWN"),
            llm_fn=llm_fn,
        )
        llm_latency_ms = round((time.time() - _start) * 1000, 2)
    else:
        explanation = explain(
            evidence,
            ontology_room_types=ontology_rooms,
            status=report.get("status", "UNKNOWN"),
            llm_fn=None,
        )
    report["explanation"] = explanation
    save_compliance_report(report, str(output_paths["report"]))

    requested_rooms = _spec_room_counts(working_spec)
    generated_rooms = _building_room_counts(building)

    llm_warning_effective = llm_warning
    if llm_fn is not None and llm_warning_effective is None and isinstance(explanation, dict):
        if explanation.get("_llm_validation_errors"):
            llm_warning_effective = "validation_failed"

    return {
        "status": "completed",
        "backend_target": "planner_direct",
        "strategy_name": "planner-direct-synthesis",
        "design_score": design_score,
        "quality_gate_passed": verification.passed,
        "verification": verification_dict,
        "verification_summary": generate_verification_summary(verification),
        "planner_guidance": planner_guidance,
        "planner_summary": {
            "source": planner_guidance.get("source"),
            "checkpoint_path": planner_guidance.get("checkpoint_path"),
            "room_order_count": len(planner_guidance.get("room_order", [])),
            "spatial_hint_count": len(planner_guidance.get("spatial_hints", {})),
            "adjacency_preferences_count": len(planner_guidance.get("adjacency_preferences", [])),
            "area_ratios_count": len(planner_guidance.get("area_ratios", {})),
        },
        "report_status": report.get("status"),
        "explanation": explanation,
        "llm": {
            "used": llm_fn is not None,
            "provider": llm_provider or "deterministic",
            "model": llm_model,
            "latency_ms": llm_latency_ms,
            "warning": llm_warning_effective,
            "evidence_hash": evidence_hash,
        },
        "requested_rooms": requested_rooms,
        "generated_rooms": generated_rooms,
        "room_coverage": _room_coverage(requested_rooms, generated_rooms),
        "violations": report.get("violations", []),
        "artifact_paths": {
            "svg": str(svg_path),
            "report": str(output_paths["report"]),
        },
        "metrics": metrics,
    }
def run_learned_backend(
    spec: Dict,
    *,
    output_dir: str = "outputs",
    output_prefix: Optional[str] = None,
    regulation_file: str = DEFAULT_REGULATION_FILE,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    device: str = "cpu",
    k: int = 10,
    top_m: int = 3,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 0,
    llm_fn=None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_warning: Optional[str] = None,
) -> Dict:
    output_paths = _output_paths(output_dir, output_prefix, "learned")
    boundary = spec.get("boundary_polygon")
    entrance = spec.get("entrance_point")
    resolved_ckpt = _resolve_checkpoint(checkpoint_path)

    best, summary = _quiet_call(
        generate_best_layout_from_model,
        spec=spec,
        boundary_poly=boundary,
        entrance=entrance,
        checkpoint_path=str(resolved_ckpt),
        regulation_file=regulation_file,
        K=max(1, top_m),
        max_attempts=max(1, k),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        device=device,
        pre_rank_top_m=max(1, top_m),
    )

    if best is None or best.get("building") is None:
        raise ValueError("No viable learned layout produced.")

    building = best["building"]

    passed, rejected, design_stats = _design_filter([best])
    debug_gate_bypassed = False
    if not passed:
        top_rejection = rejected[0] if rejected else {}
        # Debug: fall back to best rejected variant instead of raising
        print(f"[DEBUG] Design-quality gate rejected learned layout — showing anyway for debugging.")
        best["_design_score"] = top_rejection.get("score", 0.0)
        best["_design_reasons"] = top_rejection.get("reasons", [])
        debug_gate_bypassed = True
    else:
        best = passed[0]

    svg_path = _quiet_call(
        save_svg_blueprint,
        building,
        output_path=str(output_paths["svg"]),
        boundary_polygon=boundary,
        entrance_point=entrance,
        title="NL Interface - Learned Run",
    )

    chosen_result = {
        "source": "learned",
        "input_spec": spec,
        "modifications": best.get("violations", []),
        "metrics": best.get("metrics", {}),
        "bounding_box": _bbox_from_boundary(boundary),
        "raw_validity": best.get("raw_valid", False),
        "repair_trace": best.get("repair_trace", []),
        "generation_summary": _json_safe_summary(summary),
        "wall_pipeline": getattr(building, "wall_render_stats", {}),
    }
    report = build_compliance_report(chosen_result)

    ontology_rooms = _load_ontology_room_types(regulation_file)

    evidence = build_evidence(
        chosen_result,
        report,
        variant_id=chosen_result.get("strategy_name"),
        design_score=best.get("_design_score"),
        design_reasons=best.get("_design_reasons", []),
        alternatives=[],
        design_filter_stats=design_stats,
    )

    import hashlib

    evidence_hash = hashlib.sha256(json.dumps(evidence, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    llm_latency_ms = None
    if llm_fn is not None:
        import time

        _start = time.time()
        explanation = explain(
            evidence,
            ontology_room_types=ontology_rooms,
            status=report.get("status", "UNKNOWN"),
            llm_fn=llm_fn,
        )
        llm_latency_ms = round((time.time() - _start) * 1000, 2)
    else:
        explanation = explain(
            evidence,
            ontology_room_types=ontology_rooms,
            status=report.get("status", "UNKNOWN"),
            llm_fn=None,
        )
    report["explanation"] = explanation
    save_compliance_report(report, str(output_paths["report"]))

    requested_rooms = _spec_room_counts(spec)
    generated_rooms = _building_room_counts(building)

    llm_warning_effective = llm_warning
    if llm_fn is not None and llm_warning_effective is None and isinstance(explanation, dict):
        if explanation.get("_llm_validation_errors"):
            llm_warning_effective = "validation_failed"

    return {
        "status": "completed",
        "backend_target": "learned",
        "strategy_name": best.get("strategy_name", "learned-best"),
        "design_score": best.get("_design_score"),
        "design_reasons": best.get("_design_reasons", []),
        "design_filter_stats": design_stats,
        "report_status": report.get("status"),
        "explanation": explanation,
        "debug_gate_bypassed": debug_gate_bypassed,
        "llm": {
            "used": llm_fn is not None,
            "provider": llm_provider or "deterministic",
            "model": llm_model,
            "latency_ms": llm_latency_ms,
            "warning": llm_warning_effective,
            "evidence_hash": evidence_hash,
        },
        "requested_rooms": requested_rooms,
        "generated_rooms": generated_rooms,
        "room_coverage": _room_coverage(requested_rooms, generated_rooms),
        "violations": report.get("violations", []),
        "artifact_paths": {
            "svg": str(svg_path),
            "report": str(output_paths["report"]),
        },
        "metrics": {
            "fully_connected": best.get("metrics", {}).get("fully_connected"),
            "max_travel_distance": best.get("metrics", {}).get("max_travel_distance"),
            "max_allowed_travel_distance": best.get("metrics", {}).get("max_allowed_travel_distance"),
            "adjacency_satisfaction": best.get("metrics", {}).get("adjacency_satisfaction"),
            "alignment_score": best.get("metrics", {}).get("alignment_score"),
        },
        "generation_summary": {
            "raw_valid_count": summary.get("raw_valid_count", 0),
            "repaired_valid_count": summary.get("repaired_valid_count", 0),
            "total_attempts": summary.get("total_attempts", 0),
        },
    }


def run_hybrid_backend(
    spec: Dict,
    *,
    output_dir: str = "outputs",
    output_prefix: Optional[str] = None,
    regulation_file: str = DEFAULT_REGULATION_FILE,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    device: str = "cpu",
    k: int = 10,
    top_m: int = 3,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 0,
    llm_fn=None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_warning: Optional[str] = None,
) -> Dict:
    """Run both algorithmic and learned backends, pool variants, rank, and return the best."""
    output_paths = _output_paths(output_dir, output_prefix, "hybrid")
    all_variants = []
    learned_variants = []  # Track learned variants separately for logging
    algo_variants = []     # Track algorithmic variants separately for logging
    learned_spatial_hints: dict = {}

    # 1. GENERATE FROM LEARNED TRANSFORMER (run first to extract spatial hints)
    try:
        resolved_ckpt = _resolve_checkpoint(checkpoint_path)
        boundary = spec.get("boundary_polygon")
        entrance = spec.get("entrance_point")

        best_learned, learned_summary = _quiet_call(
            generate_best_layout_from_model,
            spec=spec,
            boundary_poly=boundary,
            entrance=entrance,
            checkpoint_path=str(resolved_ckpt),
            regulation_file=regulation_file,
            K=max(1, top_m),
            max_attempts=max(1, k),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            device=device,
            pre_rank_top_m=max(1, top_m),
        )
        if best_learned and best_learned.get("building"):
            # Collect spatial hints for the algorithmic packer
            learned_spatial_hints = best_learned.get("learned_spatial_hints", {})

            # Format learned variant similarly to algorithmic variants
            learned_variant = {
                "source": "learned",
                "input_spec": spec,
                "building": best_learned["building"],
                "modifications": best_learned.get("violations", []),
                "metrics": best_learned.get("metrics", {}),
                "bounding_box": _bbox_from_boundary(boundary) if boundary else None,
                "raw_validity": best_learned.get("raw_valid", False),
                "repair_trace": best_learned.get("repair_trace", []),
                "generation_summary": _json_safe_summary(learned_summary),
                "wall_pipeline": getattr(best_learned["building"], "wall_render_stats", {}),
            }
            learned_variants.append(learned_variant)
            all_variants.append(learned_variant)
    except Exception as e:
        print(f"Learned generation failed: {e}")

    # 2. GENERATE FROM ALGORITHMIC PACKING (seeded with learned spatial hints)
    try:
        repaired = validate_and_repair_spec(spec, validate_spec, max_attempts=3)
        working_spec = repaired["spec"]
        working_spec["_spec_validation"] = repaired.get("validation", {})
        working_spec["_repair"] = {"repair_attempts": repaired.get("repair_attempts", 0)}
        working_spec["learned_checkpoint"] = "__disabled_checkpoint__.pt"

        # Inject transformer spatial hints so PolygonPacker can use them for
        # bisection ordering (rooms still get polygon shapes, not rectangles)
        if learned_spatial_hints:
            working_spec["learned_spatial_hints"] = learned_spatial_hints

        algo_result = _quiet_call(generate_layout_from_spec, working_spec, regulation_file=regulation_file)
        algo_variants = [v for v in algo_result.get("layout_variants", [algo_result]) if v.get("source") == "algorithmic"]
        all_variants.extend(algo_variants)
    except Exception as e:
        print(f"Algorithmic packing failed: {e}")
        algo_variants = []


    if not all_variants:
        print(f"[GENERATION_ERROR] No variants produced. Learned: {len(learned_variants)}, Algo: {len(algo_variants)}")
        raise ValueError("Both Algorithmic and Learned generators failed to produce any variants.")

    # 3. RANK AND FILTER (Bonus is applied in rank_layout_variants)
    print(f"[GENERATION] Ranking {len(all_variants)} total variants (Learned: {len(learned_variants)}, Algo: {len(algo_variants)})")
    ranked_variants, _ = rank_layout_variants(all_variants)
    passed, rejected, design_stats = _design_filter(ranked_variants)

    if not passed:
        top_rejection = rejected[0] if rejected else {"_design_reasons": ["No variants generated"], "strategy_name": "n/a"}
        top_reasons = top_rejection.get("_design_reasons") or top_rejection.get("reasons", [])
        print(f"[GENERATION_ERROR] All variants rejected. Top reasons: {top_reasons}")
        print(f"[GENERATION_ERROR] Design stats: {design_stats}")
        if rejected:
            print(f"[GENERATION_ERROR] First 3 rejections:")
            for i, rej in enumerate(rejected[:3]):
                rejection_reasons = rej.get("_design_reasons") or rej.get("reasons", [])
                print(f"  {i+1}. {rej.get('strategy_name', 'unknown')}: {', '.join(rejection_reasons)}")
        raise ValueError(f"Design-quality gate rejected all variants from both engines: {', '.join(top_reasons)}")

    chosen = passed[0]
    building = chosen["building"]
    
    # Provide the boundary polygon and entrance specifically as they were passed to the winning variant
    boundary_to_draw = spec.get("boundary_polygon") if chosen.get("source") == "learned" else spec.get("boundary_polygon")
    entrance_to_draw = spec.get("entrance_point") if chosen.get("source") == "learned" else spec.get("entrance_point")

    svg_path = _quiet_call(
        save_svg_blueprint,
        building,
        output_path=str(output_paths["svg"]),
        boundary_polygon=boundary_to_draw,
        entrance_point=entrance_to_draw,
        title=f"NL Interface - Hybrid Run ({str(chosen.get('source') or 'unknown').capitalize()} Won)",
    )

    report = build_compliance_report(chosen)
    ontology_rooms = _load_ontology_room_types(regulation_file)

    alternatives_evidence = []
    for alt in passed[1:4]:
        alternatives_evidence.append(
            {
                "strategy_name": alt.get("strategy_name", alt.get("source", "unknown")),
                "design_score": alt.get("_design_score"),
                "design_reasons": alt.get("_design_reasons", []),
                "metrics": _safe_metric_snapshot(alt.get("metrics", {})),
                "diversity_from_selected": _diversity_score(chosen, alt) if alt.get("source") == chosen.get("source") else 1.0, # Simplified diversity across sources
                "source": alt.get("source"),
            }
        )

    evidence = build_evidence(
        chosen,
        report,
        variant_id=chosen.get("strategy_name", chosen.get("source", "hybrid-best")),
        design_score=chosen.get("_design_score"),
        design_reasons=chosen.get("_design_reasons", []),
        alternatives=alternatives_evidence,
        design_filter_stats=design_stats,
    )

    import hashlib
    evidence_hash = hashlib.sha256(json.dumps(evidence, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    llm_latency_ms = None
    if llm_fn is not None:
        import time
        _start = time.time()
        explanation = explain(
            evidence,
            ontology_room_types=ontology_rooms,
            status=report.get("status", "UNKNOWN"),
            llm_fn=llm_fn,
        )
        llm_latency_ms = round((time.time() - _start) * 1000, 2)
    else:
        explanation = explain(
            evidence,
            ontology_room_types=ontology_rooms,
            status=report.get("status", "UNKNOWN"),
            llm_fn=None,
        )
    
    report["explanation"] = explanation
    save_compliance_report(report, str(output_paths["report"]))

    requested_rooms = _spec_room_counts(spec)
    generated_rooms = _building_room_counts(building)

    llm_warning_effective = llm_warning
    if llm_fn is not None and llm_warning_effective is None and isinstance(explanation, dict):
        if explanation.get("_llm_validation_errors"):
            llm_warning_effective = "validation_failed"

    return {
        "status": "completed",
        "backend_target": "hybrid",
        "winning_source": chosen.get("source"),
        "strategy_name": chosen.get("strategy_name", f"{chosen.get('source')}-best"),
        "design_score": chosen.get("_design_score"),
        "design_reasons": chosen.get("_design_reasons", []),
        "report_status": report.get("status"),
        "explanation": explanation,
        "debug_gate_bypassed": debug_gate_bypassed,
        "llm": {
            "used": llm_fn is not None,
            "provider": llm_provider or "deterministic",
            "model": llm_model,
            "latency_ms": llm_latency_ms,
            "warning": llm_warning_effective,
            "evidence_hash": evidence_hash,
        },
        "requested_rooms": requested_rooms,
        "generated_rooms": generated_rooms,
        "room_coverage": _room_coverage(requested_rooms, generated_rooms),
        "violations": report.get("violations", []),
        "alternatives": alternatives_evidence,
        "design_filter_stats": design_stats,
        "artifact_paths": {
            "svg": str(svg_path),
            "report": str(output_paths["report"]),
        },
        "metrics": _safe_metric_snapshot(chosen.get("metrics", {})),
        "generation_summary": chosen.get("generation_summary", {}),
    }


def _output_paths(output_dir: str, output_prefix: Optional[str], backend_target: str) -> Dict[str, Path]:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or f"nl_{backend_target}"
    return {
        "svg": base / f"{prefix}_blueprint.svg",
        "report": base / f"{prefix}_compliance_report.json",
    }


def _inflate_rectangular_boundary_spec(
    spec: Dict,
    *,
    rejection_reasons: Optional[List[str]] = None,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    boundary = spec.get("boundary_polygon") or []
    if len(boundary) < 4:
        return None, None

    try:
        xs = [float(point[0]) for point in boundary]
        ys = [float(point[1]) for point in boundary]
    except (TypeError, ValueError, IndexError):
        return None, None

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return None, None

    boundary_area = width * height
    room_count = len(spec.get("rooms", []) or [])
    area_scale = 1.10
    if room_count >= 7:
        area_scale += 0.08
    if rejection_reasons and "Circulation area is too high" in rejection_reasons:
        area_scale += 0.05
    if rejection_reasons and "Corridor width below 1.0 m" in rejection_reasons:
        area_scale += 0.03

    linear_scale = math.sqrt(area_scale)
    new_width = round(width * linear_scale, 1)
    new_height = round(height * linear_scale, 1)
    if new_width <= width and new_height <= height:
        return None, None

    recovered = deepcopy(spec)
    recovered["boundary_polygon"] = [
        [x_min, y_min],
        [round(x_min + new_width, 4), y_min],
        [round(x_min + new_width, 4), round(y_min + new_height, 4)],
        [x_min, round(y_min + new_height, 4)],
    ]

    entrance_point = spec.get("entrance_point")
    if entrance_point and len(entrance_point) >= 2:
        ex = float(entrance_point[0])
        ey = float(entrance_point[1])
        rel_x = 0.5 if width <= 0 else (ex - x_min) / width
        rel_y = 0.5 if height <= 0 else (ey - y_min) / height
        recovered["entrance_point"] = [
            round(x_min + rel_x * new_width, 4),
            round(y_min + rel_y * new_height, 4),
        ]

    auto_dimensions = dict(recovered.get("auto_dimensions") or {})
    auto_dimensions["width_m"] = new_width
    auto_dimensions["height_m"] = new_height
    auto_dimensions["area_sqm"] = round(new_width * new_height, 2)
    recovered["auto_dimensions"] = auto_dimensions

    return recovered, {
        "reason": "auto_boundary_recovery",
        "prior_boundary_size": [round(width, 2), round(height, 2)],
        "new_boundary_size": [new_width, new_height],
        "prior_area_sqm": round(boundary_area, 2),
        "new_area_sqm": round(new_width * new_height, 2),
        "area_scale": round((new_width * new_height) / boundary_area, 4),
    }


def _bbox_from_boundary(boundary: List[Tuple[float, float]]) -> Dict:
    """Compute bounding box from boundary polygon. Returns empty dict if boundary is invalid."""
    if not boundary or len(boundary) < 3:
        return {}
    try:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        return {
            "x_min": min(xs),
            "y_min": min(ys),
            "x_max": max(xs),
            "y_max": max(ys),
        }
    except (TypeError, IndexError, ValueError) as e:
        print(f"[WARNING] Invalid boundary polygon: {e}")
        return {}


def _planner_direct_exit_segment(
    boundary_polygon: List[Tuple[float, float]],
    entrance_point: Tuple[float, float],
    width: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs = [p[0] for p in boundary_polygon]
    ys = [p[1] for p in boundary_polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    ex, ey = entrance_point
    distances = {
        "left": abs(ex - x_min),
        "right": abs(ex - x_max),
        "bottom": abs(ey - y_min),
        "top": abs(ey - y_max),
    }
    side = min(distances, key=distances.get)
    half = width / 2.0
    if side == "left":
        center = min(max(ey, y_min + half), y_max - half) if y_max - y_min > width else (y_min + y_max) / 2.0
        return ((x_min, center - half), (x_min, center + half))
    if side == "right":
        center = min(max(ey, y_min + half), y_max - half) if y_max - y_min > width else (y_min + y_max) / 2.0
        return ((x_max, center - half), (x_max, center + half))
    if side == "bottom":
        center = min(max(ex, x_min + half), x_max - half) if x_max - x_min > width else (x_min + x_max) / 2.0
        return ((center - half, y_min), (center + half, y_min))
    center = min(max(ex, x_min + half), x_max - half) if x_max - x_min > width else (x_min + x_max) / 2.0
    return ((center - half, y_max), (center + half, y_max))



def _attach_planner_direct_doors(
    building,
    synthesized_rooms: List[Dict],
    *,
    boundary_polygon: List[Tuple[float, float]],
    entrance_point: Optional[Tuple[float, float]],
) -> None:
    from core.door import Door
    from learned.planner.geometry_synthesis import synthesize_simple_doors

    room_lookup = {room.name: room for room in getattr(building, "rooms", [])}
    for door in synthesize_simple_doors(synthesized_rooms, boundary_polygon, entrance_point):
        room_a = room_lookup.get(door.get("from_room"))
        room_b = room_lookup.get(door.get("to_room")) if door.get("to_room") else None
        segment = door.get("segment")
        if room_a is None or segment is None:
            continue
        placed = Door(
            room_a,
            room_b,
            float(door.get("width", 0.9) or 0.9),
            segment,
            door_type=door.get("door_type", "room_to_room"),
        )
        building.add_door(placed)
        room_a.doors.append(placed)
        if room_b is not None:
            room_b.doors.append(placed)



def _planner_direct_metrics(
    building,
    *,
    verification: Dict,
    regulation_file: str,
    boundary_polygon: List[Tuple[float, float]],
    entrance_point: Optional[Tuple[float, float]],
    spec: Optional[Dict],
    skip_corridors: bool,
) -> Dict:
    from constraints.rule_engine import RuleEngine
    from generator.composition_metrics import composition_quality
    from geometry.adjacency_intent import adjacency_satisfaction_score
    from geometry.zoning import assign_room_zones
    from graph.connectivity import is_fully_connected
    from graph.door_graph_path import door_graph_travel_distance, get_room_travel_distances
    from graph.manhattan_path import max_travel_distance

    engine = RuleEngine(regulation_file)
    engine.set_plot_area_from_boundary(boundary_polygon)
    max_allowed_travel = engine.get_max_travel_distance(building.occupancy_type)

    centroid_travel = max_travel_distance(building)
    door_travel = door_graph_travel_distance(building)
    room_travel = get_room_travel_distances(building)
    reachable_distances = [dist for dist in room_travel.values() if dist < 999.0]
    connectivity_to_exit = bool(room_travel) and len(reachable_distances) == len(room_travel)
    effective_travel = centroid_travel
    if door_travel < 999.0:
        effective_travel = max(effective_travel, door_travel)

    zone_map = assign_room_zones(building, entrance_point=entrance_point)
    adjacency_score, adjacency_details = adjacency_satisfaction_score(building)
    composition = composition_quality(building, entrance_point, zone_map, adjacency_details)

    requested_rooms = {
        room.get("name"): float(room.get("area", 0.0) or 0.0)
        for room in (spec or {}).get("rooms", []) or []
        if isinstance(room, dict) and room.get("name")
    }
    max_room_area_error = 0.0
    room_area_errors: List[Dict[str, float]] = []
    for room in getattr(building, "rooms", []) or []:
        target_area = float(requested_rooms.get(room.name, getattr(room, "target_area", 0.0) or 0.0))
        actual_area = float(getattr(room, "final_area", 0.0) or 0.0)
        relative_error = abs(actual_area - target_area) / max(target_area, 1e-6) if target_area > 0 else 0.0
        room_area_errors.append(
            {
                "room": room.name,
                "target_area": round(target_area, 3),
                "actual_area": round(actual_area, 3),
                "relative_error": round(relative_error, 4),
            }
        )
        max_room_area_error = max(max_room_area_error, relative_error)

    return {
        "total_area": round(float(getattr(building, "total_area", 0.0) or 0.0), 2),
        "fully_connected": is_fully_connected(building),
        "zone_map": zone_map,
        "adjacency_satisfaction": adjacency_score,
        "adjacency_details": adjacency_details,
        "alignment_score": verification.get("metrics", {}).get("distribution_score", 0.5),
        "overlap_free_ratio": verification.get("metrics", {}).get("overlap_free_ratio", 1.0),
        "boundary_containment": verification.get("metrics", {}).get("boundary_containment", 1.0),
        "area_compliance": verification.get("metrics", {}).get("area_compliance", 0.0),
        "area_details": verification.get("metrics", {}).get("area_details", {}),
        "max_room_area_error": round(max_room_area_error, 4),
        "room_area_errors": room_area_errors,
        "required_exit_width": building.exit.width if getattr(building, "exit", None) else 0.0,
        "max_travel_distance": round(effective_travel, 2),
        "max_allowed_travel_distance": float(max_allowed_travel),
        "travel_distance_compliant": effective_travel <= max_allowed_travel,
        "corridor_width": 0.0,
        "circulation_walkable_area": 0.0,
        "connectivity_to_exit": connectivity_to_exit,
        "door_path_travel_distance": round(door_travel, 2) if door_travel < 999.0 else 999.0,
        "room_travel_distances": room_travel,
        "skip_corridors": bool(skip_corridors),
        "public_frontage_score": composition["public_frontage_score"],
        "bedroom_privacy_score": composition["bedroom_privacy_score"],
        "kitchen_living_score": composition["kitchen_living_score"],
        "bathroom_access_score": composition["bathroom_access_score"],
        "architectural_reasonableness": composition["architectural_reasonableness"],
    }

def _json_safe_summary(summary: Dict) -> Dict:
    safe = dict(summary)
    safe_candidates = []
    for cand in safe.get("all_candidates", []):
        safe_cand = {k: v for k, v in cand.items() if k not in ("raw_rooms", "building")}
        safe_candidates.append(safe_cand)
    safe["all_candidates"] = safe_candidates
    return safe


def _quiet_call(func, *args, **kwargs):
    """Call function with suppressed stdout, but preserve stderr for debugging."""
    stdout_sink = StringIO()
    stderr_sink = StringIO()
    try:
        with redirect_stdout(stdout_sink), redirect_stderr(stderr_sink):
            result = func(*args, **kwargs)
        # If there were errors, log them
        stderr_content = stderr_sink.getvalue()
        if stderr_content:
            print(f"[DEBUG] {func.__name__} stderr: {stderr_content[:500]}")
        return result
    except Exception as e:
        # Don't swallow exceptions - re-raise with context
        stderr_content = stderr_sink.getvalue()
        if stderr_content:
            print(f"[ERROR] {func.__name__} failed with stderr: {stderr_content[:500]}")
        raise


def _resolve_checkpoint(checkpoint_path: str) -> Path:
    env_override = os.getenv(CHECKPOINT_ENV_VAR)
    candidate = Path(env_override) if env_override else Path(checkpoint_path)
    if not candidate.exists():
        raise ValueError(
            f"Learned checkpoint not found at '{candidate}'. "
            f"Provide --checkpoint or set {CHECKPOINT_ENV_VAR} to a valid path."
        )
    return candidate


def _load_ontology_room_types(regulation_file: str) -> set:
    try:
        data = _json_load(regulation_file)
        rooms = set()
        for occupancy in (data or {}).values():
            rooms.update((occupancy.get("rooms", {}) or {}).keys())
        return rooms or {"Bedroom", "LivingRoom", "Kitchen", "Bathroom", "WC"}
    except Exception:
        return {"Bedroom", "LivingRoom", "Kitchen", "Bathroom", "WC"}


def _json_load(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _spec_room_counts(spec: Dict) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for room in spec.get("rooms", []) or []:
        room_type = room.get("type") if isinstance(room, dict) else None
        if not room_type:
            continue
        counts[room_type] = counts.get(room_type, 0) + 1
    return counts


def _building_room_counts(building) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for room in getattr(building, "rooms", []) or []:
        room_type = getattr(room, "room_type", None)
        if not room_type:
            continue
        counts[room_type] = counts.get(room_type, 0) + 1
    return counts


def _room_coverage(requested: Dict[str, int], generated: Dict[str, int]) -> Dict[str, List[str]]:
    missing: List[str] = []
    extra: List[str] = []
    matched: List[str] = []

    room_types = set(requested) | set(generated)
    for room_type in sorted(room_types):
        req = requested.get(room_type, 0)
        got = generated.get(room_type, 0)
        if req == got:
            matched.append(room_type)
        elif got < req:
            missing.append(f"{room_type}: expected {req}, got {got}")
        else:
            extra.append(f"{room_type}: expected {req}, got {got}")

    return {
        "matched": matched,
        "missing": missing,
        "extra": extra,
    }


def _spec_without_all_planner_guidance(spec: Dict) -> Dict:
    cleaned = deepcopy(spec)
    cleaned.pop("planner_guidance", None)
    cleaned.pop("learned_spatial_hints", None)
    return cleaned


def _spec_without_learned_planner_guidance(spec: Dict) -> Dict:
    cleaned = deepcopy(spec)
    guidance = cleaned.get("planner_guidance") or {}
    guidance_source = str(guidance.get("source") or "").strip().lower()
    if guidance_source in STRUCTURED_PLANNER_GUIDANCE_SOURCES:
        if guidance.get("spatial_hints") and not cleaned.get("learned_spatial_hints"):
            cleaned["learned_spatial_hints"] = deepcopy(guidance.get("spatial_hints", {}))
        return cleaned
    cleaned.pop("planner_guidance", None)
    cleaned.pop("learned_spatial_hints", None)
    return cleaned







def _design_filter(variants: List[Dict]) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    passed: List[Dict] = []
    rejected: List[Dict] = []
    for variant in variants:
        ok, score, reasons = _design_gate(variant)
        variant["_design_score"] = score
        variant["_design_reasons"] = reasons
        if ok:
            passed.append(variant)
        else:
            rejected.append(variant)
    passed.sort(key=lambda v: v.get("_design_score", 0), reverse=True)
    rejected.sort(key=lambda v: v.get("_design_score", 0), reverse=True)
    stats = {
        "accepted_count": len(passed),
        "rejected_count": len(rejected),
        "rejection_reasons_counts": _reason_counts(rejected),
    }
    return passed, rejected, stats
def _design_gate(variant: Dict) -> Tuple[bool, float, List[str]]:
    m = variant.get("metrics", {}) or {}
    reasons: List[str] = []
    source = variant.get("source", "algorithmic")

    # planner_direct mode has relaxed connectivity requirements
    is_planner_direct = source == "planner_direct"
    skip_corridors = bool(m.get("skip_corridors"))

    fully_connected = bool(m.get("fully_connected", False))
    if not fully_connected and not is_planner_direct:
        reasons.append("Not fully connected")

    max_travel = float(m.get("max_travel_distance", 0) or 0)
    max_allowed = float(m.get("max_allowed_travel_distance", 1) or 1)
    travel_margin = max(0.0, min(1.0, (max_allowed - max_travel) / max_allowed)) if max_allowed > 0 else 0.0
    if max_travel > max_allowed and not is_planner_direct:
        reasons.append("Travel distance exceeds limit")

    adjacency = float(m.get("adjacency_satisfaction", 0) or 0)
    if adjacency < 0.2 and not is_planner_direct:
        reasons.append("Adjacency satisfaction below 0.2")

    alignment = float(m.get("alignment_score", 0) or 0)
    if alignment < 0.45 and not is_planner_direct:
        reasons.append("Alignment score below 0.45")

    max_room_area_error = float(m.get("max_room_area_error", 0) or 0)
    if max_room_area_error > 0.35 and not is_planner_direct:
        reasons.append("Room area allocation drift too high")

    total_area = float(m.get("total_area", 0) or 0)
    circulation_area = float(m.get("circulation_walkable_area", 0) or 0)
    circulation_ratio = (circulation_area / total_area) if total_area > 0 else 0.0
    if circulation_ratio > 0.16 and not is_planner_direct and not skip_corridors:
        reasons.append("Circulation area is too high")

    public_frontage = float(m.get("public_frontage_score", 0.0) or 0.0)
    if public_frontage and public_frontage < 0.40 and not is_planner_direct:
        reasons.append("Public frontage near entrance is too weak")

    bedroom_privacy = float(m.get("bedroom_privacy_score", 0.0) or 0.0)
    if bedroom_privacy and bedroom_privacy < 0.35 and not is_planner_direct:
        reasons.append("Bedroom privacy zoning is too weak")

    kitchen_living_score = float(m.get("kitchen_living_score", 0.0) or 0.0)
    if kitchen_living_score and kitchen_living_score < 0.5 and not is_planner_direct:
        reasons.append("Kitchen to living relationship is too weak")

    bathroom_access = float(m.get("bathroom_access_score", 0.0) or 0.0)
    if bathroom_access and bathroom_access < 0.5 and not is_planner_direct:
        reasons.append("Bathroom access from bedrooms is too weak")

    architectural_reasonableness = float(m.get("architectural_reasonableness", 0.0) or 0.0)
    if architectural_reasonableness and architectural_reasonableness < 0.45 and not is_planner_direct:
        reasons.append("Overall residential composition is too weak")

    # Corridor check - skip for planner_direct since user requested no corridors
    corridor_width = float(m.get("corridor_width", 0) or 0)
    if corridor_width < 1.0 and not is_planner_direct and not skip_corridors:
        reasons.append("Corridor width below 1.0 m")

    corridor_norm = min(1.0, corridor_width / 1.5) if corridor_width > 0 else 0.0

    # Scoring differs by source
    if is_planner_direct:
        # For planner_direct: use verification metrics instead of corridor
        overlap_free = float(m.get("overlap_free_ratio", 1.0) or 1.0)
        boundary_containment = float(m.get("boundary_containment", 1.0) or 1.0)
        score = (
            0.30 * overlap_free
            + 0.25 * boundary_containment
            + 0.25 * adjacency
            + 0.20 * alignment
        )
    else:
        # Original scoring for algorithmic/learned/planner backends
        algo_bonus = 0.25 if source == "algorithmic" else 0.0
        score = (
            0.35 * adjacency
            + 0.25 * alignment
            + 0.20 * travel_margin
            + 0.20 * corridor_norm
            + 0.15 * architectural_reasonableness
            - 0.20 * min(1.0, max_room_area_error)
            - 0.20 * min(1.0, circulation_ratio / 0.2)
            + algo_bonus
        )

    passed = not reasons
    return passed, round(score, 4), reasons


def _safe_metric_snapshot(metrics: Dict) -> Dict:
    keys = (
        "fully_connected",
        "max_travel_distance",
        "max_allowed_travel_distance",
        "adjacency_satisfaction",
        "alignment_score",
        "corridor_width",
        "max_room_area_error",
        "public_frontage_score",
        "bedroom_privacy_score",
        "kitchen_living_score",
        "bathroom_access_score",
        "architectural_reasonableness",
    )
    return {k: metrics.get(k) for k in keys if k in metrics}




def _travel_margin(metrics: Dict) -> float:
    max_travel = float(metrics.get("max_travel_distance", 0) or 0)
    max_allowed = float(metrics.get("max_allowed_travel_distance", 1) or 1)
    if max_allowed <= 0:
        return 0.0
    return max(0.0, min(1.0, (max_allowed - max_travel) / max_allowed))


def _variant_vector(variant: Dict) -> Tuple[float, float, float, float]:
    m = variant.get("metrics", {}) or {}
    adjacency = float(m.get("adjacency_satisfaction", 0) or 0)
    alignment = float(m.get("alignment_score", 0) or 0)
    travel_margin = _travel_margin(m)
    corridor_width = float(m.get("corridor_width", 0) or 0)
    corridor_norm = min(1.0, corridor_width / 1.5) if corridor_width > 0 else 0.0
    return (adjacency, alignment, travel_margin, corridor_norm)


def _diversity_score(selected: Dict, alternative: Dict) -> float:
    a = _variant_vector(selected)
    b = _variant_vector(alternative)
    return round(((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2 + (a[3]-b[3])**2) ** 0.5, 4)


def _reason_counts(rejected: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in rejected:
        for reason in item.get("_design_reasons") or item.get("reasons", []) or []:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _resolve_llm_fn(
    llm_provider: Optional[str],
    llm_api_key: Optional[str],
    llm_model: Optional[str],
):
    if not llm_provider:
        return None, None, None, None

    provider = llm_provider.lower()
    warning = None
    if provider == "gemini":
        api_key = llm_api_key or os.getenv("GEMINI_API_KEY")
        model = llm_model or "gemini-pro"
        try:
            from explain.gemini_adapter import build_gemini_llm_fn

            return build_gemini_llm_fn(api_key=api_key, model=model), None, "gemini", model
        except Exception as exc:  # pragma: no cover - optional dependency
            warning = f"Gemini LLM unavailable: {exc}"
            return None, warning, "gemini", model

    warning = f"Unsupported llm_provider '{llm_provider}'"
    return None, warning, llm_provider, llm_model





