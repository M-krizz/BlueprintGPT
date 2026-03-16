"""Execute backend-ready NL responses through the existing smoke pipelines."""

from __future__ import annotations

import os
import json
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
    "run_learned_backend",
    "run_hybrid_backend",
]

DEFAULT_REGULATION_FILE = "ontology/regulation_data.json"
DEFAULT_CHECKPOINT = "learned/model/checkpoints/kaggle_test.pt"
CHECKPOINT_ENV_VAR = "BLUEPRINTGPT_CHECKPOINT"


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
    if not passed:
        top_rejection = rejected[0] if rejected else {"reasons": ["No variants generated"], "strategy_name": "n/a"}
        raise ValueError(f"Design-quality gate rejected all variants: {', '.join(top_rejection['reasons'])}")

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
            "fully_connected": chosen.get("metrics", {}).get("fully_connected"),
            "max_travel_distance": chosen.get("metrics", {}).get("max_travel_distance"),
            "max_allowed_travel_distance": chosen.get("metrics", {}).get("max_allowed_travel_distance"),
            "adjacency_satisfaction": chosen.get("metrics", {}).get("adjacency_satisfaction"),
            "alignment_score": chosen.get("metrics", {}).get("alignment_score"),
        },
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
    if not passed:
        top_rejection = rejected[0]
        if os.getenv("BLUEPRINTGPT_ALLOW_DESIGN_FAIL") == "1":
            best["_design_score"] = top_rejection.get("score", 0.0)
            best["_design_reasons"] = top_rejection.get("reasons", [])
        else:
            raise ValueError(f"Design-quality gate rejected learned layout: {', '.join(top_rejection['reasons'])}")
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
        raise ValueError("Both Algorithmic and Learned generators failed to produce any variants.")

    # 3. RANK AND FILTER (Bonus is applied in rank_layout_variants)
    ranked_variants, _ = rank_layout_variants(all_variants)
    passed, rejected, design_stats = _design_filter(ranked_variants)
    
    if not passed:
        top_rejection = rejected[0] if rejected else {"reasons": ["No variants generated"], "strategy_name": "n/a"}
        raise ValueError(f"Design-quality gate rejected all variants from both engines: {', '.join(top_rejection['reasons'])}")

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


def _bbox_from_boundary(boundary: List[Tuple[float, float]]) -> Dict:
    xs = [p[0] for p in boundary]
    ys = [p[1] for p in boundary]
    return {
        "x_min": min(xs),
        "y_min": min(ys),
        "x_max": max(xs),
        "y_max": max(ys),
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
    sink = StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return func(*args, **kwargs)


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
            rejected.append({"strategy_name": variant.get("strategy_name"), "reasons": reasons, "score": score})
    passed.sort(key=lambda v: v.get("_design_score", 0), reverse=True)
    rejected.sort(key=lambda v: v.get("score", 0), reverse=True)
    stats = {
        "accepted_count": len(passed),
        "rejected_count": len(rejected),
        "rejection_reasons_counts": _reason_counts(rejected),
    }
    return passed, rejected, stats
def _design_gate(variant: Dict) -> Tuple[bool, float, List[str]]:
    m = variant.get("metrics", {}) or {}
    reasons: List[str] = []

    fully_connected = bool(m.get("fully_connected", False))
    if not fully_connected:
        reasons.append("Not fully connected")

    max_travel = float(m.get("max_travel_distance", 0) or 0)
    max_allowed = float(m.get("max_allowed_travel_distance", 1) or 1)
    travel_margin = max(0.0, min(1.0, (max_allowed - max_travel) / max_allowed)) if max_allowed > 0 else 0.0
    if max_travel > max_allowed:
        reasons.append("Travel distance exceeds limit")

    adjacency = float(m.get("adjacency_satisfaction", 0) or 0)
    if adjacency < 0.2:
        reasons.append("Adjacency satisfaction below 0.2")

    alignment = float(m.get("alignment_score", 0) or 0)
    if alignment < 0.45:
        reasons.append("Alignment score below 0.45")

    corridor_width = float(m.get("corridor_width", 0) or 0)
    if corridor_width < 1.0:
        reasons.append("Corridor width below 1.0 m")

    corridor_norm = min(1.0, corridor_width / 1.5) if corridor_width > 0 else 0.0

    # Add a bonus if the layout came from the algorithmic packing engine
    algo_bonus = 0.25 if variant.get("source") == "algorithmic" else 0.0

    score = (
        0.35 * adjacency
        + 0.25 * alignment
        + 0.20 * travel_margin
        + 0.20 * corridor_norm
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
        for reason in item.get("reasons", []) or []:
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
