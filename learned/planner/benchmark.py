"""Benchmark planner-guided packing against the algorithmic baseline."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from nl_interface.constants import CORE_ALGORITHMIC_ROOM_TYPES
from nl_interface.constraint_analyzer import calculate_optimal_dimensions, get_layout_requirements
from nl_interface.runner import run_algorithmic_backend, run_planner_backend
from nl_interface.service import process_user_request


CANONICAL_LAYOUT_TYPES = ("1BHK", "2BHK", "3BHK", "4BHK")
CANONICAL_ENTRANCE_SIDES = ("North", "East", "South", "West")
CANONICAL_SIZE_MODES = ("minimum", "recommended", "generous")

DEFAULT_BENCHMARK_CASES: List[Dict] = [
    {
        "name": "two_bedroom_open_kitchen",
        "prompt": "Create a 2 bedroom house with open kitchen near living room and minimize corridor.",
        "resolution": {"boundary_size": [12, 15], "entrance_point": [6, 0]},
    },
    {
        "name": "three_bedroom_family_layout",
        "prompt": "Need 3 bedrooms, 2 bathrooms, 1 kitchen and 1 drawing room on a 10 marla plot with a north entrance.",
        "resolution": {"boundary_size": [15, 10]},
    },
    {
        "name": "privacy_weighted_three_bedroom",
        "prompt": "Design a 3BHK house with north entrance, keep bedroom private, kitchen near living room, and minimize corridor.",
        "resolution": {"boundary_size": [15, 12], "entrance_point": [7.5, 0]},
    },
    {
        "name": "compact_two_bedroom",
        "prompt": "Need 2 bedrooms, 1 bathroom, 1 kitchen and 1 drawing room in a compact layout with a west entrance.",
        "resolution": {"boundary_size": [11, 14], "entrance_point": [0, 7]},
    },
]

CANONICAL_ACCEPTANCE_THRESHOLDS = {
    "adjacency_satisfaction": 0.20,
    "circulation_ratio_max": 0.18,
    "max_room_area_error": 0.35,
    "public_frontage_score": 0.35,
    "bedroom_privacy_score": 0.30,
    "kitchen_living_score": 0.50,
    "bathroom_access_score": 0.50,
    "architectural_reasonableness": 0.45,
}


def _slugify(text: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in text).strip("_")


def _status_rank(status: Optional[str]) -> int:
    if status == "COMPLIANT":
        return 2
    if status == "UNKNOWN":
        return 1
    return 0


def _size_mode_target_area(requirements: Dict, size_mode: str) -> float:
    minimum = float(requirements.get("min_total_area_sqm") or requirements.get("recommended_area_sqm") or 40.0)
    recommended = float(requirements.get("recommended_area_sqm") or minimum)
    maximum = float(requirements.get("max_total_area_sqm") or max(recommended, minimum))
    if size_mode == "minimum":
        return minimum
    if size_mode == "recommended":
        return recommended
    if size_mode == "generous":
        return max(recommended * 1.12, maximum)
    raise ValueError(f"Unsupported size mode '{size_mode}'")


def _entrance_point_for_side(width: float, height: float, side: str) -> List[float]:
    side = side.lower()
    if side == "north":
        return [round(width / 2.0, 3), 0.0]
    if side == "south":
        return [round(width / 2.0, 3), round(height, 3)]
    if side == "east":
        return [round(width, 3), round(height / 2.0, 3)]
    return [0.0, round(height / 2.0, 3)]


def _canonical_prompt(layout_type: str, entrance_side: str, room_counts: Dict[str, int], target_area: float) -> str:
    bedrooms = int(room_counts.get("Bedroom", 0))
    bathrooms = int(room_counts.get("Bathroom", 0))
    dining = int(room_counts.get("DiningRoom", 0))
    room_parts = [
        f"{bedrooms} bedroom{'s' if bedrooms != 1 else ''}",
        f"{bathrooms} bathroom{'s' if bathrooms != 1 else ''}",
        "1 kitchen",
        "1 living room",
    ]
    if dining:
        room_parts.append(f"{dining} dining room")
    joined = ", ".join(room_parts[:-1]) + f", and {room_parts[-1]}"
    return (
        f"Design a {layout_type} apartment with a {entrance_side.lower()} entrance, "
        f"{joined}, and a total area near {round(target_area, 1)} sqm."
    )


def build_canonical_residential_matrix(
    *,
    layout_types: Optional[Iterable[str]] = None,
    entrance_sides: Optional[Iterable[str]] = None,
    size_modes: Optional[Iterable[str]] = None,
) -> List[Dict]:
    cases: List[Dict] = []
    for layout_type in layout_types or CANONICAL_LAYOUT_TYPES:
        requirements = get_layout_requirements(layout_type)
        room_counts: Dict[str, int] = {}
        for room in requirements.get("rooms", []):
            room_counts[str(room["type"])] = room_counts.get(str(room["type"]), 0) + 1

        room_counts.pop("DiningRoom", None)

        for entrance_side in entrance_sides or CANONICAL_ENTRANCE_SIDES:
            for size_mode in size_modes or CANONICAL_SIZE_MODES:
                target_area = _size_mode_target_area(requirements, size_mode)
                dims = calculate_optimal_dimensions(layout_type, user_total_area=target_area)
                width = float(dims["width_m"])
                height = float(dims["height_m"])
                resolution = {
                    "boundary_size": [width, height],
                    "boundary_polygon": [
                        [0.0, 0.0],
                        [width, 0.0],
                        [width, height],
                        [0.0, height],
                    ],
                    "entrance_point": _entrance_point_for_side(width, height, entrance_side),
                    "total_area": float(dims["area_sqm"]),
                    "area_unit": "sq.m",
                }
                cases.append(
                    {
                        "name": f"{layout_type.lower()}_{entrance_side.lower()}_{size_mode}",
                        "prompt": _canonical_prompt(layout_type, entrance_side, room_counts, float(dims["area_sqm"])),
                        "layout_type": layout_type,
                        "entrance_side": entrance_side,
                        "size_mode": size_mode,
                        "expected_room_counts": dict(room_counts),
                        "resolution": resolution,
                    }
                )
    return cases


def prepare_case(case: Dict) -> Dict:
    prompt = case["prompt"]
    resolution = deepcopy(case.get("resolution") or {})
    response = process_user_request(prompt, resolution=resolution)
    if not response.get("backend_ready") or not response.get("backend_spec"):
        raise ValueError(
            f"Case '{case.get('name', prompt[:30])}' is not backend-ready: "
            f"{response.get('missing_fields')} {response.get('validation_errors')}"
        )

    room_types = {room.get("type") for room in response["backend_spec"].get("rooms", [])}
    unsupported = sorted(room_types - CORE_ALGORITHMIC_ROOM_TYPES)
    if unsupported:
        raise ValueError(
            f"Case '{case.get('name', prompt[:30])}' contains unsupported benchmark room types: {unsupported}"
        )

    return {
        "name": case.get("name") or _slugify(prompt) or "planner_case",
        "prompt": prompt,
        "layout_type": case.get("layout_type"),
        "entrance_side": case.get("entrance_side"),
        "size_mode": case.get("size_mode"),
        "expected_room_counts": deepcopy(case.get("expected_room_counts") or {}),
        "resolution": resolution,
        "backend_spec": deepcopy(response["backend_spec"]),
        "backend_translation_warnings": list(response.get("backend_translation_warnings", [])),
    }


def _generated_counts_match_expected(result: Dict, expected_counts: Dict[str, int]) -> bool:
    if not expected_counts:
        return True
    generated = result.get("generated_rooms") or {}
    return all(int(generated.get(room_type, 0) or 0) == int(expected) for room_type, expected in expected_counts.items())


def evaluate_acceptance(result: Dict, case: Dict, *, thresholds: Optional[Dict] = None) -> Dict:
    thresholds = {**CANONICAL_ACCEPTANCE_THRESHOLDS, **(thresholds or {})}
    metrics = result.get("metrics", {}) or {}
    coverage = result.get("room_coverage", {}) or {}
    total_area = float(metrics.get("total_area", 0.0) or 0.0)
    circulation_area = float(metrics.get("circulation_walkable_area", 0.0) or 0.0)
    circulation_ratio = (circulation_area / total_area) if total_area > 0 else 0.0

    checks = {
        "compliant": result.get("report_status") == "COMPLIANT",
        "room_program_complete": not coverage.get("missing") and not coverage.get("extra"),
        "expected_program_match": _generated_counts_match_expected(result, case.get("expected_room_counts") or {}),
        "fully_connected": bool(metrics.get("fully_connected", False)),
        "exit_connected": bool(metrics.get("connectivity_to_exit", True)),
        "adjacency_fit": float(metrics.get("adjacency_satisfaction", 0.0) or 0.0)
        >= float(thresholds["adjacency_satisfaction"]),
        "bounded_circulation": circulation_ratio <= float(thresholds["circulation_ratio_max"]),
        "room_area_drift_ok": float(metrics.get("max_room_area_error", 0.0) or 0.0)
        <= float(thresholds["max_room_area_error"]),
        "public_frontage_ok": float(metrics.get("public_frontage_score", 0.0) or 0.0)
        >= float(thresholds["public_frontage_score"]),
        "bedroom_privacy_ok": float(metrics.get("bedroom_privacy_score", 0.0) or 0.0)
        >= float(thresholds["bedroom_privacy_score"]),
        "kitchen_living_ok": float(metrics.get("kitchen_living_score", 0.0) or 0.0)
        >= float(thresholds["kitchen_living_score"]),
        "bathroom_access_ok": float(metrics.get("bathroom_access_score", 0.0) or 0.0)
        >= float(thresholds["bathroom_access_score"]),
        "architectural_reasonableness_ok": float(metrics.get("architectural_reasonableness", 0.0) or 0.0)
        >= float(thresholds["architectural_reasonableness"]),
    }

    accepted = all(checks.values())
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "accepted": accepted,
        "failed_checks": failures,
        "checks": checks,
        "circulation_ratio": round(circulation_ratio, 4),
        "thresholds": deepcopy(thresholds),
    }


def compact_result(result: Dict) -> Dict:
    return {
        "status": result.get("status"),
        "backend_target": result.get("backend_target"),
        "winning_source": result.get("winning_source"),
        "report_status": result.get("report_status"),
        "design_score": result.get("design_score"),
        "design_reasons": result.get("design_reasons", []),
        "room_coverage": result.get("room_coverage", {}),
        "generated_rooms": result.get("generated_rooms", {}),
        "metrics": result.get("metrics", {}),
        "planner_summary": result.get("planner_summary"),
        "artifact_paths": result.get("artifact_paths", {}),
    }


def compare_results(algorithmic_result: Dict, planner_result: Dict) -> Dict:
    algo_acceptance = algorithmic_result.get("acceptance", {})
    planner_acceptance = planner_result.get("acceptance", {})
    algo_accepted = bool(algo_acceptance.get("accepted"))
    planner_accepted = bool(planner_acceptance.get("accepted"))

    algo_status_rank = _status_rank(algorithmic_result.get("report_status"))
    planner_status_rank = _status_rank(planner_result.get("report_status"))
    algo_score = float(algorithmic_result.get("design_score") or 0.0)
    planner_score = float(planner_result.get("design_score") or 0.0)

    if planner_accepted and not algo_accepted:
        winner = "planner"
        reason = "acceptance_suite"
    elif algo_accepted and not planner_accepted:
        winner = "algorithmic"
        reason = "acceptance_suite"
    elif planner_status_rank > algo_status_rank:
        winner = "planner"
        reason = "better_compliance"
    elif planner_status_rank < algo_status_rank:
        winner = "algorithmic"
        reason = "better_compliance"
    elif planner_score > algo_score:
        winner = "planner"
        reason = "better_design_score"
    elif planner_score < algo_score:
        winner = "algorithmic"
        reason = "better_design_score"
    else:
        winner = "tie"
        reason = "equal_score"

    return {
        "winner": winner,
        "reason": reason,
        "design_score_delta": round(planner_score - algo_score, 4),
        "adjacency_delta": round(
            float(planner_result.get("metrics", {}).get("adjacency_satisfaction") or 0.0)
            - float(algorithmic_result.get("metrics", {}).get("adjacency_satisfaction") or 0.0),
            4,
        ),
        "alignment_delta": round(
            float(planner_result.get("metrics", {}).get("alignment_score") or 0.0)
            - float(algorithmic_result.get("metrics", {}).get("alignment_score") or 0.0),
            4,
        ),
    }


def summarize_benchmark(case_results: List[Dict]) -> Dict:
    completed = [case for case in case_results if case.get("comparison")]
    winner_counts = {"planner": 0, "algorithmic": 0, "tie": 0}
    score_deltas: List[float] = []
    adjacency_deltas: List[float] = []
    alignment_deltas: List[float] = []
    acceptance = {
        "algorithmic": {"accepted": 0, "rejected": 0},
        "planner": {"accepted": 0, "rejected": 0},
    }

    for case in completed:
        comparison = case["comparison"]
        winner_counts[comparison["winner"]] = winner_counts.get(comparison["winner"], 0) + 1
        score_deltas.append(float(comparison["design_score_delta"]))
        adjacency_deltas.append(float(comparison["adjacency_delta"]))
        alignment_deltas.append(float(comparison["alignment_delta"]))

        for backend_name in ("algorithmic", "planner"):
            accepted = bool(case.get(backend_name, {}).get("acceptance", {}).get("accepted"))
            acceptance[backend_name]["accepted" if accepted else "rejected"] += 1

    def _avg(values: List[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "case_count": len(case_results),
        "completed_case_count": len(completed),
        "winner_counts": winner_counts,
        "acceptance_counts": acceptance,
        "planner_acceptance_delta": acceptance["planner"]["accepted"] - acceptance["algorithmic"]["accepted"],
        "planner_avg_score_delta": _avg(score_deltas),
        "planner_avg_adjacency_delta": _avg(adjacency_deltas),
        "planner_avg_alignment_delta": _avg(alignment_deltas),
    }


def run_case(case: Dict, output_root: str, regulation_file: str, device: str) -> Dict:
    prepared = prepare_case(case)
    case_dir = Path(output_root) / prepared["name"]
    case_dir.mkdir(parents=True, exist_ok=True)

    algorithmic_raw = run_algorithmic_backend(
        deepcopy(prepared["backend_spec"]),
        output_dir=str(case_dir),
        output_prefix="algorithmic",
        regulation_file=regulation_file,
    )
    planner_raw = run_planner_backend(
        deepcopy(prepared["backend_spec"]),
        output_dir=str(case_dir),
        output_prefix="planner",
        regulation_file=regulation_file,
        device=device,
    )

    algorithmic_result = compact_result(algorithmic_raw)
    planner_result = compact_result(planner_raw)
    algorithmic_result["acceptance"] = evaluate_acceptance(algorithmic_raw, prepared)
    planner_result["acceptance"] = evaluate_acceptance(planner_raw, prepared)

    return {
        "name": prepared["name"],
        "prompt": prepared["prompt"],
        "layout_type": prepared.get("layout_type"),
        "entrance_side": prepared.get("entrance_side"),
        "size_mode": prepared.get("size_mode"),
        "expected_room_counts": prepared.get("expected_room_counts"),
        "backend_translation_warnings": prepared["backend_translation_warnings"],
        "algorithmic": algorithmic_result,
        "planner": planner_result,
        "comparison": compare_results(algorithmic_result, planner_result),
    }


def run_benchmark(
    cases: List[Dict],
    *,
    output_root: str = "outputs/planner_benchmark",
    output_json: str = "outputs/planner_benchmark/summary.json",
    regulation_file: str = "ontology/regulation_data.json",
    device: str = "cpu",
) -> Dict:
    case_results = []
    for case in cases:
        try:
            case_results.append(run_case(case, output_root=output_root, regulation_file=regulation_file, device=device))
        except Exception as exc:
            case_results.append(
                {
                    "name": case.get("name") or _slugify(case.get("prompt", "")) or "planner_case",
                    "prompt": case.get("prompt", ""),
                    "layout_type": case.get("layout_type"),
                    "entrance_side": case.get("entrance_side"),
                    "size_mode": case.get("size_mode"),
                    "error": str(exc),
                }
            )

    summary = summarize_benchmark(case_results)
    payload = {
        "summary": summary,
        "cases": case_results,
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _load_cases(path: Optional[str], case_set: str) -> List[Dict]:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    if case_set == "canonical_residential":
        return build_canonical_residential_matrix()
    return deepcopy(DEFAULT_BENCHMARK_CASES)


def main():
    parser = argparse.ArgumentParser(description="Benchmark planner-guided packing against the algorithmic baseline")
    parser.add_argument("--cases-json", default="", help="Optional JSON file containing benchmark cases")
    parser.add_argument(
        "--case-set",
        default="default",
        choices=["default", "canonical_residential"],
        help="Benchmark case set to run when --cases-json is not provided",
    )
    parser.add_argument("--output-root", default="outputs/planner_benchmark", help="Directory for benchmark artifacts")
    parser.add_argument(
        "--output-json",
        default="outputs/planner_benchmark/summary.json",
        help="JSON file where the benchmark summary is written",
    )
    parser.add_argument("--regulation-file", default="ontology/regulation_data.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    payload = run_benchmark(
        _load_cases(args.cases_json, args.case_set),
        output_root=args.output_root,
        output_json=args.output_json,
        regulation_file=args.regulation_file,
        device=args.device,
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
