"""Generate synthetic planner-training records from the deterministic algorithmic backend."""

from __future__ import annotations

import argparse
import json
import random
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple

from constraints.repair_loop import validate_and_repair_spec
from constraints.spec_validator import validate_spec
from core.building import Building
from core.room import Room
from generator.layout_generator import generate_layout_from_spec
from generator.ranking import rank_layout_variants
from learned.planner.data import build_planner_record
from nl_interface.adapter import build_backend_spec
from nl_interface.runner import _design_filter


PROGRAM_LIBRARY: List[Dict] = [
    {"name": "1bhk_starter", "plot_type": "5Marla", "rooms": {"Bedroom": 1, "Bathroom": 1, "Kitchen": 1, "LivingRoom": 1}},
    {"name": "2bhk_compact", "plot_type": "5Marla", "rooms": {"Bedroom": 2, "Bathroom": 1, "Kitchen": 1, "LivingRoom": 1}},
    {"name": "2bhk_balanced", "plot_type": "10Marla", "rooms": {"Bedroom": 2, "Bathroom": 2, "Kitchen": 1, "LivingRoom": 1}},
    {"name": "3bhk_family", "plot_type": "10Marla", "rooms": {"Bedroom": 3, "Bathroom": 2, "Kitchen": 1, "LivingRoom": 1}},
    {"name": "4bhk_large", "plot_type": "20Marla", "rooms": {"Bedroom": 4, "Bathroom": 3, "Kitchen": 1, "LivingRoom": 1}},
    {"name": "5bhk_large", "plot_type": "20Marla", "rooms": {"Bedroom": 5, "Bathroom": 3, "Kitchen": 1, "LivingRoom": 1}},
]

BOUNDARY_LIBRARY: Dict[str, List[Tuple[float, float]]] = {
    "5Marla": [(8.0, 14.0), (9.0, 13.0), (10.0, 12.0)],
    "10Marla": [(12.0, 15.0), (15.0, 10.0), (14.0, 12.0)],
    "20Marla": [(16.0, 18.0), (18.0, 16.0), (20.0, 15.0)],
}

STYLE_LIBRARY: List[Dict] = [
    {
        "name": "balanced",
        "weights": {"privacy": 0.3333, "compactness": 0.3333, "corridor": 0.3334},
        "minimize_corridor": False,
    },
    {
        "name": "compact",
        "weights": {"privacy": 0.2, "compactness": 0.5, "corridor": 0.3},
        "minimize_corridor": True,
    },
    {
        "name": "privacy",
        "weights": {"privacy": 0.55, "compactness": 0.25, "corridor": 0.2},
        "minimize_corridor": False,
    },
    {
        "name": "corridor_light",
        "weights": {"privacy": 0.2, "compactness": 0.25, "corridor": 0.55},
        "minimize_corridor": True,
    },
]

ENTRANCE_SIDES = ("North", "South", "East", "West")


def _quiet_call(func, *args, **kwargs):
    stdout_sink = StringIO()
    stderr_sink = StringIO()
    with redirect_stdout(stdout_sink), redirect_stderr(stderr_sink):
        return func(*args, **kwargs)


def build_teacher_adjacency(room_counts: Dict[str, int]) -> List[List[str]]:
    adjacency = [["Kitchen", "LivingRoom", "near_to"]]
    if room_counts.get("Bedroom", 0) and room_counts.get("Bathroom", 0):
        adjacency.append(["Bedroom", "Bathroom", "near_to"])
    return adjacency


def build_teacher_current_spec(case: Dict) -> Dict:
    room_entries = []
    for room_type in ("Bedroom", "Bathroom", "Kitchen", "LivingRoom"):
        count = int(case["rooms"].get(room_type, 0))
        if count > 0:
            room_entries.append({"type": room_type, "count": count})

    return {
        "building_type": "Residential",
        "plot_type": case["plot_type"],
        "entrance_side": case["entrance_side"],
        "rooms": room_entries,
        "preferences": {
            "adjacency": build_teacher_adjacency(case["rooms"]),
            "privacy": {},
            "minimize_corridor": bool(case["style"].get("minimize_corridor", False)),
        },
        "weights": dict(case["style"]["weights"]),
    }


def build_teacher_case_catalog() -> List[Dict]:
    catalog = []
    for program in PROGRAM_LIBRARY:
        for width, height in BOUNDARY_LIBRARY[program["plot_type"]]:
            for entrance_side in ENTRANCE_SIDES:
                for style in STYLE_LIBRARY:
                    catalog.append(
                        {
                            "name": f"{program['name']}_{int(width)}x{int(height)}_{entrance_side.lower()}_{style['name']}",
                            "plot_type": program["plot_type"],
                            "boundary_size": [width, height],
                            "entrance_side": entrance_side,
                            "rooms": dict(program["rooms"]),
                            "style": deepcopy(style),
                        }
                    )
    return catalog


def building_to_plan(building: Building, backend_spec: Dict, plan_id: str) -> Dict:
    rooms = []
    for room in getattr(building, "rooms", []):
        polygon = getattr(room, "polygon", None)
        if not polygon or len(polygon) < 3:
            continue
        rooms.append(
            {
                "type": room.room_type,
                "polygon": [[float(x), float(y)] for x, y in polygon],
            }
        )

    return {
        "plan_id": plan_id,
        "building_type": backend_spec.get("building_type", "Residential"),
        "plot_type": backend_spec.get("plot_type", "Custom"),
        "boundary_polygon": [list(point) for point in backend_spec.get("boundary_polygon", [])],
        "rooms": rooms,
    }


def generate_teacher_record(
    case: Dict,
    *,
    regulation_file: str = "ontology/regulation_data.json",
    allow_rejected: bool = False,
) -> Dict:
    current_spec = build_teacher_current_spec(case)
    resolution = {"boundary_size": list(case["boundary_size"]), "area_unit": "sq.m"}
    backend_spec, warnings = build_backend_spec(current_spec, resolution)
    if backend_spec is None:
        raise ValueError(f"Failed to build backend spec for case {case['name']}: {warnings}")

    repaired = validate_and_repair_spec(backend_spec, validate_spec, max_attempts=3)
    working_spec = repaired["spec"]
    working_spec["_spec_validation"] = repaired.get("validation", {})
    working_spec["_repair"] = {"repair_attempts": repaired.get("repair_attempts", 0)}
    working_spec["learned_checkpoint"] = "__disabled_checkpoint__.pt"

    result = _quiet_call(generate_layout_from_spec, working_spec, regulation_file=regulation_file)
    algo_variants = [variant for variant in result.get("layout_variants", [result]) if variant.get("source") == "algorithmic"]
    ranked_algo, _ = rank_layout_variants(algo_variants)
    passed, rejected, _ = _design_filter(ranked_algo)

    if passed:
        chosen = passed[0]
        fallback_used = False
    elif allow_rejected and rejected:
        chosen = rejected[0]
        fallback_used = True
    else:
        raise ValueError(f"No acceptable algorithmic layout generated for case {case['name']}")

    plan = building_to_plan(chosen["building"], backend_spec, case["name"])
    record = build_planner_record(plan)
    if record is None:
        raise ValueError(f"Planner record conversion failed for case {case['name']}")

    record["teacher_metadata"] = {
        "case_name": case["name"],
        "style": case["style"]["name"],
        "entrance_side": case["entrance_side"],
        "boundary_size": list(case["boundary_size"]),
        "design_score": chosen.get("_design_score"),
        "design_reasons": chosen.get("_design_reasons", []),
        "fallback_used": fallback_used,
        "backend_translation_warnings": warnings,
    }
    return record


def build_teacher_dataset(
    output_path: str,
    *,
    num_cases: int = 64,
    seed: int = 42,
    regulation_file: str = "ontology/regulation_data.json",
    allow_rejected: bool = False,
) -> Dict:
    catalog = build_teacher_case_catalog()
    rng = random.Random(seed)
    rng.shuffle(catalog)

    selected_cases = catalog[: min(num_cases, len(catalog))]
    records = []
    failures = []

    for case in selected_cases:
        try:
            records.append(
                generate_teacher_record(
                    case,
                    regulation_file=regulation_file,
                    allow_rejected=allow_rejected,
                )
            )
        except Exception as exc:
            failures.append({"name": case["name"], "error": str(exc)})

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(record) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )

    return {
        "requested_cases": len(selected_cases),
        "written_records": len(records),
        "failed_cases": len(failures),
        "failures": failures,
        "output_path": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic planner-training records from the algorithmic backend")
    parser.add_argument("--output", default="learned/planner/data/planner_teacher_records.jsonl")
    parser.add_argument("--num-cases", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regulation-file", default="ontology/regulation_data.json")
    parser.add_argument("--allow-rejected", action="store_true")
    args = parser.parse_args()

    summary = build_teacher_dataset(
        args.output,
        num_cases=args.num_cases,
        seed=args.seed,
        regulation_file=args.regulation_file,
        allow_rejected=args.allow_rejected,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
