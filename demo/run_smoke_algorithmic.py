from __future__ import annotations

import argparse
import json
from pathlib import Path

from constraints.compliance_report import build_compliance_report, save_compliance_report
from constraints.repair_loop import validate_and_repair_spec
from constraints.spec_validator import validate_spec
from generator.layout_generator import generate_layout_from_spec
from generator.ranking import rank_layout_variants
from visualization.export_svg_blueprint import save_svg_blueprint


ALGORITHMIC_ROOM_ALIASES = {
    "DrawingRoom": "LivingRoom",
    "Lobby": "LivingRoom",
    "Passage": None,
    "Stairs": None,
    "Staircase": None,
}


def _load_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)

    upgraded = []
    for idx, item in enumerate(spec.get("rooms", []), start=1):
        rtype = item.get("type")
        if not rtype:
            continue
        mapped = ALGORITHMIC_ROOM_ALIASES.get(rtype, rtype)
        if mapped is None:
            continue
        count = int(item.get("count", 1))
        for copy_idx in range(max(1, count)):
            upgraded.append({"name": f"{mapped}_{idx}_{copy_idx+1}", "type": mapped})

    spec["rooms"] = upgraded
    spec.setdefault("occupancy", "Residential")
    spec.setdefault("total_area", 1200)
    spec.setdefault("area_unit", "sq.ft")
    spec.setdefault("allocation_strategy", "priority_weights")
    return spec


def _boundary_from_arg(boundary_str: str):
    w, h = [float(x) for x in boundary_str.split(",")]
    return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]


def main():
    ap = argparse.ArgumentParser(description="Run algorithmic baseline smoke pipeline and export artifacts")
    ap.add_argument("--spec", default="learned/data/residential_spec.json")
    ap.add_argument("--regulation", default="ontology/regulation_data.json")
    ap.add_argument("--boundary", default="15,10")
    ap.add_argument("--entrance", default="0.2,0.0")
    ap.add_argument("--output-svg", default="outputs/blueprint_algorithmic.svg")
    ap.add_argument("--output-report", default="outputs/compliance_report_algorithmic.json")
    args = ap.parse_args()

    spec = _load_spec(args.spec)
    boundary = _boundary_from_arg(args.boundary)
    ex, ey = [float(x) for x in args.entrance.split(",")]
    entrance = (ex, ey)
    spec["boundary_polygon"] = boundary
    spec["entrance_point"] = entrance

    repaired = validate_and_repair_spec(spec, validate_spec, max_attempts=3)
    spec = repaired["spec"]
    spec["_spec_validation"] = repaired.get("validation", {})
    spec["_repair"] = {"repair_attempts": repaired.get("repair_attempts", 0)}

    result = generate_layout_from_spec(spec, regulation_file=args.regulation)
    algo_variants = [v for v in result.get("layout_variants", [result]) if v.get("source") == "algorithmic"]
    ranked_algo, _ = rank_layout_variants(algo_variants)
    chosen = ranked_algo[0]
    building = chosen["building"]

    svg_path = save_svg_blueprint(
        building,
        output_path=args.output_svg,
        boundary_polygon=boundary,
        entrance_point=entrance,
        title="Algorithmic Baseline - Smoke Test",
    )

    report = build_compliance_report(chosen)
    save_compliance_report(report, args.output_report)

    metrics = chosen.get("metrics", {})
    print("Algorithmic baseline smoke complete.")
    print(f"  Variant: {chosen.get('strategy_name')}")
    print(f"  Status: {report.get('status')}")
    print(f"  Connectivity: {metrics.get('fully_connected')}")
    print(f"  Travel: {metrics.get('max_travel_distance')} / {metrics.get('max_allowed_travel_distance')}")
    print(f"  Adjacency: {metrics.get('adjacency_satisfaction')}")
    print(f"  SVG: {Path(svg_path)}")
    print(f"  Report: {Path(args.output_report)}")


if __name__ == "__main__":
    main()
