"""
run_smoke_learned.py - One-command end-to-end smoke test for learned pipeline.

Pipeline
--------
1. Load spec + checkpoint
2. Generate K constrained samples
3. Pre-rank by adjacency proxy (inside model_generation_loop)
4. Repair + evaluate + rank
5. Export:
   - outputs/blueprint_learned.svg
   - outputs/compliance_report_learned.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from constraints.compliance_report import build_compliance_report, save_compliance_report
from learned.integration.model_generation_loop import generate_best_layout_from_model
from visualization.export_svg_blueprint import save_svg_blueprint


def _load_spec(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)

    # Backward-compatible spec format:
    # old: {"rooms": [{"type": "Bedroom"}, ...]}
    # new: {"rooms": [{"type":"Bedroom","count":2}, ...]}
    upgraded_rooms = []
    for item in spec.get("rooms", []):
        rtype = item.get("type")
        if not rtype:
            continue
        count = int(item.get("count", 1))
        for _ in range(max(1, count)):
            upgraded_rooms.append({"type": rtype})

    spec["rooms"] = upgraded_rooms
    spec.setdefault("building_type", spec.get("occupancy", "Residential"))
    spec.setdefault("occupancy", spec.get("building_type", "Residential"))
    return spec


def _boundary_from_arg(boundary_str: str) -> List[Tuple[float, float]]:
    w, h = [float(x) for x in boundary_str.split(",")]
    return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]


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
    """Drop heavy/non-serializable fields (e.g., RoomBox objects) from summary."""
    safe = dict(summary)
    safe_candidates = []
    for cand in safe.get("all_candidates", []):
        safe_cand = {k: v for k, v in cand.items() if k not in ("raw_rooms", "building")}
        safe_candidates.append(safe_cand)
    safe["all_candidates"] = safe_candidates
    return safe


def main():
    ap = argparse.ArgumentParser(description="Run learned-model smoke pipeline and export artifacts")
    ap.add_argument("--spec", default="learned/data/residential_spec.json",
                    help="Path to spec JSON")
    ap.add_argument("--checkpoint", default="learned/model/checkpoints/kaggle_test.pt",
                    help="Model checkpoint")
    ap.add_argument("--regulation", default="ontology/regulation_data.json",
                    help="Regulation JSON path")
    ap.add_argument("--boundary", default="15,10",
                    help="Boundary size as 'width,height' in metres")
    ap.add_argument("--entrance", default="0.2,0.0",
                    help="Entrance point as 'x,y'")
    ap.add_argument("--k", type=int, default=10,
                    help="Number of raw generation attempts")
    ap.add_argument("--top-m", type=int, default=3,
                    help="Number of pre-ranked candidates to repair")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--output-svg", default="outputs/blueprint_learned.svg")
    ap.add_argument("--output-report", default="outputs/compliance_report_learned.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    spec = _load_spec(args.spec)
    boundary = _boundary_from_arg(args.boundary)
    ex, ey = [float(x) for x in args.entrance.split(",")]
    entrance = (ex, ey)

    best, summary = generate_best_layout_from_model(
        spec=spec,
        boundary_poly=boundary,
        entrance=entrance,
        checkpoint_path=args.checkpoint,
        regulation_file=args.regulation,
        K=max(1, args.top_m),
        max_attempts=max(1, args.k),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=args.device,
        pre_rank_top_m=max(1, args.top_m),
    )

    if best is None or best.get("building") is None:
        print("No viable learned layout produced. Summary:")
        print(json.dumps(summary, indent=2))
        raise SystemExit(1)

    building = best["building"]

    # SVG export
    svg_path = save_svg_blueprint(
        building,
        output_path=args.output_svg,
        boundary_polygon=boundary,
        entrance_point=entrance,
        title="Learned Pipeline - Smoke Test",
    )

    # Compliance report export (shape compatible with build_compliance_report)
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
    save_compliance_report(report, args.output_report)

    print("Smoke run complete.")
    print(f"  Variant: {best.get('strategy_name', 'learned-best')}")
    print(f"  Connectivity: {best.get('metrics', {}).get('fully_connected')}")
    print(
        "  Travel:",
        f"{best.get('metrics', {}).get('max_travel_distance')} / "
        f"{best.get('metrics', {}).get('max_allowed_travel_distance')}"
    )
    print(f"  Adjacency: {best.get('metrics', {}).get('adjacency_satisfaction')}")
    print(f"  SVG:    {Path(svg_path)}")
    print(f"  Report: {Path(args.output_report)}")
    print(f"  Status: {report.get('status')}")


if __name__ == "__main__":
    main()
