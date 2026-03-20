"""
evaluate_generation.py – Research-grade evaluation harness for the learned
floor-plan generator.

Metrics
-------
* Raw validity rate (before repair gate)
* Post-repair validity rate
* Avg hard violations / sample
* Avg corridor coverage ratio
* Avg travel-distance margin
* **Adjacency satisfaction** (specific pairs):
  - Kitchen ↔ DiningRoom (or Kitchen ↔ LivingRoom as proxy)
  - Bedroom ↔ Bathroom / WC
  - LivingRoom ↔ Entrance
* **Diversity** – average pairwise IoU distance across top-K layouts
* **Ablation table** (4 modes):
  1. Algorithmic-only
  2. Learned-only (no repair)
  3. Learned + deterministic repair
  4. Learned + KG semantic check + repair

Usage
-----
    python -m learned.evaluate_generation \\
        --checkpoint learned/model/checkpoints/kaggle_test.pt \\
        --K 10 --N 50 --boundary 15,10

    python -m learned.evaluate_generation --ablation --N 30
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from learned.model.sample import load_model, sample_layout
from learned.data.tokenizer_layout import LayoutTokenizer, RoomBox
from learned.integration.learned_to_building_adapter import (
    adapt_generated_layout_to_building,
)
from learned.integration.repair_gate import (
    validate_and_repair_generated_layout,
    evaluate_variant,
)
from graph.connectivity import is_fully_connected
from graph.manhattan_path import max_travel_distance
from constraints.rule_engine import RuleEngine
from learned.integration.prerank import (
    score_aspect_ratio_quality,
    score_min_dims_compliance,
    score_corridor_simplicity,
    compute_realism_score,
    estimate_repair_severity,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _fingerprint(rooms: List[RoomBox]) -> str:
    return ",".join(sorted(r.room_type for r in rooms))


def _pairwise_iou_distance(candidates: List[Dict]) -> float:
    """Average pairwise (1 − IoU) between candidate room layouts.

    We compare sets of room bounding boxes using the sum of per-room-type IoU
    as the similarity metric.  Higher distance → more diverse.
    """
    from shapely.geometry import box as shapely_box

    def _room_union(rooms):
        from shapely.ops import unary_union
        polys = []
        for r in rooms:
            if r.polygon:
                from shapely.geometry import Polygon
                polys.append(Polygon(r.polygon))
        if not polys:
            return None
        return unary_union(polys)

    unions = []
    for c in candidates:
        b = c.get("building")
        if b is None:
            continue
        u = _room_union(b.rooms)
        if u is not None:
            unions.append(u)

    if len(unions) < 2:
        return 0.0

    total_dist = 0.0
    count = 0
    for i in range(len(unions)):
        for j in range(i + 1, len(unions)):
            inter = unions[i].intersection(unions[j]).area
            union_ = unions[i].union(unions[j]).area
            iou = inter / max(union_, 1e-9)
            total_dist += 1.0 - iou
            count += 1

    return round(total_dist / max(count, 1), 4)


_ADJACENCY_PAIRS = [
    ("Kitchen", "DiningRoom", "Kitchen↔Dining"),
    ("Kitchen", "LivingRoom", "Kitchen↔Living"),
    ("Bedroom", "Bathroom", "Bedroom↔Bathroom"),
    ("Bedroom", "WC", "Bedroom↔WC"),
    ("LivingRoom", None, "Living↔Entrance"),   # special: LivingRoom near entrance
]


def _adjacency_specific(building, entrance_point) -> Dict[str, bool]:
    """Check specific adjacency pairs (touching within 0.25 m tolerance)."""
    from shapely.geometry import Polygon, Point

    results = {}
    typed = {}
    for r in building.rooms:
        typed.setdefault(r.room_type, []).append(r)

    for type_a, type_b, label in _ADJACENCY_PAIRS:
        if type_b is None:
            # Special case: LivingRoom near entrance
            if entrance_point is None:
                results[label] = False
                continue
            rooms_a = typed.get(type_a, [])
            entry = Point(entrance_point)
            sat = False
            for r in rooms_a:
                if r.polygon and Polygon(r.polygon).distance(entry) <= 1.5:
                    sat = True
                    break
            results[label] = sat
            continue

        sat = False
        for ra in typed.get(type_a, []):
            for rb in typed.get(type_b, []):
                if ra.polygon and rb.polygon:
                    pa, pb = Polygon(ra.polygon), Polygon(rb.polygon)
                    if pa.touches(pb) or pa.distance(pb) <= 0.25:
                        sat = True
                        break
            if sat:
                break
        results[label] = sat

    return results


def _raw_validity(building, boundary_polygon, entrance, regulation_file) -> Dict:
    """Quick raw validity check (no repair)."""
    try:
        engine = RuleEngine(regulation_file)
        occ = building.occupancy_type
        max_allowed = engine.get_max_travel_distance(occ)
        regs = engine.data.get(occ, {}).get("rooms", {})

        hard = []
        if not is_fully_connected(building):
            hard.append("not_connected")
        travel = max_travel_distance(building)
        if travel > max_allowed:
            hard.append("travel_exceeded")
        for r in building.rooms:
            if r.polygon is None:
                hard.append(f"{r.name}_no_polygon")
                continue
            rule = regs.get(r.room_type)
            if rule and (r.final_area or 0) < rule["min_area"]:
                hard.append(f"{r.name}_under_area")

        return {
            "valid": len(hard) == 0,
            "num_violations": len(hard),
            "hard_violations": hard,
            "travel_distance": travel,
            "travel_compliant": travel <= max_allowed,
        }
    except Exception:
        return {
            "valid": False, "num_violations": 1,
            "hard_violations": ["raw_validity_error"],
            "travel_distance": 999, "travel_compliant": False,
        }


def _compute_realism_metrics(
    decoded: List[RoomBox],
    boundary_polygon: List[Tuple[float, float]],
    plot_area_sqm: float = 100.0,
) -> Dict:
    """Compute Chapter-4 realism metrics for raw decoded layout."""
    if not decoded:
        return {"error": "empty"}

    # Get boundary dimensions
    xs = [p[0] for p in boundary_polygon]
    ys = [p[1] for p in boundary_polygon]
    bw = max(xs) - min(xs)
    bh = max(ys) - min(ys)

    aspect = score_aspect_ratio_quality(decoded)
    dims = score_min_dims_compliance(decoded, bw, bh, plot_area_sqm)
    corridor = score_corridor_simplicity(decoded)
    repair = estimate_repair_severity(decoded, bw, bh, plot_area_sqm)

    return {
        "aspect_ratio_score": aspect,
        "min_dims_score": dims,
        "corridor_score": corridor,
        "repair_severity": repair["severity"],
        "sliver_rooms": repair["issues"]["sliver_rooms"],
        "undersized_rooms": repair["issues"]["undersized_rooms"],
        "overlapping_pairs": repair["issues"]["overlapping_pairs"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(
    checkpoint_path: str,
    boundary_polygon: List[Tuple[float, float]],
    entrance_point: Tuple[float, float],
    regulation_file: str = "ontology/regulation_data.json",
    N: int = 50,
    K: int = 10,
    temperature: float = 0.85,
    building_type: str = "Residential",
    device: str = "cpu",
    spec: Optional[Dict] = None,
) -> Dict:
    """Generate N trials × K samples; evaluate and aggregate metrics."""
    model, tok = load_model(checkpoint_path, device=device)
    spec = spec or {"occupancy": building_type, "rooms": []}

    results = []
    all_fps = []
    t0 = time.time()

    for trial in range(N):
        trial_cands = []

        for k in range(K):
            decoded = sample_layout(
                model, tok,
                building_type=building_type,
                temperature=temperature,
                device=device,
            )
            if not decoded:
                trial_cands.append({"empty": True})
                continue

            fp = _fingerprint(decoded)
            all_fps.append(fp)

            # Raw validity
            raw_b = adapt_generated_layout_to_building(
                decoded, boundary_polygon, entrance=entrance_point,
                spec=spec, regulation_data=regulation_file,
            )
            raw_chk = _raw_validity(raw_b, boundary_polygon, entrance_point, regulation_file)

            # Compute realism metrics on raw decoded (before repair)
            realism_raw = _compute_realism_metrics(decoded, boundary_polygon)

            # Post-repair
            rep_b = adapt_generated_layout_to_building(
                decoded, boundary_polygon, entrance=entrance_point,
                spec=spec, regulation_data=regulation_file,
            )
            repaired, violations, status, trace = validate_and_repair_generated_layout(
                rep_b, boundary_polygon,
                entrance_point=entrance_point,
                regulation_file=regulation_file,
                spec=spec,
                run_ontology=False,          # skip OWL reasoner in batch eval
            )
            post_m = evaluate_variant(repaired, regulation_file, entrance_point)
            adj_specific = _adjacency_specific(repaired, entrance_point)

            trial_cands.append({
                "empty": False,
                "num_rooms_raw": len(decoded),
                "num_rooms_repaired": len(repaired.rooms),
                "raw_valid": raw_chk["valid"],
                "raw_violations": raw_chk["num_violations"],
                "post_valid": status == "COMPLIANT",
                "post_violations": len(violations),
                "travel_distance": post_m.get("max_travel_distance", 0),
                "travel_margin": (
                    post_m.get("max_allowed_travel_distance", 0)
                    - post_m.get("max_travel_distance", 0)
                ),
                "corridor_ratio": (
                    post_m.get("circulation_walkable_area", 0)
                    / max(post_m.get("total_area", 1), 0.01)
                ),
                "adjacency_score": post_m.get("adjacency_satisfaction", 0),
                "adjacency_specific": adj_specific,
                "connected": post_m.get("fully_connected", False),
                "fingerprint": fp,
                "building": repaired,
                # Chapter-4 realism metrics (raw, before repair)
                "realism_raw": realism_raw,
            })

        results.append(trial_cands)
        if (trial + 1) % 10 == 0:
            print(f"  [{trial + 1}/{N}] {time.time() - t0:.1f}s")

    elapsed = time.time() - t0

    # ── Aggregate ─────────────────────────────────────────────────────────
    flat = [c for trial in results for c in trial if not c.get("empty")]
    total = len(flat)
    if total == 0:
        return {"error": "all samples empty", "N": N, "K": K}

    raw_valid = sum(1 for c in flat if c.get("raw_valid"))
    post_valid = sum(1 for c in flat if c.get("post_valid"))
    avg = lambda key: round(sum(c.get(key, 0) for c in flat) / total, 4)

    # Adjacency specific rates
    adj_labels = [l for _, _, l in _ADJACENCY_PAIRS]
    adj_rates = {}
    for label in adj_labels:
        count = sum(1 for c in flat if c.get("adjacency_specific", {}).get(label, False))
        adj_rates[label] = round(count / total, 4)

    # Diversity
    diversity_unique = len(set(all_fps))
    diversity_ratio = round(diversity_unique / max(len(all_fps), 1), 4)
    iou_diversity = _pairwise_iou_distance(flat[:50])  # cap for perf

    # Chapter-4 realism metrics (aggregated)
    realism_metrics_agg = {
        "avg_aspect_ratio_score": round(
            sum(c.get("realism_raw", {}).get("aspect_ratio_score", 0) for c in flat) / total, 4
        ),
        "avg_min_dims_score": round(
            sum(c.get("realism_raw", {}).get("min_dims_score", 0) for c in flat) / total, 4
        ),
        "avg_corridor_score": round(
            sum(c.get("realism_raw", {}).get("corridor_score", 0) for c in flat) / total, 4
        ),
        "avg_repair_severity": round(
            sum(c.get("realism_raw", {}).get("repair_severity", 0) for c in flat) / total, 2
        ),
        "total_sliver_rooms": sum(c.get("realism_raw", {}).get("sliver_rooms", 0) for c in flat),
        "total_undersized_rooms": sum(c.get("realism_raw", {}).get("undersized_rooms", 0) for c in flat),
        "total_overlapping_pairs": sum(c.get("realism_raw", {}).get("overlapping_pairs", 0) for c in flat),
    }

    report = {
        "N": N, "K": K,
        "total_samples": total,
        "empty_samples": N * K - total,
        "elapsed_seconds": round(elapsed, 1),
        "raw_validity_rate": round(raw_valid / total, 4),
        "post_repair_validity_rate": round(post_valid / total, 4),
        "avg_raw_violations": avg("raw_violations"),
        "avg_post_violations": avg("post_violations"),
        "avg_travel_distance": avg("travel_distance"),
        "avg_travel_margin": avg("travel_margin"),
        "avg_corridor_ratio": avg("corridor_ratio"),
        "avg_adjacency_score": avg("adjacency_score"),
        "connectivity_rate": round(sum(1 for c in flat if c.get("connected")) / total, 4),
        "adjacency_specific": adj_rates,
        "diversity_unique_fingerprints": diversity_unique,
        "diversity_ratio": diversity_ratio,
        "pairwise_iou_diversity": iou_diversity,
        "realism": realism_metrics_agg,
    }
    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  Ablation table (4 modes)
# ═══════════════════════════════════════════════════════════════════════════════

def ablation_table(
    checkpoint_path: str,
    boundary_polygon: List[Tuple[float, float]],
    entrance_point: Tuple[float, float],
    regulation_file: str = "ontology/regulation_data.json",
    N: int = 30,
    temperature: float = 0.85,
    device: str = "cpu",
    spec: Optional[Dict] = None,
) -> List[Dict]:
    """
    Compare four modes:
    1. algorithmic_only – existing corridor-first planner
    2. learned_no_repair – raw learned output (no repair gate)
    3. learned_repair – learned + deterministic repair
    4. learned_repair_kg – learned + KG semantic check + repair
    """
    spec = spec or {"occupancy": "Residential", "rooms": [
        {"name": "BR1", "type": "Bedroom"},
        {"name": "LR", "type": "LivingRoom"},
        {"name": "KT", "type": "Kitchen"},
        {"name": "BA", "type": "Bathroom"},
    ]}
    model, tok = load_model(checkpoint_path, device=device)

    table = []

    # ── Mode 1: Algorithmic-only ──────────────────────────────────────────
    try:
        from generator.layout_generator import _build_base
        from geometry.corridor_first_planner import generate_corridor_first_variants
        from geometry.door_placer import DoorPlacer

        algo_spec = dict(spec)
        algo_spec.setdefault("total_area", 1200)
        algo_spec.setdefault("area_unit", "sq.ft")
        algo_spec.setdefault("allocation_strategy", "priority_weights")
        algo_spec.setdefault("boundary_polygon", list(boundary_polygon))
        algo_spec.setdefault("entrance_point", entrance_point)

        compliant_count = 0
        total_adj = 0.0
        total_margin = 0.0
        algo_count = 0

        for _ in range(min(N, 5)):  # algorithmic is deterministic, fewer trials
            try:
                b, eng, _, _, w, h, bp, _ = _build_base(algo_spec, regulation_file)
                min_cw = eng.data[b.occupancy_type].get("corridor", {}).get("min_width", 1.2)
                min_dw = eng.get_min_door_width(b.occupancy_type)
                variants = generate_corridor_first_variants(b, bp, entrance_point, min_cw)
                for vb, _ in variants:
                    DoorPlacer(vb, min_dw).place_doors()
                    m = evaluate_variant(vb, regulation_file, entrance_point)
                    algo_count += 1
                    if m["fully_connected"] and m["travel_distance_compliant"]:
                        compliant_count += 1
                    total_adj += m.get("adjacency_satisfaction", 0)
                    total_margin += m.get("max_allowed_travel_distance", 0) - m.get("max_travel_distance", 0)
            except Exception:
                pass

        c = max(algo_count, 1)
        table.append({
            "mode": "algorithmic_only",
            "n_samples": algo_count,
            "raw_validity_rate": round(compliant_count / c, 4),
            "post_validity_rate": round(compliant_count / c, 4),
            "avg_violations": 0.0,
            "avg_travel_margin": round(total_margin / c, 2),
            "avg_adjacency": round(total_adj / c, 4),
        })
    except Exception:
        table.append({"mode": "algorithmic_only", "error": "failed"})

    # ── Modes 2–4: Learned variants ───────────────────────────────────────
    modes = {
        "learned_no_repair":  {"repair": False, "kg": False},
        "learned_repair":     {"repair": True,  "kg": False},
        "learned_repair_kg":  {"repair": True,  "kg": True},
    }

    for mode_name, opts in modes.items():
        raw_ok = 0
        post_ok = 0
        total_violations = 0
        total_margin = 0.0
        total_adj = 0.0
        count = 0

        for _ in range(N):
            decoded = sample_layout(model, tok, temperature=temperature, device=device)
            if not decoded:
                continue
            count += 1

            building = adapt_generated_layout_to_building(
                decoded, boundary_polygon, entrance=entrance_point,
                spec=spec, regulation_data=regulation_file,
            )

            raw_chk = _raw_validity(building, boundary_polygon, entrance_point, regulation_file)
            if raw_chk["valid"]:
                raw_ok += 1

            if opts["repair"]:
                rep_b = adapt_generated_layout_to_building(
                    decoded, boundary_polygon, entrance=entrance_point,
                    spec=spec, regulation_data=regulation_file,
                )
                repaired, viol, status, _ = validate_and_repair_generated_layout(
                    rep_b, boundary_polygon,
                    entrance_point=entrance_point,
                    regulation_file=regulation_file,
                    spec=spec,
                    run_ontology=opts["kg"],
                )
                m = evaluate_variant(repaired, regulation_file, entrance_point)
                if status == "COMPLIANT":
                    post_ok += 1
                total_violations += len(viol)
                total_margin += m.get("max_allowed_travel_distance", 0) - m.get("max_travel_distance", 0)
                total_adj += m.get("adjacency_satisfaction", 0)
            else:
                m = evaluate_variant(building, regulation_file, entrance_point)
                if raw_chk["valid"]:
                    post_ok += 1
                total_violations += raw_chk["num_violations"]
                total_margin += m.get("max_allowed_travel_distance", 0) - m.get("max_travel_distance", 0)
                total_adj += m.get("adjacency_satisfaction", 0)

        c = max(count, 1)
        table.append({
            "mode": mode_name,
            "n_samples": count,
            "raw_validity_rate": round(raw_ok / c, 4),
            "post_validity_rate": round(post_ok / c, 4),
            "avg_violations": round(total_violations / c, 2),
            "avg_travel_margin": round(total_margin / c, 2),
            "avg_adjacency": round(total_adj / c, 4),
        })

    return table


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Evaluate learned layout generation quality")
    ap.add_argument("--checkpoint", default="learned/model/checkpoints/kaggle_test.pt")
    ap.add_argument("--N", type=int, default=50, help="Number of trials")
    ap.add_argument("--K", type=int, default=10, help="Samples per trial")
    ap.add_argument("--boundary", default="15,10", help="Width,Height of boundary rect")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ablation", action="store_true", help="Run ablation table only")
    ap.add_argument("--output", default="outputs/evaluation_report.json")
    args = ap.parse_args()

    w, h = [float(x) for x in args.boundary.split(",")]
    boundary = [(0, 0), (w, 0), (w, h), (0, h)]
    entrance = (0.2, 0.0)

    print("=" * 60)
    print("  Learned Generator – Evaluation Harness")
    print("=" * 60)

    if args.ablation:
        print(f"\nAblation table  (N={args.N}) …")
        tbl = ablation_table(
            args.checkpoint, boundary, entrance,
            N=args.N, temperature=args.temperature, device=args.device,
        )
        header = f"{'Mode':<25} {'RawVal':>8} {'PostVal':>8} {'AvgViol':>8} {'TravMarg':>9} {'AdjScr':>8}"
        print(f"\n{header}")
        print("-" * len(header))
        for row in tbl:
            if "error" in row:
                print(f"{row['mode']:<25}  ERROR")
                continue
            print(
                f"{row['mode']:<25} {row['raw_validity_rate']:>8.4f} "
                f"{row['post_validity_rate']:>8.4f} {row['avg_violations']:>8.2f} "
                f"{row['avg_travel_margin']:>9.2f} {row['avg_adjacency']:>8.4f}"
            )

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"ablation": tbl}, f, indent=2)
        print(f"\nReport saved → {args.output}")
    else:
        print(f"\nEvaluation  (N={args.N}, K={args.K}) …\n")
        report = evaluate(
            args.checkpoint, boundary, entrance,
            N=args.N, K=args.K, temperature=args.temperature, device=args.device,
        )
        for key, val in report.items():
            print(f"  {key:<35} {val}")

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved → {args.output}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
