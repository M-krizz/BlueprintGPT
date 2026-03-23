"""Evaluate whether a planner benchmark is strong enough for rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_benchmark(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_benchmark(payload: Dict) -> Dict:
    summary = payload.get("summary", {})
    cases = payload.get("cases", [])

    completed_case_count = int(summary.get("completed_case_count", 0))
    case_count = int(summary.get("case_count", 0))
    winner_counts = summary.get("winner_counts", {}) or {}
    planner_wins = int(winner_counts.get("planner", 0))
    algorithmic_wins = int(winner_counts.get("algorithmic", 0))
    score_delta = float(summary.get("planner_avg_score_delta", 0.0))
    adjacency_delta = float(summary.get("planner_avg_adjacency_delta", 0.0))
    alignment_delta = float(summary.get("planner_avg_alignment_delta", 0.0))

    planner_cases = [case.get("planner", {}) for case in cases if case.get("planner")]
    all_planner_compliant = all(case.get("report_status") == "COMPLIANT" for case in planner_cases)
    all_planner_connected = all(bool(case.get("metrics", {}).get("fully_connected")) for case in planner_cases)
    all_planner_room_coverage_ok = all(
        not case.get("room_coverage", {}).get("missing") and not case.get("room_coverage", {}).get("extra")
        for case in planner_cases
    )

    checks: List[Dict] = [
        {
            "name": "all_cases_completed",
            "passed": completed_case_count == case_count and case_count > 0,
            "detail": f"completed={completed_case_count}, total={case_count}",
        },
        {
            "name": "planner_not_worse_on_win_count",
            "passed": planner_wins >= algorithmic_wins,
            "detail": f"planner={planner_wins}, algorithmic={algorithmic_wins}",
        },
        {
            "name": "planner_avg_score_non_negative",
            "passed": score_delta >= 0.0,
            "detail": f"avg_score_delta={score_delta:.4f}",
        },
        {
            "name": "planner_avg_adjacency_non_negative",
            "passed": adjacency_delta >= 0.0,
            "detail": f"avg_adjacency_delta={adjacency_delta:.4f}",
        },
        {
            "name": "planner_avg_alignment_non_negative",
            "passed": alignment_delta >= 0.0,
            "detail": f"avg_alignment_delta={alignment_delta:.4f}",
        },
        {
            "name": "planner_cases_compliant",
            "passed": all_planner_compliant,
            "detail": f"planner_cases={len(planner_cases)}",
        },
        {
            "name": "planner_cases_connected",
            "passed": all_planner_connected,
            "detail": f"planner_cases={len(planner_cases)}",
        },
        {
            "name": "planner_room_coverage_exact",
            "passed": all_planner_room_coverage_ok,
            "detail": f"planner_cases={len(planner_cases)}",
        },
    ]

    passed = [check for check in checks if check["passed"]]
    failed = [check for check in checks if not check["passed"]]
    ready_for_rollout = not failed

    return {
        "ready_for_rollout": ready_for_rollout,
        "recommendation": (
            "enable planner_if_available"
            if ready_for_rollout
            else "hold algorithmic default and improve planner training/data"
        ),
        "summary": summary,
        "checks": checks,
        "passed_check_count": len(passed),
        "failed_check_count": len(failed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate planner benchmark output for rollout readiness")
    parser.add_argument(
        "--input",
        default="learned/planner/checkpoints/summary.json",
        help="Benchmark summary JSON produced by planner benchmarking",
    )
    args = parser.parse_args()

    result = evaluate_benchmark(load_benchmark(args.input))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
