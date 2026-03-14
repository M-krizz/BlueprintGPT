from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results_bundle"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _safe(path: Path) -> dict:
    return _read_json(path) if path.exists() else {}


def _run(cmd: list[str], *, cwd: Optional[Path] = None, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=capture, check=True)


def _run_pytest(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pytest", "-q"]
    result = _run(cmd, capture=True)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")


def _run_nl(prompt: str, boundary: str, prefix: str, *, llm: bool = False, extra_args: Optional[list[str]] = None) -> Dict:
    cmd = [
        sys.executable,
        "-m",
        "nl_interface.cli",
        prompt,
        "--boundary",
        boundary,
        "--run",
        "--output-prefix",
        prefix,
    ]
    if llm:
        cmd.extend(["--llm-provider", "gemini"])
    if extra_args:
        cmd.extend(extra_args)
    result = _run(cmd, capture=True)
    payload = json.loads(result.stdout)
    (OUTPUTS / f"{prefix}_execution.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _copy_if_exists(src: Path, dest: Path):
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
        return True
    return False


def _reasoner_table(report: dict) -> list[dict]:
    ontology = report.get("ontology", {})
    return [
        {
            "mode": "off",
            "reasoner": "off",
            "success": False,
            "valid": report.get("checks", {}).get("ontology_validation", report.get("status") == "COMPLIANT"),
        },
        {
            "mode": "try",
            "reasoner": ontology.get("reasoner", "not-configured"),
            "success": ontology.get("reasoner_success", False),
            "valid": report.get("status") == "COMPLIANT",
        },
        {
            "mode": "require",
            "reasoner": ontology.get("reasoner", "not-configured"),
            "success": ontology.get("reasoner_success", False),
            "valid": report.get("status") == "COMPLIANT" and ontology.get("reasoner_success", False),
        },
    ]


def _pick_demo_view_artifact() -> str:
    candidates = [
        "outputs/blueprint.svg",
        "outputs/Figure_1.png",
        "outputs/test_layout.png",
        "outputs/blueprint_learned.svg",
    ]
    for rel_path in candidates:
        if (ROOT / rel_path).exists() and rel_path != "outputs/blueprint_learned.svg":
            return rel_path
    return "outputs/blueprint_learned.svg"


def _bundle_results(algorithmic_exec: Dict, learned_exec: Dict, pytest_log: Path) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Copy primary artifacts
    algo_svg = Path(algorithmic_exec["artifact_paths"]["svg"])
    algo_report = Path(algorithmic_exec["artifact_paths"]["report"])
    learned_svg = Path(learned_exec["artifact_paths"]["svg"])
    learned_report = Path(learned_exec["artifact_paths"]["report"])

    _copy_if_exists(algo_svg, RESULTS / "blueprint_algorithmic.svg")
    _copy_if_exists(algo_svg, RESULTS / "nl_algorithmic_blueprint.svg")  # alias
    _copy_if_exists(learned_svg, RESULTS / "blueprint_learned.svg")
    _copy_if_exists(algo_report, RESULTS / "nl_algorithmic_compliance_report.json")
    _copy_if_exists(learned_report, RESULTS / "compliance_report_learned.json")
    _copy_if_exists(OUTPUTS / "evaluation_report.json", RESULTS / "evaluation_report.json")
    _copy_if_exists(pytest_log, RESULTS / "pytest_full.txt")
    _copy_if_exists(ROOT / "outputs/architecture_diagram.svg", RESULTS / "architecture_diagram.svg")

    # Save execution payloads for reproducibility
    (RESULTS / "execution_algorithmic.json").write_text(json.dumps(algorithmic_exec, indent=2), encoding="utf-8")
    (RESULTS / "execution_learned.json").write_text(json.dumps(learned_exec, indent=2), encoding="utf-8")

    _write_results_md(algorithmic_exec, learned_exec)


def _write_results_md(algorithmic_exec: Dict, learned_exec: Dict) -> None:
    learned_report = _safe(OUTPUTS / "evaluation_report.json")
    algo_llm = algorithmic_exec.get("llm", {})
    learned_llm = learned_exec.get("llm", {})

    content = []
    content.append("# Results Package")
    content.append("")
    content.append("## Demo Artifacts")
    content.append("")
    content.append("### Learned Best")
    content.append("![Learned Best](blueprint_learned.svg)")
    content.append("")
    content.append("### Algorithmic Baseline")
    content.append("![Algorithmic Baseline](blueprint_algorithmic.svg)")
    content.append("")
    content.append("## LLM Explanation Metadata")
    content.append("")
    content.append("| Run | Used | Provider | Model | Latency (ms) | Warning | Evidence Hash |")
    content.append("| --- | --- | --- | --- | --- | --- | --- |")
    content.append(
        f"| Algorithmic | {algo_llm.get('used')} | {algo_llm.get('provider')} | {algo_llm.get('model')} | "
        f"{algo_llm.get('latency_ms')} | {algo_llm.get('warning')} | {algo_llm.get('evidence_hash')} |"
    )
    content.append(
        f"| Learned | {learned_llm.get('used')} | {learned_llm.get('provider')} | {learned_llm.get('model')} | "
        f"{learned_llm.get('latency_ms')} | {learned_llm.get('warning')} | {learned_llm.get('evidence_hash')} |"
    )
    content.append("")
    content.append("## Output Summary")
    content.append("")
    content.append(f"- Learned status: `{learned_exec.get('report_status', 'missing')}`")
    content.append(f"- Algorithmic status: `{algorithmic_exec.get('report_status', 'missing')}`")
    content.append(f"- Learned design score: `{learned_exec.get('design_score')}`")
    content.append(f"- Algorithmic design score: `{algorithmic_exec.get('design_score')}`")
    content.append(f"- Alternatives: {len(algorithmic_exec.get('alternatives', []))} shown")
    content.append("")
    if learned_report:
        ablations = learned_report.get("ablation", [])
        content.append("## Ablation Table")
        content.append("")
        content.append("| Mode | Samples | Raw Validity | Post-Repair Validity | Avg Violations | Avg Travel Margin | Avg Adjacency |")
        content.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in ablations:
            content.append(
                f"| {row.get('mode')} | {row.get('n_samples', 'n/a')} | {row.get('raw_validity_rate', 'n/a')} | "
                f"{row.get('post_validity_rate', 'n/a')} | {row.get('avg_violations', 'n/a')} | "
                f"{row.get('avg_travel_margin', 'n/a')} | {row.get('avg_adjacency', 'n/a')} |"
            )
        content.append("")

    (ROOT / "RESULTS.md").write_text("\n".join(content) + "\n", encoding="utf-8")
    shutil.copy(ROOT / "RESULTS.md", RESULTS / "RESULTS.md")


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # 1) Run quick pytest and capture log
    pytest_log = OUTPUTS / "pytest_full.txt"
    _run_pytest(pytest_log)

    # 2) NL algorithmic (deterministic; uses Gemini fallback path but no key)
    algo_exec = _run_nl(
        "Need 2 bedrooms, 1 bathroom, 1 kitchen and 1 drawing room on a 10 marla plot with a north entrance and minimize corridor.",
        boundary="15,10",
        prefix="nl_algorithmic",
        llm=False,
    )["execution"]

    # 3) NL learned (force learned by adding Garage)
    os.environ["BLUEPRINTGPT_ALLOW_DESIGN_FAIL"] = "1"
    learned_exec = _run_nl(
        "Need 2 bedrooms, 1 bathroom, 1 kitchen and 1 garage on a 10 marla plot with a north entrance.",
        boundary="15,10",
        prefix="nl_learned",
        llm=False,
        extra_args=["--k", "20", "--top-m", "5"],
    )["execution"]
    os.environ.pop("BLUEPRINTGPT_ALLOW_DESIGN_FAIL", None)

    _bundle_results(algo_exec, learned_exec, pytest_log)
    print(f"Bundle written to {RESULTS}")


if __name__ == "__main__":
    main()
