import json
from pathlib import Path
from uuid import uuid4

import demo.build_results_package as pkg


def test_build_results_package_creates_bundle(monkeypatch):
    # Redirect paths to a workspace sandbox we control
    root = Path("outputs") / f"bundle_test_{uuid4().hex}"
    outputs = root / "outputs"
    results = root / "results_bundle"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "nl_algorithmic_blueprint.svg").write_text("svg", encoding="utf-8")
    (outputs / "nl_algorithmic_compliance_report.json").write_text("{}", encoding="utf-8")
    (outputs / "blueprint_learned.svg").write_text("svg", encoding="utf-8")
    (outputs / "compliance_report_learned.json").write_text("{}", encoding="utf-8")
    (outputs / "evaluation_report.json").write_text(json.dumps({"ablation": []}), encoding="utf-8")
    (outputs / "architecture_diagram.svg").write_text("<svg/>", encoding="utf-8")

    # Stub executions with artifact paths inside temp outputs
    algo_exec = {
        "artifact_paths": {
            "svg": str(outputs / "nl_algorithmic_blueprint.svg"),
            "report": str(outputs / "nl_algorithmic_compliance_report.json"),
        },
        "design_score": 0.9,
        "report_status": "COMPLIANT",
        "alternatives": [],
        "llm": {
            "used": False,
            "provider": "deterministic",
            "model": None,
            "latency_ms": None,
            "warning": None,
            "evidence_hash": "hash-a",
        },
    }
    learned_exec = {
        "artifact_paths": {
            "svg": str(outputs / "blueprint_learned.svg"),
            "report": str(outputs / "compliance_report_learned.json"),
        },
        "design_score": 0.8,
        "report_status": "COMPLIANT",
        "alternatives": [],
        "llm": {
            "used": False,
            "provider": "deterministic",
            "model": None,
            "latency_ms": None,
            "warning": None,
            "evidence_hash": "hash-b",
        },
    }

    def fake_run_pytest(log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("pytest ok", encoding="utf-8")

    def fake_run_nl(prompt, boundary, prefix, llm=False, extra_args=None):
        return {"execution": algo_exec if "garage" not in prompt.lower() else learned_exec}

    # Monkeypatch module-level paths and functions
    monkeypatch.setattr(pkg, "ROOT", root)
    monkeypatch.setattr(pkg, "OUTPUTS", outputs)
    monkeypatch.setattr(pkg, "RESULTS", results)
    monkeypatch.setattr(pkg, "_run_pytest", fake_run_pytest)
    monkeypatch.setattr(pkg, "_run_nl", fake_run_nl)

    pkg.main()

    expected_files = [
        "blueprint_algorithmic.svg",
        "blueprint_learned.svg",
        "nl_algorithmic_blueprint.svg",
        "nl_algorithmic_compliance_report.json",
        "compliance_report_learned.json",
        "evaluation_report.json",
        "pytest_full.txt",
        "RESULTS.md",
        "execution_algorithmic.json",
        "execution_learned.json",
        "architecture_diagram.svg",
    ]
    for name in expected_files:
        assert (results / name).exists(), f"Missing {name}"
