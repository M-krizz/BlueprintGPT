import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_compliance_report(result: Dict, chapter4_check: Optional[Dict] = None) -> Dict:
	"""Build comprehensive compliance report including Chapter-4 checks.

	Args:
		result: Layout generation result dict
		chapter4_check: Optional output from RuleEngine.check_chapter4_compliance()
	"""
	metrics = result["metrics"]
	ontology = result.get("ontology")
	spec_validation = result.get("spec_validation") or {}
	repair = result.get("repair") or {}
	rule_preflight = result.get("rule_preflight") or {}
	kg_precheck = result.get("kg_precheck") or {}

	checks = {
		"room_minimums": len(result["modifications"]) == 0,
		"connectivity": metrics["fully_connected"],
		"travel_distance": metrics["travel_distance_compliant"],
		"exit_width": metrics["required_exit_width"] >= 1.0,
		"circulation_to_exit": metrics.get("connectivity_to_exit", False),
		"llm_spec_valid": spec_validation.get("schema_valid", True),
		"kg_valid": spec_validation.get("kg_valid", True),
		"rule_preflight_valid": rule_preflight.get("valid", True),
		"kg_precheck_valid": kg_precheck.get("valid", True),
	}

	if ontology is not None:
		checks["ontology_validation"] = ontology.get("valid", False)

	violations = []
	if not checks["connectivity"]:
		violations.append("Layout graph is not fully connected")
	if not checks["travel_distance"]:
		violations.append(
			f"Max travel distance {metrics['max_travel_distance']} exceeds allowed {metrics['max_allowed_travel_distance']}"
		)
	if ontology is not None:
		violations.extend(v["message"] for v in ontology.get("violations", []))

	# Add Chapter-4 specific violations if provided
	chapter4_violations = []
	if chapter4_check:
		checks["chapter4_compliant"] = chapter4_check.get("compliant", True)
		for v in chapter4_check.get("violations", []):
			chapter4_violations.append({
				"rule": v.get("rule"),
				"required": v.get("required"),
				"actual": v.get("actual"),
				"message": v.get("message"),
			})
			violations.append(v.get("message", str(v)))

	report = {
		"input": result.get("input_spec", {}),
		"source": result.get("source", "algorithmic"),
		"allocation": result.get("allocation"),
		"modifications": result["modifications"],
		"checks": checks,
		"metrics": metrics,
		"circulation_space": {
			"corridor_width": metrics.get("corridor_width", 0.0),
			"walkable_area": metrics.get("circulation_walkable_area", 0.0),
			"connectivity_to_exit": metrics.get("connectivity_to_exit", False),
		},
		"grounding": {
			"llm_spec_valid": checks["llm_spec_valid"],
			"kg_valid": checks["kg_valid"],
			"rule_preflight_valid": checks["rule_preflight_valid"],
			"kg_precheck_valid": checks["kg_precheck_valid"],
			"repair_attempts": repair.get("repair_attempts", 0),
			"errors": spec_validation.get("errors", []),
			"rule_preflight": rule_preflight,
			"kg_precheck": kg_precheck,
		},
		"truth_table": [
			{
				"claim": "Connectivity",
				"algorithm_result": metrics["fully_connected"],
			},
			{
				"claim": "Travel distance within limit",
				"algorithm_result": metrics["travel_distance_compliant"],
				"margin": round(
					metrics.get("max_allowed_travel_distance", 0.0)
					- metrics.get("max_travel_distance", 0.0),
					2,
				),
			},
			{
				"claim": "Exit width sufficient",
				"algorithm_result": metrics["required_exit_width"] >= 1.0,
			},
		],
		"status": "COMPLIANT" if not violations else "NON_COMPLIANT",
		"violations": violations,
		"bounding_box": result["bounding_box"],
	}

	# Add Chapter-4 detailed report section
	if chapter4_check:
		report["chapter4"] = {
			"compliant": chapter4_check.get("compliant", True),
			"plot_bucket": chapter4_check.get("plot_bucket"),
			"plot_area_sqm": chapter4_check.get("plot_area_sqm"),
			"violations": chapter4_violations,
			"checks": chapter4_check.get("checks", {}),
		}

	if ontology is not None:
		report["ontology"] = {
			"reasoner": ontology.get("reasoner"),
			"reasoner_success": ontology.get("reasoner_success"),
			"reasoner_error": ontology.get("reasoner_error"),
			"ontology_loaded": ontology.get("ontology_loaded"),
			"load_error": ontology.get("load_error"),
		}

	# ── Learned-generator specific fields ─────────────────────────────────
	if result.get("source") == "learned":
		report["raw_validity"] = result.get("raw_validity", False)
		report["repair_trace"] = result.get("repair_trace", [])
		report["generation_summary"] = result.get("generation_summary", {})
		report["wall_pipeline"] = result.get("wall_pipeline", {})

	return report


def format_violation(violation: Dict) -> str:
	"""Format a Chapter-4 violation for display."""
	rule = violation.get("rule", "unknown")
	required = violation.get("required")
	actual = violation.get("actual")
	msg = violation.get("message", "")

	if required is not None and actual is not None:
		return f"[{rule}] Required: {required}, Actual: {actual} — {msg}"
	return f"[{rule}] {msg}"

	return report


def save_compliance_report(report, output_file):
	output_path = Path(output_file)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(report, f, indent=2)
