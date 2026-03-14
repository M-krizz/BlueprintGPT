# BlueprintGPT Mentor Demo (≈2 minutes)

## 1) NL generation (deterministic, no external LLM)
```
.\.venv\Scripts\python.exe -m nl_interface.cli "Need 2 bedrooms, 1 bathroom, 1 kitchen on a 10 marla plot with a north entrance." --boundary 15,10 --run --output-prefix nl_algorithmic
```
What to show:
- `outputs/nl_algorithmic_blueprint.svg`
- `outputs/nl_algorithmic_compliance_report.json`
- `execution_algorithmic.json` (design_score, design_reasons, alternatives, llm.used=false)

## 2) NL generation with Gemini polish (optional)
```
$env:GEMINI_API_KEY="YOUR_KEY"
.\.venv\Scripts\python.exe -m nl_interface.cli "Need 2 bedrooms, 1 bathroom, 1 kitchen on a 10 marla plot with a north entrance." --boundary 15,10 --run --output-prefix nl_algorithmic_llm --llm-provider gemini --llm-model gemini-1.5-flash
```
Show:
- `llm.used=true`, latency, evidence_hash, and that numbers match the evidence.

## 3) Alternatives + why chosen
- Open `execution_algorithmic.json` → `alternatives` array with design_scores and diversity.
- Point out `design_reasons` and `explanation` fields.

## 4) Edit mode (live violations)
- Launch existing GUI edit flow (already wired): move/resize a room, see live violation panel. (Command unchanged from existing demo.)

## 5) Export
- Show final SVG (blueprint) and compliance report.

## 6) Optional learned run
```
.\.venv\Scripts\python.exe -m nl_interface.cli "Need 2 bedrooms, 1 bathroom, 1 kitchen and 1 garage on a 10 marla plot with a north entrance." --boundary 15,10 --run --output-prefix nl_learned
```
- Highlights learned path gating + same explanation guardrails.

Artifacts to attach for mentors:
- `results_bundle/` contents (SVGs, reports, executions, pytest log, RESULTS.md, architecture_diagram.svg).
