# BlueprintGPT Architecture & Usage

## Architecture (text diagram)
```
NL Request -> nl_interface.service (parse/normalize) -> nl_interface.adapter (route backend)
    -> nl_interface.runner.execute_response (orchestrate) ->
        [Algorithmic] generator.layout_generator -> geometry/*, graph/*, constraints.rule_engine
            -> visualization.export_svg_blueprint -> outputs/*.svg + compliance_report.json
        [Learned] learned.integration.model_generation_loop -> repair gate -> constraints + geometry
            -> visualization.export_svg_blueprint -> outputs/*.svg + compliance_report.json
Interactive UI (main.py): form -> generation -> picker -> plot_layout (edit mode + live validation)
Ontology/KG: ontology.ontology_bridge + regulation_data.json/regulatory.owl used in precheck/validation
```

## Main pipeline steps
1. Parse/repair spec: `constraints.spec_validator`, `constraints.repair_loop`.
2. Preflight rules: `constraints.rule_engine.preflight_validate_spec`.
3. Generate variants:
   - Algorithmic: corridor strategies + door placement + snapping.
   - Learned: sampled layouts -> adapt -> repair gate -> compliance metrics.
4. Rank variants: `generator.ranking` (hard compliance, compactness, adjacency, travel margin, alignment).
5. Validate & report: connectivity, travel, area, exit width, KG/ontology; `constraints.compliance_report`.
6. Render/export: `visualization.export_svg_blueprint` (merged walls, carved doors/windows, dimensions, title).
7. Interact/edit (optional): `visualization.plot_layout` live checks; GUI flow in `main.py`.

## How the KG constrains interaction
- Ontology precheck in `ontology.ontology_bridge.validate_spec_semantics` rejects unsupported room types.
- KG validation per variant (if enabled) flags violations (min area, travel, exit width, corridor width) and surfaces messages used by reports and live edit panel.
- Zone/intents from KG inform adjacency scoring and zoning helpers.

## Learned model repair + verification
- Sampling via `learned.integration.model_generation_loop.generate_best_layout_from_model`.
- Raw validity checks (connectivity, travel, min areas) before repair.
- Repair gate enforces geometry bounds; compliance metrics recomputed.
- Summary captures raw vs repaired validity counts, failure reasons, and wall pipeline stats.

## Commands (demos & tests)
- NL CLI parse+run (algorithmic default):
  `python -m nl_interface.cli "Need 2 bedrooms, 1 bathroom, 1 kitchen and 1 drawing room on a 10 marla plot with a north entrance and minimize corridor." --boundary 15,10 --run --output-dir outputs`
- Algorithmic smoke: `python -m demo.run_smoke_algorithmic`
- Learned smoke (needs checkpoint): `python -m learned.integration.run_smoke_learned --checkpoint <path>`
- Results package: `python -m demo.build_results_package`
- Full test suite: `python -m pytest`

## Web app (FastAPI + chat UI)
- Start server: `uvicorn api.server:app --reload --port 8000`
- Open UI: `http://127.0.0.1:8000/ui`
- Prompt-first flow: type a natural-language requirement and click Generate.

### API endpoints
- `GET /health` -> service health.
- `POST /chat/generate` -> natural-language prompt + boundary to full execution.
- `POST /generate` -> structured payload execution (rooms + boundary).
- `GET /report?path=outputs/<file>.json` -> load saved report JSON.

### Example prompt payload
```json
{
    "prompt": "Need 2 bedrooms, 1 bathroom, 1 kitchen and 1 drawing room on a 10 marla plot with a north entrance and minimize corridor.",
    "boundary": {"width": 15, "height": 10},
    "output_prefix": "chat_run"
}
```

## Checkpoint configuration (learned backend)
- Env override: set `BLUEPRINTGPT_CHECKPOINT` to the checkpoint path.
- CLI flag: `--checkpoint <path>` when using `nl_interface.cli --run`.
- If missing, learned execution raises a clear error before model load.
