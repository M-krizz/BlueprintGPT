from generator.layout_generator import generate_layout_from_spec
from constraints.compliance_report import build_compliance_report, save_compliance_report
from constraints.rule_engine import RuleEngine
from constraints.spec_validator import validate_spec
from constraints.repair_loop import validate_and_repair_spec
from ontology.ontology_bridge import OntologyBridge
from visualization.plot_layout import plot_layout
from gui.layout_form import LayoutForm
from gui.layout_picker import LayoutPicker


def summarize_reasoner_error(error_text):
    if not error_text:
        return None
    lines = [line.strip() for line in str(error_text).splitlines() if line.strip()]
    for line in lines:
        if line == "Java error message is:":
            continue
        if line.lower().startswith("exception") or "error" in line.lower():
            return line
    return lines[0] if lines else None


def main():
    # ── Step 1: Collect input from the GUI ────────────────────────────────────
    form = LayoutForm()
    form_data = form.run()

    if not form_data:
        print("No input provided. Exiting.")
        return

    room_inputs = form_data.get("rooms", [])
    if not room_inputs:
        print("No valid room definitions provided. Exiting.")
        return

    boundary_polygon = form_data.get("boundary_polygon")  # list[(x,y)] in metres

    spec = {
        "occupancy":           "Residential",
        "total_area":          form_data.get("total_area"),
        "area_unit":           form_data.get("area_unit", "sq.ft"),
        "allocation_strategy": form_data.get("allocation_strategy", "priority_weights"),
        "rooms": [
            {"name": name, "type": room_type}
            for name, room_type in room_inputs
        ],
        "boundary_polygon": boundary_polygon,
        "entrance_point":   form_data.get("entrance_point"),
        "adjacency": [],
        "preferences": {},
    }

    repaired = validate_and_repair_spec(spec, validate_spec, max_attempts=3)
    spec = repaired["spec"]
    spec["_spec_validation"] = repaired.get("validation", {})
    spec["_repair"] = {"repair_attempts": repaired.get("repair_attempts", 0)}

    # ── Step 2: Generate all layout variants ─────────────────────────────────
    ontology_validator = OntologyBridge("ontology/regulatory.owl")
    try:
        result = generate_layout_from_spec(
            spec,
            regulation_file="ontology/regulation_data.json",
            ontology_validator=ontology_validator,
        )
    except ValueError as exc:
        print("\nGeneration blocked by validation gate:", exc)
        return

    # ── Step 3: Show picker GUI – user selects a corridor strategy ───────────
    variants = result.get("layout_variants", [result])
    picker = LayoutPicker(
        variants,
        boundary_polygon=boundary_polygon,
        recommended_idx=result.get("recommended_index", 0),
    )
    chosen_idx = picker.run()
    chosen = variants[chosen_idx]

    building = chosen["building"]
    width    = chosen["bounding_box"]["width"]
    height   = chosen["bounding_box"]["height"]
    strategy = chosen.get("strategy_name", "Selected Layout")

    # ── Step 4: Print metrics ─────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Selected Strategy: {strategy}")
    print(f"{'='*55}")

    print("\nBounding Box:")
    print("  Width:", width, "m")
    print("  Height:", height, "m")

    print("\nRoom Coordinates:")
    for room in building.rooms:
        print(f"  {room.name} -> {room.polygon}")

    print("\nCorridors:")
    for corr in building.corridors:
        print(f"  {corr}")

    print("\nDoors placed:", len(building.doors))
    print("Fully connected:", chosen["metrics"]["fully_connected"])
    print("Max travel distance:", chosen["metrics"]["max_travel_distance"])
    print("Circulation walkable area:", chosen["metrics"].get("circulation_walkable_area"))
    print("Corridor width:", chosen["metrics"].get("corridor_width"))

    preflight = chosen.get("rule_preflight", {})
    if preflight:
        print("\nRULE PREFLIGHT:")
        print("  Valid:", preflight.get("valid"))
        for warning in preflight.get("warnings", []):
            print("  Warning:", warning)

    kg_precheck = chosen.get("kg_precheck", {})
    if kg_precheck:
        print("\nKG SEMANTIC PREFLIGHT:")
        print("  Valid:", kg_precheck.get("valid"))
        print("  Ontology loaded:", kg_precheck.get("ontology_loaded"))
        if kg_precheck.get("warnings"):
            for warning in kg_precheck.get("warnings", []):
                print("  Warning:", warning)

    print("\nMODIFICATIONS:")
    for m in chosen.get("modifications", []):
        print(" -", m)

    print("\nROOMS:")
    for room in building.rooms:
        print(" ", room)

    print("\nTotal Area (with circulation):", chosen["metrics"]["total_area"])
    print("Occupant Load:", chosen["metrics"]["occupant_load"])
    print("Required Exit Width:", chosen["metrics"]["required_exit_width"], "m")

    allocation = chosen.get("allocation")
    if allocation:
        print("\nALLOCATION BREAKDOWN:")
        print("  Input Total Area:", allocation.get("input_total_area"),
              allocation.get("input_unit"))
        print("  Input Total Area (sq.m):", allocation.get("input_total_area_sqm"))
        print("  Target Usable Area (sq.m):", allocation.get("target_usable_area_sqm"))
        print("  Min Required Area (sq.m):", allocation.get("min_required_area_sqm"))
        print("  Surplus Area (sq.m):", allocation.get("surplus_area_sqm"))
        print("  Strategy:", allocation.get("allocation_strategy"))
        for room_alloc in allocation.get("rooms", []):
            print(
                f"  - {room_alloc.get('name')} ({room_alloc.get('type')})"
                f"  min={room_alloc.get('min_area')}  "
                f"allocated={room_alloc.get('allocated_area')}  "
                f"weight={room_alloc.get('weight')}"
            )

    # ── Step 5: Compliance report ─────────────────────────────────────────────
    report = build_compliance_report(chosen)
    save_compliance_report(report, "outputs/compliance_report.json")
    print("\nCompliance Status:", report["status"])

    ontology_info = report.get("ontology", {})
    if ontology_info:
        print(
            "Ontology Reasoner:", ontology_info.get("reasoner"),
            "| Success:", ontology_info.get("reasoner_success"),
        )
        if ontology_info.get("reasoner_error"):
            error_summary = summarize_reasoner_error(ontology_info.get("reasoner_error"))
            if error_summary:
                print("Ontology Reasoner Error:", error_summary)

    print("Report written to outputs/compliance_report.json")

    # ── Step 6: Full-size floor plan ─────────────────────────────────────────
    plot_layout(
        building, width, height,
        boundary_polygon=boundary_polygon,
        entrance_point=form_data.get("entrance_point"),
        title=f"Floor Plan – {strategy}",
        enable_edit_mode=True,
        rule_engine=RuleEngine("ontology/regulation_data.json"),
        ontology_validator=ontology_validator,
    )


if __name__ == "__main__":
    main()