import pytest
from constraints.repair_loop import validate_and_repair_spec
from constraints.spec_validator import validate_spec
from generator.layout_generator import generate_layout_from_spec
from generator.ranking import rank_layout_variants
from constraints.compliance_report import build_compliance_report
from visualization.export_svg_blueprint import render_svg_blueprint
from ontology.ontology_bridge import OntologyBridge

def test_full_algorithmic_pipeline():
    spec = {
        "occupancy": "Residential",
        "total_area": 120,
        "area_unit": "sq.m",
        "allocation_strategy": "priority_weights",
        "rooms": [
            {"name": "Bed1", "type": "Bedroom"},
            {"name": "Kit", "type": "Kitchen"},
            {"name": "Liv", "type": "LivingRoom"}
        ],
        "boundary_polygon": [(0, 0), (10, 0), (10, 12), (0, 12)],
        "entrance_point": (0, 6)
    }

    # 1. Validate & Repair
    repaired = validate_and_repair_spec(spec, validate_spec, max_attempts=3)
    spec_repaired = repaired["spec"]
    
    assert repaired["validation"]["valid"] is True

    # 2. Generate Layout
    bridge = OntologyBridge("ontology/regulatory.owl")
    result = generate_layout_from_spec(
        spec_repaired,
        regulation_file="ontology/regulation_data.json",
        ontology_validator=bridge
    )
    
    variants = result.get("layout_variants", [result])
    assert len(variants) > 0

    # 3. Ranking
    ranked_variants, rec_idx = rank_layout_variants(variants)
    best = ranked_variants[rec_idx]
    
    # 4. Compliance Report
    report = build_compliance_report(best)
    assert report is not None
    assert "status" in report
    
    # 5. SVG Render
    svg = render_svg_blueprint(
        best["building"], 
        boundary_polygon=spec_repaired["boundary_polygon"],
        entrance_point=spec_repaired["entrance_point"]
    )
    assert len(svg) > 100
    assert "svg" in svg
