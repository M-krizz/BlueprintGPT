# NBC Compliance Solver - Quick Reference Guide

## File Organization

```
ground_truth/
├── ground_truth.yml              Main rules file (YAML) - input data
├── compiler.py                   Rule → KG compiler
├── validator.py                  Configuration validator
├── test_runner.py                Integration test suite
├── knowledge_graph.json          Compiled KG (generated)
├── constraints.json              Solver constraints (generated)
├── validation_report.json        Compilation report (generated)
├── IMPLEMENTATION_STATUS.md      Detailed status
├── ARCHITECTURE.md               System design
└── QUICK_REFERENCE.md           This file
```

## Quick Start

### 1. Run Tests
```bash
cd d:\Projects\BlueprintGPT
& .\.venv\Scripts\Activate.ps1
python ground_truth\test_runner.py
```

**Expected Output:** "🎉 All tests passed!"

### 2. Check Compliance of a Building
```python
from ground_truth.validator import ComplianceValidator, BuildingConfiguration

# Create building design
config = BuildingConfiguration(
    occupancy_group="B",           # Business
    construction_type="II-B",       # Non-combustible
    height_feet=75,                 # 75 feet tall
    area_sqft=50_000,               # 50,000 sq ft
    num_stories=5,
    has_sprinklers=True
)

# Validate
validator = ComplianceValidator(
    "ground_truth/knowledge_graph.json",
    "ground_truth/constraints.json"
)
score = validator.validate_configuration(config)

# Review results
print(f"Status: {score.status}")              # COMPLIANT | NON_COMPLIANT | CONDITIONAL
print(f"Score: {score.score}/100")            # 0-100
for v in score.violations:
    print(f"  [{v.severity}] {v.rule_description}")
```

### 3. Generate Detailed Report
```python
from ground_truth.validator import ComplianceReporter

# Text report
print(ComplianceReporter.report_text(score))

# JSON report (for APIs, storage)
report_dict = ComplianceReporter.report_json(score)
```

## Key Classes

### BuildingConfiguration
```python
BuildingConfiguration(
    occupancy_group: str,          # "A-1" through "U"
    construction_type: str,         # "I-A" through "V-B"
    height_feet: float,
    area_sqft: float,
    num_stories: int,
    has_sprinklers: bool = False,           # Fire suppression system?
    has_fire_alarm: bool = False,           # Fire alarm system?
    fire_separation_hours: int = 0,         # Hours for mixed occupancy
    mixed_occupancies: List[str] = []       # For mixed-use buildings
)
```

### ComplianceScore
```python
ComplianceScore:
  configuration: BuildingConfiguration
  score: float                     # 0.0 to 100.0
  violations: List[ComplianceViolation]
  status: str                      # "COMPLIANT" | "NON_COMPLIANT" | "CONDITIONAL"
  details: Dict[str, Any]
```

### ComplianceViolation
```python
ComplianceViolation:
  rule_id: str
  rule_description: str
  severity: str                    # "CRITICAL" | "MAJOR" | "MINOR"
  building_value: Any              # What the building has
  required_value: Any              # What the code requires
  remediation: str                 # How to fix it
  rule_section: str                # NBC section/table reference
```

## Occupancy Group Quick Reference

| Code | Name | Examples |
|------|------|----------|
| A-1 | Assembly - Fixed Seating | Theaters, auditoriums, concert halls |
| A-2 | Assembly - Liquid Refreshment | Restaurants, bars, lounges |
| A-3 | Assembly - General | Conference centers, chapels, bowling |
| A-4 | Assembly - Indoor Recreation | Tennis, squash, racquetball courts |
| A-5 | Assembly - Outdoor | Stadiums, bleachers, grandstands |
| B | Business | Offices, professional services |
| E | Educational | Schools, universities, training |
| F-1 | Factory - Moderate Hazard | Manufacturing with moderate fire risk |
| F-2 | Factory - Low Hazard | Manufacturing with low fire risk |
| H-1 | High-Hazard - Detonation | Explosive materials |
| H-2 | High-Hazard - Flammable Solid | Flammable chemical storage |
| H-3 | High-Hazard - Oxidizing | Peroxide, oxidizing agents |
| H-4 | High-Hazard - Pyrophoric | Spontaneously ignitable materials |
| H-5 | High-Hazard - Unstable | Unstable/reactive chemicals |
| I-1 | Institutional - Residential Care | Assisted living, group homes |
| I-2 | Institutional - Hospital | Hospitals, medical care facilities |
| I-3 | Institutional - Detention | Prisons, jails, correctional |
| I-4 | Institutional - Young Children | Daycare, nurseries |
| M | Mercantile | Stores, shopping centers |
| R-1 | Residential - Hotels | Hotels, motels, inns |
| R-2 | Residential - Multi-family | Apartments, condos |
| R-3 | Residential - One/Two Family | Single houses, small homes |
| R-4 | Residential - Care (16+) | Care facilities with 16+ occupants |
| S-1 | Storage - Moderate Hazard | Storage with moderate fire risk |
| S-2 | Storage - Low Hazard | General storage, low risk |
| U | Utility/Miscellaneous | Farm buildings, utility structures |

## Construction Type Quick Reference

| Code | Name | Fire Ratings | Notes |
|------|------|--------------|-------|
| I-A | Fire Resistant | 3/3/3 hrs | Most stringent |
| I-B | Modified | 2/2/2 hrs | Non-combustible |
| II-A | Non-Combustible | 1/1/1 hrs | |
| II-B | Protected | 0/0/0 hrs | Non-combustible unprotected |
| II-C | Unprotected | 0/0/0 hrs | Non-combustible unprotected |
| III-A | Combustible | 1/1/1 hrs | Exterior non-combustible |
| III-B | Unprotected | 0/0/0 hrs | Combustible interior/exterior |
| IV-A | Heavy Timber | 0/2/1 hrs | Unrated frame, rated walls |
| V-A | Protected | 1/1/1 hrs | Combustible with protection |
| V-B | Unprotected | 0/0/0 hrs | Combustible throughout |

*Fire ratings format: Structural Frame / Exterior Walls / Interior Bearing Walls*

## Common Violations & Fixes

### Height Exceeds Limit
**Problem:** Building taller than allowed height for occupation/type combination
**Fixes:**
- Reduce building height
- Upgrade construction type (e.g., V-B → II-B)
- Change occupancy (if feasible)

**Example:** Type V-B assembly building limited to 65 ft; design is 85 ft
→ Upgrade to Type II or reduce height

### Area Exceeds Limit
**Problem:** Floor area exceeds maximum allowed
**Fixes (in order of cost):**
- Install automatic sprinkler system (adds 25% area credit)
- Reduce building footprint
- Upgrade construction type
- Change occupancy

**Example:** Business occupancy max 65,000 sq ft in Type II-C; design is 80,000 sq ft
→ Install sprinklers (allows 81,250 sq ft) OR reduce area OR upgrade type

### Illegal Occupancy/Type Combination
**Problem:** Occupancy not permitted in construction type
**Fixes:**
- Upgrade construction type only (change Type V → Type II or better)
- Cannot keep structure type if occupancy is prohibited

**Example:** H-2 (flammable manufacturing) not allowed in Type V
→ Must use Type I-A, I-B, or II-A

### Inadequate Fire Separation
**Problem:** Mixed-occupancy building lacks required fire-resistance-rated barrier
**Fixes:**
- Install 2-4 hour fire-resistance-rated wall
- Separate occupancies with walls
- Reconfigure to single occupancy

**Example:** A-3 (assembly) floor above B (business) requires 0-hr separation (allowed)
but H-2 floor above B requires 3-hr separation
→ Install rated wall or rearrange occupancies

## Compliance Scoring

### Score Calculation
- Base: 100 points
- CRITICAL violation: -20 points
- MAJOR violation: -10 points  
- MINOR violation: -5 points

### Status Determination
- 100 points: **COMPLIANT** ✓
- >0 points (only MINOR violations): **CONDITIONAL** ⚠
- ≤0 points (has CRITICAL/MAJOR): **NON_COMPLIANT** ✗

### Examples
```
Configuration 1: No violations
  Score: 100 → COMPLIANT ✓

Configuration 2: One MINOR (unusual type choice)
  Score: 95 → CONDITIONAL ⚠
  Reason: Allowed but not typical

Configuration 3: One CRITICAL (height violation)
  Score: 80 → NON_COMPLIANT ✗
  Reason: Must be fixed before approval
```

## Troubleshooting

### "YAML file not found"
**Solution:** Run from project root directory
```bash
cd d:\Projects\BlueprintGPT
python ground_truth\test_runner.py
```

### "knowledge_graph.json not found"
**Solution:** Compile rules first
```bash
python -c "from ground_truth.compiler import NBCRuleCompiler; c = NBCRuleCompiler('ground_truth/ground_truth.yml'); c.compile(); c.export_knowledge_graph('ground_truth/knowledge_graph.json')"
```

### Validator returns wrong results
**Solution:** Ensure you're using latest generated files
```bash
python ground_truth\test_runner.py    # Regenerate all files
```

## Performance Notes

- **Single validation:** < 100ms
- **Compilation (YAML → KG):** < 500ms
- **Report generation:** < 10ms

For high-volume checking (100+ buildings), cache validator instance:
```python
validator = ComplianceValidator("kg.json", "constraints.json")  # Init once
for config in list_of_configs:
    score = validator.validate_configuration(config)  # Reuse
```

## Advanced Usage

### Custom Validation Rules
Extend `ComplianceValidator._check_*` methods:
```python
class CustomValidator(ComplianceValidator):
    def _check_custom_rule(self, config):
        # Your logic here
        violation = ComplianceViolation(...)
        self.violations.append(violation)
```

### Batch Processing
```python
configs = [
    BuildingConfiguration(...),
    BuildingConfiguration(...),
    # ... more configs
]

validator = ComplianceValidator("kg.json", "constraints.json")
results = []
for config in configs:
    score = validator.validate_configuration(config)
    results.append(score)

# Generate summary report
compliant_count = sum(1 for r in results if r.status == "COMPLIANT")
print(f"Compliance rate: {compliant_count}/{len(results)}")
```

## References

- **Implementation Status:** `IMPLEMENTATION_STATUS.md`
- **System Architecture:** `ARCHITECTURE.md`
- **Source Data:** `ground_truth.yml`
- **NBC 2018:** International Building Code, Chapters 3-5

## Support

For issues or questions, refer to:
1. Test suite output: `python ground_truth\test_runner.py`
2. Validation report: `ground_truth/validation_report.json`
3. Architecture documentation: `ARCHITECTURE.md`

---
Last Updated: 2026-03-16  
Status: Implementation Complete (Phases 1-4)  
Next: Phase 5 - Rule Audit
