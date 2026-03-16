# NBC 2018 Compliance Solver - Architecture Guide

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│           NBC 2018 BUILDING CODE COMPLIANCE SOLVER                  │
└─────────────────────────────────────────────────────────────────────┘

Level 1: DATA LAYER
┌─────────────────────────────────────────────────────────────────────┐
│  ground_truth.yml                                                   │
│  - Occupancy groups (A-U)                                           │
│  - Construction types (I-A through V-B)                             │
│  - Height/area tables (Table 504.3, 506.2)                         │
│  - Fire-resistance separations (Table 508.4)                       │
│  - Special requirements (egress, accessibility, finishes)          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
Level 2: COMPILATION LAYER
┌─────────────────────────────────────────────────────────────────────┐
│  compiler.py                                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ NBCRuleCompiler                                             │   │
│  │ • Load YAML rules                                           │   │
│  │ • Parse occupancy/construction hierarchies               │   │
│  │ • Extract numeric constraints                            │   │
│  │ • Generate KG nodes & edges                              │   │
│  │ • Formulate CSP constraints                              │   │
│  │ • Validate rule consistency                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Output:                                                            │
│  - knowledge_graph.json (47 nodes, 85 edges)                       │
│  - constraints.json (123 solver constraints)                       │
│  - validation_report.json                                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
Level 3: VALIDATION LAYER
┌─────────────────────────────────────────────────────────────────────┐
│  validator.py                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ ComplianceValidator                                         │   │
│  │ • Load compiled KG & constraints                            │   │
│  │ • Accept building configurations                           │   │
│  │ • Run 5 validation checks:                                 │   │
│  │   1. Height limits                                         │   │
│  │   2. Area limits (with sprinkler mitigation)              │   │
│  │   3. Construction type restrictions                       │   │
│  │   4. Fire-resistance ratings                              │   │
│  │   5. Mixed-occupancy rules                                │   │
│  │ • Detect violations                                        │   │
│  │ • Score compliance (0-100)                                │   │
│  │ • Generate reports                                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Data Classes:                                                      │
│  • BuildingConfiguration: Input design parameters                  │
│  • ComplianceViolation: Individual rule violations                │
│  • ComplianceScore: Overall assessment result                     │
│                                                                     │
│  Output:                                                            │
│  • ComplianceScore object                                          │
│  • Text format report                                              │
│  • JSON format report                                              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
Level 4: INTERFACE LAYER
┌─────────────────────────────────────────────────────────────────────┐
│  Application Integration Points                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │   CLI Tool   │  │   REST API   │  │  BlueprintGPT│             │
│  │              │  │              │  │  Integration │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Model

### Knowledge Graph Nodes

```python
KGNode:
  node_id: str                # Unique identifier
  node_type: str              # "occupancy" | "construction_type" | "property"
  label: str                  # Human-readable name
  properties: Dict[str, Any]  # Type-specific metadata

Example Nodes:
  - ID: "occupancy_A-3"
    Type: "occupancy"
    Label: "Assembly - General"
    Properties: {code: "A-3", capacity_threshold: 100, examples: [...]}

  - ID: "construction_type_II-B"
    Type: "construction_type"
    Label: "Type II - Non-Combustible"
    Properties: {fire_rating_structural_frame: 1, ...}

  - ID: "property_height_limit"
    Type: "property"
    Label: "Height Limit (feet)"
    Properties: {unit: "feet", category: "dimensional"}
```

### Knowledge Graph Edges

```python
KGEdge:
  source_id: str              # From node
  target_id: str              # To node
  edge_type: str              # Relationship type
  label: str                  # Readable description
  properties: Dict            # Metadata (rules, ratings, etc.)

Example Edges:
  - Source: "construction_type_II-B"
    Target: "property_height_limit"
    Type: "has_limit"
    Label: "Type II-B has height limit"
    Properties: {rule_section: "Table 504.3"}

  - Source: "occupancy_A-1"
    Target: "occupancy_R-2"
    Type: "separation_requires"
    Label: "A-1 vs R-2: 0-hr rating"
    Properties: {fire_rating_hours: 0, rule_section: "Table 508.4"}
```

### Solver Constraints

```python
Constraint:
  constraint_id: str          # Unique identifier
  constraint_type: Enum       # EQUALITY | INEQUALITY_* | IMPLICATION | etc.
  variables: List[str]        # Variables involved
  operator: str               # "<", ">", "<=", ">=", "=="
  value: Any                  # Right-hand side value
  rule_references: List[str]  # Citations (e.g., "Table 504.3")

Example Constraints:
  - ID: "constraint_height_II-B_A"
    Type: INEQUALITY_LTE
    Variables: ["building_height_feet"]
    Operator: "<="
    Value: 85
    References: ["Table 504.3 II-B/A"]

  - ID: "constraint_fire_rating_H-2_E"
    Type: INEQUALITY_GTE
    Variables: ["separation_fire_rating_hours"]
    Operator: ">="
    Value: 3
    References: ["Table 508.4 H-2 vs E"]
```

## Module Interfaces

### NBCRuleCompiler

```python
class NBCRuleCompiler:
    def __init__(yaml_path: str) -> None
    def compile() -> Tuple[Dict[KGNode], List[KGEdge], List[Constraint]]
    
    def export_knowledge_graph(output_path: str) -> None
        # Export to JSON
    
    def export_constraints(output_path: str) -> None
        # Export to JSON
    
    def export_validation_report(output_path: str) -> None
        # Export compilation diagnostics

# Usage:
compiler = NBCRuleCompiler("ground_truth/ground_truth.yml")
nodes, edges, constraints = compiler.compile()
compiler.export_knowledge_graph("output/kg.json")
```

### ComplianceValidator

```python
class ComplianceValidator:
    def __init__(kg_file: str, constraints_file: str) -> None
    
    def validate_configuration(config: BuildingConfiguration) -> ComplianceScore
        # Primary validation entry point

    def _check_height_limits(config: BuildingConfiguration) -> None
        # Height constraint checking
    
    def _check_area_limits(config: BuildingConfiguration) -> None
        # Area constraint checking
    
    def _check_construction_type_allowed(config: BuildingConfiguration) -> None
        # Type restrictions
    
    def _check_fire_resistance(config: BuildingConfiguration) -> None
        # Fire-resistance checking
    
    def _check_mixed_occupancy(config: BuildingConfiguration) -> None
        # Mixed-use building rules

# Usage:
validator = ComplianceValidator("kg.json", "constraints.json")
config = BuildingConfiguration(occupancy_group="B", ...)
score = validator.validate_configuration(config)

if score.status == "COMPLIANT":
    print("✓ Building meets all code requirements")
else:
    for v in score.violations:
        print(f"Violation: {v.rule_description}")
```

## Data Flow Diagram

```
User Input
   ↓
BuildingConfiguration object
├── occupancy_group: "B"
├── construction_type: "II-B"
├── height_feet: 75
├── area_sqft: 50000
├── num_stories: 5
└── has_sprinklers: true
   ↓
ComplianceValidator.validate_configuration()
   ├─→ _check_height_limits()      ─→ violation? → ComplianceViolation
   ├─→ _check_area_limits()        ─→ violation? → ComplianceViolation
   ├─→ _check_construction_type()  ─→ violation? → ComplianceViolation
   ├─→ _check_fire_resistance()    ─→ violation? → ComplianceViolation
   └─→ _check_mixed_occupancy()    ─→ violation? → ComplianceViolation
   ↓
ComplianceScore object
├── configuration: BuildingConfiguration
├── score: 100.0 (or less if violations)
├── status: "COMPLIANT" | "NON_COMPLIANT" | "CONDITIONAL"
├── violations: List[ComplianceViolation]
└── details: Dict with summary statistics
   ↓
ComplianceReporter.report_text() or report_json()
   ↓
Output
├── Text Report
│   ├── Building Configuration
│   ├── Compliance Status
│   ├── Violations (if any)
│   └── Remediation suggestions
│
└── JSON Report
    ├── configuration details
    ├── compliance score/status
    ├── violation details
    └── summary statistics
```

## Constraint Satisfaction Problem (CSP) Formulation

### Variables
```
Building Design:
  • occupancy_group ∈ {A-1, A-2, ..., U}
  • construction_type ∈ {I-A, I-B, ..., V-B}
  • building_height_feet ∈ [0, ∞)
  • building_area_sqft ∈ [0, ∞)
  • num_stories ∈ [1, ∞)
  • has_sprinklers ∈ {true, false}
  • fire_separation_hours ∈ [0, 4]

Environmental:
  • mixed_occupancies ∈ 2^{A-U}
  • fire_rating_required ∈ [0, 4]
```

### Constraint Examples

**Height Constraint:**
```
For construction_type = "V-B" AND occupancy_group = "A-3":
  building_height_feet <= 65

∀ type ∀ occupancy:
  IF (construction_type = type) AND (occupancy_group = occupancy)
  THEN building_height_feet <= height_limit[type][occupancy]
```

**Area Constraint with Sprinkler Mitigation:**
```
base_area_limit = area_limit[construction_type][occupancy_group]
sprinkler_factor = 1.0 (no sprinklers) OR 1.25 (with sprinklers)
effective_limit = base_area_limit × sprinkler_factor

building_area_sqft <= effective_limit
```

**Occupancy Type Restriction:**
```
FOR occupancy_group IN {H-1, H-2, H-3, H-4, H-5}:
  construction_type IN {I-A, I-B, II-A}  [Only these types allowed]
```

## Validation Logic Flowchart

```
Start: Building Configuration
  ↓
[Height Check]
  ├─ True (violates) ──→ Add CRITICAL violation → -20 pts
  └─ False ──→ Continue
  ↓
[Area Check]
  ├─ True (base violates)
  │  ├─ Sprinklers installed?
  │  │  ├─ Yes → Recalculate → Check again
  │  │  │  ├─ Still violates → Add MAJOR violation → -10 pts
  │  │  │  └─ Now complies → Continue
  │  │  └─ No → Add CRITICAL violation → -20 pts
  └─ False ──→ Continue
  ↓
[Construction Type Check]
  ├─ H-occupancy in non-I/II type? → Add CRITICAL violation → -20 pts
  ├─ Unusual but allowed? → Add MINOR violation → -5 pts
  └─ Allowed → Continue
  ↓
[Fire Resistance Check]
  ├─ Mixed occupancy without separation? → Add MAJOR/CRITICAL → -10/-20 pts
  └─ Adequate separation → Continue
  ↓
[Mixed Occupancy Check]
  ├─ Classification not mostrestrictive? → Add MINOR violation → -5 pts
  └─ Correct classification → Continue
  ↓
[Calculate Score]
  Score = 100 - Σ(violation_penalties)
  ↓
[Determine Status]
  ├─ Score = 100 → "COMPLIANT"
  ├─ All violations MINOR → "CONDITIONAL"
  └─ Has CRITICAL/MAJOR → "NON_COMPLIANT"
  ↓
End: ComplianceScore
```

## Integration Points

### Future: REST API

```
POST /api/compliance/validate
Content-Type: application/json

{
  "occupancy_group": "B",
  "construction_type": "II-B",
  "height_feet": 75,
  "area_sqft": 50000,
  "num_stories": 5,
  "has_sprinklers": true
}

Response:
{
  "status": "COMPLIANT",
  "score": 100,
  "violations": [],
  "timestamp": "2026-03-16T10:30:00Z"
}
```

### Future: CLI Interface

```bash
$ compliance-check \
    --occupancy B \
    --type II-B \
    --height 75 \
    --area 50000 \
    --stories 5 \
    --sprinklers true \
    --output report.json

✓ COMPLIANT (100.0/100)
Report saved to report.json
```

### Integration with BlueprintGPT

```python
# In BlueprintGPT's building generator:
from ground_truth.validator import ComplianceValidator, BuildingConfiguration

config = BuildingConfiguration(
    occupancy_group=design.occupancy,
    construction_type=design.construction_type,
    height_feet=design.height,
    area_sqft=design.total_area,
    num_stories=design.stories,
    has_sprinklers=design.fire_systems.sprinklers
)

validator = ComplianceValidator(...)
score = validator.validate_configuration(config)

if score.status != "COMPLIANT":
    # Suggest design modifications
    suggestions = generate_compliance_fixes(score.violations, design)
    design.apply_suggestions(suggestions)
```

## Scalability Considerations

### Current Capacity
- Single building analysis: < 100ms
- 47 nodes, 85 edges, 123 constraints
- Suitable for demonstration and small-scale use

### Future Scaling

**For Large Mixed-Occupancy Buildings:**
- Estimated constraints: 10,000+
- Recommended: Z3 SMT solver or OR-Tools
- Expected runtime: 1-10 seconds

**For Real-time Interactive Feedback:**
- Constraint compilation: cache for 24 hours
- Incremental validation: validate only changed parameters
- Parallel checking: multiple configurations simultaneously

**For Nationwide Compliance Database:**
- PostgreSQL for rule storage
- Redis for constraint caching
- Distributed solver farm (multiple nodes)
- API gateway for request routing

---

## Summary

The NBC 2018 Compliance Solver is a **modular, extensible architecture** that:

1. **Extracts** building code rules into structured format (YAML)
2. **Compiles** rules into a semantic knowledge graph + CSP constraints
3. **Validates** building configurations against extracted rules
4. **Reports** violations with remediation suggestions
5. **Scales** from demonstration to enterprise use

Each module is independent, testable, and can be replaced/upgraded independently. The system is designed to integrate with multiple backends (solvers) and frontends (CLI, API, GUI).
