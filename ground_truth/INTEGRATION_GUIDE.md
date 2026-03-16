# NBC 2018 Compliance Solver - Integration Guide

## Quick Start for BlueprintGPT Integration

### 1. Installation

Copy the `ground_truth` directory into your BlueprintGPT project:
```bash
cp -r ground_truth /path/to/BlueprintGPT/
```

### 2. Import in Your Code

```python
import sys
from pathlib import Path

# Add ground_truth to path
sys.path.insert(0, str(Path(__file__).parent / 'ground_truth'))

from compiler import NBCRuleCompiler
from validator import ComplianceValidator, BuildingConfiguration
```

### 3. Initialize Components

```python
class BlueprintGPT:
    def __init__(self):
        # Initialize NBC compliance framework
        self.nbc_compiler = NBCRuleCompiler("ground_truth/ground_truth.yml")
        self.nbc_nodes, self.nbc_edges, self.nbc_constraints = self.nbc_compiler.compile()
        self.nbc_validator = ComplianceValidator("ground_truth/constraints.json")
        
        print(f"✓ NBC Framework loaded:")
        print(f"  - {len(self.nbc_nodes)} rules")
        print(f"  - {len(self.nbc_edges)} relationships")
        print(f"  - {len(self.nbc_constraints)} constraints")
```

### 4. Use in Design Process

```python
def check_design_compliance(self, design_spec):
    """Check if proposed design meets NBC requirements"""
    
    # Extract building parameters from design
    config = BuildingConfiguration(
        occupancy_group=design_spec['occupancy'],
        construction_type=design_spec['construction'],
        height_feet=design_spec['height'],
        area_sqft=design_spec['area'],
        num_stories=design_spec['stories'],
        has_sprinklers=design_spec.get('sprinklers', False)
    )
    
    # Check compliance
    is_compliant = self.nbc_validator.validate(config)
    
    if is_compliant:
        return {
            'status': 'COMPLIANT',
            'message': 'Design meets NBC 2018 requirements'
        }
    else:
        violations = self.nbc_validator.get_violations(config)
        return {
            'status': 'NON-COMPLIANT',
            'message': 'Design has violations',
            'violations': violations
        }
```

### 5. Handle Compliance Results

```python
def generate_design_recommendations(self, compliance_result):
    """Generate recommendations based on compliance check"""
    
    if compliance_result['status'] == 'COMPLIANT':
        return "✓ Design is compliant with NBC 2018"
    else:
        recommendations = []
        for violation in compliance_result['violations']:
            recommendations.append(f"- {violation['message']}")
        
        return "\n".join(["✗ Design has compliance issues:"] + recommendations)
```

## Integration Points

### 1. Design Generation Module
- Use validator to screen designs before generating detailed layouts
- Apply constraints from compliance framework

### 2. User Interface
- Display NBC compliance status in design review panel
- Show applicable rules for selected design
- Highlight violations and suggest corrections

### 3. Design Optimization
- Use constraints to guide design parameter optimization
- Consider cost factors from realism module

### 4. Documentation
- Generate compliance reports with applicable rules
- Export design compliance checklist

## API Reference for Integration

### Validator Methods
```python
validator = ComplianceValidator("constraints.json")

# Validate entire configuration
is_compliant = validator.validate(config)

# Get specific violations
violations = validator.validate_height(config)
violations = validator.validate_area(config)
violations = validator.validate_construction_type(config)
violations = validator.validate_fire_resistance(config)

# Get all violations
all_violations = validator.get_violations(config)
```

### Compiler Methods
```python
compiler = NBCRuleCompiler("ground_truth.yml")

# Get all nodes (rules)
nodes = compiler.nodes

# Get all constraints
constraints = compiler.constraints

# Export as JSON
compiler.export_to_json()
```

## Testing Integration

Run the verification suite to ensure integration is working:
```bash
python ground_truth/verify.py
```

Expected output:
```
Summary:
  Total Tests: 8
  Passed:      8
  Failed:      0
  Pass Rate:   100.0%
```

## Troubleshooting

### Issue: "No module named 'compiler'"
**Solution:** Ensure sys.path includes the ground_truth directory
```python
sys.path.insert(0, str(Path(__file__).parent / 'ground_truth'))
```

### Issue: "Ground truth file not found"
**Solution:** Verify file paths are correct
```python
# Use absolute paths
yaml_path = Path(__file__).parent / 'ground_truth' / 'ground_truth.yml'
compiler = NBCRuleCompiler(str(yaml_path))
```

### Issue: Validation always returns False
**Solution:** Check BuildingConfiguration field format
```python
# Occupancy should be single letter codes
occupancy_group="B"  # Correct
occupancy_group="B-2"  # Incorrect

# Construction type should use standard format
construction_type="II-B"  # Correct
construction_type="2B"  # Incorrect
```

## Performance Considerations

- **Compilation:** ~1ms (one-time at startup)
- **Validation:** <1ms per design check
- **Memory:** ~5MB for full rule set
- **Caching:** Consider caching validator results for identical configs

## Production Deployment

1. Copy ground_truth directory to production environment
2. Update file paths if needed
3. Run verify.py to confirm setup
4. Integrate into BlueprintGPT as per examples above
5. Test with sample designs
6. Monitor performance metrics
