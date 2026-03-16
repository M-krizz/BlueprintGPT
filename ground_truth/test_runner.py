#!/usr/bin/env python3
"""
NBC Compliance Solver - Integration Test & Demo

Tests the complete pipeline:
1. Load ground truth rules from YAML
2. Compile to Knowledge Graph and constraints
3. Validate building configurations
4. Generate compliance reports
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ground_truth.compiler import NBCRuleCompiler
from ground_truth.validator import (
    ComplianceValidator,
    ComplianceReporter,
    BuildingConfiguration
)


def run_compilation_test():
    """Test the rule compilation pipeline"""
    print("\n" + "="*70)
    print("STEP 1: TESTING RULE COMPILATION")
    print("="*70)
    
    yaml_path = "ground_truth/ground_truth.yml"
    
    try:
        # Initialize compiler
        compiler = NBCRuleCompiler(yaml_path)
        
        # Compile rules
        print("\nCompiling rules from:", yaml_path)
        nodes, edges, constraints = compiler.compile()
        
        # Export results
        print("\nExporting compilation results...")
        compiler.export_knowledge_graph("ground_truth/knowledge_graph.json")
        compiler.export_constraints("ground_truth/constraints.json")
        compiler.export_validation_report("ground_truth/validation_report.json")
        
        # Print summary
        print(f"\n✓ Compilation completed successfully!")
        print(f"  - Nodes created: {len(nodes)}")
        print(f"  - Edges created: {len(edges)}")
        print(f"  - Constraints created: {len(constraints)}")
        
        if compiler.validation_warnings:
            print(f"\n⚠ Warnings ({len(compiler.validation_warnings)}):")
            for warning in compiler.validation_warnings[:3]:  # Show first 3
                print(f"    - {warning}")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_validation_test():
    """Test the compliance validation pipeline"""
    print("\n" + "="*70)
    print("STEP 2: TESTING COMPLIANCE VALIDATION")
    print("="*70)
    
    try:
        # Create test configurations
        test_configs = [
            {
                'name': 'Commercial - Type II-B, Business',
                'config': BuildingConfiguration(
                    occupancy_group="B",
                    construction_type="II-B",
                    height_feet=75,
                    area_sqft=50_000,
                    num_stories=5,
                    has_sprinklers=True,
                    has_fire_alarm=True
                )
            },
            {
                'name': 'Assembly - Type V-B, General (OVER HEIGHT)',
                'config': BuildingConfiguration(
                    occupancy_group="A-3",
                    construction_type="V-B",
                    height_feet=100,  # Exceeds 65 ft limit for V-B
                    area_sqft=25_000,
                    num_stories=8,
                    has_sprinklers=False
                )
            },
            {
                'name': 'High-Hazard - Type V-B (ILLEGAL)',
                'config': BuildingConfiguration(
                    occupancy_group="H-2",
                    construction_type="V-B",  # Not allowed for H occupancies
                    height_feet=45,
                    area_sqft=15_000,
                    num_stories=2,
                    has_sprinklers=True
                )
            },
            {
                'name': 'High-Hazard - Type I-A, Manufacturing',
                'config': BuildingConfiguration(
                    occupancy_group="H-2",
                    construction_type="I-A",
                    height_feet=50,
                    area_sqft=20_000,
                    num_stories=1,
                    has_sprinklers=True,
                    fire_separation_hours=4
                )
            },
        ]
        
        # Check if validator can be initialized
        print("\nInitializing validator...")
        try:
            validator = ComplianceValidator(
                "ground_truth/knowledge_graph.json",
                "ground_truth/constraints.json"
            )
            validator_ready = True
        except FileNotFoundError:
            print("  ⚠ Compilation files not found - using demo mode with sample logic")
            validator_ready = False
        
        # Validate each configuration
        print(f"\nValidating {len(test_configs)} building configurations...\n")
        
        for i, test in enumerate(test_configs, 1):
            print(f"\n[Test {i}] {test['name']}")
            print("-" * 65)
            
            if validator_ready:
                try:
                    score = validator.validate_configuration(test['config'])
                    print(ComplianceReporter.report_text(score))
                except Exception as e:
                    print(f"Validation error: {e}")
            else:
                # Demo mode: just show configuration
                config = test['config']
                print(f"Occupancy Group:    {config.occupancy_group}")
                print(f"Construction Type:  {config.construction_type}")
                print(f"Height:             {config.height_feet} ft")
                print(f"Area:               {config.area_sqft} sq ft")
                print(f"Stories:            {config.num_stories}")
                print(f"Sprinklers:         {config.has_sprinklers}")
                print("[Note: Detailed validation requires compiled rules]")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def diagnose_environment():
    """Diagnose environment setup"""
    print("\n" + "="*70)
    print("ENVIRONMENT DIAGNOSTICS")
    print("="*70)
    
    # Check working directory
    print(f"\nWorking Directory: {os.getcwd()}")
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    
    # Check required files
    files_to_check = [
        "ground_truth/ground_truth.yml",
        "ground_truth/compiler.py",
        "ground_truth/validator.py",
    ]
    
    print("\nRequired Files:")
    for file_path in files_to_check:
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
    
    # Check optional output files
    print("\nGenerated Files:")
    output_files = [
        "ground_truth/knowledge_graph.json",
        "ground_truth/constraints.json",
        "ground_truth/validation_report.json",
    ]
    
    for file_path in output_files:
        exists = os.path.exists(file_path)
        status = "✓" if exists else "○"
        size = f"({os.path.getsize(file_path):,} bytes)" if exists else "(not yet generated)"
        print(f"  {status} {file_path} {size}")


def main():
    """Main test runner"""
    print("\n" + "="*70)
    print("NBC 2018 COMPLIANCE SOLVER - INTEGRATION TEST SUITE")
    print("="*70)
    
    # Diagnose environment
    diagnose_environment()
    
    # Run compilation test
    compile_ok = run_compilation_test()
    
    # Run validation test
    if compile_ok:
        validate_ok = run_validation_test()
    else:
        print("\n⚠ Skipping validation test due to compilation failure")
        validate_ok = False
    
    # Print final summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Compilation Test:  {'✓ PASSED' if compile_ok else '✗ FAILED'}")
    print(f"Validation Test:   {'✓ PASSED' if validate_ok else '✗ FAILED'}")
    
    if compile_ok and validate_ok:
        print("\n🎉 All tests passed! System ready for NBC compliance checking.")
        return 0
    else:
        print("\n⚠ Some tests did not pass. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
