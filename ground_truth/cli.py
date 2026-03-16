#!/usr/bin/env python3
"""
NBC 2018 Compliance Solver - CLI Tool
Command-line interface for validating buildings against NBC rules
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from compiler import NBCRuleCompiler
from validator import ComplianceValidator, BuildingConfiguration


def main():
    parser = argparse.ArgumentParser(
        description="NBC 2018 Compliance Solver - CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a building
  python cli.py validate --occupancy B --construction II-B --height 75 --area 50000 --stories 5 --sprinklers
  
  # Show rules
  python cli.py show-rules --occupancy B
  
  # Compile rules
  python cli.py compile --input ground_truth.yml --output rules.json
  
  # Run tests
  python cli.py test
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a building design')
    validate_parser.add_argument('--occupancy', required=True, help='Occupancy group')
    validate_parser.add_argument('--construction', required=True, help='Construction type')
    validate_parser.add_argument('--height', type=float, required=True, help='Height in feet')
    validate_parser.add_argument('--area', type=float, required=True, help='Area in sq ft')
    validate_parser.add_argument('--stories', type=int, required=True, help='Number of stories')
    validate_parser.add_argument('--sprinklers', action='store_true', help='Has sprinklers')
    
    # Show rules command
    rules_parser = subparsers.add_parser('show-rules', help='Show applicable rules')
    rules_parser.add_argument('--occupancy', help='Occupancy group (optional)')
    
    # Compile command
    compile_parser = subparsers.add_parser('compile', help='Compile rules from YAML')
    compile_parser.add_argument('--input', default='ground_truth.yml', help='Input YAML file')
    compile_parser.add_argument('--output', default='rules.json', help='Output JSON file')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run verification tests')
    
    args = parser.parse_args()
    
    if args.command == 'validate':
        cmd_validate(args)
    elif args.command == 'show-rules':
        cmd_show_rules(args)
    elif args.command == 'compile':
        cmd_compile(args)
    elif args.command == 'test':
        cmd_test(args)
    else:
        parser.print_help()


def cmd_validate(args):
    """Validate a building design"""
    try:
        # Load validator
        validator = ComplianceValidator("constraints.json")
        
        # Create building configuration
        config = BuildingConfiguration(
            occupancy_group=args.occupancy,
            construction_type=args.construction,
            height_feet=args.height,
            area_sqft=args.area,
            num_stories=args.stories,
            has_sprinklers=args.sprinklers
        )
        
        # Validate
        is_compliant = validator.validate(config)
        
        print(f"\nBuilding Configuration:")
        print(f"  Occupancy Group:    {args.occupancy}")
        print(f"  Construction Type:  {args.construction}")
        print(f"  Height:             {args.height} ft")
        print(f"  Area:               {args.area:,} sq ft")
        print(f"  Stories:            {args.stories}")
        print(f"  Sprinklers:         {'Yes' if args.sprinklers else 'No'}")
        
        if is_compliant:
            print(f"\n✓ COMPLIANT - Building meets NBC 2018 requirements")
        else:
            print(f"\n✗ NON-COMPLIANT - Building has violations")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_show_rules(args):
    """Show applicable rules"""
    try:
        compiler = NBCRuleCompiler("ground_truth.yml")
        nodes, edges, constraints = compiler.compile()
        
        print(f"\nNBC 2018 Rules Summary:")
        print(f"  Total Nodes:        {len(nodes)}")
        print(f"  Total Edges:        {len(edges)}")
        print(f"  Total Constraints:  {len(constraints)}")
        
        if args.occupancy:
            occ_constraints = [c for c in constraints 
                             if args.occupancy in c.get('occupancy_groups', [])]
            print(f"\n  Constraints for {args.occupancy}: {len(occ_constraints)}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_compile(args):
    """Compile rules from YAML"""
    try:
        print(f"Compiling rules from {args.input}...")
        compiler = NBCRuleCompiler(args.input)
        nodes, edges, constraints = compiler.compile()
        
        # Save to output file
        output = {
            "nodes": len(nodes),
            "edges": len(edges),
            "constraints": len(constraints),
            "timestamp": str(Path(__file__).stat().st_mtime)
        }
        
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Compilation complete")
        print(f"  Nodes:        {len(nodes)}")
        print(f"  Edges:        {len(edges)}")
        print(f"  Constraints:  {len(constraints)}")
        print(f"  Output:       {args.output}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_test(args):
    """Run verification tests"""
    try:
        # Import verify module
        from verify import VerificationSuite
        
        suite = VerificationSuite()
        suite.run_all_tests()
        suite.print_report()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
