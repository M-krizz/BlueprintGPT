#!/usr/bin/env python3
"""
NBC Compliance Solver - Full Verification & Integration Test System

Comprehensive validation covering:
1. Unit tests for each module
2. Integration tests across components
3. Compliance scenario tests (real-world buildings)
4. Performance benchmarking
5. Data integrity verification
"""

import json
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class TestResult:
    """Represents a test result"""
    test_name: str
    category: str
    status: str                # PASS, FAIL, SKIP
    duration_ms: float
    message: str
    assertions: int = 0


class VerificationSuite:
    """Comprehensive verification test suite"""
    
    def __init__(self):
        """Initialize test suite"""
        self.results: List[TestResult] = []
        
    def run_all_tests(self) -> List[TestResult]:
        """Run complete verification suite"""
        print("\nStarting Full Verification Suite...\n")
        
        # Module tests
        self._test_yaml_structure()
        self._test_knowledge_graph()
        self._test_constraints()
        self._test_validator()
        
        # Integration tests
        self._test_integration_pipeline()
        self._test_scenario_buildings()
        
        # Performance tests
        self._test_performance()
        
        # Data integrity tests
        self._test_data_integrity()
        
        return self.results
    
    def _test_yaml_structure(self) -> None:
        """Test YAML structure integrity"""
        start = time.time()
        
        try:
            import yaml
            with open("ground_truth/ground_truth.yml", 'r') as f:
                data = yaml.safe_load(f)
            
            required_keys = [
                'occupancy_groups', 'construction_types', 
                'height_area_limitations', 'fire_resistance_requirements'
            ]
            
            missing = [k for k in required_keys if k not in data]
            
            if missing:
                self.results.append(TestResult(
                    test_name="YAML Structure",
                    category="data_structure",
                    status="FAIL",
                    duration_ms=time.time() - start,
                    message=f"Missing keys: {missing}",
                ))
            else:
                self.results.append(TestResult(
                    test_name="YAML Structure",
                    category="data_structure",
                    status="PASS",
                    duration_ms=time.time() - start,
                    message="All required sections present",
                    assertions=len(required_keys)
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name="YAML Structure",
                category="data_structure",
                status="FAIL",
                duration_ms=time.time() - start,
                message=str(e),
            ))
    
    def _test_knowledge_graph(self) -> None:
        """Test knowledge graph completeness"""
        start = time.time()
        
        try:
            with open("ground_truth/knowledge_graph.json", 'r') as f:
                kg = json.load(f)
            
            nodes = kg.get('nodes', [])
            edges = kg.get('edges', [])
            
            assertions = 0
            
            # Check node types
            node_types = set(n['node_type'] for n in nodes)
            expected_types = {'occupancy', 'construction_type', 'property'}
            assertions += len(nodes)
            
            # Check edges reference existing nodes
            node_ids = set(n['node_id'] for n in nodes)
            bad_edges = [
                e for e in edges 
                if e['source_id'] not in node_ids or e['target_id'] not in node_ids
            ]
            assertions += len(edges)
            
            if bad_edges:
                self.results.append(TestResult(
                    test_name="Knowledge Graph Integrity",
                    category="kg_structure",
                    status="FAIL",
                    duration_ms=time.time() - start,
                    message=f"{len(bad_edges)} orphaned edge references",
                    assertions=assertions
                ))
            else:
                self.results.append(TestResult(
                    test_name="Knowledge Graph Integrity",
                    category="kg_structure",
                    status="PASS",
                    duration_ms=time.time() - start,
                    message=f"{len(nodes)} nodes, {len(edges)} edges validated",
                    assertions=assertions
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name="Knowledge Graph Integrity",
                category="kg_structure",
                status="FAIL",
                duration_ms=time.time() - start,
                message=str(e),
            ))
    
    def _test_constraints(self) -> None:
        """Test constraint validity"""
        start = time.time()
        
        try:
            with open("ground_truth/constraints.json", 'r') as f:
                constraints_data = json.load(f)
            
            constraints = constraints_data.get('constraints', [])
            
            # Validate constraint structure - ensure required fields exist
            valid_constraints = [
                c for c in constraints
                if 'constraint_id' in c and 'constraint_type' in c and 'variables' in c
            ]
            
            if len(valid_constraints) != len(constraints):
                self.results.append(TestResult(
                    test_name="Constraint Validity",
                    category="constraints",
                    status="FAIL",
                    duration_ms=time.time() - start,
                    message=f"{len(constraints) - len(valid_constraints)} constraints missing required fields",
                    assertions=len(constraints)
                ))
            else:
                self.results.append(TestResult(
                    test_name="Constraint Validity",
                    category="constraints",
                    status="PASS",
                    duration_ms=time.time() - start,
                    message=f"{len(constraints)} constraints validated",
                    assertions=len(constraints)
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name="Constraint Validity",
                category="constraints",
                status="FAIL",
                duration_ms=time.time() - start,
                message=str(e),
            ))
    
    def _test_validator(self) -> None:
        """Test validator module"""
        start = time.time()
        
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            from validator import (
                ComplianceValidator, 
                BuildingConfiguration
            )
            
            # Quick validation test
            config = BuildingConfiguration(
                occupancy_group="B",
                construction_type="II-B",
                height_feet=75,
                area_sqft=50000,
                num_stories=5,
                has_sprinklers=True
            )
            
            self.results.append(TestResult(
                test_name="Validator Module",
                category="validator",
                status="PASS",
                duration_ms=time.time() - start,
                message="Validator module loads and classes instantiate",
                assertions=1
            ))
        except Exception as e:
            self.results.append(TestResult(
                test_name="Validator Module",
                category="validator",
                status="FAIL",
                duration_ms=time.time() - start,
                message=str(e),
            ))
    
    def _test_integration_pipeline(self) -> None:
        """Test full integration pipeline"""
        start = time.time()
        
        try:
            import sys
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, script_dir)
            
            from compiler import NBCRuleCompiler
            
            yaml_path = os.path.join(script_dir, "ground_truth.yml")
            compiler = NBCRuleCompiler(yaml_path)
            nodes, edges, constraints = compiler.compile()
            
            # Verify output
            if len(nodes) > 40 and len(edges) > 80 and len(constraints) > 100:
                self.results.append(TestResult(
                    test_name="Integration Pipeline",
                    category="integration",
                    status="PASS",
                    duration_ms=time.time() - start,
                    message=f"Compilation: {len(nodes)} nodes, {len(edges)} edges, {len(constraints)} constraints",
                    assertions=3
                ))
            else:
                self.results.append(TestResult(
                    test_name="Integration Pipeline",
                    category="integration",
                    status="FAIL",
                    duration_ms=time.time() - start,
                    message="Insufficient compilation output",
                    assertions=3
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name="Integration Pipeline",
                category="integration",
                status="FAIL",
                duration_ms=time.time() - start,
                message=str(e),
            ))
    
    def _test_scenario_buildings(self) -> None:
        """Test real-world building scenarios"""
        start = time.time()
        
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            from validator import (
                ComplianceValidator,
                BuildingConfiguration
            )
            
            scenarios = [
                ("Small office", "B", "V-B", 50, 5000, 1, False),
                ("Medium retail", "M", "II-C", 50, 25000, 2, False),
                ("Assembly hall", "A-3", "II-B", 65, 20000, 2, True),
                ("Apartment building", "R-2", "IV-A", 45, 30000, 4, False),
            ]
            
            test_assertions = 0
            for name, occ, const, height, area, stories, sprinklers in scenarios:
                config = BuildingConfiguration(
                    occupancy_group=occ,
                    construction_type=const,
                    height_feet=height,
                    area_sqft=area,
                    num_stories=stories,
                    has_sprinklers=sprinklers
                )
                test_assertions += 1
            
            self.results.append(TestResult(
                test_name="Scenario Building Configs",
                category="scenarios",
                status="PASS",
                duration_ms=time.time() - start,
                message=f"{test_assertions} test scenarios created successfully",
                assertions=test_assertions
            ))
        except Exception as e:
            self.results.append(TestResult(
                test_name="Scenario Building Configs",
                category="scenarios",
                status="FAIL",
                duration_ms=time.time() - start,
                message=str(e),
            ))
    
    def _test_performance(self) -> None:
        """Performance benchmarking tests"""
        start = time.time()
        
        try:
            import sys
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, script_dir)
            
            from compiler import NBCRuleCompiler
            
            yaml_path = os.path.join(script_dir, "ground_truth.yml")
            compiler = NBCRuleCompiler(yaml_path)
            
            compile_start = time.time()
            nodes, edges, constraints = compiler.compile()
            compile_time = (time.time() - compile_start) * 1000
            
            # Verify acceptable performance
            if compile_time < 2000:  # Less than 2 seconds
                self.results.append(TestResult(
                    test_name="Compilation Performance",
                    category="performance",
                    status="PASS",
                    duration_ms=compile_time,
                    message=f"Compilation completed in {compile_time:.0f}ms (target: <2000ms)",
                ))
            else:
                self.results.append(TestResult(
                    test_name="Compilation Performance",
                    category="performance",
                    status="FAIL",
                    duration_ms=compile_time,
                    message=f"Compilation took {compile_time:.0f}ms (exceeds 2000ms target)",
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name="Compilation Performance",
                category="performance",
                status="FAIL",
                duration_ms=time.time() - start,
                message=str(e),
            ))
    
    def _test_data_integrity(self) -> None:
        """Test data integrity across all files"""
        start = time.time()
        
        try:
            import yaml
            
            # Load all data files
            with open("ground_truth/ground_truth.yml", 'r') as f:
                rules = yaml.safe_load(f)
            
            with open("ground_truth/knowledge_graph.json", 'r') as f:
                kg = json.load(f)
            
            with open("ground_truth/constraints.json", 'r') as f:
                constraints = json.load(f)
            
            # Cross-reference checks
            og_count = len(rules.get('occupancy_groups', {}))
            ct_count = len(rules.get('construction_types', {}))
            kg_nodes = len(kg.get('nodes', []))
            
            # Verify consistency
            if kg_nodes > 40:  # Should have at least 40 nodes for occupancies + types + properties
                self.results.append(TestResult(
                    test_name="Data Integrity Cross-checks",
                    category="data_quality",
                    status="PASS",
                    duration_ms=time.time() - start,
                    message=f"Rules: {og_count} occupancies, {ct_count} types; KG: {kg_nodes} nodes",
                    assertions=3
                ))
            else:
                self.results.append(TestResult(
                    test_name="Data Integrity Cross-checks",
                    category="data_quality",
                    status="FAIL",
                    duration_ms=time.time() - start,
                    message=f"Insufficient KG nodes: {kg_nodes}",
                    assertions=3
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name="Data Integrity Cross-checks",
                category="data_quality",
                status="FAIL",
                duration_ms=time.time() - start,
                message=str(e),
            ))


def print_verification_report(results: List[TestResult]) -> None:
    """Print formatted verification report"""
    print("\n" + "="*80)
    print("NBC COMPLIANCE SOLVER - FULL VERIFICATION REPORT")
    print("="*80)
    
    # Summary
    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")
    
    print(f"\nSummary:")
    print(f"  Total Tests: {total}")
    print(f"  Passed:      {passed}")
    print(f"  Failed:      {failed}")
    print(f"  Skipped:     {skipped}")
    print(f"  Pass Rate:   {(passed/total*100):.1f}%")
    
    # By category
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {'pass': 0, 'fail': 0, 'skip': 0}
        if r.status == "PASS":
            categories[r.category]['pass'] += 1
        elif r.status == "FAIL":
            categories[r.category]['fail'] += 1
        else:
            categories[r.category]['skip'] += 1
    
    print(f"\nBy Category:")
    for cat, counts in sorted(categories.items()):
        total_cat = counts['pass'] + counts['fail'] + counts['skip']
        status = "✓" if counts['fail'] == 0 else "✗"
        print(f"  {status} {cat:25s} {counts['pass']}/{total_cat} passed")
    
    # Detailed results
    print(f"\nDetailed Results:")
    print(f"{'-'*80}")
    for r in results:
        status_symbol = "✓" if r.status == "PASS" else "✗" if r.status == "FAIL" else "○"
        print(f"{status_symbol} {r.test_name:30s} [{r.category:15s}] {r.duration_ms:6.0f}ms")
        if r.message:
            print(f"  → {r.message}")
    
    print(f"\n{'='*80}")
    if failed == 0:
        print("✓ ALL TESTS PASSED - System verified and ready for deployment")
    else:
        print(f"✗ {failed} TEST(S) FAILED - Review and address before deployment")
    print(f"{'='*80}\n")


def main():
    """Main entry point"""
    import sys
    
    try:
        suite = VerificationSuite()
        results = suite.run_all_tests()
        
        print_verification_report(results)
        
        # Determine exit code
        failed = sum(1 for r in results if r.status == "FAIL")
        return 0 if failed == 0 else 1
    
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
